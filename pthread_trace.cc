#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <thread>

#include "perfetto.h"
#include "proto.h"

#define NOINLINE __attribute__((noinline))
#define INLINE __attribute__((always_inline))

using namespace perfetto;

namespace {

constexpr proto::buffer<2> sequence_flags_cleared(
    {{make_tag(TracePacket::sequence_flags, proto::wire_type::varint), SequenceFlags::INCREMENTAL_STATE_CLEARED}},
    /*size=*/2);

// The trace events we support.
enum class event_type : uint8_t {
  none = 0,
  cond_broadcast,
  cond_signal,
  cond_timedwait,
  cond_wait,
  join,
  mutex_lock,
  mutex_trylock,
  mutex_unlock,
  once,
  yield,
  sleep,
  usleep,
  nanosleep,
  interned_count,

  mutex_locked,
};

constexpr uint8_t name_iid_tag = make_tag(TrackEvent::name_iid, proto::wire_type::varint);
constexpr uint8_t track_event_tag = make_tag(TracePacket::track_event, proto::wire_type::len);
constexpr uint8_t track_event_type_tag = make_tag(TrackEvent::type, proto::wire_type::varint);

constexpr proto::buffer<6> make_slice_begin(event_type event) {
  return proto::buffer<6>(
      {{track_event_tag, 4, track_event_type_tag, EventType::SLICE_BEGIN, name_iid_tag, static_cast<uint8_t>(event)}},
      /*size=*/6);
}

constexpr proto::buffer<4> slice_end({{track_event_tag, 2, track_event_type_tag, EventType::SLICE_END}}, /*size=*/4);

constexpr auto slice_begin_cond_broadcast = make_slice_begin(event_type::cond_broadcast);
constexpr auto slice_begin_cond_signal = make_slice_begin(event_type::cond_signal);
constexpr auto slice_begin_cond_timedwait = make_slice_begin(event_type::cond_timedwait);
constexpr auto slice_begin_cond_wait = make_slice_begin(event_type::cond_wait);
constexpr auto slice_begin_join = make_slice_begin(event_type::join);
constexpr auto slice_begin_mutex_lock = make_slice_begin(event_type::mutex_lock);
constexpr auto slice_begin_mutex_trylock = make_slice_begin(event_type::mutex_trylock);
constexpr auto slice_begin_mutex_unlock = make_slice_begin(event_type::mutex_unlock);
constexpr auto slice_begin_mutex_locked = make_slice_begin(event_type::mutex_locked);
constexpr auto slice_begin_once = make_slice_begin(event_type::once);
constexpr auto slice_begin_yield = make_slice_begin(event_type::yield);
constexpr auto slice_begin_sleep = make_slice_begin(event_type::sleep);
constexpr auto slice_begin_usleep = make_slice_begin(event_type::usleep);
constexpr auto slice_begin_nanosleep = make_slice_begin(event_type::nanosleep);

const char* to_interned_string(event_type t) {
  switch (t) {
  case event_type::cond_broadcast: return "pthread_cond_broadcast";
  case event_type::cond_signal: return "pthread_cond_signal";
  case event_type::cond_timedwait: return "pthread_cond_timedwait";
  case event_type::cond_wait: return "pthread_cond_wait";
  case event_type::join: return "pthread_join";
  case event_type::mutex_lock: return "pthread_mutex_lock";
  case event_type::mutex_trylock: return "pthread_mutex_trylock";
  case event_type::mutex_unlock: return "pthread_mutex_unlock";
  case event_type::once: return "pthread_once";
  case event_type::yield: return "sched_yield";
  case event_type::sleep: return "sleep";
  case event_type::usleep: return "usleep";
  case event_type::nanosleep: return "nanosleep";
  case event_type::none:
  case event_type::mutex_locked:  // Handled elsewhere, doesn't need to be interned.
  case event_type::interned_count: break;
  }
  return nullptr;
}

template <size_t N>
void write_interned_data(proto::buffer<N>& buf) {
  proto::buffer<N> event_names;
  for (size_t i = 1; i < static_cast<size_t>(event_type::interned_count); ++i) {
    proto::buffer<64> event_name;
    event_name.write_tagged(EventName::iid, i);
    event_name.write_tagged(EventName::name, to_interned_string(static_cast<event_type>(i)));
    event_names.write_tagged(InternedData::event_names, event_name);
  }
  buf.write_tagged(TracePacket::interned_data, event_names);
}

constexpr size_t block_size_kb = 32;
constexpr size_t block_size = block_size_kb * 1024;

class circular_file {
  int fd = 0;
  uint8_t* buffer_ = 0;
  size_t size_ = 0;
  size_t blocks_ = 0;

  std::atomic<size_t> next_{0};

  void close() {
    if (buffer_) {
      munmap(buffer_, size_);
      buffer_ = nullptr;
    }
    if (fd >= 0) {
      size_t size = next_.load();
      if (size < size_) {
        // Remove blocks we didn't write anything to.
        int result = ftruncate(fd, size);
        (void)result;
      }
      ::close(fd);
      fd = -1;

      fprintf(stderr, "pthread_trace: Recorded %zu KB trace\n", size / 1024);
    }
  }

public:
  circular_file(const char* path, size_t blocks) {
    // Don't use O_TRUNC here, it might truncate the file after we started writing it from another thread.
    fd = ::open(path, O_CREAT | O_RDWR | O_TRUNC, S_IRWXU);
    if (fd < 0) {
      fprintf(stderr, "pthread_trace: Error opening file '%s': %s\n", path, strerror(errno));
      exit(1);
    }

    blocks_ = blocks;
    size_ = block_size * blocks;
    int result = ftruncate(fd, size_);
    if (result < 0) {
      close();
      fprintf(stderr, "pthread_trace: Error allocating space in file '%s': %s\n", path, strerror(errno));
      exit(1);
    }

    buffer_ = static_cast<uint8_t*>(mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    if (buffer_ == (void*)-1) {
      close();
      fprintf(stderr, "pthread_trace: Error mapping file '%s': %s\n", path, strerror(errno));
      exit(1);
    }
    next_ = 0;

    // Initialize all the blocks with padding.
    for (size_t i = 0; i < size_; i += block_size) {
      proto::write_padding(buffer_ + i, Trace::padding_tag, block_size);
    }
  }

  ~circular_file() { close(); }

  void write_block(const uint8_t* data) {
    size_t offset = next_.fetch_add(block_size) % size_;
    memcpy(buffer_ + offset, data, block_size);
  }
};

// Don't use globals with non-constant initialization, because global constructors might run after the first thread
// starts tracing.
std::unique_ptr<circular_file> file;
std::unique_ptr<proto::buffer<512>> interned_data;

// Sequence IDs are fixed size 3 byte varints.
using sequence_id_type = std::array<uint8_t, 4>;

static std::atomic<int> next_sequence_id{0};

sequence_id_type new_sequence_id() {
  static constexpr uint8_t tag = make_tag(TracePacket::trusted_packet_sequence_id, proto::wire_type::varint);
  uint64_t id = proto::to_varint(++next_sequence_id);
  sequence_id_type result;
  result[0] = tag;
  result[1] = static_cast<uint8_t>(id) | proto::varint_continuation;
  result[2] = static_cast<uint8_t>(id >> 8) | proto::varint_continuation;
  result[3] = static_cast<uint8_t>(id >> 16) & ~proto::varint_continuation;
  return result;
}

// Set this to 0 to disable incremental timestamps (for debugging).
constexpr uint64_t clock_id = 64;

using timestamp_type = proto::buffer<12>;

class incremental_clock {
  uint64_t t0;

  static uint64_t now_ns() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return static_cast<uint64_t>(t.tv_sec) * 1000000000 + t.tv_nsec;
  }

public:
  // Write a timestamp for the current time for use in the sequence `sequence_id` passed to `write_clock_snapshot`.
  void update_timestamp(timestamp_type& timestamp) {
    uint64_t now = now_ns();
    uint64_t value;
    if (clock_id) {
      value = now - t0;
      t0 = now;
    } else {
      value = now;
    }
    timestamp.clear();
    timestamp.write_tagged(TracePacket::timestamp, value);
  }

  template <size_t N>
  NOINLINE void write_clock_snapshot(proto::buffer<N>& buffer, const sequence_id_type& sequence_id) {
    if (!clock_id) return;

    t0 = now_ns();

    proto::buffer<32> clock;
    clock.write_tagged(Clock::clock_id, clock_id);
    clock.write_tagged(Clock::timestamp, static_cast<uint64_t>(0));
    clock.write_tagged(Clock::is_incremental, true);

    proto::buffer<32> boottime_clock;
    boottime_clock.write_tagged(Clock::clock_id, static_cast<uint64_t>(BuiltinClocks::BOOTTIME));
    boottime_clock.write_tagged(Clock::timestamp, t0);

    proto::buffer<64> clocks;
    clocks.write_tagged(ClockSnapshot::clocks, clock);
    clocks.write_tagged(ClockSnapshot::clocks, boottime_clock);

    proto::buffer<64> clock_snapshot;
    clock_snapshot.write_tagged(TracePacket::clock_snapshot, clocks);

    buffer.write_tagged(Trace::trace_packet_tag, clock_snapshot, sequence_id);
  }
};

static std::atomic<int> next_track_id{0};

class track {
  int id;
  std::array<uint8_t, 4> sequence_id;

  proto::buffer<block_size> buffer;

  // The previous timestamp emitted on this thread.
  incremental_clock clock;

  // We write mutex events to a track with an id that includes the mutex pointer. This means we can write mutex events
  // to this thread's buffer. This avoids needing to find a way to synchronize access to a shared structure for mutex
  // events. The trace parser will reassemble the events from the different tracks (with the same IDs) into one track.
  struct mutex_track {
    const void* mutex;
    sequence_id_type sequence_id;
    incremental_clock clock;
  };

  // Remember the sequence ID and clock for each mutex we see in this block.
  static constexpr size_t mutexes_per_thread = 256;
  std::array<mutex_track, mutexes_per_thread> mutex_tracks;

  NOINLINE void write_track_descriptor(
      uint64_t id, const char* name_str, const proto::buffer<512>& interned_data, const sequence_id_type& sequence_id) {
    proto::buffer<12> uuid;
    uuid.write_tagged(TrackDescriptor::uuid, id);

    proto::buffer<256> name;
    name.write_tagged(TrackDescriptor::name, name_str);

    proto::buffer<256> track_descriptor;
    track_descriptor.write_tagged(TracePacket::track_descriptor, uuid, name);

    proto::buffer<8> timestamp_clock_id;
    if (clock_id) {
      timestamp_clock_id.write_tagged(TracePacketDefaults::timestamp_clock_id, clock_id);
    }

    proto::buffer<12> track_uuid;
    track_uuid.write_tagged(TrackEventDefaults::track_uuid, id);

    proto::buffer<16> track_event_defaults;
    track_event_defaults.write_tagged(TracePacketDefaults::track_event_defaults, track_uuid);

    proto::buffer<32> trace_packet_defaults;
    trace_packet_defaults.write_tagged(TracePacket::trace_packet_defaults, timestamp_clock_id, track_event_defaults);

    buffer.write_tagged(Trace::trace_packet_tag, track_descriptor, trace_packet_defaults, interned_data, sequence_id,
        sequence_flags_cleared);
  }

  void write_track_descriptor() {
    // Use the thread id as the thread name, pad it so it sorts alphabetically in numerical order.
    // TODO: Use real thread names?
    char thread_name[16];
    snprintf(thread_name, sizeof(thread_name), "thread %04d", id);

    write_track_descriptor(id, thread_name, *interned_data, sequence_id);
  }

  NOINLINE void write_mutex_track_descriptor(mutex_track& track, const void* mutex) {
    flush(256);

    track.mutex = mutex;
    track.sequence_id = new_sequence_id();

    char mutex_locked_str[16];
    snprintf(mutex_locked_str, sizeof(mutex_locked_str), "(locked by %d)", id);

    proto::buffer<32> event_name;
    event_name.write_tagged(static_cast<uint64_t>(EventName::iid), static_cast<uint64_t>(event_type::mutex_locked));
    event_name.write_tagged(EventName::name, static_cast<const char*>(mutex_locked_str));

    proto::buffer<32> event_names;
    event_names.write_tagged(InternedData::event_names, event_name);

    proto::buffer<512> interned_data;
    interned_data.write_tagged(TracePacket::interned_data, event_names);

    char track_name[32];
    snprintf(track_name, sizeof(track_name), "mutex %p", mutex);
    write_track_descriptor(reinterpret_cast<uint64_t>(mutex), track_name, interned_data, track.sequence_id);
    track.clock.write_clock_snapshot(buffer, track.sequence_id);
  }

  mutex_track& get_mutex_track(const void* mutex) {
    for (mutex_track& i : mutex_tracks) {
      if (i.mutex == mutex) {
        return i;
      } else if (!i.mutex) {
        write_mutex_track_descriptor(i, mutex);
        return i;
      }
    }
    // Flush the current block and try again.
    fprintf(stderr, "pthread_trace: No track for mutex %p, prematurely flushing block with %zu of %zu bytes used\n",
        mutex, buffer.size(), buffer.capacity());
    flush();
    return get_mutex_track(mutex);
  }

  NOINLINE void begin_block(bool first = false) {
    if (first || clock_id) {
      sequence_id = new_sequence_id();
    }

    write_track_descriptor();
    clock.write_clock_snapshot(buffer, sequence_id);
    memset(mutex_tracks.data(), 0, sizeof(mutex_tracks));
  }

  NOINLINE void flush() {
    buffer.write_tagged_padding(Trace::padding_tag, block_size - buffer.size());
    assert(buffer.size() == block_size);
    file->write_block(buffer.data());
    buffer.clear();
    begin_block();
  }

  void flush(size_t size) {
    constexpr size_t padding_capacity = 2;
    if (buffer.size() + size + padding_capacity >= block_size) {
      flush();
    }
  }

public:
  NOINLINE track() : id(next_track_id++) { begin_block(/*first=*/true); }

  NOINLINE ~track() {
    if (buffer.size() > 0) {
      buffer.write_tagged_padding(Trace::padding_tag, block_size - buffer.size());
      assert(buffer.size() == block_size);
      file->write_block(buffer.data());
    }
  }

  // This is inline so we can see constexpr track_events.
  template <typename TrackEvent>
  INLINE void write_begin(const TrackEvent& track_event) {
    constexpr size_t message_capacity = 32;
    flush(message_capacity);

    timestamp_type timestamp;
    clock.update_timestamp(timestamp);
    buffer.write_tagged(Trace::trace_packet_tag, sequence_id, track_event, timestamp);
  }

  NOINLINE void write_end() {
    constexpr size_t message_capacity = 32;
    flush(message_capacity);

    timestamp_type timestamp;
    clock.update_timestamp(timestamp);
    buffer.write_tagged(Trace::trace_packet_tag, sequence_id, slice_end, timestamp);
  }

  // Write a (mutex %p locked) event. These always have the same timestamp as the previous event.
  NOINLINE void write_begin_mutex_locked(const void* mutex) {
    constexpr size_t message_capacity = 32;
    flush(message_capacity);

    mutex_track& track = get_mutex_track(mutex);

    timestamp_type timestamp;
    track.clock.update_timestamp(timestamp);
    buffer.write_tagged(Trace::trace_packet_tag, track.sequence_id, slice_begin_mutex_locked, timestamp);
  }

  NOINLINE void write_end_mutex_locked(const void* mutex) {
    constexpr size_t message_capacity = 32;
    flush(message_capacity);

    mutex_track& track = get_mutex_track(mutex);

    timestamp_type timestamp;
    track.clock.update_timestamp(timestamp);
    buffer.write_tagged(Trace::trace_packet_tag, track.sequence_id, slice_end, timestamp);
  }

  static track& get_thread() {
    thread_local track t;
    return t;
  }
};

const char* getenv_or(const char* env, const char* def) {
  const char* s = getenv(env);
  return s ? s : def;
}

pthread_once_t init_once = PTHREAD_ONCE_INIT;

NOINLINE void init_trace() {
  // Use the real pthread_once to handle initialization.
  typedef int (*once_fn_t)(pthread_once_t*, void (*)());
  once_fn_t real_once = (once_fn_t)dlsym(RTLD_NEXT, "pthread_once");

  real_once(&init_once, []() {
    const char* path = getenv_or("PTHREAD_TRACE_PATH", "pthread_trace.proto");
    const char* buffer_size_str = getenv_or("PTHREAD_TRACE_BUFFER_SIZE_KB", "65536");
    int buffer_size_kb = atoi(buffer_size_str);
    int blocks = (buffer_size_kb + block_size_kb - 1) / block_size_kb;
    if (blocks <= 0) {
      fprintf(stderr, "pthread_trace: Invalid buffer size %d (%s).\n", buffer_size_kb, buffer_size_str);
      exit(1);
    }

    file = std::make_unique<circular_file>(path, blocks);
    fprintf(stderr, "pthread_trace: Writing trace to '%s'\n", path);

    interned_data = std::make_unique<proto::buffer<512>>();
    write_interned_data(*interned_data);
  });
}

namespace hooks {

unsigned int (*sleep)(unsigned int) = nullptr;
int (*usleep)(useconds_t) = nullptr;
int (*nanosleep)(const struct timespec*, struct timespec*) = nullptr;
int (*sched_yield)() = nullptr;
int (*pthread_cond_broadcast)(pthread_cond_t*) = nullptr;
int (*pthread_cond_signal)(pthread_cond_t*) = nullptr;
int (*pthread_cond_timedwait)(pthread_cond_t* cond, pthread_mutex_t* mutex, const struct timespec* abstime) = nullptr;
int (*pthread_cond_wait)(pthread_cond_t* cond, pthread_mutex_t* mutex) = nullptr;
int (*pthread_join)(pthread_t thread, void** value_ptr) = nullptr;
int (*pthread_mutex_lock)(pthread_mutex_t*) = nullptr;
int (*pthread_mutex_trylock)(pthread_mutex_t*) = nullptr;
int (*pthread_mutex_unlock)(pthread_mutex_t*) = nullptr;
int (*pthread_once)(pthread_once_t* once_control, void (*init_routine)()) = nullptr;

template <typename T>
NOINLINE void init(T& hook, const char* name, const char* version = nullptr) {
  init_trace();
  T result;
  if (version) {
    result = (T)dlvsym(RTLD_NEXT, name, version);
  } else {
    result = (T)dlsym(RTLD_NEXT, name);
  }
  if (!result) {
    fprintf(stderr, "Failed to find %s\n", name);
    exit(1);
  }

  // This might run on more than one thread, this is a benign race.
  __atomic_store_n(&hook, result, __ATOMIC_RELAXED);
}

}  // namespace hooks

}  // namespace

extern "C" {

unsigned int sleep(unsigned int secs) {
  if (!hooks::sleep) hooks::init(hooks::sleep, "sleep");

  auto& t = track::get_thread();
  t.write_begin(slice_begin_sleep);
  unsigned int result = hooks::sleep(secs);
  t.write_end();  // slice_begin_sleep
  return result;
}

int usleep(useconds_t usecs) {
  if (!hooks::usleep) hooks::init(hooks::usleep, "usleep");

  auto& t = track::get_thread();
  t.write_begin(slice_begin_usleep);
  int result = hooks::usleep(usecs);
  t.write_end();  // slice_begin_usleep
  return result;
}

int nanosleep(const struct timespec* duration, struct timespec* rem) {
  if (!hooks::nanosleep) hooks::init(hooks::nanosleep, "nanosleep");

  auto& t = track::get_thread();
  t.write_begin(slice_begin_nanosleep);
  int result = hooks::nanosleep(duration, rem);
  t.write_end();  // slice_begin_nanosleep
  return result;
}

int sched_yield() {
  if (!hooks::sched_yield) hooks::init(hooks::sched_yield, "sched_yield");

  auto& t = track::get_thread();
  t.write_begin(slice_begin_yield);
  int result = hooks::sched_yield();
  t.write_end();  // slice_begin_yield
  return result;
}

int pthread_cond_broadcast(pthread_cond_t* cond) {
  if (!hooks::pthread_cond_broadcast)
    hooks::init(hooks::pthread_cond_broadcast, "pthread_cond_broadcast", "GLIBC_2.3.2");

  auto& t = track::get_thread();
  t.write_begin(slice_begin_cond_broadcast);
  int result = hooks::pthread_cond_broadcast(cond);
  t.write_end();  // slice_begin_cond_broadcast
  return result;
}

int pthread_cond_signal(pthread_cond_t* cond) {
  if (!hooks::pthread_cond_signal) hooks::init(hooks::pthread_cond_signal, "pthread_cond_signal", "GLIBC_2.3.2");

  auto& t = track::get_thread();
  t.write_begin(slice_begin_cond_signal);
  int result = hooks::pthread_cond_signal(cond);
  t.write_end();  // slice_begin_cond_signal
  return result;
}

int pthread_cond_timedwait(pthread_cond_t* cond, pthread_mutex_t* mutex, const struct timespec* abstime) {
  if (!hooks::pthread_cond_timedwait)
    hooks::init(hooks::pthread_cond_timedwait, "pthread_cond_timedwait", "GLIBC_2.3.2");

  // When we wait on a cond var, the mutex gets unlocked, and then relocked before returning.
  auto& t = track::get_thread();
  t.write_end_mutex_locked(mutex);
  t.write_begin(slice_begin_cond_timedwait);
  int result = hooks::pthread_cond_timedwait(cond, mutex, abstime);
  t.write_end();  // slice_begin_cond_timedwait
  t.write_begin_mutex_locked(mutex);
  return result;
}

int pthread_cond_wait(pthread_cond_t* cond, pthread_mutex_t* mutex) {
  if (!hooks::pthread_cond_wait) hooks::init(hooks::pthread_cond_wait, "pthread_cond_wait", "GLIBC_2.3.2");

  // When we wait on a cond var, the mutex gets unlocked, and then relocked before returning.
  auto& t = track::get_thread();
  t.write_end_mutex_locked(mutex);
  t.write_begin(slice_begin_cond_wait);
  int result = hooks::pthread_cond_wait(cond, mutex);
  t.write_end();  // slice_begin_cond_wait
  t.write_begin_mutex_locked(mutex);
  return result;
}

int pthread_join(pthread_t thread, void** value_ptr) {
  if (!hooks::pthread_join) hooks::init(hooks::pthread_join, "pthread_join");

  auto& t = track::get_thread();
  t.write_begin(slice_begin_join);
  int result = hooks::pthread_join(thread, value_ptr);
  t.write_end();  // slice_begin_join
  return result;
}

int pthread_mutex_lock(pthread_mutex_t* mutex) {
  if (!hooks::pthread_mutex_lock) hooks::init(hooks::pthread_mutex_lock, "pthread_mutex_lock");

  auto& t = track::get_thread();
  t.write_begin(slice_begin_mutex_lock);
  int result = hooks::pthread_mutex_lock(mutex);
  t.write_end();  // slice_begin_mutex_lock
  t.write_begin_mutex_locked(mutex);
  return result;
}

int pthread_mutex_trylock(pthread_mutex_t* mutex) {
  if (!hooks::pthread_mutex_trylock) hooks::init(hooks::pthread_mutex_trylock, "pthread_mutex_trylock");

  auto& t = track::get_thread();
  t.write_begin(slice_begin_mutex_trylock);
  int result = hooks::pthread_mutex_trylock(mutex);
  t.write_end();  // slice_begin_mutex_trylock
  if (result == 0) {
    t.write_begin_mutex_locked(mutex);
  }
  return result;
}

int pthread_mutex_unlock(pthread_mutex_t* mutex) {
  if (!hooks::pthread_mutex_unlock) hooks::init(hooks::pthread_mutex_unlock, "pthread_mutex_unlock");

  auto& t = track::get_thread();
  t.write_end_mutex_locked(mutex);
  t.write_begin(slice_begin_mutex_unlock);
  int result = hooks::pthread_mutex_unlock(mutex);
  t.write_end();  // slice_begin_mutex_unlock
  return result;
}

int pthread_once(pthread_once_t* once_control, void (*init_routine)(void)) {
  if (!hooks::pthread_once) hooks::init(hooks::pthread_once, "pthread_once");

  auto& t = track::get_thread();
  t.write_begin(slice_begin_once);
  int result = hooks::pthread_once(once_control, init_routine);
  t.write_end();  // slice_begin_once
  return result;
}

}  // extern "C"