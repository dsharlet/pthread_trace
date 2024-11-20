#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <optional>
#include <type_traits>
#include <utility>

#include "perfetto.h"
#include "proto.h"

using namespace perfetto;

#define NOINLINE __attribute__((noinline))
#define INLINE __attribute__((always_inline))
#define WEAK __attribute__((weak))

extern "C" {

// In some environments, dlsym itself uses malloc which uses pthreads/sleep, which breaks
// our hooking mechanism. These same environments have these non-standard-looking functions,
// so we can skip dlsym. Mark these as weak in case they don't exist everywhere.
WEAK unsigned int __sleep(unsigned int);
WEAK int __usleep(useconds_t);
WEAK int __nanosleep(const struct timespec*, struct timespec*);
WEAK int __sched_yield();

WEAK int __pthread_cond_broadcast(pthread_cond_t*);
WEAK int __pthread_cond_signal(pthread_cond_t*);
WEAK int __pthread_cond_timedwait(pthread_cond_t*, pthread_mutex_t*, const struct timespec*);
WEAK int __pthread_cond_wait(pthread_cond_t*, pthread_mutex_t*);
WEAK int __pthread_join(pthread_t, void**);
WEAK int __pthread_mutex_lock(pthread_mutex_t*);
WEAK int __pthread_mutex_trylock(pthread_mutex_t*);
WEAK int __pthread_mutex_unlock(pthread_mutex_t*);
WEAK int __pthread_once(pthread_once_t*, void (*)());
WEAK int __pthread_barrier_wait(pthread_barrier_t*);

struct sem_t;
WEAK int __sem_wait(sem_t*);
WEAK int __sem_timedwait(sem_t*, const struct timespec*);
WEAK int __sem_trywait(sem_t*);
WEAK int __sem_post(sem_t*);

}  // extern "C"

namespace {

namespace hooks {

unsigned int (*sleep)(unsigned int) = nullptr;
int (*usleep)(useconds_t) = nullptr;
int (*nanosleep)(const struct timespec*, struct timespec*) = nullptr;
int (*sched_yield)() = nullptr;

int (*pthread_cond_broadcast)(pthread_cond_t*) = nullptr;
int (*pthread_cond_signal)(pthread_cond_t*) = nullptr;
int (*pthread_cond_timedwait)(pthread_cond_t*, pthread_mutex_t*, const struct timespec*) = nullptr;
int (*pthread_cond_wait)(pthread_cond_t*, pthread_mutex_t*) = nullptr;
int (*pthread_join)(pthread_t, void**) = nullptr;
int (*pthread_mutex_lock)(pthread_mutex_t*) = nullptr;
int (*pthread_mutex_trylock)(pthread_mutex_t*) = nullptr;
int (*pthread_mutex_unlock)(pthread_mutex_t*) = nullptr;
int (*pthread_once)(pthread_once_t*, void (*)()) = nullptr;
int (*pthread_barrier_wait)(pthread_barrier_t*) = nullptr;

int (*sem_wait)(sem_t*) = nullptr;
int (*sem_timedwait)(sem_t*, const struct timespec*) = nullptr;
int (*sem_trywait)(sem_t*) = nullptr;
int (*sem_post)(sem_t*) = nullptr;

template <typename T>
NOINLINE void init(T& hook, T def, const char* name, const char* version = nullptr) {
  if (hook) return;

  T result = def;
  if (!result) {
    if (version) {
      result = (T)dlvsym(RTLD_NEXT, name, version);
    } else {
      result = (T)dlsym(RTLD_NEXT, name);
    }
  }
  if (!result) {
    fprintf(stderr, "Failed to find %s\n", name);
    exit(1);
  }

  // This might run on more than one thread, this is a benign race.
  __atomic_store_n(&hook, result, __ATOMIC_RELAXED);
}

}  // namespace hooks

constexpr proto::buffer<2> sequence_flags_cleared(
    {{make_tag(TracePacket::sequence_flags, proto::wire_type::varint), SequenceFlags::INCREMENTAL_STATE_CLEARED}},
    /*size=*/2);

// The trace events we support.
enum class event_type : uint8_t {
  none = 0,

  sleep,
  usleep,
  nanosleep,
  yield,

  cond_broadcast,
  cond_signal,
  cond_timedwait,
  cond_wait,
  join,
  mutex_lock,
  mutex_trylock,
  mutex_unlock,
  once,
  barrier_wait,

  sem_wait,
  sem_timedwait,
  sem_trywait,
  sem_post,

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

constexpr auto slice_begin_sleep = make_slice_begin(event_type::sleep);
constexpr auto slice_begin_usleep = make_slice_begin(event_type::usleep);
constexpr auto slice_begin_nanosleep = make_slice_begin(event_type::nanosleep);
constexpr auto slice_begin_yield = make_slice_begin(event_type::yield);

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
constexpr auto slice_begin_barrier_wait = make_slice_begin(event_type::barrier_wait);

constexpr auto slice_begin_sem_wait = make_slice_begin(event_type::sem_wait);
constexpr auto slice_begin_sem_timedwait = make_slice_begin(event_type::sem_timedwait);
constexpr auto slice_begin_sem_trywait = make_slice_begin(event_type::sem_trywait);
constexpr auto slice_begin_sem_post = make_slice_begin(event_type::sem_post);

const char* to_interned_string(event_type t) {
  switch (t) {
  case event_type::sleep: return "sleep";
  case event_type::usleep: return "usleep";
  case event_type::nanosleep: return "nanosleep";
  case event_type::yield: return "sched_yield";

  case event_type::cond_broadcast: return "pthread_cond_broadcast";
  case event_type::cond_signal: return "pthread_cond_signal";
  case event_type::cond_timedwait: return "pthread_cond_timedwait";
  case event_type::cond_wait: return "pthread_cond_wait";
  case event_type::join: return "pthread_join";
  case event_type::mutex_lock: return "pthread_mutex_lock";
  case event_type::mutex_trylock: return "pthread_mutex_trylock";
  case event_type::mutex_unlock: return "pthread_mutex_unlock";
  case event_type::once: return "pthread_once";
  case event_type::barrier_wait: return "pthread_barrier_wait";

  case event_type::sem_wait: return "sem_wait";
  case event_type::sem_timedwait: return "sem_timedwait";
  case event_type::sem_trywait: return "sem_trywait";
  case event_type::sem_post: return "sem_post";

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
  size_t blocks_ = 0;

  std::atomic<size_t> next_{0};

  void close() {
    if (buffer_) {
      munmap(buffer_, size());
      buffer_ = nullptr;
    }
    if (fd >= 0) {
      size_t blocks = next_.load();
      if (blocks < blocks_) {
        // Remove blocks we didn't write anything to.
        int result = ftruncate(fd, blocks * block_size);
        blocks_ = blocks;
        (void)result;
      }
      ::close(fd);
      fd = -1;

      fprintf(stderr, "pthread_trace: Recorded %zu KB trace, saved most recent %zu KB\n", blocks * block_size_kb,
          blocks_ * block_size_kb);
    }
  }

public:
  circular_file(const char* path, size_t blocks) {
    fd = ::open(path, O_CREAT | O_RDWR | O_TRUNC, S_IRWXU);
    if (fd < 0) {
      fprintf(stderr, "pthread_trace: Error opening file '%s': %s\n", path, strerror(errno));
      exit(1);
    }

    blocks_ = blocks;
    int result = ftruncate(fd, size());
    if (result < 0) {
      close();
      fprintf(stderr, "pthread_trace: Error allocating space in file '%s': %s\n", path, strerror(errno));
      exit(1);
    }

    buffer_ = static_cast<uint8_t*>(mmap(nullptr, size(), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    if (buffer_ == (void*)-1) {
      close();
      fprintf(stderr, "pthread_trace: Error mapping file '%s': %s\n", path, strerror(errno));
      exit(1);
    }
    next_ = 0;

    // Initialize all the blocks with padding.
    for (size_t i = 0; i < size(); i += block_size) {
      proto::write_padding(buffer_ + i, Trace::padding_tag, block_size);
    }
  }
  circular_file(circular_file&& other) {
    std::swap(fd, other.fd);
    std::swap(buffer_, other.buffer_);
    std::swap(blocks_, other.blocks_);
  }

  ~circular_file() { close(); }

  size_t size() const { return blocks_ * block_size; }

  void write_block(const uint8_t* data) {
    size_t offset = (next_.fetch_add(1) % blocks_) * block_size;
    memcpy(buffer_ + offset, data, block_size);
  }
};

// These are optional to avoid non-trivial global constructors. An alternative would be to use
// a global pointer, but we also want to avoid malloc.
std::optional<circular_file> file = std::nullopt;
std::optional<proto::buffer<512>> interned_data = std::nullopt;

const char* getenv_or(const char* env, const char* def) {
  const char* s = getenv(env);
  return s ? s : def;
}

std::atomic<bool> initializing{false};
std::atomic<bool> initialized{false};

NOINLINE void init_trace() {
  // It seems like we could initialize the pthread_once hook, and use it here. However,
  // some pthread_once implementations use pthread_mutex internally (ask me how I know),
  // so we can't use that. We're just going to implement a call_once with spinning :(

  bool not_initializing = false;
  if (initializing.compare_exchange_strong(not_initializing, true)) {
    const char* path = getenv_or("PTHREAD_TRACE_PATH", "pthread_trace.proto");
    const char* buffer_size_str = getenv_or("PTHREAD_TRACE_BUFFER_SIZE_KB", "65536");
    int buffer_size_kb = atoi(buffer_size_str);
    int blocks = (buffer_size_kb + block_size_kb - 1) / block_size_kb;
    if (blocks <= 0) {
      fprintf(stderr, "pthread_trace: Invalid buffer size %d (%s).\n", buffer_size_kb, buffer_size_str);
      exit(1);
    }

    file.emplace(path, blocks);
    fprintf(stderr, "pthread_trace: Writing trace to '%s'\n", path);

    interned_data = proto::buffer<512>();
    write_interned_data(*interned_data);

    initialized = true;
  } else {
    while (!initialized.load()) {
      // Don't use anything that might recursively call init_trace (sched_yield).
    }
  }
}

// Sequence IDs are fixed size 3 byte varints.
using sequence_id_type = std::array<uint8_t, 4>;

static std::atomic<int> next_sequence_id{0};

sequence_id_type new_sequence_id() {
  static constexpr uint8_t tag = make_tag(TracePacket::trusted_packet_sequence_id, proto::wire_type::varint);
  uint64_t id = ++next_sequence_id;
  sequence_id_type result;
  result[0] = tag;
  result[1] = static_cast<uint8_t>(id) | proto::varint_continuation;
  result[2] = static_cast<uint8_t>(id >> 7) | proto::varint_continuation;
  result[3] = static_cast<uint8_t>(id >> 14) & ~proto::varint_continuation;
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
  circular_file* global_buffer;
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

  NOINLINE void write_mutex_track_descriptor(mutex_track& track, const char* type, const void* mutex) {
    flush(256);

    track.mutex = mutex;
    track.sequence_id = new_sequence_id();

    char mutex_locked_str[16];
    snprintf(mutex_locked_str, sizeof(mutex_locked_str), "(acquired by %d)", id);

    proto::buffer<32> event_name;
    event_name.write_tagged(EventName::iid, static_cast<uint64_t>(event_type::mutex_locked));
    event_name.write_tagged(EventName::name, static_cast<const char*>(mutex_locked_str));

    proto::buffer<32> event_names;
    event_names.write_tagged(InternedData::event_names, event_name);

    proto::buffer<512> interned_data;
    interned_data.write_tagged(TracePacket::interned_data, event_names);

    // Make a track id that is a "hash" of the mutex and the type. Collisions are relatively common when mutexes are
    // allocated on the stack.
    const uint64_t track_id = reinterpret_cast<uintptr_t>(mutex) ^ reinterpret_cast<uintptr_t>(type);

    char track_name[32];
    snprintf(track_name, sizeof(track_name), "%s %p", type, mutex);
    write_track_descriptor(track_id, track_name, interned_data, track.sequence_id);
    track.clock.write_clock_snapshot(buffer, track.sequence_id);
  }

  mutex_track& get_mutex_track(const char* type, const void* mutex) {
    for (mutex_track& i : mutex_tracks) {
      if (i.mutex == mutex) {
        return i;
      } else if (!i.mutex) {
        write_mutex_track_descriptor(i, type, mutex);
        return i;
      }
    }
    // Flush the current block and try again.
    fprintf(stderr, "pthread_trace: No track for %s %p, prematurely flushing block with %zu of %zu bytes used\n", type,
        mutex, buffer.size(), buffer.capacity());
    flush();
    return get_mutex_track(type, mutex);
  }

  NOINLINE void begin_block(bool first = false) {
    buffer.clear();
    if (first || clock_id) {
      sequence_id = new_sequence_id();
    }

    write_track_descriptor();
    clock.write_clock_snapshot(buffer, sequence_id);
    memset(mutex_tracks.data(), 0, sizeof(mutex_tracks));
  }

  void write_block() {
    if (!global_buffer || buffer.empty()) return;
    buffer.write_tagged_padding(Trace::padding_tag, block_size - buffer.size());
    assert(buffer.size() == block_size);
    global_buffer->write_block(buffer.data());
  }

  NOINLINE void flush() {
    write_block();
    begin_block();
  }

  void flush(size_t size) {
    constexpr size_t padding_capacity = 2;
    if (buffer.size() + size + padding_capacity >= block_size) {
      flush();
    }
  }

  void construct() {}

  // We need a dummy track to return when hooks are recursively called that eats events.
  static track dummy;
  track(std::false_type) : global_buffer(nullptr) {}

  track() : global_buffer(nullptr), id(next_track_id++) {
    init_trace();
    global_buffer = &*file;
    begin_block(/*first=*/true);
  }

  NOINLINE ~track() noexcept { write_block(); }

public:
  static track& get_thread() {
    // We've eliminated all uses of malloc or other things that might recursively call functions we hook. However, the
    // thread_local destructor mechanism uses malloc, defeating the purpose of all those efforts :( To avoid recursive
    // calls, we need to detect recursive calls and only attempt to initialize the thread_local on the first call.
    thread_local bool recursive = false;
    if (recursive) return dummy;
    recursive = true;
    thread_local track t;
    recursive = false;
    return t;
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

  NOINLINE void write_begin_mutex_locked(const char* type, const void* mutex) {
    constexpr size_t message_capacity = 32;
    flush(message_capacity);

    mutex_track& track = get_mutex_track(type, mutex);

    timestamp_type timestamp;
    track.clock.update_timestamp(timestamp);
    buffer.write_tagged(Trace::trace_packet_tag, track.sequence_id, slice_begin_mutex_locked, timestamp);
  }

  NOINLINE void write_end_mutex_locked(const char* type, const void* mutex) {
    constexpr size_t message_capacity = 32;
    flush(message_capacity);

    mutex_track& track = get_mutex_track(type, mutex);

    timestamp_type timestamp;
    track.clock.update_timestamp(timestamp);
    buffer.write_tagged(Trace::trace_packet_tag, track.sequence_id, slice_end, timestamp);
  }
};

track track::dummy{std::false_type()};

}  // namespace

extern "C" {

unsigned int sleep(unsigned int secs) {
  auto& t = track::get_thread();
  t.write_begin(slice_begin_sleep);
  if (!hooks::sleep) hooks::init(hooks::sleep, __sleep, "sleep");
  unsigned int result = hooks::sleep(secs);
  t.write_end();
  return result;
}

int usleep(useconds_t usecs) {
  auto& t = track::get_thread();
  t.write_begin(slice_begin_usleep);
  if (!hooks::usleep) hooks::init(hooks::usleep, __usleep, "usleep");
  int result = hooks::usleep(usecs);
  t.write_end();
  return result;
}

int nanosleep(const struct timespec* duration, struct timespec* rem) {
  auto& t = track::get_thread();
  t.write_begin(slice_begin_nanosleep);
  if (!hooks::nanosleep) hooks::init(hooks::nanosleep, __nanosleep, "nanosleep");
  int result = hooks::nanosleep(duration, rem);
  t.write_end();
  return result;
}

int sched_yield() {
  auto& t = track::get_thread();
  t.write_begin(slice_begin_yield);
  if (!hooks::sched_yield) hooks::init(hooks::sched_yield, __sched_yield, "sched_yield");
  int result = hooks::sched_yield();
  t.write_end();
  return result;
}

int pthread_cond_broadcast(pthread_cond_t* cond) {
  auto& t = track::get_thread();
  t.write_begin(slice_begin_cond_broadcast);
  if (!hooks::pthread_cond_broadcast)
    hooks::init(hooks::pthread_cond_broadcast, __pthread_cond_broadcast, "pthread_cond_broadcast", "GLIBC_2.3.2");
  int result = hooks::pthread_cond_broadcast(cond);
  t.write_end();
  return result;
}

int pthread_cond_signal(pthread_cond_t* cond) {
  auto& t = track::get_thread();
  t.write_begin(slice_begin_cond_signal);
  if (!hooks::pthread_cond_signal)
    hooks::init(hooks::pthread_cond_signal, __pthread_cond_signal, "pthread_cond_signal", "GLIBC_2.3.2");
  int result = hooks::pthread_cond_signal(cond);
  t.write_end();
  return result;
}

int pthread_cond_timedwait(pthread_cond_t* cond, pthread_mutex_t* mutex, const struct timespec* abstime) {
  // When we wait on a cond var, the mutex gets unlocked, and then relocked before returning.
  auto& t = track::get_thread();
  t.write_end_mutex_locked("mutex", mutex);
  t.write_begin(slice_begin_cond_timedwait);
  if (!hooks::pthread_cond_timedwait)
    hooks::init(hooks::pthread_cond_timedwait, __pthread_cond_timedwait, "pthread_cond_timedwait", "GLIBC_2.3.2");
  int result = hooks::pthread_cond_timedwait(cond, mutex, abstime);
  t.write_end();
  t.write_begin_mutex_locked("mutex", mutex);
  return result;
}

int pthread_cond_wait(pthread_cond_t* cond, pthread_mutex_t* mutex) {
  // When we wait on a cond var, the mutex gets unlocked, and then relocked before returning.
  auto& t = track::get_thread();
  t.write_end_mutex_locked("mutex", mutex);
  t.write_begin(slice_begin_cond_wait);
  if (!hooks::pthread_cond_wait)
    hooks::init(hooks::pthread_cond_wait, __pthread_cond_wait, "pthread_cond_wait", "GLIBC_2.3.2");
  int result = hooks::pthread_cond_wait(cond, mutex);
  t.write_end();
  t.write_begin_mutex_locked("mutex", mutex);
  return result;
}

int pthread_join(pthread_t thread, void** value_ptr) {
  auto& t = track::get_thread();
  t.write_begin(slice_begin_join);
  if (!hooks::pthread_join) hooks::init(hooks::pthread_join, __pthread_join, "pthread_join");
  int result = hooks::pthread_join(thread, value_ptr);
  t.write_end();
  return result;
}

int pthread_mutex_lock(pthread_mutex_t* mutex) {
  auto& t = track::get_thread();
  t.write_begin(slice_begin_mutex_lock);
  if (!hooks::pthread_mutex_lock) hooks::init(hooks::pthread_mutex_lock, __pthread_mutex_lock, "pthread_mutex_lock");
  int result = hooks::pthread_mutex_lock(mutex);
  t.write_end();
  t.write_begin_mutex_locked("mutex", mutex);
  return result;
}

int pthread_mutex_trylock(pthread_mutex_t* mutex) {
  auto& t = track::get_thread();
  t.write_begin(slice_begin_mutex_trylock);
  if (!hooks::pthread_mutex_trylock)
    hooks::init(hooks::pthread_mutex_trylock, __pthread_mutex_trylock, "pthread_mutex_trylock");
  int result = hooks::pthread_mutex_trylock(mutex);
  t.write_end();
  if (result == 0) {
    t.write_begin_mutex_locked("mutex", mutex);
  }
  return result;
}

int pthread_mutex_unlock(pthread_mutex_t* mutex) {
  auto& t = track::get_thread();
  t.write_end_mutex_locked("mutex", mutex);
  t.write_begin(slice_begin_mutex_unlock);
  if (!hooks::pthread_mutex_unlock)
    hooks::init(hooks::pthread_mutex_unlock, __pthread_mutex_unlock, "pthread_mutex_unlock");
  int result = hooks::pthread_mutex_unlock(mutex);
  t.write_end();
  return result;
}

int pthread_once(pthread_once_t* once_control, void (*init_routine)(void)) {
  auto& t = track::get_thread();
  t.write_begin(slice_begin_once);
  if (!hooks::pthread_once) hooks::init(hooks::pthread_once, __pthread_once, "pthread_once");
  int result = hooks::pthread_once(once_control, init_routine);
  t.write_end();
  return result;
}

int pthread_barrier_wait(pthread_barrier_t* barrier) {
  auto& t = track::get_thread();
  t.write_begin(slice_begin_barrier_wait);
  hooks::init(hooks::pthread_barrier_wait, __pthread_barrier_wait, "pthread_barrier_wait");
  int result = hooks::pthread_barrier_wait(barrier);
  t.write_end();
  return result;
}

int sem_wait(sem_t* sem) {
  auto& t = track::get_thread();
  t.write_begin(slice_begin_sem_wait);
  if (!hooks::sem_wait) hooks::init(hooks::sem_wait, __sem_wait, "sem_wait");
  int result = hooks::sem_wait(sem);
  t.write_end();
  t.write_begin_mutex_locked("semaphore", sem);
  return result;
}

int sem_timedwait(sem_t* sem, const struct timespec* abstime) {
  auto& t = track::get_thread();
  t.write_begin(slice_begin_sem_timedwait);
  if (!hooks::sem_timedwait) hooks::init(hooks::sem_timedwait, __sem_timedwait, "sem_timedwait");
  int result = hooks::sem_timedwait(sem, abstime);
  t.write_end();
  if (result == 0) {
    t.write_begin_mutex_locked("semaphore", sem);
  }
  return result;
}

int sem_trywait(sem_t* sem) {
  auto& t = track::get_thread();
  t.write_begin(slice_begin_sem_trywait);
  if (!hooks::sem_trywait) hooks::init(hooks::sem_trywait, __sem_trywait, "sem_trywait");
  int result = hooks::sem_trywait(sem);
  t.write_end();
  if (result == 0) {
    t.write_begin_mutex_locked("semaphore", sem);
  }
  return result;
}

int sem_post(sem_t* sem) {
  auto& t = track::get_thread();
  t.write_end_mutex_locked("semaphore", sem);
  t.write_begin(slice_begin_sem_post);
  if (!hooks::sem_post) hooks::init(hooks::sem_post, __sem_post, "sem_post");
  int result = hooks::sem_post(sem);
  t.write_end();
  return result;
}

}  // extern "C"