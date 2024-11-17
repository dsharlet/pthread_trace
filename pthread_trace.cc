#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <assert.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <sched.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <thread>

#define NOINLINE __attribute__((noinline))

namespace {

size_t sum() { return 0; }
template <class... Args>
size_t sum(size_t a, Args... args) {
  return a + sum(args...);
}

namespace proto {

enum class wire_type {
  varint = 0,
  i64 = 1,
  len = 2,
  i32 = 5,
};

size_t write_varint(uint8_t* dst, uint64_t value) {
  constexpr uint8_t continuation = 0x80;
  // clang-format on
  size_t result = 0;
  while (value > 0x7f) {
    dst[result++] = static_cast<uint8_t>(value | continuation);
    value >>= 7;
  }
  dst[result++] = static_cast<uint8_t>(value);
  return result;
}

size_t write_tag(uint8_t* dst, uint64_t tag, wire_type type) {
  return write_varint(dst, (tag << 3) | static_cast<uint64_t>(type));
}

size_t write_padding(uint8_t* dst, uint64_t tag, uint64_t size) {
  size_t result = 0;
  while (size != 0) {
    size_t tag_size = write_tag(dst, tag, wire_type::len);
    dst += tag_size;
    result += tag_size;
    assert(size >= tag_size);
    size -= tag_size;
    // We need to write a size that includes the bytes occupied by itself, which is tricky.
    // Trial and error seems like the way to go.
    for (size_t attempt = 1; attempt < 10; ++attempt) {
      size_t actual = write_varint(dst, size - attempt);
      if (actual == attempt) {
        // We wrote the right number of bytes for this size.
        return result + size;
      }
    }
    // If we got here, the size bumped over a threshold of varint size. Write 0 and try again.
    *dst++ = 0;
    result += 1;
    size -= 1;
  }
  return result;
}

// Writing protobufs is a bit tricky, because you need to know the size of child messages before writing the parent
// message header. The approach used here is to use fixed size stack buffers for everything, and just copy them into
// nested buffers after constructing them (so we know the size). This approach would be bad for deeply nested protos,
// but we just don't have that much nesting in this case.
template <size_t Capacity>
class buffer {
  std::array<uint8_t, Capacity> buf_;
  size_t size_ = 0;

  // varint is 7 bits at a time, with the MSB indicating if there is another 7
  // bits remaining.
  void write_varint(uint64_t value) { size_ += proto::write_varint(&buf_[size_], value); }

  // sint uses "zigzag" encoding: positive x -> 2*x, negative x -> -2*x - 1
  // void write_varint(buffer& buf, int64_t value) {
  // write_varint(buf, static_cast<uint64_t>(value < 0 ? -2 * value - 1 : 2 *
  // value));
  //}

  constexpr buffer(uint8_t d0, uint8_t d1) : buf_({{d0, d1}}), size_(2) {}

  void write_all() {}

  template <typename Field0, typename... Fields>
  void write_all(const Field0& first, const Fields&... rest) {
    write(first);
    write_all(rest...);
  }

public:
  constexpr buffer() : size_(0) {}

  static constexpr size_t capacity() { return Capacity; }

  static constexpr buffer<2> make(uint8_t tag, uint8_t value) {
    uint8_t tag_type = static_cast<uint8_t>(tag << 3) | static_cast<uint8_t>(wire_type::varint);
    return buffer<2>(tag_type, value);
  }

  size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }
  const uint8_t* data() const { return buf_.data(); }
  void clear() { size_ = 0; }

  // Write objects directly to the buffer, without any tags.
  void write(const void* s, size_t n) {
    assert(size_ + n <= Capacity);
    std::memcpy(&buf_[size_], s, n);
    size_ += n;
  }

  template <size_t M>
  void write(const buffer<M>& buf) {
    write(buf.data(), buf.size());
  }

  void write(const buffer<2>& buf) {
    assert(buf.size() == 2);
    assert(size_ + 2 <= Capacity);
    memcpy(&buf_[size_], buf.data(), 2);
    size_ += 2;
  }
  void write(const buffer<4>& buf) {
    assert(buf.size() >= 2);
    assert(size_ + buf.size() <= Capacity);
    switch (buf.size()) {
    case 4: memcpy(&buf_[size_], buf.data(), 4); break;
    case 3: memcpy(&buf_[size_], buf.data(), 3); break;
    case 2: memcpy(&buf_[size_], buf.data(), 2); break;
    }
    size_ += buf.size();
  }

  // Write a tag.
  void write_tag(uint64_t tag, wire_type type) { write_varint((tag << 3) | static_cast<uint64_t>(type)); }

  // Write tagged values.
  void write_tagged(uint64_t tag, uint64_t value) {
    write_tag(tag, wire_type::varint);
    write_varint(value);
  }

  void write_tagged(uint64_t tag, bool value) {
    write_tag(tag, wire_type::varint);
    write_varint(value);
  }

  // void write(uint64_t tag, int64_t value) {
  //   write_tag(tag, wire_type::varint);
  //   write_varint(value);
  // }

  void write_tagged_padding(uint64_t tag, uint64_t size) { size_ += write_padding(&buf_[size_], tag, size); }

  void write_tagged(uint64_t tag, const char* str) {
    write_tag(tag, wire_type::len);
    std::size_t len = strlen(str);
    write_varint(len);
    write(str, len);
  }

  template <typename... Fields>
  void write_tagged(uint64_t tag, const Fields&... fields) {
    write_tag(tag, wire_type::len);
    write_varint(sum(fields.size()...));
    write_all(fields...);
  }
};

}  // namespace proto

namespace perfetto {

// These protobuf messages are for 'perfetto', excerpted from
// https://cs.android.com/android/platform/superproject/main/+/main:external/perfetto/protos/perfetto/trace/trace_packet.proto

enum class EventName {
  /*optional uint64*/ iid = 1,
  /*optional string*/ name = 2,
};

enum class InternedData {
  /*repeated EventName*/ event_names = 2,
};

enum class TrackDescriptor {
  /*optional uint64*/ uuid = 1,
  /*optional uint64*/ parent_uuid = 5,
  // oneof {
  /*optional string*/ name = 2,
  /*optional string*/ static_name = 10,
  // }
  /*optional ProcessDescriptor*/ process = 3,
  /*optional ThreadDescriptor*/ thread = 4,
};

enum class ThreadDescriptor {
  /*optional int32*/ pid = 1,
  /*optional int32*/ tid = 2,
  /*optional string*/ thread_name = 5,
};

enum class EventType {
  UNSPECIFIED = 0,
  SLICE_BEGIN = 1,
  SLICE_END = 2,
  INSTANT = 3,
  COUNTER = 4,
};

enum class TrackEvent {
  /*uint64*/ name_iid = 10,
  /*string*/ name = 23,
  /*optional EventType*/ type = 9,
  /*optional uint64*/ track_uuid = 11,
  /*repeated uint64*/ extra_counter_track_uuids = 31,
  /*repeated int64*/ extra_counter_values = 12,
};

enum class SequenceFlags {
  UNSPECIFIED = 0,
  INCREMENTAL_STATE_CLEARED = 1,
  NEEDS_INCREMENTAL_STATE = 2,
};

enum class TracePacket {
  /*ClockSnapshot*/ clock_snapshot = 6,
  /*optional uint64*/ timestamp = 8,
  /*optional uint32*/ timestamp_clock_id = 58,
  /*TrackEvent*/ track_event = 11,
  /*TrackDescriptor*/ track_descriptor = 60,
  /*optional TracePacketDefaults*/ trace_packet_defaults = 59,
  /*uint32*/ trusted_packet_sequence_id = 10,
  /*optional InternedData*/ interned_data = 12,
  /*optional uint32*/ sequence_flags = 13,
};

enum class TracePacketDefaults {
  /*optional uint32*/ timestamp_clock_id = 58,
};

// https://cs.android.com/android/platform/superproject/main/+/main:external/perfetto/protos/perfetto/trace/clock_snapshot.proto
enum class Clock {
  /*optional uint32*/ clock_id = 1,
  /*optional uint64*/ timestamp = 2,
  /*optional bool*/ is_incremental = 3,
};

enum class BuiltinClocks {
  UNKNOWN = 0,
  REALTIME = 1,
  REALTIME_COARSE = 2,
  MONOTONIC = 3,
  MONOTONIC_COARSE = 4,
  MONOTONIC_RAW = 5,
  BOOTTIME = 6,
  PROCESS_CPUTIME = 7,
  THREAD_CPUTIME = 8,
  BUILTIN_CLOCK_MAX_ID = 63,
};

// A snapshot of clock readings to allow for trace alignment.
enum class ClockSnapshot {
  /*repeated Clock*/ clocks = 1,
};

constexpr auto sequence_flags_cleared = proto::buffer<2>::make(static_cast<uint64_t>(TracePacket::sequence_flags),
    static_cast<uint64_t>(SequenceFlags::INCREMENTAL_STATE_CLEARED));
constexpr auto sequence_flags_needed = proto::buffer<2>::make(
    static_cast<uint64_t>(TracePacket::sequence_flags), static_cast<uint64_t>(SequenceFlags::NEEDS_INCREMENTAL_STATE));

constexpr auto slice_begin =
    proto::buffer<2>::make(static_cast<uint64_t>(TrackEvent::type), static_cast<uint64_t>(EventType::SLICE_BEGIN));
constexpr auto slice_end =
    proto::buffer<2>::make(static_cast<uint64_t>(TrackEvent::type), static_cast<uint64_t>(EventType::SLICE_END));
constexpr auto instant =
    proto::buffer<2>::make(static_cast<uint64_t>(TrackEvent::type), static_cast<uint64_t>(EventType::INSTANT));

constexpr uint64_t trace_packet_tag = 1;
constexpr uint64_t padding_tag = 2;

}  // namespace perfetto

using namespace perfetto;

// The trace events we support.
enum class event_type {
  none = 0,
  cond_broadcast,
  cond_signal,
  cond_timedwait,
  cond_wait,
  join,
  mutex_lock,
  mutex_trylock,
  mutex_unlock,
  mutex_locked,
  once,
  yield,
  sleep,
  usleep,
  nanosleep,
  count,
};

constexpr auto event_cond_broadcast = proto::buffer<2>::make(
    static_cast<uint64_t>(TrackEvent::name_iid), static_cast<uint64_t>(event_type::cond_broadcast));
constexpr auto event_cond_signal =
    proto::buffer<2>::make(static_cast<uint64_t>(TrackEvent::name_iid), static_cast<uint64_t>(event_type::cond_signal));
constexpr auto event_cond_timedwait = proto::buffer<2>::make(
    static_cast<uint64_t>(TrackEvent::name_iid), static_cast<uint64_t>(event_type::cond_timedwait));
constexpr auto event_cond_wait =
    proto::buffer<2>::make(static_cast<uint64_t>(TrackEvent::name_iid), static_cast<uint64_t>(event_type::cond_wait));
constexpr auto event_join =
    proto::buffer<2>::make(static_cast<uint64_t>(TrackEvent::name_iid), static_cast<uint64_t>(event_type::join));
constexpr auto event_mutex_lock =
    proto::buffer<2>::make(static_cast<uint64_t>(TrackEvent::name_iid), static_cast<uint64_t>(event_type::mutex_lock));
constexpr auto event_mutex_trylock = proto::buffer<2>::make(
    static_cast<uint64_t>(TrackEvent::name_iid), static_cast<uint64_t>(event_type::mutex_trylock));
constexpr auto event_mutex_unlock = proto::buffer<2>::make(
    static_cast<uint64_t>(TrackEvent::name_iid), static_cast<uint64_t>(event_type::mutex_unlock));
constexpr auto event_mutex_locked = proto::buffer<2>::make(
    static_cast<uint64_t>(TrackEvent::name_iid), static_cast<uint64_t>(event_type::mutex_locked));
constexpr auto event_once =
    proto::buffer<2>::make(static_cast<uint64_t>(TrackEvent::name_iid), static_cast<uint64_t>(event_type::once));
constexpr auto event_yield =
    proto::buffer<2>::make(static_cast<uint64_t>(TrackEvent::name_iid), static_cast<uint64_t>(event_type::yield));
constexpr auto event_sleep =
    proto::buffer<2>::make(static_cast<uint64_t>(TrackEvent::name_iid), static_cast<uint64_t>(event_type::sleep));
constexpr auto event_usleep =
    proto::buffer<2>::make(static_cast<uint64_t>(TrackEvent::name_iid), static_cast<uint64_t>(event_type::usleep));
constexpr auto event_nanosleep =
    proto::buffer<2>::make(static_cast<uint64_t>(TrackEvent::name_iid), static_cast<uint64_t>(event_type::nanosleep));

constexpr auto timestamp_zero =
    proto::buffer<2>::make(static_cast<uint64_t>(TracePacket::timestamp), static_cast<uint64_t>(0));

const char* to_string(event_type t) {
  switch (t) {
  case event_type::cond_broadcast: return "pthread_cond_broadcast";
  case event_type::cond_signal: return "pthread_cond_signal";
  case event_type::cond_timedwait: return "pthread_cond_timedwait";
  case event_type::cond_wait: return "pthread_cond_wait";
  case event_type::join: return "pthread_join";
  case event_type::mutex_lock: return "pthread_mutex_lock";
  case event_type::mutex_trylock: return "pthread_mutex_trylock";
  case event_type::mutex_unlock: return "pthread_mutex_unlock";
  case event_type::mutex_locked: return "(mutex locked)";
  case event_type::once: return "pthread_once";
  case event_type::yield: return "sched_yield";
  case event_type::sleep: return "sleep";
  case event_type::usleep: return "usleep";
  case event_type::nanosleep: return "nanosleep";
  case event_type::none:
  case event_type::count: break;
  }
  return nullptr;
}

const auto& get_interned_data() {
  static proto::buffer<512> interned_data = []() {
    proto::buffer<512> event_names;
    for (size_t i = 1; i < static_cast<size_t>(event_type::count); ++i) {
      proto::buffer<64> event_name;
      event_name.write_tagged(static_cast<uint64_t>(EventName::iid), i);
      event_name.write_tagged(static_cast<uint64_t>(EventName::name), to_string(static_cast<event_type>(i)));
      event_names.write_tagged(static_cast<uint64_t>(InternedData::event_names), event_name);
    }
    proto::buffer<512> interned_data;
    interned_data.write_tagged(static_cast<uint64_t>(TracePacket::interned_data), event_names);
    return interned_data;
  }();

  return interned_data;
}

constexpr size_t block_size = 1024 * 32;

class circular_file {
  int fd;
  uint8_t* buffer_;
  size_t size_;
  size_t blocks_;

  std::atomic<bool>* allocated_;

  std::atomic<size_t> next_;

public:
  void open(const char* path, size_t blocks) {
    // Don't use O_TRUNC here, it might truncate the file after we started writing it from another thread.
    fd = ::open(path, O_CREAT | O_RDWR | O_TRUNC, S_IRWXU);
    if (fd < 0) {
      fprintf(stderr, "Error opening file '%s': %s\n", path, strerror(errno));
      exit(1);
    }

    blocks_ = blocks;
    size_ = block_size * blocks;
    int result = ftruncate(fd, size_);
    if (result < 0) {
      close();
      fprintf(stderr, "Error allocating space in file '%s': %s\n", path, strerror(errno));
      exit(1);
    }

    buffer_ = static_cast<uint8_t*>(mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    if (buffer_ == (void*)-1) {
      close();
      fprintf(stderr, "Error mapping file '%s': %s\n", path, strerror(errno));
      exit(1);
    }
    next_ = 0;

    allocated_ = new std::atomic<bool>[blocks];
    memset(allocated_, 0, blocks * sizeof(std::atomic<bool>));

    // Initialize all the blocks with padding.
    for (size_t i = 0; i < size_; i += block_size) {
      proto::write_padding(buffer_ + i, 2, block_size);
    }
  }

  void close() {
    if (buffer_) {
      munmap(buffer_, size_);
      buffer_ = nullptr;
    }
    if (fd >= 0) {
      ::close(fd);
      fd = -1;
    }
  }

  void write_block(const uint8_t* data) {
    size_t offset = next_.fetch_add(block_size) % size_;
    memcpy(buffer_ + offset, data, block_size);
  }
};

circular_file file;

constexpr uint64_t clock_id = 64;

// We want to avoid flushing blocks with open slices, because dropped blocks could result in unclosed slices.
// This is not something we can do with 100% reliability, but we can approximate it in most cases by assuming
// there are no more than `max_depth` slices open at once.
constexpr size_t max_depth = 8;
constexpr size_t message_capacity = 32;

static std::atomic<int> next_thread_id{0};
static std::atomic<int> next_sequence_id{1};

class thread_state {
  int id;
  proto::buffer<4> track_uuid;
  proto::buffer<4> trusted_packet_sequence_id;

  proto::buffer<block_size> buffer;

  // The previous timestamp emitted on this thread.
  std::chrono::time_point<std::chrono::high_resolution_clock> t0;

  void write_clock_snapshot() {
    proto::buffer<32> clock;
    clock.write_tagged(static_cast<uint64_t>(Clock::clock_id), clock_id);
    clock.write_tagged(static_cast<uint64_t>(Clock::timestamp), static_cast<uint64_t>(0));
    clock.write_tagged(static_cast<uint64_t>(Clock::is_incremental), true);

    proto::buffer<32> boottime_clock;
    boottime_clock.write_tagged(static_cast<uint64_t>(Clock::clock_id), static_cast<uint64_t>(BuiltinClocks::BOOTTIME));
    boottime_clock.write_tagged(static_cast<uint64_t>(Clock::timestamp),
        static_cast<uint64_t>(std::chrono::time_point_cast<std::chrono::nanoseconds>(t0).time_since_epoch().count()));

    proto::buffer<64> clocks;
    clocks.write_tagged(static_cast<uint64_t>(ClockSnapshot::clocks), clock);
    clocks.write_tagged(static_cast<uint64_t>(ClockSnapshot::clocks), boottime_clock);

    proto::buffer<64> clock_snapshot;
    clock_snapshot.write_tagged(static_cast<uint64_t>(TracePacket::clock_snapshot), clocks);

    buffer.write_tagged(trace_packet_tag, clock_snapshot, trusted_packet_sequence_id);
  }

  void write_track_descriptor() {
    // Write the thread descriptor once.
    proto::buffer<4> uuid;
    uuid.write_tagged(static_cast<uint64_t>(TrackDescriptor::uuid), static_cast<uint64_t>(id));

    // Use the thread id as the thread name, pad it so it sorts alphabetically in numerical order.
    // TODO: Use real thread names?
    std::string thread_name = std::to_string(id);
    thread_name = std::string(4 - thread_name.size(), '0') + thread_name;

    proto::buffer<256> name;
    name.write_tagged(static_cast<uint64_t>(TrackDescriptor::name), thread_name.c_str());

    proto::buffer<256> track_descriptor;
    track_descriptor.write_tagged(static_cast<uint64_t>(TracePacket::track_descriptor), uuid, name);

    proto::buffer<32> timestamp_clock_id;
    timestamp_clock_id.write_tagged(static_cast<uint64_t>(TracePacketDefaults::timestamp_clock_id), clock_id);

    proto::buffer<32> trace_packet_defaults;
    trace_packet_defaults.write_tagged(static_cast<uint64_t>(TracePacket::trace_packet_defaults), timestamp_clock_id);

    buffer.write_tagged(trace_packet_tag, track_descriptor, trace_packet_defaults, get_interned_data(),
        trusted_packet_sequence_id, sequence_flags_cleared);
  }

  void write_sequence_header() {
    trusted_packet_sequence_id.clear();
    trusted_packet_sequence_id.write_tagged(
        static_cast<uint64_t>(TracePacket::trusted_packet_sequence_id), static_cast<uint64_t>(next_sequence_id++));

    write_track_descriptor();
    write_clock_snapshot();
  }

  void flush(size_t size) {
    constexpr size_t padding_capacity = 5;
    if (buffer.size() + size + padding_capacity >= block_size) {
      buffer.write_tagged_padding(padding_tag, block_size - buffer.size());
      assert(buffer.size() == block_size);
      file.write_block(buffer.data());
      buffer.clear();
      write_sequence_header();
    }
  }

  uint64_t delta_timestamp() {
    auto now = std::chrono::high_resolution_clock::now();
    uint64_t delta = std::chrono::duration_cast<std::chrono::nanoseconds>(now - t0).count();
    t0 = now;
    return delta;
  }

  void make_delta_timestamp(proto::buffer<16>& timestamp) {
    timestamp.write_tagged(static_cast<uint64_t>(TracePacket::timestamp), delta_timestamp());
  }

  template <size_t TimestampSize, typename EventField>
  void write_begin_with_delta(size_t flush_capacity, const proto::buffer<TimestampSize>& timestamp, const EventField& field) {
    flush(flush_capacity);

    proto::buffer<16> track_event;
    track_event.write_tagged(static_cast<uint64_t>(TracePacket::track_event), slice_begin, field, track_uuid);

    // Write the trace packet directly into the buffer to avoid a memcpy from a temporary protobuf.
    buffer.write_tagged(trace_packet_tag, trusted_packet_sequence_id, sequence_flags_needed, track_event, timestamp);
  }

public:
  thread_state() : id(next_thread_id++) {
    t0 = std::chrono::high_resolution_clock::now();
    track_uuid.write_tagged(static_cast<uint64_t>(TrackEvent::track_uuid), static_cast<uint64_t>(id));

    write_sequence_header();
  }

  ~thread_state() { flush(buffer.capacity()); }


  template <size_t TimestampSize, typename EventField>
  void write_begin_with_delta(const proto::buffer<TimestampSize>& timestamp, const EventField& field) {
    // Avoid flushing because delta 0 timestamps should go in the same block if possible.
    constexpr size_t flush_capacity = message_capacity;
    write_begin_with_delta(flush_capacity, timestamp, field);
  }

  template <typename EventField>
  void write_begin(const EventField& field) {
    proto::buffer<16> timestamp;
    make_delta_timestamp(timestamp);

    constexpr size_t flush_capacity = max_depth * message_capacity;
    write_begin_with_delta(flush_capacity, timestamp, field);
  }

  void write_end() {
    flush(message_capacity);

    proto::buffer<16> timestamp;
    make_delta_timestamp(timestamp);

    proto::buffer<16> track_event;
    track_event.write_tagged(static_cast<uint64_t>(TracePacket::track_event), slice_end, track_uuid);

    buffer.write_tagged(trace_packet_tag, trusted_packet_sequence_id, sequence_flags_needed, track_event, timestamp);
  }

  static thread_state& get() {
    thread_local thread_state t;
    return t;
  }
};

int getenv_or(const char* env, int def) {
  const char* s = getenv(env);
  return s ? std::atoi(s) : def;
}

pthread_once_t init_once = PTHREAD_ONCE_INIT;

void init_trace() {
  // Use the real pthread_once to handle initialization.
  typedef int (*once_fn_t)(pthread_once_t*, void (*)());
  once_fn_t real_once = (once_fn_t)dlsym(RTLD_NEXT, "pthread_once");

  real_once(&init_once, []() {
    const char* path = getenv("PTHREAD_TRACE_PATH");
    if (!path) {
      path = "pthread_trace.proto";
    }
    size_t buffer_size = getenv_or("PTHREAD_TRACE_BUFFER_SIZE", 1024 * 1024 * 64);

    file.open(path, buffer_size / block_size);
    fprintf(stderr, "pthread_trace: Writing trace to '%s'\n", path);
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

typedef unsigned int (*hook_t)(unsigned int);

unsigned int sleep(unsigned int secs) {
  if (!hooks::sleep) hooks::init(hooks::sleep, "sleep");

  auto& t = thread_state::get();
  t.write_begin(event_sleep);
  unsigned int result = hooks::sleep(secs);
  t.write_end();  // event_sleep
  return result;
}

int usleep(useconds_t usecs) {
  if (!hooks::usleep) hooks::init(hooks::usleep, "usleep");

  auto& t = thread_state::get();
  t.write_begin(event_usleep);
  int result = hooks::usleep(usecs);
  t.write_end();  // event_usleep
  return result;
}

int nanosleep(const struct timespec* duration, struct timespec* rem) {
  if (!hooks::nanosleep) hooks::init(hooks::nanosleep, "nanosleep");

  auto& t = thread_state::get();
  t.write_begin(event_nanosleep);
  int result = hooks::nanosleep(duration, rem);
  t.write_end();  // event_nanosleep
  return result;
}

int sched_yield() {
  if (!hooks::sched_yield) hooks::init(hooks::sched_yield, "sched_yield");

  auto& t = thread_state::get();
  t.write_begin(event_yield);
  int result = hooks::sched_yield();
  t.write_end();  // event_yield
  return result;
}

int pthread_cond_broadcast(pthread_cond_t* cond) {
  if (!hooks::pthread_cond_broadcast)
    hooks::init(hooks::pthread_cond_broadcast, "pthread_cond_broadcast", "GLIBC_2.3.2");

  auto& t = thread_state::get();
  t.write_begin(event_cond_broadcast);
  int result = hooks::pthread_cond_broadcast(cond);
  t.write_end();  // event_cond_broadcast
  return result;
}

int pthread_cond_signal(pthread_cond_t* cond) {
  if (!hooks::pthread_cond_signal) hooks::init(hooks::pthread_cond_signal, "pthread_cond_signal", "GLIBC_2.3.2");

  auto& t = thread_state::get();
  t.write_begin(event_cond_signal);
  int result = hooks::pthread_cond_signal(cond);
  t.write_end();  // event_cond_signal
  return result;
}

int pthread_cond_timedwait(pthread_cond_t* cond, pthread_mutex_t* mutex, const struct timespec* abstime) {
  if (!hooks::pthread_cond_timedwait)
    hooks::init(hooks::pthread_cond_timedwait, "pthread_cond_timedwait", "GLIBC_2.3.2");

  // When we wait on a cond var, the mutex gets unlocked, and then relocked before returning.
  auto& t = thread_state::get();
  t.write_end();  // event_mutex_locked
  t.write_begin_with_delta(timestamp_zero, event_cond_timedwait);
  int result = hooks::pthread_cond_timedwait(cond, mutex, abstime);
  t.write_end();  // event_cond_timedwait
  t.write_begin_with_delta(timestamp_zero, event_mutex_locked);
  return result;
}

int pthread_cond_wait(pthread_cond_t* cond, pthread_mutex_t* mutex) {
  if (!hooks::pthread_cond_wait) hooks::init(hooks::pthread_cond_wait, "pthread_cond_wait", "GLIBC_2.3.2");

  // When we wait on a cond var, the mutex gets unlocked, and then relocked before returning.
  auto& t = thread_state::get();
  t.write_end();  // event_mutex_locked
  t.write_begin_with_delta(timestamp_zero, event_cond_wait);
  int result = hooks::pthread_cond_wait(cond, mutex);
  t.write_end();  // event_cond_wait
  t.write_begin_with_delta(timestamp_zero, event_mutex_locked);
  return result;
}

int pthread_join(pthread_t thread, void** value_ptr) {
  if (!hooks::pthread_join) hooks::init(hooks::pthread_join, "pthread_join");

  auto& t = thread_state::get();
  t.write_begin(event_join);
  int result = hooks::pthread_join(thread, value_ptr);
  t.write_end();  // event_join
  return result;
}

int pthread_mutex_lock(pthread_mutex_t* mutex) {
  if (!hooks::pthread_mutex_lock) hooks::init(hooks::pthread_mutex_lock, "pthread_mutex_lock");

  auto& t = thread_state::get();
  t.write_begin(event_mutex_lock);
  int result = hooks::pthread_mutex_lock(mutex);
  t.write_end();  // event_mutex_lock
  t.write_begin_with_delta(timestamp_zero, event_mutex_locked);
  return result;
}

int pthread_mutex_trylock(pthread_mutex_t* mutex) {
  if (!hooks::pthread_mutex_trylock) hooks::init(hooks::pthread_mutex_trylock, "pthread_mutex_trylock");

  auto& t = thread_state::get();
  t.write_begin(event_mutex_trylock);
  int result = hooks::pthread_mutex_trylock(mutex);
  t.write_end();  // event_mutex_trylock
  if (result == 0) {
    t.write_begin_with_delta(timestamp_zero, event_mutex_locked);
  }
  return result;
}

int pthread_mutex_unlock(pthread_mutex_t* mutex) {
  if (!hooks::pthread_mutex_unlock) hooks::init(hooks::pthread_mutex_unlock, "pthread_mutex_unlock");

  // It makes some sense that we should report the mutex as locked until this returns.
  // However, it seems that other threads are able to lock the mutex well before pthread_mutex_unlock returns, so this
  // results in confusing traces.
  auto& t = thread_state::get();
  t.write_end();  // event_mutex_locked
  t.write_begin_with_delta(timestamp_zero, event_mutex_unlock);
  int result = hooks::pthread_mutex_unlock(mutex);
  t.write_end();  // event_mutex_unlock
  return result;
}

int pthread_once(pthread_once_t* once_control, void (*init_routine)(void)) {
  if (!hooks::pthread_once) hooks::init(hooks::pthread_once, "pthread_once");

  auto& t = thread_state::get();
  t.write_begin(event_once);
  int result = hooks::pthread_once(once_control, init_routine);
  t.write_end();  // event_once
  return result;
}

}  // extern "C"