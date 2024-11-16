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
  void write_varint(uint64_t value) {
    constexpr uint8_t continuation = 0x80;
    while (value > 0x7f) {
      buf_[size_++] = static_cast<uint8_t>(value | continuation);
      value >>= 7;
    }
    buf_[size_++] = static_cast<uint8_t>(value);
  }

  // sint uses "zigzag" encoding: positive x -> 2*x, negative x -> -2*x - 1
  // void write_varint(buffer& buf, int64_t value) {
  // write_varint(buf, static_cast<uint64_t>(value < 0 ? -2 * value - 1 : 2 *
  // value));
  //}

  constexpr buffer(uint8_t d0, uint8_t d1) : buf_({{d0, d1}}), size_(2) {}

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

  void write() {}

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

  template <size_t N, typename... Fields>
  void write(const buffer<N>& first, const Fields&... rest) {
    write(first);
    write(rest...);
  }

  // Write a tag.
  void write_tag(uint64_t tag, wire_type type) { write_varint((tag << 3) | static_cast<uint64_t>(type)); }

  // Write tagged values.
  void write(uint64_t tag, uint64_t value) {
    write_tag(tag, wire_type::varint);
    write_varint(value);
  }

  // void write(uint64_t tag, int64_t value) {
  //   write_tag(tag, wire_type::varint);
  //   write_varint(value);
  // }

  void write(uint64_t tag, const char* str) {
    std::size_t len = strlen(str);
    write_tag(tag, wire_type::len);
    write_varint(len);
    write(str, len);
  }

  template <size_t N, typename... Fields>
  void write(uint64_t tag, const buffer<N>& first, const Fields&... rest) {
    write_tag(tag, wire_type::len);
    write_varint(first.size() + sum(rest.size()...));
    write(first);
    write(rest...);
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
  // Optional name of the event for its display in trace viewer. May be left
  // unspecified for events with typed arguments.
  //
  // Note that metrics should not rely on event names, as they are prone to
  // changing. Instead, they should use typed arguments to identify the events
  // they are interested in.
  // oneof {
  /*uint64*/ name_iid = 10,
  /*string*/ name = 23,
  //}

  /*optional EventType*/ type = 9,

  /*optional uint64*/ track_uuid = 11,

  // Deprecated. Use the |timestamp| and |timestamp_clock_id| fields in
  // TracePacket instead.
  // oneof timestamp {

  /*repeated uint64*/ extra_counter_track_uuids = 31,
  /*repeated int64*/ extra_counter_values = 12,

  // Deprecated. Use |extra_counter_values| and |extra_counter_track_uuids| to
  // encode thread time instead.
  //
  // CPU time for the current thread (e.g., CLOCK_THREAD_CPUTIME_ID) in
  // microseconds.
  // oneof thread_time {
};

enum class SequenceFlags {
  UNSPECIFIED = 0,
  INCREMENTAL_STATE_CLEARED = 1,
  NEEDS_INCREMENTAL_STATE = 2,
};

enum class TracePacket {
  /*optional uint64*/ timestamp = 8,

  // Specifies the ID of the clock used for the TracePacket |timestamp|. Can be
  // one of the built-in types from ClockSnapshot::BuiltinClocks, or a
  // producer-defined clock id.
  // If unspecified and if no default per-sequence value has been provided via
  // TracePacketDefaults, it defaults to BuiltinClocks::BOOTTIME.
  /*optional uint32*/ timestamp_clock_id = 58,
  /*TrackEvent*/ track_event = 11,
  /*TrackDescriptor*/ track_descriptor = 60,

  /*optional TracePacketDefaults*/ trace_packet_defaults = 59,

  /*uint32*/ trusted_packet_sequence_id = 10,

  /*optional InternedData*/ interned_data = 12,
  /*optional uint32*/ sequence_flags = 13,
};

auto trusted_packet_sequence_id =
    proto::buffer<2>::make(static_cast<uint64_t>(TracePacket::trusted_packet_sequence_id), 1);

auto sequence_flags_cleared = proto::buffer<2>::make(static_cast<uint64_t>(TracePacket::sequence_flags),
    static_cast<uint64_t>(SequenceFlags::INCREMENTAL_STATE_CLEARED));
auto sequence_flags = proto::buffer<2>::make(
    static_cast<uint64_t>(TracePacket::sequence_flags), static_cast<uint64_t>(SequenceFlags::NEEDS_INCREMENTAL_STATE));

auto slice_begin =
    proto::buffer<2>::make(static_cast<uint64_t>(TrackEvent::type), static_cast<uint64_t>(EventType::SLICE_BEGIN));
auto slice_end =
    proto::buffer<2>::make(static_cast<uint64_t>(TrackEvent::type), static_cast<uint64_t>(EventType::SLICE_END));
auto instant =
    proto::buffer<2>::make(static_cast<uint64_t>(TrackEvent::type), static_cast<uint64_t>(EventType::INSTANT));

constexpr uint64_t trace_packet_tag = 1;

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
  flush,
  count,
};

auto event_cond_broadcast = proto::buffer<2>::make(
    static_cast<uint64_t>(TrackEvent::name_iid), static_cast<uint64_t>(event_type::cond_broadcast));
auto event_cond_signal =
    proto::buffer<2>::make(static_cast<uint64_t>(TrackEvent::name_iid), static_cast<uint64_t>(event_type::cond_signal));
auto event_cond_timedwait = proto::buffer<2>::make(
    static_cast<uint64_t>(TrackEvent::name_iid), static_cast<uint64_t>(event_type::cond_timedwait));
auto event_cond_wait =
    proto::buffer<2>::make(static_cast<uint64_t>(TrackEvent::name_iid), static_cast<uint64_t>(event_type::cond_wait));
auto event_join =
    proto::buffer<2>::make(static_cast<uint64_t>(TrackEvent::name_iid), static_cast<uint64_t>(event_type::join));
auto event_mutex_lock =
    proto::buffer<2>::make(static_cast<uint64_t>(TrackEvent::name_iid), static_cast<uint64_t>(event_type::mutex_lock));
auto event_mutex_trylock = proto::buffer<2>::make(
    static_cast<uint64_t>(TrackEvent::name_iid), static_cast<uint64_t>(event_type::mutex_trylock));
auto event_mutex_unlock = proto::buffer<2>::make(
    static_cast<uint64_t>(TrackEvent::name_iid), static_cast<uint64_t>(event_type::mutex_unlock));
auto event_mutex_locked = proto::buffer<2>::make(
    static_cast<uint64_t>(TrackEvent::name_iid), static_cast<uint64_t>(event_type::mutex_locked));
auto event_once =
    proto::buffer<2>::make(static_cast<uint64_t>(TrackEvent::name_iid), static_cast<uint64_t>(event_type::once));
auto event_yield =
    proto::buffer<2>::make(static_cast<uint64_t>(TrackEvent::name_iid), static_cast<uint64_t>(event_type::yield));
auto event_sleep =
    proto::buffer<2>::make(static_cast<uint64_t>(TrackEvent::name_iid), static_cast<uint64_t>(event_type::sleep));
auto event_usleep =
    proto::buffer<2>::make(static_cast<uint64_t>(TrackEvent::name_iid), static_cast<uint64_t>(event_type::usleep));
auto event_nanosleep =
    proto::buffer<2>::make(static_cast<uint64_t>(TrackEvent::name_iid), static_cast<uint64_t>(event_type::nanosleep));
auto event_flush =
    proto::buffer<2>::make(static_cast<uint64_t>(TrackEvent::name_iid), static_cast<uint64_t>(event_type::flush));

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
  case event_type::flush: return "(flush)";
  case event_type::none:
  case event_type::count: break;
  }
  return nullptr;
}

std::atomic<bool> initialized = false;
std::atomic<int> fd = -1;

std::atomic<size_t> file_offset = 0;

std::atomic<int> next_thread_id = 0;

constexpr uint64_t root_track_uuid = 0;

auto get_start_time() {
  static auto t0 = std::chrono::high_resolution_clock::now();
  return t0;
}

class thread_state {
  proto::buffer<4> track_uuid;

  // To minimize contention while writing to the file, we accumulate messages in this local buffer, and
  // flush it to the file when its full.
  proto::buffer<1024 * 64> buffer;

  // The previous timestamp emitted on this thread.
  std::chrono::time_point<std::chrono::high_resolution_clock> t0 = get_start_time();

public:
  thread_state() {
    int id = next_thread_id++;

    track_uuid.write(static_cast<uint64_t>(TrackEvent::track_uuid), id);

    // Write the thread descriptor once.
    proto::buffer<4> uuid;
    uuid.write(static_cast<uint64_t>(TrackDescriptor::uuid), id);

    auto parent_uuid = proto::buffer<2>::make(static_cast<uint64_t>(TrackDescriptor::parent_uuid), root_track_uuid);

    // Use the thread id as the thread name, pad it so it sorts alphabetically in numerical order.
    // TODO: Use real thread names?
    std::string thread_name = std::to_string(id);
    thread_name = std::string(4 - thread_name.size(), '0') + thread_name;

    proto::buffer<256> name;
    name.write(static_cast<uint64_t>(TrackDescriptor::name), thread_name.c_str());

    proto::buffer<256> track_descriptor;
    track_descriptor.write(static_cast<uint64_t>(TracePacket::track_descriptor), uuid, parent_uuid, name);

    proto::buffer<4096> trace_packet;
    trace_packet.write(trace_packet_tag, track_descriptor, trusted_packet_sequence_id);

    write(trace_packet);
  }
  ~thread_state() {
    if (!buffer.empty()) {
      size_t at = file_offset.fetch_add(buffer.size());
      ssize_t result = pwrite(fd.load(), buffer.data(), buffer.size(), at);
      (void)result;
    }
  }

  const proto::buffer<4>& get_track_uuid() const { return track_uuid; }

  void make_timestamp(proto::buffer<16>& timestamp) {
    auto now = std::chrono::high_resolution_clock::now();
    uint64_t delta = std::chrono::duration_cast<std::chrono::nanoseconds>(now - t0).count();
    timestamp.write(static_cast<uint64_t>(TracePacket::timestamp), delta);
  }

  template <size_t N>
  void write(const proto::buffer<N>& message) {
    if (message.size() + buffer.size() > buffer.capacity()) {
      // We can't write the flush slice before we flush, so get the timestamp now.
      proto::buffer<16> timestamp;
      make_timestamp(timestamp);

      // Our buffer is full, flush it.
      size_t at = file_offset.fetch_add(buffer.size());
      ssize_t result = pwrite(fd.load(), buffer.data(), buffer.size(), at);
      (void)result;
      buffer.clear();

      // Now write the two flush events, using the timestamp from above.
      proto::buffer<64> flush;
      make_trace_packet(flush, timestamp, slice_begin, event_flush);
      timestamp.clear();
      make_timestamp(timestamp);
      make_trace_packet(flush, timestamp, slice_end);
      buffer.write(flush);
    }
    buffer.write(message);
  }

  template <size_t N, typename... TrackEventFields>
  void make_trace_packet(proto::buffer<N>& buf, const proto::buffer<16>& timestamp, const TrackEventFields&... fields) {
    proto::buffer<16> track_event;
    track_event.write(static_cast<uint64_t>(TracePacket::track_event), fields..., get_track_uuid());

    buf.write(trace_packet_tag, trusted_packet_sequence_id, sequence_flags, track_event, timestamp);
  }

  template <typename... TrackEventFields>
  void write_trace_packet(const proto::buffer<16>& timestamp, const TrackEventFields&... fields) {
    proto::buffer<32> trace_packet;
    make_trace_packet(trace_packet, timestamp, fields...);

    write(trace_packet);
  }

  template <typename... TrackEventFields>
  void write_trace_packet(const TrackEventFields&... fields) {
    proto::buffer<16> timestamp;
    make_timestamp(timestamp);

    write_trace_packet(timestamp, fields...);
  }

  static thread_state& get() {
    thread_local thread_state t;
    return t;
  }
};

void write_trace_header() {
  auto uuid = proto::buffer<2>::make(static_cast<uint8_t>(TrackDescriptor::uuid), root_track_uuid);

  proto::buffer<8> track_descriptor;
  track_descriptor.write(static_cast<uint64_t>(TracePacket::track_descriptor), uuid);

  proto::buffer<4096> event_names;
  for (size_t i = 1; i < static_cast<size_t>(event_type::count); ++i) {
    proto::buffer<64> event_name;
    event_name.write(static_cast<uint64_t>(EventName::iid), i);
    event_name.write(static_cast<uint64_t>(EventName::name), to_string(static_cast<event_type>(i)));
    event_names.write(static_cast<uint64_t>(InternedData::event_names), event_name);
  }
  proto::buffer<4096> interned_data;
  interned_data.write(static_cast<uint64_t>(TracePacket::interned_data), event_names);

  proto::buffer<4096> trace_packet;
  trace_packet.write(
      trace_packet_tag, track_descriptor, interned_data, trusted_packet_sequence_id, sequence_flags_cleared);

  size_t at = file_offset.fetch_add(trace_packet.size());
  ssize_t written = pwrite(fd.load(), trace_packet.data(), trace_packet.size(), at);
  (void)written;
}

void trace_init() {
  if (initialized) {
    return;
  }
  const char* path = getenv("PTHREAD_TRACE_PATH");
  if (!path) {
    path = "pthread_trace.proto";
  }
  // Don't use O_TRUNC here, it might truncate the file after we started writing it from another thread.
  int maybe_fd = open(path, O_CREAT | O_WRONLY, S_IRWXU);
  if (maybe_fd < 0) {
    fprintf(stderr, "Error opening file '%s': %s\n", path, strerror(errno));
    exit(1);
  }
  // Not sure we can use pthread_once after all the hooking of pthreads we do, so handle it
  // ourselves :(
  int negative_one = -1;
  if (fd.compare_exchange_strong(negative_one, maybe_fd)) {
    // We opened the file, we should truncate it.
    int result = ftruncate(fd.load(), 0);
    (void)result;

    fprintf(stderr, "pthread_trace: Writing trace to '%s'\n", path);

    write_trace_header();

    // We're done initializing.
    initialized = true;
  } else {
    // Another thread must have opened the file already, close our fd and wait
    // for it to say we're initialized.
    close(maybe_fd);
    while (!initialized) {
      std::this_thread::yield();
    }
    assert(fd.load() >= 0);
  }
}

}  // namespace

extern "C" {

unsigned int sleep(unsigned int secs) {
  typedef unsigned int (*hook_t)(unsigned int);
  static hook_t hook = nullptr;
  if (!hook) {
    trace_init();
    hook = (hook_t)dlsym(RTLD_NEXT, "sleep");
    if (!hook) {
      fprintf(stderr, "Failed to find sleep\n");
      exit(1);
    }
  }
  thread_state::get().write_trace_packet(slice_begin, event_sleep);
  unsigned int result = hook(secs);
  thread_state::get().write_trace_packet(slice_end);  // event_sleep
  return result;
}

int usleep(useconds_t usecs) {
  typedef int (*hook_t)(useconds_t);
  static hook_t hook = nullptr;
  if (!hook) {
    trace_init();
    hook = (hook_t)dlsym(RTLD_NEXT, "usleep");
    if (!hook) {
      fprintf(stderr, "Failed to find usleep\n");
      exit(1);
    }
  }
  thread_state::get().write_trace_packet(slice_begin, event_usleep);
  int result = hook(usecs);
  thread_state::get().write_trace_packet(slice_end);  // event_usleep
  return result;
}

int nanosleep(const struct timespec* duration, struct timespec* rem) {
  typedef int (*hook_t)(const struct timespec*, struct timespec*);
  static hook_t hook = nullptr;
  if (!hook) {
    trace_init();
    hook = (hook_t)dlsym(RTLD_NEXT, "nanosleep");
    if (!hook) {
      fprintf(stderr, "Failed to find nanosleep\n");
      exit(1);
    }
  }
  thread_state::get().write_trace_packet(slice_begin, event_nanosleep);
  int result = hook(duration, rem);
  thread_state::get().write_trace_packet(slice_end);  // event_nanosleep
  return result;
}

int sched_yield() {
  typedef int (*hook_t)();
  static hook_t hook = nullptr;
  if (!hook) {
    trace_init();
    hook = (hook_t)dlsym(RTLD_NEXT, "sched_yield");
    if (!hook) {
      fprintf(stderr, "Failed to find sched_yield\n");
      exit(1);
    }
  }
  thread_state::get().write_trace_packet(slice_begin, event_yield);
  int result = hook();
  thread_state::get().write_trace_packet(slice_end);  // event_yield
  return result;
}

int pthread_cond_broadcast(pthread_cond_t* cond) {
  typedef int (*hook_t)(pthread_cond_t*);
  static hook_t hook = nullptr;
  if (!hook) {
    trace_init();
    // TODO: properly handle different versions.
    hook = (hook_t)dlvsym(RTLD_NEXT, "pthread_cond_broadcast", "GLIBC_2.3.2");
    if (!hook) {
      fprintf(stderr, "Failed to find pthread_cond_broadcast\n");
      exit(1);
    }
  }
  thread_state::get().write_trace_packet(slice_begin, event_cond_broadcast);
  int result = hook(cond);
  thread_state::get().write_trace_packet(slice_end);  // event_cond_broadcast
  return result;
}

int pthread_cond_signal(pthread_cond_t* cond) {
  typedef int (*hook_t)(pthread_cond_t*);
  static hook_t hook = nullptr;
  if (!hook) {
    trace_init();
    hook = (hook_t)dlvsym(RTLD_NEXT, "pthread_cond_signal", "GLIBC_2.3.2");
    if (!hook) {
      fprintf(stderr, "Failed to find pthread_cond_signal\n");
      exit(1);
    }
  }
  thread_state::get().write_trace_packet(slice_begin, event_cond_signal);
  int result = hook(cond);
  thread_state::get().write_trace_packet(slice_end);  // event_cond_signal
  return result;
}

int pthread_cond_timedwait(pthread_cond_t* cond, pthread_mutex_t* mutex, const struct timespec* abstime) {
  typedef int (*hook_t)(pthread_cond_t*, pthread_mutex_t*, const struct timespec*);
  static hook_t hook = nullptr;
  if (!hook) {
    trace_init();
    hook = (hook_t)dlvsym(RTLD_NEXT, "pthread_cond_timedwait", "GLIBC_2.3.2");
    if (!hook) {
      fprintf(stderr, "Failed to find pthread_cond_timedwait\n");
      exit(1);
    }
  }
  // When we wait on a cond var, the mutex gets unlocked, and then relocked before returning.
  auto& t = thread_state::get();
  proto::buffer<16> timestamp;
  t.make_timestamp(timestamp);
  t.write_trace_packet(timestamp, slice_end);  // event_mutex_locked
  t.write_trace_packet(timestamp, slice_begin, event_cond_timedwait);
  int result = hook(cond, mutex, abstime);
  timestamp.clear();
  t.make_timestamp(timestamp);
  t.write_trace_packet(timestamp, slice_end);  // event_cond_timedwait
  t.write_trace_packet(timestamp, slice_begin, event_mutex_locked);
  return result;
}

int pthread_cond_wait(pthread_cond_t* cond, pthread_mutex_t* mutex) {
  typedef int (*hook_t)(pthread_cond_t*, pthread_mutex_t*);
  static hook_t hook = nullptr;
  if (!hook) {
    trace_init();
    hook = (hook_t)dlvsym(RTLD_NEXT, "pthread_cond_wait", "GLIBC_2.3.2");
    if (!hook) {
      fprintf(stderr, "Failed to find pthread_cond_wait\n");
      exit(1);
    }
  }
  // When we wait on a cond var, the mutex gets unlocked, and then relocked before returning.
  auto& t = thread_state::get();
  proto::buffer<16> timestamp;
  t.make_timestamp(timestamp);
  t.write_trace_packet(timestamp, slice_end);  // event_mutex_locked
  t.write_trace_packet(timestamp, slice_begin, event_cond_wait);
  int result = hook(cond, mutex);
  timestamp.clear();
  t.make_timestamp(timestamp);
  t.write_trace_packet(timestamp, slice_end);  // event_cond_wait
  t.write_trace_packet(timestamp, slice_begin, event_mutex_locked);
  return result;
}

int pthread_join(pthread_t thread, void** value_ptr) {
  typedef int (*hook_t)(pthread_t, void**);
  static hook_t hook = nullptr;
  if (!hook) {
    trace_init();
    hook = (hook_t)dlsym(RTLD_NEXT, "pthread_join");
    if (!hook) {
      fprintf(stderr, "Failed to find pthread_join\n");
      exit(1);
    }
  }
  auto& t = thread_state::get();
  t.write_trace_packet(slice_begin, event_join);
  int result = hook(thread, value_ptr);
  t.write_trace_packet(slice_end);  // event_join
  return result;
}

int pthread_mutex_lock(pthread_mutex_t* mutex) {
  typedef int (*hook_t)(pthread_mutex_t*);
  static hook_t hook = nullptr;
  if (!hook) {
    trace_init();
    hook = (hook_t)dlsym(RTLD_NEXT, "pthread_mutex_lock");
    if (!hook) {
      fprintf(stderr, "Failed to find pthread_mutex_lock\n");
      exit(1);
    }
  }
  auto& t = thread_state::get();
  t.write_trace_packet(slice_begin, event_mutex_lock);
  int result = hook(mutex);
  proto::buffer<16> timestamp;
  t.make_timestamp(timestamp);
  t.write_trace_packet(timestamp, slice_end);  // event_mutex_lock
  t.write_trace_packet(timestamp, slice_begin, event_mutex_locked);
  return result;
}

int pthread_mutex_trylock(pthread_mutex_t* mutex) {
  typedef int (*hook_t)(pthread_mutex_t*);
  static hook_t hook = nullptr;
  if (!hook) {
    trace_init();
    hook = (hook_t)dlsym(RTLD_NEXT, "pthread_mutex_trylock");
    if (!hook) {
      fprintf(stderr, "Failed to find pthread_mutex_trylock\n");
      exit(1);
    }
  }
  auto& t = thread_state::get();
  t.write_trace_packet(slice_begin, event_mutex_trylock);
  int result = hook(mutex);
  t.write_trace_packet(slice_end);  // event_mutex_trylock
  if (result == 0) {
    t.write_trace_packet(slice_begin, event_mutex_locked);
  }
  return result;
}

int pthread_mutex_unlock(pthread_mutex_t* mutex) {
  typedef int (*hook_t)(pthread_mutex_t*);
  static hook_t hook = nullptr;
  if (!hook) {
    trace_init();
    hook = (hook_t)dlsym(RTLD_NEXT, "pthread_mutex_unlock");
    if (!hook) {
      fprintf(stderr, "Failed to find pthread_mutex_unlock\n");
      exit(1);
    }
  }

  // It makes some sense that we should report the mutex as locked until this returns.
  // However, it seems that other threads are able to lock the mutex well before pthread_mutex_unlock returns, so this
  // results in confusing traces.
  auto& t = thread_state::get();
  proto::buffer<16> timestamp;
  t.make_timestamp(timestamp);
  t.write_trace_packet(timestamp, slice_end);  // event_mutex_locked
  t.write_trace_packet(timestamp, slice_begin, event_mutex_unlock);
  int result = hook(mutex);
  t.write_trace_packet(slice_end);  // event_mutex_unlock
  return result;
}

int pthread_once(pthread_once_t* once_control, void (*init_routine)(void)) {
  typedef int (*hook_t)(pthread_once_t*, void (*)());
  static hook_t hook = nullptr;
  if (!hook) {
    trace_init();
    hook = (hook_t)dlsym(RTLD_NEXT, "pthread_once");
    if (!hook) {
      fprintf(stderr, "Failed to find pthread_once\n");
      exit(1);
    }
  }
  auto& t = thread_state::get();
  t.write_trace_packet(slice_begin, event_once);
  int result = hook(once_control, init_routine);
  t.write_trace_packet(slice_end);  // event_once
  return result;
}

}  // extern "C"