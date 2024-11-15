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
#include <map>
#include <mutex>
#include <thread>
#include <vector>

namespace {

namespace proto {

enum class wire_type {
  varint = 0,
  i64 = 1,
  len = 2,
  i32 = 5,
};

size_t sum() { return 0; }
template <class... Args>
size_t sum(size_t a, Args... args) {
  return a + sum(args...);
}

// Writing protobufs is a bit tricky, because you need to know the size of child messages before writing the parent
// message header. The approach used here is to use fixed size stack buffers for everything, and just copy them into
// nested buffers after constructing them (so we know the size). This approach would be bad for deeply nested protos,
// but we just don't have that much nesting in this case.
template <size_t Capacity>
class buffer {
  std::array<char, Capacity> buf_;
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

public:
  static constexpr size_t capacity() { return Capacity; }

  size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }
  const char* data() const { return buf_.data(); }
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

// These protobuf messages are for 'perfetto', excerpted from
// https://cs.android.com/android/platform/superproject/main/+/main:external/perfetto/protos/perfetto/trace/trace_packet.proto
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
};

// The trace events we support.
enum class trace_type {
  cond_broadcast,
  cond_signal,
  cond_timedwait,
  cond_wait,
  join,
  mutex_lock,
  mutex_trylock,
  mutex_unlock,
  mutex_locked,
};

const char* to_string(trace_type t) {
  switch (t) {
  case trace_type::cond_broadcast: return "cond_broadcast";
  case trace_type::cond_signal: return "cond_signal";
  case trace_type::cond_timedwait: return "cond_timedwait";
  case trace_type::cond_wait: return "cond_wait";
  case trace_type::join: return "join";
  case trace_type::mutex_lock: return "mutex_lock";
  case trace_type::mutex_trylock: return "mutex_trylock";
  case trace_type::mutex_unlock: return "mutex_unlock";
  case trace_type::mutex_locked: return "mutex_locked";
  default: return nullptr;
  }
}

std::atomic<bool> initialized = false;
std::atomic<int> fd = -1;

auto t0_ = std::chrono::high_resolution_clock::now();

std::atomic<int> next_thread_id = 0;

constexpr uint64_t root_track_uuid = 0;

class thread_info {
  int id;

  // To minimize contention while writing to the file, we accumulate messages in this local buffer, and
  // flush it to the file when its full.
  proto::buffer<4096> buffer;

public:
  thread_info() {
    id = next_thread_id++;

    // Write the thread descriptor once.
    proto::buffer<8> uuid;
    uuid.write(static_cast<uint64_t>(TrackDescriptor::uuid), id);

    proto::buffer<8> parent_uuid;
    parent_uuid.write(static_cast<uint64_t>(TrackDescriptor::parent_uuid), root_track_uuid);

    // Use the thread id as the thread name, pad it so it sorts alphabetically in numerical order.
    // TODO: Use real thread names?
    std::string thread_name = std::to_string(id);
    thread_name = std::string(4 - thread_name.size(), '0') + thread_name;

    proto::buffer<256> name;
    name.write(static_cast<uint64_t>(TrackDescriptor::name), thread_name.c_str());

    proto::buffer<256> track_descriptor;
    track_descriptor.write(static_cast<uint64_t>(TracePacket::track_descriptor), uuid, parent_uuid, name);

    proto::buffer<256> trace_packet;
    trace_packet.write(1, track_descriptor);

    write(trace_packet);
  }
  ~thread_info() {
    if (!buffer.empty()) {
      ssize_t result = ::write(fd.load(), buffer.data(), buffer.size());
      (void)result;
    }
  }

  int get_id() const { return id; }

  template <size_t N>
  void write(const proto::buffer<N>& message) {
    if (message.size() + buffer.size() > buffer.capacity()) {
      // Our buffer is full, flush it.
      ssize_t result = ::write(fd.load(), buffer.data(), buffer.size());
      (void)result;
      buffer.clear();
    }
    buffer.write(message);
  }
};

thread_local thread_info thread;

void write_trace_begin(trace_type e, EventType event_type = EventType::SLICE_BEGIN) {
  auto t = std::chrono::high_resolution_clock::now();
  auto ts = std::chrono::duration_cast<std::chrono::nanoseconds>(t - t0_).count();

  proto::buffer<8> track_uuid;
  track_uuid.write(static_cast<uint64_t>(TrackEvent::track_uuid), thread.get_id());

  proto::buffer<4> type;
  type.write(static_cast<uint64_t>(TrackEvent::type), static_cast<uint64_t>(event_type));

  proto::buffer<256> name_buf;
  name_buf.write(static_cast<uint64_t>(TrackEvent::name), to_string(e));

  proto::buffer<256> track_event;
  track_event.write(static_cast<uint64_t>(TracePacket::track_event), name_buf, type, track_uuid);

  proto::buffer<16> timestamp;
  timestamp.write(static_cast<uint64_t>(TracePacket::timestamp), ts);

  proto::buffer<4> trusted_packet_sequence_id;
  trusted_packet_sequence_id.write(static_cast<uint64_t>(TracePacket::trusted_packet_sequence_id), 1);

  proto::buffer<256> trace_packet;
  trace_packet.write(1, timestamp, track_event, trusted_packet_sequence_id);

  thread.write(trace_packet);
}

void write_trace_event(trace_type e) { write_trace_begin(e, EventType::INSTANT); }

void write_trace_end() {
  auto t = std::chrono::high_resolution_clock::now();
  auto ts = std::chrono::duration_cast<std::chrono::nanoseconds>(t - t0_).count();

  proto::buffer<8> track_uuid;
  track_uuid.write(static_cast<uint64_t>(TrackEvent::track_uuid), thread.get_id());

  proto::buffer<4> type;
  type.write(static_cast<uint64_t>(TrackEvent::type), static_cast<uint64_t>(EventType::SLICE_END));

  proto::buffer<256> track_event;
  track_event.write(static_cast<uint64_t>(TracePacket::track_event), type, track_uuid);

  proto::buffer<16> timestamp;
  timestamp.write(static_cast<uint64_t>(TracePacket::timestamp), ts);

  proto::buffer<4> trusted_packet_sequence_id;
  trusted_packet_sequence_id.write(static_cast<uint64_t>(TracePacket::trusted_packet_sequence_id), 1);

  proto::buffer<256> trace_packet;
  trace_packet.write(1, timestamp, track_event, trusted_packet_sequence_id);

  thread.write(trace_packet);
}

void write_trace_header() {
  proto::buffer<8> uuid;
  uuid.write(static_cast<uint64_t>(TrackDescriptor::uuid), root_track_uuid);

  proto::buffer<8> track_descriptor;
  track_descriptor.write(static_cast<uint64_t>(TracePacket::track_descriptor), uuid);

  proto::buffer<32> trace_packet;
  trace_packet.write(1, track_descriptor);

  ssize_t written = write(fd.load(), trace_packet.data(), trace_packet.size());
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
  write_trace_event(trace_type::cond_broadcast);
  return hook(cond);
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
  write_trace_event(trace_type::cond_signal);
  return hook(cond);
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
  write_trace_end();
  write_trace_begin(trace_type::cond_timedwait);
  int result = hook(cond, mutex, abstime);
  write_trace_end();
  write_trace_begin(trace_type::mutex_locked);
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
  write_trace_end();
  write_trace_begin(trace_type::cond_wait);
  int result = hook(cond, mutex);
  write_trace_end();
  write_trace_begin(trace_type::mutex_locked);
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
  write_trace_begin(trace_type::join);
  int result = hook(thread, value_ptr);
  write_trace_end();
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
  write_trace_begin(trace_type::mutex_lock);
  int result = hook(mutex);
  write_trace_end();
  write_trace_begin(trace_type::mutex_locked);
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
  write_trace_event(trace_type::mutex_trylock);
  int result = hook(mutex);
  if (result == 0) {
    write_trace_begin(trace_type::mutex_locked);
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
  write_trace_end();
  return hook(mutex);
}
}