#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <assert.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <sched.h>
#include <stdio.h>
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
#include <iostream>
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
  const char* data() const { return buf_.data(); }
  void clear() { size_ = 0; }

  void write(const void* s, size_t n) {
    assert(size_ + n <= Capacity);
    std::memcpy(&buf_[size_], s, n);
    size_ += n;
  }

  template <size_t M>
  void write(const buffer<M>& buf) {
    write(buf.data(), buf.size());
  }

  void write_tag(uint64_t field_number, wire_type type) {
    write_varint((field_number << 3) | static_cast<uint64_t>(type));
  }

  void write_len_tag(uint64_t field_number, uint64_t len) {
    write_tag(field_number, wire_type::len);
    write_varint(len);
  }

  void write(uint64_t field_number, uint64_t value) {
    write_tag(field_number, wire_type::varint);
    write_varint(value);
  }

  template <size_t M>
  void write(uint64_t field_number, const buffer<M>& field) {
    write_len_tag(field_number, field.size());
    write(field.data(), field.size());
  }

  // void write(uint64_t field_number, int64_t value) {
  //   write_tag(field_number, wire_type::varint);
  //   write_varint(value);
  // }

  void write(uint64_t field_number, const char* str) {
    std::size_t len = strlen(str);
    write_len_tag(field_number, len);
    write(str, len);
  }
};

}  // namespace proto

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
  TYPE_UNSPECIFIED = 0,
  TYPE_SLICE_BEGIN = 1,
  TYPE_SLICE_END = 2,
  TYPE_INSTANT = 3,
  TYPE_COUNTER = 4,
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

class per_thread_data {
  int id;

public:
  per_thread_data() {
    id = next_thread_id++;

    // Write the thread descriptor once.
    proto::buffer<8> uuid;
    uuid.write(static_cast<uint64_t>(TrackDescriptor::uuid), id);

    proto::buffer<8> parent_uuid;
    parent_uuid.write(static_cast<uint64_t>(TrackDescriptor::parent_uuid), (uint64_t)0);

    proto::buffer<256> name;
    name.write(static_cast<uint64_t>(TrackDescriptor::name), std::to_string(id).c_str());

    proto::buffer<256> track_descriptor;
    track_descriptor.write_len_tag(
        static_cast<uint64_t>(TracePacket::track_descriptor), uuid.size() + parent_uuid.size() + name.size());
    track_descriptor.write(uuid);
    track_descriptor.write(parent_uuid);
    track_descriptor.write(name);

    proto::buffer<256> trace_packet;
    trace_packet.write(1, track_descriptor);

    ssize_t result = write(fd.load(), trace_packet.data(), trace_packet.size());
    (void)result;
  }

  int get_id() const { return id; }
};

thread_local per_thread_data per_thread;

void write_trace_begin(trace_type e) {
  auto t = std::chrono::high_resolution_clock::now();
  auto ts = std::chrono::duration_cast<std::chrono::nanoseconds>(t - t0_).count();

  proto::buffer<8> track_uuid;
  track_uuid.write(static_cast<uint64_t>(TrackEvent::track_uuid), per_thread.get_id());

  proto::buffer<4> type;
  type.write(static_cast<uint64_t>(TrackEvent::type), static_cast<uint64_t>(EventType::TYPE_SLICE_BEGIN));

  proto::buffer<256> name_buf;
  name_buf.write(static_cast<uint64_t>(TrackEvent::name), to_string(e));

  proto::buffer<256> track_event;
  track_event.write_len_tag(
      static_cast<uint64_t>(TracePacket::track_event), name_buf.size() + type.size() + track_uuid.size());
  track_event.write(name_buf);
  track_event.write(type);
  track_event.write(track_uuid);

  proto::buffer<16> timestamp;
  timestamp.write(static_cast<uint64_t>(TracePacket::timestamp), ts);

  proto::buffer<4> trusted_packet_sequence_id;
  trusted_packet_sequence_id.write(static_cast<uint64_t>(TracePacket::trusted_packet_sequence_id), 1);

  proto::buffer<256> buf;
  buf.write_len_tag(1, track_event.size() + trusted_packet_sequence_id.size() + timestamp.size());
  buf.write(timestamp);
  buf.write(track_event);
  buf.write(trusted_packet_sequence_id);

  ssize_t result = write(fd.load(), buf.data(), buf.size());
  (void)result;
}

void write_trace_end() {
  auto t = std::chrono::high_resolution_clock::now();
  auto ts = std::chrono::duration_cast<std::chrono::nanoseconds>(t - t0_).count();

  proto::buffer<8> track_uuid;
  track_uuid.write(static_cast<uint64_t>(TrackEvent::track_uuid), per_thread.get_id());

  proto::buffer<4> type;
  type.write(static_cast<uint64_t>(TrackEvent::type), static_cast<uint64_t>(EventType::TYPE_SLICE_END));

  proto::buffer<256> track_event;
  track_event.write_len_tag(static_cast<uint64_t>(TracePacket::track_event), type.size() + track_uuid.size());
  track_event.write(type);
  track_event.write(track_uuid);

  proto::buffer<16> timestamp;
  timestamp.write(static_cast<uint64_t>(TracePacket::timestamp), ts);

  proto::buffer<4> trusted_packet_sequence_id;
  trusted_packet_sequence_id.write(static_cast<uint64_t>(TracePacket::trusted_packet_sequence_id), 1);

  proto::buffer<256> buf;
  buf.write_len_tag(1, track_event.size() + trusted_packet_sequence_id.size() + timestamp.size());
  buf.write(timestamp);
  buf.write(track_event);
  buf.write(trusted_packet_sequence_id);

  ssize_t result = write(fd.load(), buf.data(), buf.size());
  (void)result;
}

void write_trace_event(trace_type e) {
  auto t = std::chrono::high_resolution_clock::now();
  auto ts = std::chrono::duration_cast<std::chrono::nanoseconds>(t - t0_).count();

  proto::buffer<8> track_uuid;
  track_uuid.write(static_cast<uint64_t>(TrackEvent::track_uuid), per_thread.get_id());

  proto::buffer<4> type;
  type.write(static_cast<uint64_t>(TrackEvent::type), static_cast<uint64_t>(EventType::TYPE_INSTANT));

  proto::buffer<256> name_buf;
  name_buf.write(static_cast<uint64_t>(TrackEvent::name), to_string(e));

  proto::buffer<256> track_event;
  track_event.write_len_tag(
      static_cast<uint64_t>(TracePacket::track_event), name_buf.size() + type.size() + track_uuid.size());
  track_event.write(name_buf);
  track_event.write(type);
  track_event.write(track_uuid);

  proto::buffer<16> timestamp;
  timestamp.write(static_cast<uint64_t>(TracePacket::timestamp), ts);

  proto::buffer<4> trusted_packet_sequence_id;
  trusted_packet_sequence_id.write(static_cast<uint64_t>(TracePacket::trusted_packet_sequence_id), 1);

  proto::buffer<256> buf;
  buf.write_len_tag(1, track_event.size() + trusted_packet_sequence_id.size() + timestamp.size());
  buf.write(timestamp);
  buf.write(track_event);
  buf.write(trusted_packet_sequence_id);

  ssize_t written = write(fd.load(), buf.data(), buf.size());
  (void)written;
}

void write_trace_header() {
  proto::buffer<8> uuid;
  uuid.write(static_cast<uint64_t>(TrackDescriptor::uuid), (uint64_t)0);

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
  int maybe_fd = open(path, O_CREAT | O_WRONLY, S_IRWXU);
  if (maybe_fd < 0) {
    fprintf(stderr, "Error opening file '%s': %s\n", path, strerror(errno));
    exit(1);
  }
  // Not sure we can use pthread_once after all the hooking we do, so handle it
  // ourselves :(
  int negative_one = -1;
  if (fd.compare_exchange_strong(negative_one, maybe_fd)) {
    // We opened the file, we should truncate it.
    int result = ftruncate(fd.load(), 0);
    (void)result;

    // We're done initializing.
    initialized = true;

    fprintf(stderr, "pthread_trace: Writing trace to '%s'\n", path);

    write_trace_header();
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
  write_trace_begin(trace_type::cond_timedwait);
  int result = hook(cond, mutex, abstime);
  write_trace_end();
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
  write_trace_begin(trace_type::cond_wait);
  int result = hook(cond, mutex);
  write_trace_end();
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