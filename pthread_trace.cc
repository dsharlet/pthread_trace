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

constexpr uint8_t make_tag(uint8_t tag, wire_type type) { return (tag << 3) | static_cast<uint8_t>(type); }

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

  void write_all() {}

  template <typename Field0, typename... Fields>
  void write_all(const Field0& first, const Fields&... rest) {
    write(first);
    write_all(rest...);
  }

public:
  constexpr buffer() : size_(0) {}
  explicit constexpr buffer(const std::array<uint8_t, Capacity>& raw) : buf_(raw), size_(raw.size()) {}

  static constexpr size_t capacity() { return Capacity; }

  size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }
  const uint8_t* data() const { return buf_.data(); }
  void clear() { size_ = 0; }

  constexpr uint8_t operator[](size_t i) const { return buf_[i]; }

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

  void write(std::initializer_list<uint8_t> data) {
    for (uint8_t i : data) {
      buf_[size_++] = i;
    }
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

enum class TrackEventDefaults {
  /*optional uint64*/ track_uuid = 11,
};

enum class TracePacketDefaults {
  /*optional uint32*/ timestamp_clock_id = 58,
  /*optional TrackEventDefaults*/ track_event_defaults = 11,
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

constexpr proto::buffer<2> sequence_flags_cleared(
    {{make_tag(static_cast<uint64_t>(TracePacket::sequence_flags), proto::wire_type::varint),
        static_cast<uint64_t>(SequenceFlags::INCREMENTAL_STATE_CLEARED)}});

constexpr uint64_t trace_packet_tag = 1;
constexpr uint64_t padding_tag = 2;

}  // namespace perfetto

using namespace perfetto;

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
  mutex_locked,
  once,
  yield,
  sleep,
  usleep,
  nanosleep,
  count,
};

constexpr uint8_t name_iid_tag = make_tag(static_cast<uint64_t>(TrackEvent::name_iid), proto::wire_type::varint);
constexpr uint8_t track_event_tag = make_tag(static_cast<uint64_t>(TracePacket::track_event), proto::wire_type::len);
constexpr uint8_t track_event_type_tag = make_tag(static_cast<uint64_t>(TrackEvent::type), proto::wire_type::varint);

constexpr proto::buffer<6> make_slice_begin(event_type event) {
  return proto::buffer<6>({{track_event_tag, 4, track_event_type_tag, static_cast<uint64_t>(EventType::SLICE_BEGIN),
      name_iid_tag, static_cast<uint8_t>(event)}});
}

constexpr proto::buffer<4> slice_end(
    {{track_event_tag, 2, track_event_type_tag, static_cast<uint64_t>(EventType::SLICE_END)}});

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

constexpr proto::buffer<2> prev_timestamp(
    {{make_tag(static_cast<uint64_t>(TracePacket::timestamp), proto::wire_type::varint), static_cast<uint64_t>(0)}});

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

template <size_t N>
void write_interned_data(proto::buffer<N>& buf) {
  proto::buffer<512> event_names;
  for (size_t i = 1; i < static_cast<size_t>(event_type::count); ++i) {
    proto::buffer<64> event_name;
    event_name.write_tagged(static_cast<uint64_t>(EventName::iid), i);
    event_name.write_tagged(static_cast<uint64_t>(EventName::name), to_string(static_cast<event_type>(i)));
    event_names.write_tagged(static_cast<uint64_t>(InternedData::event_names), event_name);
  }
  buf.write_tagged(static_cast<uint64_t>(TracePacket::interned_data), event_names);
}

constexpr size_t block_size_kb = 32;
constexpr size_t block_size = block_size_kb * 1024;

// Incremental timestamps can be tricky, this disables them for debugging purposes.
constexpr bool enable_incremental_timestamps = true;

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
      if (next_.load() < size_) {
        // Remove blocks we didn't write anything to.
        ftruncate(fd, next_.load());
      }
      ::close(fd);
      fd = -1;
    }
  }

public:
  circular_file(const char* path, size_t blocks) {
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

    // Initialize all the blocks with padding.
    for (size_t i = 0; i < size_; i += block_size) {
      proto::write_padding(buffer_ + i, 2, block_size);
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

constexpr uint64_t clock_id = 64;

// We can only report up to (this many) different mutexes uniquely in one block.
// After this many mutexes, we just report (mutex locked).
// This might seem like a really strict limitation, but this limit only applies to one block in one thread.
constexpr size_t max_unique_mutex = 16;

static std::atomic<int> next_thread_id{0};

class thread_state {
  int id;
  proto::buffer<4> trusted_packet_sequence_id;

  proto::buffer<block_size> buffer;

  // The previous timestamp emitted on this thread.
  std::chrono::time_point<std::chrono::high_resolution_clock> t0;

  std::array<std::pair<const void*, proto::buffer<6>>, max_unique_mutex> mutex_locked_event;

  void write_clock_snapshot() {
    proto::buffer<32> clock;
    clock.write_tagged(static_cast<uint64_t>(Clock::clock_id), clock_id);
    clock.write_tagged(static_cast<uint64_t>(Clock::timestamp), static_cast<uint64_t>(0));
    clock.write_tagged(static_cast<uint64_t>(Clock::is_incremental), enable_incremental_timestamps);

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

    proto::buffer<16> track_uuid;
    track_uuid.write_tagged(static_cast<uint64_t>(TrackEventDefaults::track_uuid), static_cast<uint64_t>(id));

    proto::buffer<32> track_event_defaults;
    track_event_defaults.write_tagged(static_cast<uint64_t>(TracePacketDefaults::track_event_defaults), track_uuid);

    proto::buffer<32> trace_packet_defaults;
    trace_packet_defaults.write_tagged(
        static_cast<uint64_t>(TracePacket::trace_packet_defaults), timestamp_clock_id, track_event_defaults);

    buffer.write_tagged(trace_packet_tag, track_descriptor, trace_packet_defaults, *interned_data,
        trusted_packet_sequence_id, sequence_flags_cleared);

    // We've invalidated the uniquely named mutex events.
    for (auto& i : mutex_locked_event) {
      i.first = nullptr;
      i.second.clear();
    }
  }

  NOINLINE void write_sequence_header(bool first = false) {
    if (first || enable_incremental_timestamps) {
      t0 = std::chrono::high_resolution_clock::now();
    }

    write_track_descriptor();
    write_clock_snapshot();
  }

  NOINLINE void flush() {
    buffer.write_tagged_padding(padding_tag, block_size - buffer.size());
    assert(buffer.size() == block_size);
    file->write_block(buffer.data());
    buffer.clear();
    write_sequence_header();
  }

  void flush(size_t size) {
    constexpr size_t padding_capacity = 2;
    if (buffer.size() + size + padding_capacity >= block_size) {
      flush();
    }
  }

  void make_timestamp(proto::buffer<16>& timestamp) {
    // For some reason I haven't been able to figure out, timestamps can get out of order when crossing block
    // boundaries. We can fix this by subtracting a small error from the timestamp deltas we emit. Error accumulates
    // within each block. However, at block boundaries, we reset the incremental timestamps to the global absolute time,
    // so the error accumulation is limited.
    constexpr uint64_t fudge_time_ns = enable_incremental_timestamps ? 1 : 0;
    auto now = std::chrono::high_resolution_clock::now();
    uint64_t delta = std::chrono::duration_cast<std::chrono::nanoseconds>(now - t0).count() - fudge_time_ns;
    if (enable_incremental_timestamps) {
      t0 = now;
    }
    timestamp.write_tagged(static_cast<uint64_t>(TracePacket::timestamp), delta);
  }

public:
  thread_state() : id(next_thread_id++) {
    // This can't be 0, just use the thread id + 1 instead.
    trusted_packet_sequence_id.write_tagged(
        static_cast<uint64_t>(TracePacket::trusted_packet_sequence_id), static_cast<uint64_t>(id + 1));
    write_sequence_header(/*first=*/true);
  }

  ~thread_state() {
    if (buffer.size() > 0) {
      buffer.write_tagged_padding(padding_tag, block_size - buffer.size());
      assert(buffer.size() == block_size);
      file->write_block(buffer.data());
    }
  }

  template <size_t TimestampSize, typename TrackEvent>
  void write_begin_with_timestamp(const proto::buffer<TimestampSize>& timestamp, const TrackEvent& track_event) {
    constexpr size_t message_capacity = 32;
    flush(message_capacity);

    // Write the trace packet directly into the buffer to avoid a memcpy from a temporary protobuf.
    buffer.write_tagged(trace_packet_tag, trusted_packet_sequence_id, track_event, timestamp);
  }

  template <typename TrackEvent>
  void write_begin(const proto::buffer<2>& timestamp, const TrackEvent& track_event) {
    if (enable_incremental_timestamps) {
      write_begin_with_timestamp(timestamp, track_event);
    } else {
      write_begin(track_event);
    }
  }

  template <typename TrackEvent>
  void write_begin(const TrackEvent& track_event) {
    proto::buffer<16> timestamp;
    make_timestamp(timestamp);

    write_begin_with_timestamp(timestamp, track_event);
  }

  void write_end() {
    constexpr size_t message_capacity = 32;
    flush(message_capacity);

    proto::buffer<16> timestamp;
    make_timestamp(timestamp);

    buffer.write_tagged(trace_packet_tag, trusted_packet_sequence_id, slice_end, timestamp);
  }

  // Write a (mutex %p locked) event. These always have the same timestamp as the previous event.
  void write_begin_mutex_locked(const void* mutex) {
    for (size_t i = 0; i < max_unique_mutex; ++i) {
      if (mutex_locked_event[i].first == mutex) {
        // We've already named this mutex locked event.
        write_begin(prev_timestamp, mutex_locked_event[i].second);
        return;
      } else if (!mutex_locked_event[i].first) {
        // Add a new mutex locked event.
        proto::buffer<64> interned_data;
        size_t name_iid = static_cast<size_t>(event_type::count) + i;

        char name_buf[32];
        snprintf(name_buf, sizeof(name_buf), "(mutex %p locked)", mutex);

        proto::buffer<64> event_name;
        event_name.write_tagged(static_cast<uint64_t>(EventName::iid), name_iid);
        event_name.write_tagged(static_cast<uint64_t>(EventName::name), static_cast<const char*>(name_buf));

        proto::buffer<64> event_names;
        event_names.write_tagged(static_cast<uint64_t>(InternedData::event_names), event_name);
        interned_data.write_tagged(static_cast<uint64_t>(TracePacket::interned_data), event_names);

        mutex_locked_event[i].first = mutex;
        mutex_locked_event[i].second.write({track_event_tag, 4, track_event_type_tag,
            static_cast<uint64_t>(EventType::SLICE_BEGIN), name_iid_tag, static_cast<uint8_t>(name_iid)});

        constexpr size_t message_capacity = 128;
        flush(message_capacity);

        // mutex locked always occurs immediately after the previous event.
        if (enable_incremental_timestamps) {
          buffer.write_tagged(trace_packet_tag, trusted_packet_sequence_id, mutex_locked_event[i].second, interned_data,
              prev_timestamp);
        } else {
          proto::buffer<16> timestamp;
          make_timestamp(timestamp);

          buffer.write_tagged(
              trace_packet_tag, trusted_packet_sequence_id, mutex_locked_event[i].second, interned_data, timestamp);
        }
        return;
      }
    }

    // If we got here, we didn't find this mutex in our list, and we don't have room for any more unqiue mutex messages.
    // Just write a generic (mutex locked) message.
    write_begin(prev_timestamp, slice_begin_mutex_locked);
  }

  static thread_state& get() {
    thread_local thread_state t;
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
    if (blocks < 0) {
      fprintf(stderr, "pthread_trace: invalid buffer size %d (%s).\n", buffer_size_kb, buffer_size_str);
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

typedef unsigned int (*hook_t)(unsigned int);

unsigned int sleep(unsigned int secs) {
  if (!hooks::sleep) hooks::init(hooks::sleep, "sleep");

  auto& t = thread_state::get();
  t.write_begin(slice_begin_sleep);
  unsigned int result = hooks::sleep(secs);
  t.write_end();  // slice_begin_sleep
  return result;
}

int usleep(useconds_t usecs) {
  if (!hooks::usleep) hooks::init(hooks::usleep, "usleep");

  auto& t = thread_state::get();
  t.write_begin(slice_begin_usleep);
  int result = hooks::usleep(usecs);
  t.write_end();  // slice_begin_usleep
  return result;
}

int nanosleep(const struct timespec* duration, struct timespec* rem) {
  if (!hooks::nanosleep) hooks::init(hooks::nanosleep, "nanosleep");

  auto& t = thread_state::get();
  t.write_begin(slice_begin_nanosleep);
  int result = hooks::nanosleep(duration, rem);
  t.write_end();  // slice_begin_nanosleep
  return result;
}

int sched_yield() {
  if (!hooks::sched_yield) hooks::init(hooks::sched_yield, "sched_yield");

  auto& t = thread_state::get();
  t.write_begin(slice_begin_yield);
  int result = hooks::sched_yield();
  t.write_end();  // slice_begin_yield
  return result;
}

int pthread_cond_broadcast(pthread_cond_t* cond) {
  if (!hooks::pthread_cond_broadcast)
    hooks::init(hooks::pthread_cond_broadcast, "pthread_cond_broadcast", "GLIBC_2.3.2");

  auto& t = thread_state::get();
  t.write_begin(slice_begin_cond_broadcast);
  int result = hooks::pthread_cond_broadcast(cond);
  t.write_end();  // slice_begin_cond_broadcast
  return result;
}

int pthread_cond_signal(pthread_cond_t* cond) {
  if (!hooks::pthread_cond_signal) hooks::init(hooks::pthread_cond_signal, "pthread_cond_signal", "GLIBC_2.3.2");

  auto& t = thread_state::get();
  t.write_begin(slice_begin_cond_signal);
  int result = hooks::pthread_cond_signal(cond);
  t.write_end();  // slice_begin_cond_signal
  return result;
}

int pthread_cond_timedwait(pthread_cond_t* cond, pthread_mutex_t* mutex, const struct timespec* abstime) {
  if (!hooks::pthread_cond_timedwait)
    hooks::init(hooks::pthread_cond_timedwait, "pthread_cond_timedwait", "GLIBC_2.3.2");

  // When we wait on a cond var, the mutex gets unlocked, and then relocked before returning.
  auto& t = thread_state::get();
  t.write_end();  // slice_begin_mutex_locked
  t.write_begin(prev_timestamp, slice_begin_cond_timedwait);
  int result = hooks::pthread_cond_timedwait(cond, mutex, abstime);
  t.write_end();  // slice_begin_cond_timedwait
  t.write_begin_mutex_locked(mutex);
  return result;
}

int pthread_cond_wait(pthread_cond_t* cond, pthread_mutex_t* mutex) {
  if (!hooks::pthread_cond_wait) hooks::init(hooks::pthread_cond_wait, "pthread_cond_wait", "GLIBC_2.3.2");

  // When we wait on a cond var, the mutex gets unlocked, and then relocked before returning.
  auto& t = thread_state::get();
  t.write_end();  // slice_begin_mutex_locked
  t.write_begin(prev_timestamp, slice_begin_cond_wait);
  int result = hooks::pthread_cond_wait(cond, mutex);
  t.write_end();  // slice_begin_cond_wait
  t.write_begin_mutex_locked(mutex);
  return result;
}

int pthread_join(pthread_t thread, void** value_ptr) {
  if (!hooks::pthread_join) hooks::init(hooks::pthread_join, "pthread_join");

  auto& t = thread_state::get();
  t.write_begin(slice_begin_join);
  int result = hooks::pthread_join(thread, value_ptr);
  t.write_end();  // slice_begin_join
  return result;
}

int pthread_mutex_lock(pthread_mutex_t* mutex) {
  if (!hooks::pthread_mutex_lock) hooks::init(hooks::pthread_mutex_lock, "pthread_mutex_lock");

  auto& t = thread_state::get();
  t.write_begin(slice_begin_mutex_lock);
  int result = hooks::pthread_mutex_lock(mutex);
  t.write_end();  // slice_begin_mutex_lock
  t.write_begin_mutex_locked(mutex);
  return result;
}

int pthread_mutex_trylock(pthread_mutex_t* mutex) {
  if (!hooks::pthread_mutex_trylock) hooks::init(hooks::pthread_mutex_trylock, "pthread_mutex_trylock");

  auto& t = thread_state::get();
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

  // It makes some sense that we should report the mutex as locked until this returns.
  // However, it seems that other threads are able to lock the mutex well before pthread_mutex_unlock returns, so this
  // results in confusing traces.
  auto& t = thread_state::get();
  t.write_end();  // slice_begin_mutex_locked
  t.write_begin(prev_timestamp, slice_begin_mutex_unlock);
  int result = hooks::pthread_mutex_unlock(mutex);
  t.write_end();  // slice_begin_mutex_unlock
  return result;
}

int pthread_once(pthread_once_t* once_control, void (*init_routine)(void)) {
  if (!hooks::pthread_once) hooks::init(hooks::pthread_once, "pthread_once");

  auto& t = thread_state::get();
  t.write_begin(slice_begin_once);
  int result = hooks::pthread_once(once_control, init_routine);
  t.write_end();  // slice_begin_once
  return result;
}

}  // extern "C"