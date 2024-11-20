#ifndef PTHREAD_TRACE_PROTO_H
#define PTHREAD_TRACE_PROTO_H

#include <array>
#include <initializer_list>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace proto {

namespace internal {

static inline constexpr size_t sum() { return 0; }
template <class... Args>
static inline constexpr size_t sum(size_t a, Args... args) {
  return a + sum(args...);
}

}  // namespace internal

enum class wire_type {
  varint = 0,
  i64 = 1,
  len = 2,
  i32 = 5,
};

constexpr uint8_t varint_continuation = 0x80;

// Convert an integer to a varint. May overflow if the result doesn't fit in 64 bits, but can be used constexpr.
static inline constexpr uint64_t to_varint(uint64_t value) {
  uint64_t result = 0;
  while (value > 0x7f) {
    result |= static_cast<uint8_t>(value | varint_continuation);
    result <<= 8;
    value >>= 7;
  }
  result |= static_cast<uint8_t>(value);
  return result;
}

static inline size_t write_varint(uint8_t* dst, uint64_t value) {
  size_t result = 0;
  while (value > 0x7f) {
    dst[result++] = static_cast<uint8_t>(value | varint_continuation);
    value >>= 7;
  }
  dst[result++] = static_cast<uint8_t>(value);
  return result;
}

static inline size_t read_varint(uint64_t& result, const uint8_t* data, size_t size) {
  result = 0;
  uint64_t shift = 0;
  for (size_t i = 1; i <= size; ++i) {
    uint8_t b = *data++;
    result |= (b & 0x7f) << shift;
    shift += 7;
    if ((b & 0x80) == 0) return i;
  }
  return 0;
}

static inline constexpr uint64_t make_tag(uint64_t tag, wire_type type) {
  return to_varint((tag << 3) | static_cast<uint8_t>(type));
}

static inline size_t write_tag(uint8_t* dst, uint64_t tag, wire_type type) {
  return write_varint(dst, (tag << 3) | static_cast<uint64_t>(type));
}

// Size must be >= 2.
static inline size_t write_padding(uint8_t* dst, uint64_t tag, uint64_t size) {
  size_t result = 0;
  while (size != 0) {
    assert(size != 1);
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
    assert(size > 0);
    size -= 1;
  }
  return result;
}

template <size_t Capacity>
class buffer;

namespace internal {

template <typename T>
struct capacity_of {};

template <size_t N>
struct capacity_of<buffer<N>> {
  static constexpr size_t value = N;
};
template <typename T, size_t N>
struct capacity_of<std::array<T, N>> {
  static constexpr size_t value = N;
};

}  // namespace internal

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
  explicit constexpr buffer(const std::array<uint8_t, Capacity>& raw) : buffer(raw, raw.size()) {}
  explicit constexpr buffer(const std::array<uint8_t, Capacity>& raw, size_t size) : buf_(raw), size_(size) {}

  static constexpr size_t capacity() { return Capacity; }

  size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }
  const uint8_t* data() const { return buf_.data(); }
  void clear() { size_ = 0; }

  constexpr uint8_t operator[](size_t i) const { return buf_[i]; }

  // Write objects directly to the buffer, without any tags.
  void write(const uint8_t* s, size_t n) {
    assert(size_ + n <= Capacity);
    if (n == 2) {
      // It is extremely common to write 2 bytes (one byte tag + one byte varint).
      buf_[size_] = s[0];
      buf_[size_ + 1] = s[1];
    } else {
      std::memcpy(&buf_[size_], s, n);
    }
    size_ += n;
  }

  void write(const buffer<2>& buf) {
    // This special case gives a constexpr size of 2.
    assert(buf.size() == 2);
    write(buf.data(), 2);
  }

  template <size_t M>
  void write(const buffer<M>& buf) {
    write(buf.data(), buf.size());
  }

  template <size_t M>
  void write(const std::array<uint8_t, M>& buf) {
    write(buf.data(), buf.size());
  }

  void write(std::initializer_list<uint8_t> data) {
    assert(size_ + data.size() <= Capacity);
    for (uint8_t i : data) {
      buf_[size_++] = i;
    }
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
    write(reinterpret_cast<const uint8_t*>(str), len);
  }

  template <typename... Fields>
  void write_tagged(uint64_t tag, const Fields&... fields) {
    write_tag(tag, wire_type::len);
    // This branch avoids varint encoding for most use cases.
    constexpr size_t capacity = internal::sum(internal::capacity_of<Fields>::value...);
    size_t size = internal::sum(fields.size()...);
    assert(size <= capacity);
    if (capacity < 0x80) {
      buf_[size_++] = size;
    } else {
      write_varint(size);
    }
    write_all(fields...);
  }
};

}  // namespace proto

#endif  // PTHREAD_TRACE_PROTO_H
