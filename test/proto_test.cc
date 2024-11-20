#include "proto.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace proto {

uint64_t decode_varint(const std::vector<uint8_t>& bytes) {
  uint64_t result = 0;
  read_varint(result, bytes.data(), bytes.size());
  return result;
}

uint64_t decode_varint(uint64_t bytes) {
  uint64_t result = 0;
  read_varint(result, reinterpret_cast<uint8_t*>(&bytes), sizeof(bytes));
  return result;
}

TEST(proto, to_varint) {
  for (uint64_t i = 0; i <= 0x7f; ++i) {
    ASSERT_EQ(to_varint(i), i);
  }
  ASSERT_EQ(to_varint(0x80), 0x0180);
  ASSERT_EQ(to_varint(0x81), 0x0181);
  ASSERT_EQ(to_varint(0x82), 0x0182);
}

// Wrappers that make testing easier.
void write_varint(std::vector<uint8_t>& to, uint64_t x) {
  to.resize(12);
  size_t size = write_varint(to.data(), x);
  ASSERT_LE(size, 10);
  to.resize(size);
}

std::vector<uint8_t> write_varint(uint64_t x) {
  std::vector<uint8_t> result;
  write_varint(result, x);
  return result;
}

TEST(proto, write_varint) {
  for (uint64_t i = 0; i <= 0x7f; ++i) {
    ASSERT_THAT(write_varint(i), testing::ElementsAre(i));
  }

  ASSERT_THAT(write_varint(0x80), testing::ElementsAre(0x80, 0x01));
  ASSERT_THAT(write_varint(0x81), testing::ElementsAre(0x81, 0x01));
  ASSERT_THAT(write_varint(0x82), testing::ElementsAre(0x82, 0x01));

  std::vector<uint8_t> varint;
  for (size_t i = 0; i < 1ull << 20; ++i) {
    write_varint(varint, i);
    ASSERT_EQ(decode_varint(varint), i) << i;
  }

  write_varint(varint, std::numeric_limits<uint64_t>::max());
  ASSERT_EQ(decode_varint(varint), std::numeric_limits<uint64_t>::max());
}

TEST(proto, to_varint_matches_write_varint) {
  for (size_t i = 0; i < 1ull << 20; ++i) {
    uint64_t to = to_varint(i);
    ASSERT_EQ(decode_varint(to), i) << i;
  }
}

TEST(proto, write_padding) {
  std::array<uint8_t, 32> buffer;
  for (uint64_t i = 2; i <= 0x7f; ++i) {
    ASSERT_EQ(write_padding(buffer.data(), 0, i), i);
  }
}

}  // namespace proto