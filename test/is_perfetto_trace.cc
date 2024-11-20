#include "proto.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <perfetto trace proto file>" << std::endl;
    return 1;
  }
  const char* filename = argv[1];
  std::cout << "Reading file " << filename << std::endl;
  std::ifstream file(argv[1], std::ios::binary);
  std::vector<uint8_t> buffer(std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{});
  if (buffer.empty()) {
    std::cerr << "File " << filename << " is empty.";
    return 1;
  }

  int trace_packets = 0;
  size_t trace_packet_size = 0;
  int padding = 0;
  size_t padding_size = 0;

  size_t at = 0;
  while (at < buffer.size()) {
    uint64_t tag = 0;
    at += proto::read_varint(tag, &buffer[at], buffer.size() - at);
    if (tag == 0) {
      std::cerr << "Invalid tag at offset " << at << std::endl;
      return 1;
    }

    proto::wire_type type = static_cast<proto::wire_type>(tag & 0x7);
    if (type != proto::wire_type::len) {
      std::cerr << "Tag was not wire_type::len" << std::endl;
      return 1;
    }
    tag >>= 3;

    uint64_t len = -1;
    at += proto::read_varint(len, &buffer[at], buffer.size() - at);
    if (len > buffer.size() - at) {
      std::cerr << "Protobuf len is larger than file size" << std::endl;
      return 1;
    }
    at += len;

    switch (tag) {
    case 1:
      trace_packets++;
      trace_packet_size += len;
      // TODO: Dig into deeper protobufs.
      break;
    case 2:
      padding++;
      padding_size += len;
      break;
    }
  }

  std::cout << "End of protobuf" << std::endl;
  std::cout << "  Trace packets: " << trace_packets << " (" << trace_packet_size << " bytes)" << std::endl;
  std::cout << "  Padding: " << padding << " (" << padding_size << " bytes)" << std::endl;
  return 0;
}