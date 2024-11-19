#ifndef PTHREAD_TRACE_PERFETTO_H
#define PTHREAD_TRACE_PERFETTO_H

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

constexpr uint64_t trace_packet_tag = 1;
constexpr uint64_t padding_tag = 2;

}  // namespace perfetto

#endif  // PTHREAD_TRACE_PERFETTO_H