#!/bin/bash

is_perfetto_trace="$1"
pthread_trace="$2"
target="$3"

shift
shift
shift

fail() { rc=$?; (( $# )) && printf '%s\n' "$*" >&2; exit $(( rc == 0 ? 1 : rc )); }

trace=$(mktemp)

PTHREAD_TRACE_BUFFER_SIZE_KB=1 PTHREAD_TRACE_PATH=$trace LD_PRELOAD="$pthread_trace" "$target" "$@"

 $is_perfetto_trace $trace || fail "ERROR: $trace is not a valid perfetto trace"