#!/bin/bash

pthread_trace="$1"
target="$2"

shift
shift

fail() { rc=$?; (( $# )) && printf '%s\n' "$*" >&2; exit $(( rc == 0 ? 1 : rc )); }

trace=$(mktemp)

exec 5>&1
output=$(PTHREAD_TRACE_BUFFER_SIZE_KB=1 PTHREAD_TRACE_PATH=$trace LD_PRELOAD="$pthread_trace" "$target" "$@" 2>&1 | tee >(cat - >&5))

echo "$output" | grep "pthread_trace: Recorded.*KB trace" > /dev/null || fail "Did not find pthread_trace report in output"