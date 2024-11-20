package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//visibility:private"],
)

cc_test(
    name = "proto_test",
    srcs = ["proto_test.cc"],
    deps = [
        "//:proto",
        "@googletest//:gtest_main",
    ],
    size = "small",
)

# We don't have good tests of tracing itself. It's hard to test:
# - Traces are nondeterministic (they contain pointers and timestamps)
# - I haven't figured out how to use it without LD_PRELOAD
# The best we have for now is just running one of the benchmarks that use pthreads with tracing,
# and verifying it doesn't crash and we see expected pthread_trace output.
sh_test(
    name = "ld_preload_test_thread_benchmark",
    srcs = ["ld_preload_test.sh"],
    data = [
        ":thread_benchmark",
        "//:pthread_trace.so",
    ],
    args = [
        "$(location //:pthread_trace.so)", 
        "$(location :thread_benchmark)",
        "--benchmark_min_time=1x",
    ],
    size = "small",
)

cc_test(
    name = "thread_benchmark",
    srcs = ["thread_benchmark.cc"],
    deps = ["@google_benchmark//:benchmark_main"],
    args = ["--benchmark_min_time=1x"],
    size = "small",
)

cc_test(
    name = "traced_thread_benchmark",
    srcs = ["thread_benchmark.cc"],
    deps = [
        "//:pthread_trace",
        "@google_benchmark//:benchmark_main",
    ],
    args = ["--benchmark_min_time=1x"],
    size = "small",
)

sh_test(
    name = "linked_test_thread_benchmark",
    srcs = ["linked_test.sh"],
    data = [":traced_thread_benchmark"],
    args = [
        "$(location :traced_thread_benchmark)",
        "--benchmark_min_time=1x",
    ],
    size = "small",
)

cc_test(
    name = "proto_benchmark",
    srcs = ["proto_benchmark.cc"],
    deps = [
        "//:proto",
        "@google_benchmark//:benchmark_main"
    ],
    args = ["--benchmark_min_time=1x"],
    size = "small",
)
