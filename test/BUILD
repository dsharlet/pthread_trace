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

cc_test(
    name = "benchmark",
    srcs = ["benchmark.cc"],
    deps = ["@google_benchmark//:benchmark_main"],
    args=["--benchmark_min_time=1x"],
    size = "small",
)

cc_test(
    name = "proto_benchmark",
    srcs = ["proto_benchmark.cc"],
    deps = [
        "//:proto",
        "@google_benchmark//:benchmark_main"
    ],
    args=["--benchmark_min_time=1x"],
    size = "small",
)
