load("@rules_license//rules:license.bzl", "license")

package(
    default_applicable_licenses = [":license"],
    default_visibility = ["//visibility:public"],
)

license(name = "license")

licenses(["notice"])

exports_files(["LICENSE"])

cc_library(
    name = "proto",
    hdrs = ["proto.h"],
)

# This can also be used by linking to this library, instead of using LD_PRELOAD with the target below.
cc_library(
    name = "pthread_trace",
    srcs = [
        "pthread_trace.cc",
        "perfetto.h",
    ],
    deps = [":proto"],
    alwayslink = 1,
    linkopts = ["-ldl", "-lpthread", "-lc"],
)

# This can also be compiled easily without bazel:
# c++ -shared -fPIC -ldl -O2 -DNDEBUG -std=c++14 pthread_trace.cc -o pthread_trace.so
cc_binary(
    name = "pthread_trace.so",
    deps = [":pthread_trace"],
    visibility = ["//visibility:public"],
    linkshared = True,
)
