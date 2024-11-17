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

cc_binary(
    name = "pthread_trace.so",
    srcs = [
        "pthread_trace.cc",
        "perfetto.h",
    ],
    deps = [":proto"],
    visibility = ["//visibility:public"],
    linkshared = True,
)
