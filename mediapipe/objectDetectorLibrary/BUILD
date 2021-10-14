cc_library(
    name = "object_detector_wrapper",
    srcs = ["object_detector_wrapper.cc"],
    hdrs = ["object_detector_wrapper.h"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/calculators/core:packet_presence_calculator",
        "//mediapipe/framework/formats:detection_cc_proto",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/graphs/hand_tracking:desktop_tflite_calculators",
         "//mediapipe/graphs/object_detection:desktop_tflite_calculators",
    ],
    visibility = ["//visibility:public"],
    alwayslink = True,
)


cc_library(
    name = "mediapipe_c",
    deps = [
        ":object_detector_wrapper",
        "//mediapipe/graphs/object_detection:desktop_tflite_calculators",
    ],
    visibility = ["//visibility:public"],
    alwayslink = True,
)

cc_binary(
    name = "libmediapipe_c.so",
    deps = [":mediapipe_c"],
    linkshared = True,
    linkstatic = True,
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "libmediapipe_c.dylib",
    deps = [":mediapipe_c"],
    linkshared = True,
    linkstatic = True,
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "libmediapipe_c.dll",
    deps = [":mediapipe_c"],
    linkshared = True,
    linkstatic = True,
    visibility = ["//visibility:public"],
)

alias(
    name = "libmediapipe_c",
    actual = select({
        "@bazel_tools//src/conditions:darwin": ":libmediapipe_c.dylib",
        "@bazel_tools//src/conditions:windows": ":libmediapipe_c.dll",
        "//conditions:default": ":libmediapipe_c.so",
    }),
)
