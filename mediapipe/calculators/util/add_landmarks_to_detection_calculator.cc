// Copyright 2020 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <memory>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_macros.h"

namespace mediapipe {

namespace {

constexpr char kDetectionTag[] = "DETECTION";
constexpr char kLandmarksTag[] = "LANDMARKS";


absl::Status addLandmarksToDetection(const Detection& detection,
                                         const NormalizedLandmarkList& landmarks,
                                         Detection* detection_out) {
  
  detection_out->CopyFrom(detection);
  auto landmarks_out = detection_out->mutable_landmark_list();
  landmarks_out->CopyFrom(landmarks);
  return absl::OkStatus();
}

}  // namespace

// Adds the landmarks in to a detection.
//
// Input:
//   DETECTION - `Detection`
//     A detection to be converted.
//   LANDMARKS - 'NormalizedLandmarkList"
//     The landmarks to be added.
//
// Output:
//   DETECTION - `Detection`
//     A converted Detection including the normalized landmark list.
//
// Example:
//
//   node {
//     calculator: "AddLandmarksToDetectionCalculator"
//     input_stream: "DETECTION:detection"
//     input_stream: "LANDMARKS:landmarks"
//     output_stream: "DETECTION:detection_out"
//   }
//
class AddLandmarksToDetectionCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {

    cc->Inputs().Tag(kDetectionTag).Set<Detection>();
    cc->Inputs().Tag(kLandmarksTag).Set<NormalizedLandmarkList>();
    cc->Outputs().Tag(kDetectionTag).Set<Detection>();

    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    const auto& detection = cc->Inputs().Tag(kDetectionTag).Get<Detection>();
    const auto& landmarks = cc->Inputs().Tag(kLandmarksTag).Get<NormalizedLandmarkList>();

    auto detection_out = absl::make_unique<Detection>();

    //auto landmarks = absl::make_unique<NormalizedLandmarkList>();
    MP_RETURN_IF_ERROR(addLandmarksToDetection(detection, landmarks, detection_out.get()));

    cc->Outputs()
        .Tag(kDetectionTag)
        .Add(detection_out.release(), cc->InputTimestamp());

    return absl::OkStatus();
  }
};

REGISTER_CALCULATOR(AddLandmarksToDetectionCalculator);

}  // namespace mediapipe
