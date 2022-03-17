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
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/status_macros.h"

namespace mediapipe {

namespace {

constexpr char kDetectionTag[] = "DETECTION";


absl::Status ConvertDetectionToLabelId(const Detection& detection,
                                         int32* labelId) {
  if (detection.label_id_size() > 0)
    *labelId = detection.label_id(0);
  return absl::OkStatus();
}

}  // namespace

// Gets the label_id from a detection 

// Input:
//   DETECTION - `Detection`
//     A detection to be converted.
//
// Output:
//   label_id index
//
class DetectionToLabelidCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    RET_CHECK(cc->Inputs().HasTag(kDetectionTag));

    cc->Inputs().Tag(kDetectionTag).Set<Detection>();
    //cc->Outputs().Tag("OUT").Set<int32>();
    cc->Outputs().Index(0).Set<int32>();
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    const auto& detection = cc->Inputs().Tag(kDetectionTag).Get<Detection>();
    
    auto labelId = absl::make_unique<int32>();
    
    //labelId = 0;

    MP_RETURN_IF_ERROR(ConvertDetectionToLabelId(detection, labelId.get()));
    
    cc->Outputs().Index(0).Add(labelId.release(), cc->InputTimestamp());

    return absl::OkStatus();
  }
};

REGISTER_CALCULATOR(DetectionToLabelidCalculator);

}  // namespace mediapipe
