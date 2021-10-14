// Copyright 2019 The MediaPipe Authors.
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
//
// An example of sending OpenCV webcam frames into a MediaPipe graph.
#include "object_detector_wrapper.h"

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"

#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"

    //constexpr char kInputStream[] = "input_video";
    //constexpr char kOutputStream[] = "output_detections";
    //std::string labelToDetect;


struct MediapipeObjectDetectorLibrary::impl {
  mediapipe::CalculatorGraph *graph;
  cv::VideoCapture capture;
  //mediapipe::OutputStreamPoller * poller;
  std::unique_ptr<mediapipe::OutputStreamPoller> poller_;
  std::unique_ptr<mediapipe::OutputStreamPoller> poller_det;
  absl::Status run_status;
  void (*resultCallback)(void*, RelativeBoundingBox) = nullptr;
  void (*resultCallbackForLandmarks)(void*, std::vector<std::vector<RelativeLandmark>>) = nullptr;
  void* resultCallbackContext = nullptr;


  absl::Status _initGraph(const char* customGraph) {
    mediapipe::CalculatorGraphConfig config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(customGraph);
    
    LOG(INFO) << "Initialize the calculator graph.";
    //mediapipe::CalculatorGraph graph;
    graph = new mediapipe::CalculatorGraph();
    //MP_RETURN_IF_ERROR(graph->Initialize(config));
    return graph->Initialize(config);
  }

  absl::Status _startGraph(std::string os, std::string osType,std::string labelTD) {
    LOG(INFO) << "Start running the calculator graph.";

    if (osType == "DETECTIONS")
    {
      MP_RETURN_IF_ERROR(
        graph->ObserveOutputStream(
          os,
          [this, labelTD](const mediapipe::Packet& packet) -> ::mediapipe::Status 
          {
            RelativeBoundingBox ret;
            auto& output_Det = packet.Get<std::vector<mediapipe::Detection>>();
            LOG(INFO) << "Number of detections:" << output_Det.size();
            float score = 0;
            for (const ::mediapipe::Detection& detection : output_Det) {
              if (detection.label_size()>0)
              {
                if (detection.label(0) == labelTD) {
                  ret.valid = true;
                  ret.xmin = detection.location_data().relative_bounding_box().xmin();
                  ret.ymin = detection.location_data().relative_bounding_box().ymin();
                  ret.width = detection.location_data().relative_bounding_box().width();
                  ret.height = detection.location_data().relative_bounding_box().height();
                }
                if (resultCallback != nullptr){
                  resultCallback(resultCallbackContext, ret);
                }
              }else{
                LOG(INFO) << "Detection does not have label," << detection.location_data().format();
                if (detection.score(0) > score){
                  LOG(INFO) << "score:" << detection.score(0);
                  score = detection.score(0);
                  ret.valid = true;
                  ret.xmin = detection.location_data().relative_bounding_box().xmin();
                  ret.ymin = detection.location_data().relative_bounding_box().ymin();
                  ret.width = detection.location_data().relative_bounding_box().width();
                  ret.height = detection.location_data().relative_bounding_box().height();
                }
              }              
            }
            if (resultCallback != nullptr){
              resultCallback(resultCallbackContext, ret);
            }

            return mediapipe::OkStatus();
          }
        )
      );
    } else if (osType == "MULTILANDMARKS")
    {

      MP_RETURN_IF_ERROR(
        graph->ObserveOutputStream(
          os,
          [this, labelTD](const mediapipe::Packet& packet) -> ::mediapipe::Status 
          {
            std::vector<std::vector<RelativeLandmark>> ret;
            auto& output_Det = packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
            LOG(INFO) << "Number of Landmarks:" << output_Det.size() << std::endl;
            for (const ::mediapipe::NormalizedLandmarkList& landmarkList : output_Det) {
              LOG(INFO) << "LandmarkList size:" << landmarkList.landmark_size();
              LOG(INFO) << "LandmarkList(0):" << landmarkList.landmark(0).x() << ", " << landmarkList.landmark(0).y() << ", " << landmarkList.landmark(0).z();
              LOG(INFO) << "LandmarkList(5):" << landmarkList.landmark(5).x() << ", " << landmarkList.landmark(5).y() << ", " << landmarkList.landmark(5).z();
              std::vector<RelativeLandmark> resultLandmarks;
              for (int i=0; i<landmarkList.landmark_size(); i++){
                RelativeLandmark resultLandmark;
                resultLandmark.x = landmarkList.landmark(i).x();
                resultLandmark.y = landmarkList.landmark(i).y();
                resultLandmark.z = landmarkList.landmark(i).z();
                resultLandmarks.push_back(resultLandmark);
              }
              ret.push_back(resultLandmarks);
            }
            if (resultCallbackForLandmarks != nullptr){
              resultCallbackForLandmarks(resultCallbackContext, ret);
            }
            return mediapipe::OkStatus();
          }
        )
      );


    }
    MP_RETURN_IF_ERROR(graph->StartRun({}));
    return absl::OkStatus();
  }

  absl::Status ShutdownMPPGraph_(std::string is) {

    LOG(INFO) << "Shutting down.";
    MP_RETURN_IF_ERROR(graph->CloseInputStream(is));
    return graph->WaitUntilDone();
  }

};

MediapipeObjectDetectorLibrary::MediapipeObjectDetectorLibrary(
  const char* inputStreamName, 
  const char* ouputStreamName, 
  const char* ouputStreamType, 
  const char* label ) : pImpl(std::make_unique<impl>())
{
  kInputStream = inputStreamName;
  kOutputStream = ouputStreamName;
  kOutputStreamType = ouputStreamType;
  labelToDetect = label;
}

MediapipeObjectDetectorLibrary::~MediapipeObjectDetectorLibrary()
{

}

int MediapipeObjectDetectorLibrary::initApp(const char * logtostderr) {
  google::InitGoogleLogging(logtostderr);
  return 0;
}

int MediapipeObjectDetectorLibrary::endApp() {

  if (!pImpl->run_status.ok()) {
    LOG(ERROR) << "Failed to run the graph: " << pImpl->run_status.message();
    return EXIT_FAILURE;
  } else {
    LOG(INFO) << "Success!";
  }
  return 0;
}

int MediapipeObjectDetectorLibrary::initGraph(const char* customGraph) {
  absl::Status retStatus = pImpl->_initGraph(customGraph);
  if (!retStatus.ok()) {
    LOG(ERROR) << "Failed initGraph: " << retStatus.message();
    return EXIT_FAILURE;
  } else {
    LOG(INFO) << "initGraph Success!";
  }
  return 0;
}

int MediapipeObjectDetectorLibrary::startGraph() {
  absl::Status retStatus = pImpl->_startGraph(kOutputStream, kOutputStreamType, labelToDetect);
  if (!retStatus.ok()) {
    LOG(ERROR) << "Failed startGraph: " << retStatus.message();
    return EXIT_FAILURE;
  } else {
    LOG(INFO) << "startGraph Success!";
  }
  return 0;
}

int MediapipeObjectDetectorLibrary::AddFrameToInputStream(unsigned char const * const inFrame) {

  cv::Mat frame(480,640, CV_8UC3, (void *) inFrame);

  cv::Mat camera_frame;
  cv::cvtColor(frame, camera_frame, cv::COLOR_BGR2RGB);

  auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
        mediapipe::ImageFrame::kDefaultAlignmentBoundary);
  cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
  input_frame->uuid = std::rand();
  //LOG(INFO) << "Creating ImageFrame:" << input_frame->uuid;
  camera_frame.copyTo(input_frame_mat);
  // Send image packet into the graph.
  size_t frame_timestamp_us =
      (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
  absl::Status retStatus = pImpl->graph->AddPacketToInputStream(
      kInputStream, mediapipe::Adopt(input_frame.release())
                        .At(mediapipe::Timestamp(frame_timestamp_us)));
  if (!retStatus.ok()) {
    LOG(ERROR) << "Failed AddFrameToInputStream: " << retStatus.message();
  }
  return 0;
}

int MediapipeObjectDetectorLibrary::ShutdownMPPGraph() {
   pImpl->run_status = pImpl->ShutdownMPPGraph_(kInputStream); 
   delete pImpl->graph;
   return 0;
}

void MediapipeObjectDetectorLibrary::setResultCallback(void* context, void (*callback)(void*, RelativeBoundingBox))
{
  pImpl->resultCallback = callback;
  pImpl->resultCallbackContext = context;
}

void MediapipeObjectDetectorLibrary::setResultCallbackForLandmarks(void* context, void (*callback)(void*, std::vector<std::vector<RelativeLandmark>>))
{
  pImpl->resultCallbackForLandmarks = callback;
  pImpl->resultCallbackContext = context;
}