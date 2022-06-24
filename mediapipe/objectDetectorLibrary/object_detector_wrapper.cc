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
#include "json.hpp"

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
#include "mediapipe/framework/formats/classification.pb.h"

#include <google/protobuf/util/json_util.h>


using json = nlohmann::json;

void to_json(json& j, const RelativeLandmarkMP& p)
{
    j = {{"x", p.x}, {"y", p.y}, {"z", p.z}};
}

struct MediapipeObjectDetectorLibrary::impl {
  mediapipe::CalculatorGraph *graph;
  cv::VideoCapture capture;
  //mediapipe::OutputStreamPoller * poller;
  std::unique_ptr<mediapipe::OutputStreamPoller> poller_;
  std::unique_ptr<mediapipe::OutputStreamPoller> poller_det;
  absl::Status run_status;
  void (*resultCallback)(void*, RelativeBoundingBoxMP) = nullptr;
  void (*resultCallbackJSON)(void*, std::string) = nullptr;
  ResultCallbackJSONLambdaSignature resultCallbackJSONLambda = nullptr;
  void* resultCallbackContext = nullptr;
  json otherInputsJSON;


  absl::Status _initGraph(const char* customGraph) {
    mediapipe::CalculatorGraphConfig config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(customGraph);
    
    LOG(INFO) << "Initialize the calculator graph.";
    //mediapipe::CalculatorGraph graph;
    graph = new mediapipe::CalculatorGraph();
    //MP_RETURN_IF_ERROR(graph->Initialize(config));
    return graph->Initialize(config);
  }

  absl::Status _startGraph(std::string configurationString) {
    LOG(INFO) << "Start running the calculator graph.";

      json configJSON = json::parse(configurationString);
      auto outputStreams = configJSON["outputStreams"];
      for (auto& outputStream:outputStreams){
        if (outputStream["type"] == "MULTILANDMARKS")
        {
          std::cout << "MULTILANDMARKS" << std::endl;
          MP_RETURN_IF_ERROR(
            graph->ObserveOutputStream(
              outputStream["name"],
              [this, outputStream](const mediapipe::Packet& packet) -> ::mediapipe::Status 
              {
                std::vector<std::vector<RelativeLandmarkMP>> ret;
                //TODO: add packet.IsEmpty()
                if (!packet.IsEmpty())
                {
                  auto& output_Det = packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
                  LOG(INFO) << "Number of Landmarks:" << output_Det.size() << std::endl;
                  for (const ::mediapipe::NormalizedLandmarkList& landmarkList : output_Det) {
                    LOG(INFO) << "LandmarkList size:" << landmarkList.landmark_size();
                    LOG(INFO) << "LandmarkList(0):" << landmarkList.landmark(0).x() << ", " << landmarkList.landmark(0).y() << ", " << landmarkList.landmark(0).z();
                    LOG(INFO) << "LandmarkList(5):" << landmarkList.landmark(5).x() << ", " << landmarkList.landmark(5).y() << ", " << landmarkList.landmark(5).z();
                    std::vector<RelativeLandmarkMP> resultLandmarks;
                    for (int i=0; i<landmarkList.landmark_size(); i++){
                      RelativeLandmarkMP resultLandmark;
                      resultLandmark.x = landmarkList.landmark(i).x();
                      resultLandmark.y = landmarkList.landmark(i).y();
                      resultLandmark.z = landmarkList.landmark(i).z();
                      resultLandmarks.push_back(resultLandmark);
                    }
                    ret.push_back(resultLandmarks);
                  }
                  json JSONret;
                  JSONret["name"] = outputStream["name"];
                  JSONret["type"] = outputStream["type"];
                  JSONret["timestamp"] = packet.Timestamp().Value();
                  JSONret["ret"] = ret;
                  if (resultCallbackJSON != nullptr){
                    resultCallbackJSON(resultCallbackContext, JSONret.dump());
                  }
                  if (resultCallbackJSONLambda != nullptr){
                    resultCallbackJSONLambda(JSONret.dump());
                  }

                }
                return mediapipe::OkStatus();
              }
            )
          );
        }
        if (outputStream["type"] == "LANDMARKS")
        {
          std::cout << "LANDMARKS" << std::endl;
          MP_RETURN_IF_ERROR(
            graph->ObserveOutputStream(
              outputStream["name"],
              [this, outputStream](const mediapipe::Packet& packet) -> ::mediapipe::Status 
              {
                std::vector<RelativeLandmarkMP> ret;
                auto& landmarkList = packet.Get<mediapipe::NormalizedLandmarkList>();
                LOG(INFO) << "LandmarkList size:" << landmarkList.landmark_size();
                for (int i=0; i<landmarkList.landmark_size(); i++){
                  RelativeLandmarkMP resultLandmark;
                  resultLandmark.x = landmarkList.landmark(i).x();
                  resultLandmark.y = landmarkList.landmark(i).y();
                  resultLandmark.z = landmarkList.landmark(i).z();
                  ret.push_back(resultLandmark);
                }                  
                json JSONret;
                JSONret["name"] = outputStream["name"];
                JSONret["type"] = outputStream["type"];
                JSONret["timestamp"] = packet.Timestamp().Value();
                JSONret["ret"] = ret;
                if (resultCallbackJSON != nullptr){
                  resultCallbackJSON(resultCallbackContext, JSONret.dump());
                }
                if (resultCallbackJSONLambda != nullptr){
                  resultCallbackJSONLambda(JSONret.dump());
                }
                return mediapipe::OkStatus();
              }
            )
          );
        }        
        if (outputStream["type"] == "HANDEDNESS")
        {
          std::cout << "HANDEDNESS" << std::endl;
          MP_RETURN_IF_ERROR(
            graph->ObserveOutputStream(
              outputStream["name"],
              [this,outputStream](const mediapipe::Packet& packet) -> ::mediapipe::Status 
              {
                std::vector<std::string> ret;
                auto& output_Det = packet.Get<std::vector<mediapipe::ClassificationList>>();
                for (const mediapipe::ClassificationList& handednessList : output_Det) {
                  if (handednessList.classification_size()>0) {
                    for (int i=0; i<handednessList.classification_size(); i++){
                      ret.push_back(handednessList.classification(i).label());
                    }
                  }
                }
                json JSONret;
                JSONret["name"] = outputStream["name"];
                JSONret["type"] = outputStream["type"];
                JSONret["timestamp"] = packet.Timestamp().Value();
                JSONret["ret"] = ret;
                if (resultCallbackJSON != nullptr){
                  resultCallbackJSON(resultCallbackContext, JSONret.dump());
                }
                if (resultCallbackJSONLambda != nullptr){
                  resultCallbackJSONLambda(JSONret.dump());
                }
                return mediapipe::OkStatus();
              }
            )
          );
        }
        if (outputStream["type"] == "INT32") {
          std::cout << "INT" << std::endl;
          MP_RETURN_IF_ERROR(
            graph->ObserveOutputStream(
              outputStream["name"],
              [this,outputStream](const mediapipe::Packet& packet) -> ::mediapipe::Status 
              {
                int32 ret = packet.Get<int32>();                
                json JSONret;
                JSONret["name"] = outputStream["name"];
                JSONret["type"] = outputStream["type"];
                JSONret["timestamp"] = packet.Timestamp().Value();
                JSONret["ret"] = ret;
                if (resultCallbackJSON != nullptr){
                  resultCallbackJSON(resultCallbackContext, JSONret.dump());
                }
                if (resultCallbackJSONLambda != nullptr){
                  resultCallbackJSONLambda(JSONret.dump());
                }
                return mediapipe::OkStatus();
              }
            )
          );

        }
        if (outputStream["type"] == "DETECTIONS")
        {
          MP_RETURN_IF_ERROR(
            graph->ObserveOutputStream(
              outputStream["name"],
              [this, outputStream](const mediapipe::Packet& packet) -> ::mediapipe::Status 
              {
                RelativeBoundingBoxMP ret;
                auto& output_Det = packet.Get<std::vector<mediapipe::Detection>>();
                
                LOG(INFO) << "Number of detections:" << output_Det.size();
                float score = 0;
                for (const ::mediapipe::Detection& detection : output_Det) 
                {
                  std::string retJson;
                  google::protobuf::util::MessageToJsonString(detection, &retJson);
                  json JSONret;
                  JSONret["name"] = outputStream["name"];
                  JSONret["type"] = outputStream["type"];
                  JSONret["timestamp"] = packet.Timestamp().Value();
                  JSONret["ret"] = retJson;
                  if (resultCallbackJSON != nullptr){
                    resultCallbackJSON(resultCallbackContext, JSONret.dump());
                  }
                  if (resultCallbackJSONLambda != nullptr){
                    resultCallbackJSONLambda(JSONret.dump());
                  }
                  
                  if (detection.label_size()>0)
                  {
                    for (auto&detectLabel : outputStream["detectLabel"])
                    {
                      if (detection.label(0) == detectLabel) 
                      {
                        ret.valid = true;
                        ret.label = detection.label(0);
                        ret.xmin = detection.location_data().relative_bounding_box().xmin();
                        ret.ymin = detection.location_data().relative_bounding_box().ymin();
                        ret.width = detection.location_data().relative_bounding_box().width();
                        ret.height = detection.location_data().relative_bounding_box().height();
                      }
                      if (resultCallback != nullptr)
                      {
                        resultCallback(resultCallbackContext, ret);
                      }
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
        }
      }
    MP_RETURN_IF_ERROR(graph->StartRun({}));
    return absl::OkStatus();
  }

  absl::Status ShutdownMPPGraph_(std::string is) {

    LOG(INFO) << "Shutting down.";
    MP_RETURN_IF_ERROR(graph->CloseInputStream(is));
    for (auto& otherInput: this->otherInputsJSON)
    {
      MP_RETURN_IF_ERROR(graph->CloseInputStream(otherInput["name"]));
    }
    return graph->WaitUntilDone();
  }
};

MediapipeObjectDetectorLibrary::MediapipeObjectDetectorLibrary() : pImpl(std::make_unique<impl>())
{
}

MediapipeObjectDetectorLibrary::~MediapipeObjectDetectorLibrary()
{

}

void MediapipeObjectDetectorLibrary::setOtherInputsString(std::string otherInputsString)
{
  pImpl->otherInputsJSON = json::parse(otherInputsString);
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

int MediapipeObjectDetectorLibrary::initGraph(const char* customGraph, const char* configJSON) {
  configString = configJSON;
  json jsonC = json::parse(configString);
  kInputStream = jsonC["inputStreams"]["inputStream"];
  pImpl->otherInputsJSON = jsonC["inputStreams"]["others"];  
  
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
  absl::Status retStatus = pImpl->_startGraph(configString);
  if (!retStatus.ok()) {
    LOG(ERROR) << "Failed startGraph: " << retStatus.message();
    return EXIT_FAILURE;
  } else {
    LOG(INFO) << "startGraph Success!";
  }
  return 0;
}

int MediapipeObjectDetectorLibrary::AddFrameToInputStream(FrameInfo const * const inFrame) {

  cv::Mat inputFrame(inFrame->Height, inFrame->Width, CV_8UC3, (void*)(inFrame->Data));
  cv::Mat frame;
  cv::resize(inputFrame, frame, cv::Size(640,480));
  
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
  for (auto& otherInput: pImpl->otherInputsJSON)
  {
    if (otherInput["type"].get<std::string>() == "bool") {
      absl::Status retStatus2 = pImpl->graph->AddPacketToInputStream(
        otherInput["name"].get<std::string>(), mediapipe::Adopt(new bool(otherInput["value"].get<bool>()))
                          .At(mediapipe::Timestamp(frame_timestamp_us)));
      if (!retStatus2.ok()) {
        LOG(ERROR) << "Failed AddFrameToInputStream: " << otherInput["name"].get<std::string>()<< ", " << retStatus2.message();
      }
  
    } else if (otherInput["type"].get<std::string>() == "int") {
      absl::Status retStatus3 = pImpl->graph->AddPacketToInputStream(
        otherInput["name"].get<std::string>(), mediapipe::Adopt(new int(otherInput["value"].get<int>()))
                          .At(mediapipe::Timestamp(frame_timestamp_us)));  
      if (!retStatus3.ok()) {
        LOG(ERROR) << "Failed AddFrameToInputStream: " << otherInput["name"].get<std::string>()<< ", " << retStatus3.message();
      }

    } else if (otherInput["type"].get<std::string>() == "string") {
      absl::Status retStatus4 = pImpl->graph->AddPacketToInputStream(
        otherInput["name"].get<std::string>(), mediapipe::Adopt(new std::string(otherInput["value"].get<std::string>()))
                          .At(mediapipe::Timestamp(frame_timestamp_us)));  
    } 
  }
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

void MediapipeObjectDetectorLibrary::setResultCallback(void* context, void (*callback)(void*, RelativeBoundingBoxMP))
{
  pImpl->resultCallback = callback;
  pImpl->resultCallbackContext = context;
}


void MediapipeObjectDetectorLibrary::setResultCallbackJSON(void* context, void (*callback)(void*, std::string))
{
  pImpl->resultCallbackJSON = callback;
  pImpl->resultCallbackContext = context;
}


void MediapipeObjectDetectorLibrary::setResultCallbackJSONLambda(ResultCallbackJSONLambdaSignature callback)
{
  pImpl->resultCallbackJSONLambda = callback;
}