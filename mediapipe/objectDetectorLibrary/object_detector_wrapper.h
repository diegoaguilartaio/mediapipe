#include <functional>
#include <cstdlib>
#include <memory>
#include <experimental/propagate_const>
#include <string>
#include <vector>
#include "Common.h"

typedef std::function<void(std::string)> ResultCallbackJSONLambdaSignature;

struct RelativeBoundingBoxMP {
    bool valid = false;
    bool endCapture = false;
    std::string label;
    float xmin;
    float ymin;
    float width;
    float height;
};

struct RelativeLandmarkMP {
    float x;
    float y;
    float z;
};



class MediapipeObjectDetectorLibrary {

private:
    struct impl;
    std::experimental::propagate_const<std::unique_ptr<impl>> pImpl;
    std::string kInputStream;
    std::string configString;

public:
    /**
     * @brief Construct a new Mediapipe Object Detector Library object
     * 
     * @param configJSON String containing a JSON with the configuration
     */
    MediapipeObjectDetectorLibrary();
    ~MediapipeObjectDetectorLibrary();

    void setOtherInputsString(std::string otherInputsString);
    int initApp(const char * logtostderr);
    int endApp();
    int initGraph(const char* customGraph, const char* configJSON);
    int startGraph();
    int AddFrameToInputStream(FrameInfo const * const inFrame);
    int ShutdownMPPGraph();
    void setResultCallback(void* context, void (*callback)(void*, RelativeBoundingBoxMP));
    void setResultCallbackJSON(void* context, void (*callback)(void*, std::string));
    void setResultCallbackJSONLambda(ResultCallbackJSONLambdaSignature callback);
};

