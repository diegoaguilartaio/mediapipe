#include <cstdlib>
#include <memory>
#include <experimental/propagate_const>
#include <string>
#include <vector>

struct RelativeBoundingBox {
    bool valid = false;
    bool endCapture = false;
    float xmin;
    float ymin;
    float width;
    float height;
};

struct RelativeLandmark {
    float x;
    float y;
    float z;
};



class MediapipeObjectDetectorLibrary {

private:
    struct impl;
    std::experimental::propagate_const<std::unique_ptr<impl>> pImpl;
    std::string kInputStream;
    std::string kOutputStream;
    std::string kOutputStreamType;

    std::string labelToDetect;

public:
    /**
     * @brief Construct a new Mediapipe Object Detector Library object
     * 
     * @param inputStreamName String containing the name of the graph input stream
     * @param ouputStreamName String containing the name of the graph ouput stream
     * @param label String containing the label to find Ex. "Jar"
     */
    MediapipeObjectDetectorLibrary(const char* inputStreamName, const char* ouputStreamName, const char* outputStreamType, const char* label);
    ~MediapipeObjectDetectorLibrary();

    int initApp(const char * logtostderr);
    int endApp();
    int initGraph(const char* customGraph);
    int startGraph();
    int AddFrameToInputStream(unsigned char const * const inFrame);
    int ShutdownMPPGraph();
    void setResultCallback(void* context, void (*callback)(void*, RelativeBoundingBox));
    void setResultCallbackForLandmarks(void* context, void (*callback)(void*, std::vector<std::vector<RelativeLandmark>>));
};

