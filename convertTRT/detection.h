#ifndef __REMOTE_DETECTION__
#define __REMOTE_DETECTION__

#include <string>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>

#include "logging.h"
#include "utils.h"
#include "calibrator.h"
#include "yololayer.h"


using namespace nvinfer1;

#define YOLOV5_USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
#define YOLOV5_DEVICE 0  // GPU id
#define YOLOV5_NMS_THRESH 0.4
#define YOLOV5_CONF_THRESH 0.5
#define YOLOV5_BATCH_SIZE 1

#define YOLOV5_INPUT_H  Yolo::INPUT_H
#define YOLOV5_INPUT_W  Yolo::INPUT_W
#define YOLOV5_CLASS_NUM  Yolo::CLASS_NUM
#define YOLOV5_OUTPUT_SIZE  (Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1)  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
#define YOLOV5_INPUT_BLOB_NAME  "data"
#define YOLOV5_OUTPUT_BLOB_NAME  "prob"


class Barrier
{
public:
    Barrier() {
        id = 0;
        classid = 0;
        x = 0;
        y = 0;
        width = 0;
        height = 0;
        confidence = 0;
        elevation = 0;
        lon = 0;
        lat = 0;
    };
    Barrier(int id, float x, float y, float width, float height, int classid, float confidence) :id(id), x(x), y(y), width(width), height(height), classid(classid), confidence(confidence) {};
    ~Barrier() {};

public:
    int id;
    int classid;
    float x;
    float y;
    float width;
    float height;

    float lon;
    float lat;
    float elevation;

    float confidence;


public:
    int get_id() { return id; }
    float get_x() { return x; }
    float get_y() { return y; }
    float get_width() { return width; }
    float get_height() { return height; }
    int get_classid() { return classid; }

    float get_lon() { return lon; }
    float get_lat() { return lat; }
    float get_elevation() { return elevation; }
    float get_conf() { return confidence; }



    void set_lon(float para) { lon = para; }
    void set_lat(float para) { lat = para; }
    void set_elevation(float para) { elevation = para; }
};






class detection
{
    Logger gLogger;

    float data[YOLOV5_BATCH_SIZE * 3 * YOLOV5_INPUT_H * YOLOV5_INPUT_W];
    float prob[YOLOV5_BATCH_SIZE * YOLOV5_OUTPUT_SIZE];

    IRuntime* runtime;
    ICudaEngine* engine;
    IExecutionContext* context;
    cudaStream_t stream;

    int inputIndex;
    int outputIndex;
   
    void* buffers[2];

    split_param s_param;
    std::vector<cv::Rect> split_result;

    // convert param
    std::string wts_name;
    std::string engine_name;
    bool is_p6;
    float gd, gw;

public:

    detection(std::string wts_name, std::string engine_name="defualt.engine", std::string net="m", bool is_p6 = false, float gd = 0.0f, float gw = 0.0f);
    detection(std::string engine_name, split_param param);
    ~detection();

    void convert2engine();
    void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream, bool& is_p6, float& gd, float& gw, std::string& wts_name);

    int preprocess(std::vector<cv::Mat> image, std::vector <cv::Rect> rects, int bs);
    void process(std::vector<cv::Mat> image, std::vector<Barrier*>& output);
    void split();
    void merge(std::vector<std::vector<Yolo::Detection> > yolo_result, std::vector <Barrier*>& output);
    void doInference(IExecutionContext& context, cudaStream_t& stream, void** buffers, float* input, float* output, int batchSize);
};

#endif // !__REMOTE_DETECTION__


