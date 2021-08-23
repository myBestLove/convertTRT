#ifndef __REMOTE_SEMANTIC__
#define __REMOTE_SEMANTIC__

#include <string>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>

#include "utils.h"
#include "logging.h"


using namespace nvinfer1;


#define SEG_USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
#define SEG_DEVICE 0  // GPU id
#define SEG_BATCH_SIZE 1

#define SEG_INPUT_H  512 
#define SEG_INPUT_W  512
#define SEG_CLASS_NUM  3
#define SEG_OUTPUT_SIZE  (SEG_CLASS_NUM*SEG_INPUT_H*SEG_INPUT_W)  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
#define SEG_INPUT_BLOB_NAME  "data"
#define SEG_OUTPUT_BLOB_NAME  "prob"



class semantic
{
    Logger gLogger;

    float data[SEG_BATCH_SIZE * 3 * SEG_INPUT_H * SEG_INPUT_W];
    float prob[SEG_BATCH_SIZE * SEG_OUTPUT_SIZE];

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


public:

    semantic(std::string wts_name, std::string engine_name="defualt.engine");
    semantic(std::string engine_name, split_param param);
    ~semantic();

    void convert2engine();
    void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream);

    int preprocess(std::vector<cv::Mat> image, std::vector <cv::Rect> rects, int bs);
    void process(std::vector<cv::Mat> image, cv::Mat& output);
    void split();
    void merge(float* probs, cv::Mat& output);
    void doInference(IExecutionContext& context, cudaStream_t& stream, void** buffers, float* input, float* output, int batchSize);
};

#endif // !__REMOTE_SEMANTIC__


