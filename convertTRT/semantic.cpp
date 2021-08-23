#include <fstream>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include "semantic.h"


// TensorRT weight files have a simple space delimited format:
   // [type] [size] <data x size in hex>
std::map<std::string, Weights> UNET_loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file. please check if the .wts file path is right!!!!!!");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{ DataType::kFLOAT, nullptr, 0 };
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;

        wt.count = size;
        weightMap[name] = wt;
        //std::cout << name << std::endl;
    }

    return weightMap;
}

IScaleLayer* UNET_addBatchNorm2d(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float* gamma = (float*)weightMap[lname + ".weight"].values;
    float* beta = (float*)weightMap[lname + ".bias"].values;
    float* mean = (float*)weightMap[lname + ".running_mean"].values;
    float* var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;

    float* scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{ DataType::kFLOAT, scval, len };

    float* shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{ DataType::kFLOAT, shval, len };

    float* pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{ DataType::kFLOAT, pval, len };

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

ILayer* UNET_convBlock(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, int s, int g, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    int p = ksize / 2;
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ ksize, ksize }, weightMap[lname + ".conv.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{ s, s });
    conv1->setPaddingNd(DimsHW{ p, p });
    conv1->setNbGroups(g);
    IScaleLayer* bn1 = UNET_addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".bn", 1e-3);

    // hard_swish = x * hard_sigmoid
    auto hsig = network->addActivation(*bn1->getOutput(0), ActivationType::kHARD_SIGMOID);
    assert(hsig);
    hsig->setAlpha(1.0 / 6.0);
    hsig->setBeta(0.5);
    auto ew = network->addElementWise(*bn1->getOutput(0), *hsig->getOutput(0), ElementWiseOperation::kPROD);
    assert(ew);
    return ew;
}

ILayer* doubleConv(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, std::string lname, int midch) {
    // Weights emptywts{DataType::kFLOAT, nullptr, 0};
    // int p = ksize / 2;
    // if (midch==NULL){
    //     midch = outch;
    // }
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, midch, DimsHW{ ksize, ksize }, weightMap[lname + ".double_conv.0.weight"], weightMap[lname + ".double_conv.0.bias"]);
    conv1->setStrideNd(DimsHW{ 1, 1 });
    conv1->setPaddingNd(DimsHW{ 1, 1 });
    conv1->setNbGroups(1);
    IScaleLayer* bn1 = UNET_addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".double_conv.1", 0);
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kLEAKY_RELU);
    IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), outch, DimsHW{ 3, 3 }, weightMap[lname + ".double_conv.3.weight"], weightMap[lname + ".double_conv.3.bias"]);
    conv2->setStrideNd(DimsHW{ 1, 1 });
    conv2->setPaddingNd(DimsHW{ 1, 1 });
    conv2->setNbGroups(1);
    IScaleLayer* bn2 = UNET_addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + ".double_conv.4", 0);
    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kLEAKY_RELU);
    assert(relu2);
    return relu2;
}

ILayer* down(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int p, std::string lname) {

    IPoolingLayer* pool1 = network->addPoolingNd(input, PoolingType::kMAX, DimsHW{ 2, 2 });
    assert(pool1);
    ILayer* dcov1 = doubleConv(network, weightMap, *pool1->getOutput(0), outch, 3, lname, outch);
    assert(dcov1);
    return dcov1;
}

ILayer* up(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input1, ITensor& input2, int resize, int outch, int midch, std::string lname, bool bilinear = true) {
    if (bilinear)
    {
        float* deval = reinterpret_cast<float*>(malloc(sizeof(float) * resize * 2 * 2));
        for (int i = 0; i < resize * 2 * 2; i++) {
            deval[i] = 1.0;
        }
        Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
        Weights deconvwts1{ DataType::kFLOAT, deval, resize * 2 * 2 };
        IDeconvolutionLayer* deconv1 = network->addDeconvolutionNd(input1, resize, DimsHW{ 2, 2 }, deconvwts1, emptywts);
        deconv1->setStrideNd(DimsHW{ 2, 2 });
        deconv1->setNbGroups(resize);
        // weightMap["deconvwts."+lname] = deconvwts1;

        int diffx = input2.getDimensions().d[1] - deconv1->getOutput(0)->getDimensions().d[1];
        int diffy = input2.getDimensions().d[2] - deconv1->getOutput(0)->getDimensions().d[2];
        // IPoolingLayer* pool1 = network->addPooling(dcov1, PoolingType::kMAX, DimsHW{2, 2});
        // pool1->setStrideNd(DimsHW{2, 2});
        // dcov1->add_pading
        ILayer* pad1 = network->addPaddingNd(*deconv1->getOutput(0), DimsHW{ diffx / 2, diffy / 2 }, DimsHW{ diffx - (diffx / 2), diffy - (diffy / 2) });
        // dcov1->setPaddingNd(DimsHW{diffx / 2, diffx - diffx / 2},DimsHW{diffy / 2, diffy - diffy / 2});
        ITensor* inputTensors[] = { &input2,pad1->getOutput(0) };
        auto cat = network->addConcatenation(inputTensors, 2);
        assert(cat);
        if (midch == 64) {
            ILayer* dcov1 = doubleConv(network, weightMap, *cat->getOutput(0), outch, 3, lname + ".conv", outch);
            assert(dcov1);
            return dcov1;
        }
        else {
            int midch1 = outch / 2;
            ILayer* dcov1 = doubleConv(network, weightMap, *cat->getOutput(0), midch1, 3, lname + ".conv", outch);
            assert(dcov1);
            return dcov1;
        }
    }
    else
    {
        int midch1 = outch / 2;
        IDeconvolutionLayer* deconv1 = network->addDeconvolutionNd(input1, outch, DimsHW{ 2, 2 }, weightMap[lname + ".up.weight"], weightMap[lname + ".up.bias"]);
        //deconv1->setStrideNd(DimsHW{ 2, 2 });
        //deconv1->setNbGroups(resize);

        int diffx = input2.getDimensions().d[1] - deconv1->getOutput(0)->getDimensions().d[1];
        int diffy = input2.getDimensions().d[2] - deconv1->getOutput(0)->getDimensions().d[2];

        ILayer* pad1 = network->addPaddingNd(*deconv1->getOutput(0), DimsHW{ diffx / 2, diffy / 2 }, DimsHW{ diffx - (diffx / 2), diffy - (diffy / 2) });
        ITensor* inputTensors[] = { &input2,pad1->getOutput(0) };
        auto cat = network->addConcatenation(inputTensors, 2);
        ILayer* dcov1 = doubleConv(network, weightMap, *cat->getOutput(0), outch, 3, lname + ".conv", outch);
        assert(dcov1);
        return dcov1;
    }

    // assert(dcov1);

    // return dcov1;
}

ILayer* outConv(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, std::string lname) {
    // Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv1 = network->addConvolutionNd(input, SEG_CLASS_NUM, DimsHW{ 1, 1 }, weightMap[lname + ".conv.weight"], weightMap[lname + ".conv.bias"]);
    assert(conv1);
    conv1->setStrideNd(DimsHW{ 1, 1 });
    conv1->setPaddingNd(DimsHW{ 0, 0 });
    conv1->setNbGroups(1);
    return conv1;
}



ICudaEngine* createEngine_l(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, const std::string & wts_name) {
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(SEG_INPUT_BLOB_NAME, dt, Dims3{ 3, SEG_INPUT_H, SEG_INPUT_W });
    assert(data);

    std::map<std::string, Weights> weightMap = UNET_loadWeights(wts_name);
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    // build network
    auto x1 = doubleConv(network, weightMap, *data, 64, 3, "inc", 64);
    auto x2 = down(network, weightMap, *x1->getOutput(0), 128, 1, "down1.maxpool_conv.1");
    auto x3 = down(network, weightMap, *x2->getOutput(0), 256, 1, "down2.maxpool_conv.1");
    auto x4 = down(network, weightMap, *x3->getOutput(0), 512, 1, "down3.maxpool_conv.1");
    auto x5 = down(network, weightMap, *x4->getOutput(0), 512, 1, "down4.maxpool_conv.1");  // biliear
    // auto x5 = down(network, weightMap, *x4->getOutput(0), 1024, 1, "down4.maxpool_conv.1"); // deconv
    ILayer* x6 = up(network, weightMap, *x5->getOutput(0), *x4->getOutput(0), 512, 512, 512, "up1");
    ILayer* x7 = up(network, weightMap, *x6->getOutput(0), *x3->getOutput(0), 256, 256, 256, "up2");
    ILayer* x8 = up(network, weightMap, *x7->getOutput(0), *x2->getOutput(0), 128, 128, 128, "up3");
    ILayer* x9 = up(network, weightMap, *x8->getOutput(0), *x1->getOutput(0), 64, 64, 64, "up4");
    ILayer* x10 = outConv(network, weightMap, *x9->getOutput(0), SEG_OUTPUT_SIZE, "outc");
    std::cout << "set name out" << std::endl;
    x10->getOutput(0)->setName(SEG_OUTPUT_BLOB_NAME);
    network->markOutput(*x10->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#ifdef USE_FP16
    config->setFlag(BuilderFlag::kFP16);
#endif
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));
    }

    return engine;
}

void semantic::APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    // ICudaEngine* engine = (CREATENET(NET))(maxBatchSize, builder, config, DataType::kFLOAT);
    ICudaEngine* engine = createEngine_l(maxBatchSize, builder, config, DataType::kFLOAT, wts_name);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}



semantic::semantic(std::string wts_name, std::string engine_name)
{
    cudaSetDevice(SEG_DEVICE);

    this->wts_name = wts_name;
    this->engine_name = engine_name;

}

semantic::semantic(std::string engine_name, split_param param)
{
    cudaSetDevice(SEG_DEVICE);

    s_param = param;

    // deserialize the .engine and run inference
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        return ;
    }
    char* trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    this->runtime = createInferRuntime(gLogger);
    assert(this->runtime != nullptr);
    this->engine = this->runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    this->context = this->engine->createExecutionContext();
    assert(this->context != nullptr);
    delete[] trtModelStream;
    assert(this->engine->getNbBindings() == 2);

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    this->inputIndex = this->engine->getBindingIndex(SEG_INPUT_BLOB_NAME);
    this->outputIndex = this->engine->getBindingIndex(SEG_OUTPUT_BLOB_NAME);
    assert(this->inputIndex == 0);
    assert(this->outputIndex == 1);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[this->inputIndex], SEG_BATCH_SIZE * 3 * SEG_INPUT_H * SEG_INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[this->outputIndex], SEG_BATCH_SIZE * SEG_OUTPUT_SIZE * sizeof(float)));
    // Create stream
    CHECK(cudaStreamCreate(&this->stream));

}

semantic::~semantic()
{
    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
}

void semantic::convert2engine()
{
    cudaSetDevice(SEG_DEVICE);
    // create a model using the API directly and serialize it to a stream
    char* trtModelStream{ nullptr };
    size_t size{ 0 };
    if (!wts_name.empty()) {
        IHostMemory* modelStream{ nullptr };
        APIToModel(SEG_BATCH_SIZE, &modelStream);
        assert(modelStream != nullptr);
        std::cout << "start write engine model" << std::endl;
        std::ofstream p(engine_name, std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return ;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return ;
    }
   
}

// 输入为r，g，b
int semantic::preprocess(std::vector<cv::Mat> image, std::vector <cv::Rect> rects, int bs)
{
    cv::Mat r_image, g_image, b_image;

    if (image.size() >= 3)
    {
        r_image = image[0];
        g_image = image[1];
        b_image = image[2];
    }
    else {
        std::cout << "input image only " << image.size() << "channel" << std::endl;
        return -1;
    }

    if (s_param.crop_h != SEG_INPUT_H || s_param.crop_w != SEG_INPUT_W)
    {
        // TODO
    }


    for (int b = 0; b < bs; b++)
    {
        cv::Rect rect = rects[b];
        int x1 = rect.tl().x;
        int y1 = rect.tl().y;
        int x2 = rect.br().x;
        int y2 = rect.br().y;
        int w = rect.width;
        int h = rect.height;

        assert(w == SEG_INPUT_W && h == SEG_INPUT_H);

        int dst_pos = 0;
        for (int row = 0; row < SEG_INPUT_H; ++row)
        {
            for (int col = 0; col < SEG_INPUT_W; ++col)
            {
                int pos = (row + y1) * s_param.merge_w + (col + x1);
                data[b * 3 * SEG_INPUT_H * SEG_INPUT_W + dst_pos] = (float)r_image.data[pos] / 255.0;
                data[b * 3 * SEG_INPUT_H * SEG_INPUT_W + dst_pos + SEG_INPUT_H * SEG_INPUT_W] = (float)g_image.data[pos] / 255.0;
                data[b * 3 * SEG_INPUT_H * SEG_INPUT_W + dst_pos + 2 * SEG_INPUT_H * SEG_INPUT_W] = (float)b_image.data[pos] / 255.0;
                dst_pos++;
            }
        }
    }

}

void semantic::split()
{
    s_param.stride_w = std::ceil(s_param.crop_w * (1 - s_param.overlap_w));
    s_param.stride_h = std::ceil(s_param.crop_h * (1 - s_param.overlap_h));
    s_param.tile_rows = int(std::ceil((s_param.merge_w - s_param.crop_w) / s_param.stride_w) + 1);  // # strided convolution formula
    s_param.tile_cols = int(std::ceil((s_param.merge_h - s_param.crop_h) / s_param.stride_h) + 1);

    s_param.tile_counter = 0;
    
    for (int row = 0; row < s_param.tile_rows; row++)
    {
        for (int col = 0; col < s_param.tile_cols; col++)
        {
            int x1 = int(col * s_param.stride_w);
            int y1 = int(row * s_param.stride_h);
            int x2 = (std::min)(x1 + s_param.crop_w, s_param.merge_w);
            int y2 = (std::min)(y1 + s_param.crop_h, s_param.merge_h);
            x1 = (std::max)(int(x2 - s_param.crop_w), 0); // for portrait images the x1 underflows sometimes
            y1 = (std::max)(int(y2 - s_param.crop_h), 0);  // for very few rows y1 underflows

            split_result.push_back(cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)));
            s_param.tile_counter++;

        }
    }
}



void semantic::merge(float* probs, cv::Mat& output)
{
    int idx = 0;
    int big_h = s_param.merge_h;
    int big_w = s_param.merge_w;
    int big_plane = big_h * big_w;
    int small_w = s_param.crop_w;
    int small_h = s_param.crop_h;
    int small_plane = small_h * small_w;
    float* big_prob = new float[big_plane * SEG_CLASS_NUM];
    int* big_count = new int[big_plane * SEG_CLASS_NUM];
    memset(big_prob, 0, big_plane * SEG_CLASS_NUM * sizeof(float));
    memset(big_count, 0, big_plane * SEG_CLASS_NUM * sizeof(int));
    for (int i = 0; i < s_param.tile_counter; i++)
    {
        cv::Rect rect = split_result[i];
        int x1 = rect.tl().x;
        int y1 = rect.tl().y;
        int x2 = rect.br().x;
        int y2 = rect.br().y;
        int w = rect.width;
        int h = rect.height;

        float* small_probs = probs + i * SEG_BATCH_SIZE * SEG_OUTPUT_SIZE;


        // softmax
        for (int r = 0; r < small_h; r++)
        {
            for (int c = 0; c < small_w; c++)
            {
                float sum = 0;
                for (int k = 0; k < SEG_CLASS_NUM; k++)
                {
                    float* pixel = small_probs + k * small_plane + (r)*small_w + (c);
                    *pixel = std::exp(*pixel);
                    sum += *pixel;

                }
                for (int k = 0; k < SEG_CLASS_NUM; k++)
                {
                    float* pixel = small_probs + k * small_plane + (r)*small_w + (c);
                    *pixel /= sum;
                }

            }
        }


        for (int k = 0; k < SEG_CLASS_NUM; k++)
        {
            for (int r = y1; r < y2; r++)
            {
                for (int c = x1; c < x2; c++)
                {
                    *(big_prob + k * big_plane + r * big_w + c) += *(small_probs + k * small_plane + (r - y1) * small_w + (c - x1));
                    *(big_count + k * big_plane + r * big_w + c) += 1;
                }
            }
        }

    }

    float* big_prob_ptr = big_prob;
    int* big_count_ptr = big_count;
    for (int k = 0; k < SEG_CLASS_NUM; k++)
    {
        for (int r = 0; r < big_h; r++)
        {
            for (int c = 0; c < big_w; c++)
            {
                *big_prob_ptr++ /= *big_count_ptr++;
            }
        }
    }


    // argmax
    int* labels = new int[big_plane];
    memset(labels, 0, big_plane * sizeof(int));
    int* labels_ptr = labels;
    float max_value = FLT_MIN;
    int max_index = 0;
    for (int r = 0; r < big_h; r++)
    {
        for (int c = 0; c < big_w; c++)
        {
            max_value = FLT_MIN;
            max_index = 0;
            for (int k = 0; k < SEG_CLASS_NUM; k++)
            {
                float value = *(big_prob + k * big_plane + r * big_w + c);
                if (k == 0)
                {
                    max_value = value;
                    max_index = k;
                    continue;
                }
                if (value > max_value)
                {
                    max_value = value;
                    max_index = k;
                }
            }
            *labels_ptr++ = max_index;


        }
    }

    output = cv::Mat(cv::Size(big_w, big_h), CV_8UC1, labels);


    delete[]big_prob;
    delete[]big_count;
}

void semantic::process(std::vector<cv::Mat> image, cv::Mat& output)
{
    memset(data,0, SEG_BATCH_SIZE * 3 * SEG_INPUT_H * SEG_INPUT_W*sizeof(float));
    memset(prob, 0, SEG_BATCH_SIZE * SEG_OUTPUT_SIZE * sizeof(float));

    split();

        
    float* probs = new float[SEG_BATCH_SIZE * SEG_OUTPUT_SIZE * s_param.tile_counter];
    for (int i = 0; i < s_param.tile_counter; )
    {
        int bs = (std::min)(SEG_BATCH_SIZE, s_param.tile_counter -i );

        std::vector<cv::Rect> crop_rects(split_result.begin()+i, split_result.begin()+i+bs);
        preprocess(image, crop_rects, bs);

        // Run inference
        auto start = std::chrono::system_clock::now();
        doInference(*context, stream, buffers, data, prob, bs);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        memcpy(probs + i * SEG_BATCH_SIZE * SEG_OUTPUT_SIZE, prob, sizeof(float) * SEG_BATCH_SIZE * SEG_OUTPUT_SIZE);

        i += SEG_BATCH_SIZE;
    }
        
    merge(probs, output);


#ifdef DEBUG
    for (int b = 0; b < bs; b++) {
        auto& res = batch_res[b];
        cv::Mat img = cv::imread(img_dir + "/" + file_names[f - fcount + 1 + b]);
        for (size_t j = 0; j < res.size(); j++) {
            cv::Rect r = get_rect(img, res[j].bbox);
            cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
            cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
        }
        cv::imwrite("_" + file_names[f - fcount + 1 + b], img);
    }
#endif // DEBUG

}


void semantic::doInference(IExecutionContext& context, cudaStream_t& stream, void** buffers, float* input, float* output, int batchSize) {

    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * SEG_INPUT_H * SEG_INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * SEG_OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    //流同步：通过cudaStreamSynchronize()来协调。
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);

}




