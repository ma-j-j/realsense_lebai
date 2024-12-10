#include "yolov5.h"

Yolov5 yolo;

// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int CLASS_NUM = Yolo::CLASS_NUM;
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1; // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
const char *INPUT_BLOB_NAME = "data";
const char *OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;

static int get_width(int x, float gw, int divisor = 8)
{

    // return math.ceil(x / divisor) * divisor
    if (int(x * gw) % divisor == 0)
    {
        return int(x * gw);
    }
    return (int(x * gw / divisor) + 1) * divisor;
}

static int get_depth(int x, float gd)
{
    if (x == 1)
    {
        return 1;
    }
    else
    {
        return round(x * gd) > 1 ? round(x * gd) : 1;
    }
}
// #创建engine和network
ICudaEngine *Yolov5::build_engine(unsigned int maxBatchSize, IBuilder *builder, IBuilderConfig *config, nvinfer1::DataType dt, float &gd, float &gw, std::string &wts_name)
{
    INetworkDefinition *network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor *data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights(wts_name);

    /* ------ yolov5 backbone------ */
    auto focus0 = focus(network, weightMap, *data, 3, get_width(64, gw), 3, "model.0");
    auto conv1 = convBlock(network, weightMap, *focus0->getOutput(0), get_width(128, gw), 3, 2, 1, "model.1");
    auto bottleneck_CSP2 = C3(network, weightMap, *conv1->getOutput(0), get_width(128, gw), get_width(128, gw), get_depth(3, gd), true, 1, 0.5, "model.2");
    auto conv3 = convBlock(network, weightMap, *bottleneck_CSP2->getOutput(0), get_width(256, gw), 3, 2, 1, "model.3");
    auto bottleneck_csp4 = C3(network, weightMap, *conv3->getOutput(0), get_width(256, gw), get_width(256, gw), get_depth(9, gd), true, 1, 0.5, "model.4");
    auto conv5 = convBlock(network, weightMap, *bottleneck_csp4->getOutput(0), get_width(512, gw), 3, 2, 1, "model.5");
    auto bottleneck_csp6 = C3(network, weightMap, *conv5->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(9, gd), true, 1, 0.5, "model.6");
    auto conv7 = convBlock(network, weightMap, *bottleneck_csp6->getOutput(0), get_width(1024, gw), 3, 2, 1, "model.7");
    auto spp8 = SPP(network, weightMap, *conv7->getOutput(0), get_width(1024, gw), get_width(1024, gw), 5, 9, 13, "model.8");

    /* ------ yolov5 head ------ */
    auto bottleneck_csp9 = C3(network, weightMap, *spp8->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.9");
    auto conv10 = convBlock(network, weightMap, *bottleneck_csp9->getOutput(0), get_width(512, gw), 1, 1, 1, "model.10");

    auto upsample11 = network->addResize(*conv10->getOutput(0));
    assert(upsample11);
    upsample11->setResizeMode(ResizeMode::kNEAREST);
    upsample11->setOutputDimensions(bottleneck_csp6->getOutput(0)->getDimensions());

    ITensor *inputTensors12[] = {upsample11->getOutput(0), bottleneck_csp6->getOutput(0)};
    auto cat12 = network->addConcatenation(inputTensors12, 2);
    auto bottleneck_csp13 = C3(network, weightMap, *cat12->getOutput(0), get_width(1024, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.13");
    auto conv14 = convBlock(network, weightMap, *bottleneck_csp13->getOutput(0), get_width(256, gw), 1, 1, 1, "model.14");

    auto upsample15 = network->addResize(*conv14->getOutput(0));
    assert(upsample15);
    upsample15->setResizeMode(ResizeMode::kNEAREST);
    upsample15->setOutputDimensions(bottleneck_csp4->getOutput(0)->getDimensions());

    ITensor *inputTensors16[] = {upsample15->getOutput(0), bottleneck_csp4->getOutput(0)};
    auto cat16 = network->addConcatenation(inputTensors16, 2);

    auto bottleneck_csp17 = C3(network, weightMap, *cat16->getOutput(0), get_width(512, gw), get_width(256, gw), get_depth(3, gd), false, 1, 0.5, "model.17");

    // yolo layer 0
    IConvolutionLayer *det0 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);
    auto conv18 = convBlock(network, weightMap, *bottleneck_csp17->getOutput(0), get_width(256, gw), 3, 2, 1, "model.18");
    ITensor *inputTensors19[] = {conv18->getOutput(0), conv14->getOutput(0)};
    auto cat19 = network->addConcatenation(inputTensors19, 2);
    auto bottleneck_csp20 = C3(network, weightMap, *cat19->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.20");
    // yolo layer 1
    IConvolutionLayer *det1 = network->addConvolutionNd(*bottleneck_csp20->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["model.24.m.1.weight"], weightMap["model.24.m.1.bias"]);
    auto conv21 = convBlock(network, weightMap, *bottleneck_csp20->getOutput(0), get_width(512, gw), 3, 2, 1, "model.21");
    ITensor *inputTensors22[] = {conv21->getOutput(0), conv10->getOutput(0)};
    auto cat22 = network->addConcatenation(inputTensors22, 2);
    auto bottleneck_csp23 = C3(network, weightMap, *cat22->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.23");
    IConvolutionLayer *det2 = network->addConvolutionNd(*bottleneck_csp23->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["model.24.m.2.weight"], weightMap["model.24.m.2.bias"]);

    auto yolo = addYoLoLayer(network, weightMap, "model.24", std::vector<IConvolutionLayer *>{det0, det1, det2});
    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20)); // 16MB
#if defined(USE_FP16)
    config->setFlag(BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(BuilderFlag::kINT8);
    Int8EntropyCalibrator2 *calibrator = new Int8EntropyCalibrator2(1, INPUT_W, INPUT_H, "./coco_calib/", "int8calib.table", INPUT_BLOB_NAME);
    config->setInt8Calibrator(calibrator);
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto &mem : weightMap)
    {
        free((void *)(mem.second.values));
    }

    return engine;
}

ICudaEngine *Yolov5::build_engine_p6(unsigned int maxBatchSize, IBuilder *builder, IBuilderConfig *config, nvinfer1::DataType dt, float &gd, float &gw, std::string &wts_name)
{
    INetworkDefinition *network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor *data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights(wts_name);

    /* ------ yolov5 backbone------ */
    auto focus0 = focus(network, weightMap, *data, 3, get_width(64, gw), 3, "model.0");
    auto conv1 = convBlock(network, weightMap, *focus0->getOutput(0), get_width(128, gw), 3, 2, 1, "model.1");
    auto c3_2 = C3(network, weightMap, *conv1->getOutput(0), get_width(128, gw), get_width(128, gw), get_depth(3, gd), true, 1, 0.5, "model.2");
    auto conv3 = convBlock(network, weightMap, *c3_2->getOutput(0), get_width(256, gw), 3, 2, 1, "model.3");
    auto c3_4 = C3(network, weightMap, *conv3->getOutput(0), get_width(256, gw), get_width(256, gw), get_depth(9, gd), true, 1, 0.5, "model.4");
    auto conv5 = convBlock(network, weightMap, *c3_4->getOutput(0), get_width(512, gw), 3, 2, 1, "model.5");
    auto c3_6 = C3(network, weightMap, *conv5->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(9, gd), true, 1, 0.5, "model.6");
    auto conv7 = convBlock(network, weightMap, *c3_6->getOutput(0), get_width(768, gw), 3, 2, 1, "model.7");
    auto c3_8 = C3(network, weightMap, *conv7->getOutput(0), get_width(768, gw), get_width(768, gw), get_depth(3, gd), true, 1, 0.5, "model.8");
    auto conv9 = convBlock(network, weightMap, *c3_8->getOutput(0), get_width(1024, gw), 3, 2, 1, "model.9");
    auto spp10 = SPP(network, weightMap, *conv9->getOutput(0), get_width(1024, gw), get_width(1024, gw), 3, 5, 7, "model.10");
    auto c3_11 = C3(network, weightMap, *spp10->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.11");

    /* ------ yolov5 head ------ */
    auto conv12 = convBlock(network, weightMap, *c3_11->getOutput(0), get_width(768, gw), 1, 1, 1, "model.12");
    auto upsample13 = network->addResize(*conv12->getOutput(0));
    assert(upsample13);
    upsample13->setResizeMode(ResizeMode::kNEAREST);
    upsample13->setOutputDimensions(c3_8->getOutput(0)->getDimensions());
    ITensor *inputTensors14[] = {upsample13->getOutput(0), c3_8->getOutput(0)};
    auto cat14 = network->addConcatenation(inputTensors14, 2);
    auto c3_15 = C3(network, weightMap, *cat14->getOutput(0), get_width(1536, gw), get_width(768, gw), get_depth(3, gd), false, 1, 0.5, "model.15");

    auto conv16 = convBlock(network, weightMap, *c3_15->getOutput(0), get_width(512, gw), 1, 1, 1, "model.16");
    auto upsample17 = network->addResize(*conv16->getOutput(0));
    assert(upsample17);
    upsample17->setResizeMode(ResizeMode::kNEAREST);
    upsample17->setOutputDimensions(c3_6->getOutput(0)->getDimensions());
    ITensor *inputTensors18[] = {upsample17->getOutput(0), c3_6->getOutput(0)};
    auto cat18 = network->addConcatenation(inputTensors18, 2);
    auto c3_19 = C3(network, weightMap, *cat18->getOutput(0), get_width(1024, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.19");

    auto conv20 = convBlock(network, weightMap, *c3_19->getOutput(0), get_width(256, gw), 1, 1, 1, "model.20");
    auto upsample21 = network->addResize(*conv20->getOutput(0));
    assert(upsample21);
    upsample21->setResizeMode(ResizeMode::kNEAREST);
    upsample21->setOutputDimensions(c3_4->getOutput(0)->getDimensions());
    ITensor *inputTensors21[] = {upsample21->getOutput(0), c3_4->getOutput(0)};
    auto cat22 = network->addConcatenation(inputTensors21, 2);
    auto c3_23 = C3(network, weightMap, *cat22->getOutput(0), get_width(512, gw), get_width(256, gw), get_depth(3, gd), false, 1, 0.5, "model.23");

    auto conv24 = convBlock(network, weightMap, *c3_23->getOutput(0), get_width(256, gw), 3, 2, 1, "model.24");
    ITensor *inputTensors25[] = {conv24->getOutput(0), conv20->getOutput(0)};
    auto cat25 = network->addConcatenation(inputTensors25, 2);
    auto c3_26 = C3(network, weightMap, *cat25->getOutput(0), get_width(1024, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.26");

    auto conv27 = convBlock(network, weightMap, *c3_26->getOutput(0), get_width(512, gw), 3, 2, 1, "model.27");
    ITensor *inputTensors28[] = {conv27->getOutput(0), conv16->getOutput(0)};
    auto cat28 = network->addConcatenation(inputTensors28, 2);
    auto c3_29 = C3(network, weightMap, *cat28->getOutput(0), get_width(1536, gw), get_width(768, gw), get_depth(3, gd), false, 1, 0.5, "model.29");

    auto conv30 = convBlock(network, weightMap, *c3_29->getOutput(0), get_width(768, gw), 3, 2, 1, "model.30");
    ITensor *inputTensors31[] = {conv30->getOutput(0), conv12->getOutput(0)};
    auto cat31 = network->addConcatenation(inputTensors31, 2);
    auto c3_32 = C3(network, weightMap, *cat31->getOutput(0), get_width(2048, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.32");

    /* ------ detect ------ */
    IConvolutionLayer *det0 = network->addConvolutionNd(*c3_23->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["model.33.m.0.weight"], weightMap["model.33.m.0.bias"]);
    IConvolutionLayer *det1 = network->addConvolutionNd(*c3_26->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["model.33.m.1.weight"], weightMap["model.33.m.1.bias"]);
    IConvolutionLayer *det2 = network->addConvolutionNd(*c3_29->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["model.33.m.2.weight"], weightMap["model.33.m.2.bias"]);
    IConvolutionLayer *det3 = network->addConvolutionNd(*c3_32->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["model.33.m.3.weight"], weightMap["model.33.m.3.bias"]);

    auto yolo = addYoLoLayer(network, weightMap, "model.33", std::vector<IConvolutionLayer *>{det0, det1, det2, det3});
    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20)); // 16MB
#if defined(USE_FP16)
    config->setFlag(BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(BuilderFlag::kINT8);
    Int8EntropyCalibrator2 *calibrator = new Int8EntropyCalibrator2(1, INPUT_W, INPUT_H, "./coco_calib/", "int8calib.table", INPUT_BLOB_NAME);
    config->setInt8Calibrator(calibrator);
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto &mem : weightMap)
    {
        free((void *)(mem.second.values));
    }

    return engine;
}

void Yolov5::APIToModel(unsigned int maxBatchSize, IHostMemory **modelStream, float &gd, float &gw, std::string &wts_name)
{
    // Create builder
    IBuilder *builder = createInferBuilder(gLogger);
    IBuilderConfig *config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine *engine = build_engine(maxBatchSize, builder, config, nvinfer1::DataType::kFLOAT, gd, gw, wts_name);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
    config->destroy();
}

void Yolov5::doInference(IExecutionContext &context, cudaStream_t &stream, void **buffers, float *input, float *output, int batchSize)
{
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

bool Yolov5::parse_args(int argc, char **argv, std::string &engine)
{
    // if (argc < 3)
    //     return false;
    // if (std::string(argv[1]) == "-v" && argc == 3)
    // {
    //     engine = std::string(argv[2]);
    // }
    // else
    // {
    //     return false;
    // }
    // return true;

    return true;
}

void Yolov5::yolo_ROI(cv::Mat inputImage, std::vector<Yolo::Detection> Input,
                      cv::Mat &color_ROI, cv::Mat &depth_ROI,
                      cv::Mat &black_color_image, cv::Mat &black_depth_image,
                      cv::Mat in_depth, cv::Mat out_depth,
                      std::vector<cv::Point2d> &out_Center)
{

    for (size_t j = 0; j < Input.size(); j++)
    {
        // 获取边框
        cv::Rect r = get_rect(inputImage, Input[j].bbox);
        // cv::rectangle(inputImage, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
        std::string label = my_classes[(int)Input[j].class_id]; // 标签
        // cv::putText(inputImage, label, cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);

        // 放中心点
        float center_x = r.x + r.width / 2;
        float center_y = r.y + r.height / 2;
        cv::Point2d ROI_center = cv::Point(center_x, center_y);
        out_Center.push_back(ROI_center);
        // std::cout << "out_Center:" << out_Center.size() << std::endl;

        // 确保边框不会超出深度图像的尺寸
        r = r & cv::Rect(0, 0, in_depth.cols, in_depth.rows);

        // 检查bbox大小是否有效，避免边框大小为零的情况
        if (r.width > 0 && r.height > 0)
        {
            // 将检测框内的深度数据复制到新的深度图中
            in_depth(r).copyTo(out_depth(r));

            color_ROI = inputImage(r);
            depth_ROI = in_depth(r);
            // cv::imwrite("color_ROI.png", color_ROI);
            // if (!color_ROI.empty() && !depth_ROI.empty())
            // {
            //     cv::imshow("color_ROI", color_ROI);
            //     cv::imshow("depth_ROI", depth_ROI);
            // }

            // 创建一个与原图像大小相同的黑色图像
            black_color_image = cv::Mat::zeros(inputImage.size(), inputImage.type());
            black_depth_image = cv::Mat::zeros(in_depth.size(), in_depth.type());

            // 将原图中的ROI区域复制到黑色图像中
            inputImage(r).copyTo(black_color_image(r));
            in_depth(r).copyTo(black_depth_image(r));
        }
    }

    //
}

double Yolov5::PCA_2d(std::vector<cv::Point> &pts, cv::Mat &image, cv::Mat &color_image, double angle_p1_x_degrees, double angle_p2_y_degrees)
{
    int size = static_cast<int>(pts.size());
    cv::Mat data_pts = cv::Mat(size, 2, CV_64FC1);
    for (int i = 0; i < size; i++)
    {
        data_pts.at<double>(i, 0) = pts[i].x;
        data_pts.at<double>(i, 1) = pts[i].y;
    }

    // PCA
    cv::PCA pca_analysi(data_pts, cv::Mat(), 0);
    cv::Point cnt = cv::Point(static_cast<int>(pca_analysi.mean.at<double>(0, 0)),
                              static_cast<int>(pca_analysi.mean.at<double>(0, 1)));
    cv::circle(image, cnt, 2, cv::Scalar(0, 255, 0), 2, 8, 0);

    std::vector<cv::Point2d> vecs(2);
    std::vector<double> vals(2);
    for (int i = 0; i < 2; i++)
    {
        vals[i] = pca_analysi.eigenvalues.at<double>(i, 0);
        vecs[i] = cv::Point2d(pca_analysi.eigenvectors.at<double>(i, 0),
                              pca_analysi.eigenvectors.at<double>(i, 1));
    }
    cv::Point p1 = cnt + 0.02 * cv::Point(static_cast<int>(vecs[0].x * vals[0]), static_cast<int>(vecs[0].y * vals[0]));
    cv::Point p2 = cnt - 0.05 * cv::Point(static_cast<int>(vecs[1].x * vals[1]), static_cast<int>(vecs[1].y * vals[1]));

    // 在深度图上划线
    cv::line(image, cnt, p1, cv::Scalar(255, 0, 0), 2, 8, 0);
    cv::line(image, cnt, p2, cv::Scalar(255, 255, 0), 2, 8, 0);

    // 在原图上划线
    cv::line(color_image, cnt, p1, cv::Scalar(255, 0, 0), 2, 8, 0);   // x
    cv::line(color_image, cnt, p2, cv::Scalar(255, 255, 0), 2, 8, 0); // y

    // 划水平和竖直的线
    int line_length = 100;
    cv::line(color_image, cv::Point(cnt.x - line_length, cnt.y), cv::Point(cnt.x + line_length, cnt.y), cv::Scalar(0, 255, 0), 2, 8, 0); // x
    cv::line(color_image, cv::Point(cnt.x, cnt.y - line_length), cv::Point(cnt.x, cnt.y + line_length), cv::Scalar(0, 255, 0), 2, 8, 0); // y

    // 计算 p1 和 p2 与水平方向（x轴）的夹角
    double angle_p1_x = atan2(p1.y - cnt.y, p1.x - cnt.x); // p1 与水平轴（x轴）的夹角
    double angle_p2_x = atan2(p2.y - cnt.y, p2.x - cnt.x); // p2 与水平轴（x轴）的夹角

    // 将角度从弧度转换为度数
    angle_p1_x_degrees = angle_p1_x * 180.0 / CV_PI;
    double angle_p2_x_degrees = angle_p2_x * 180.0 / CV_PI;

    std::cout << "p1 与水平线(x轴)的夹角: " << angle_p1_x_degrees << "°" << std::endl;
    // std::cout << "p2 与水平线(x轴)的夹角: " << angle_p2_x_degrees << "°" << std::endl;

    // 计算 p1 和 p2 与竖直方向（y轴）的夹角
    // 对于竖直方向，计算与水平方向的差值，并使用 90 度减去相应角度
    // double angle_p1_y_degrees = 90.0 - fabs(angle_p1_x_degrees); // p1 与竖直线（y轴）的夹角
    angle_p2_y_degrees = 90.0 - fabs(angle_p2_x_degrees); // p2 与竖直线（y轴）的夹角

    // std::cout << "p1 与竖直线(y轴)的夹角: " << angle_p1_y_degrees << "°" << std::endl;
    std::cout << "p2 与竖直线(y轴)的夹角: " << angle_p2_y_degrees << "°" << std::endl;

    return 0;
}

void Yolov5::PCAdraw_2d(cv::Mat depth_masked, cv::Mat color_image)
{
    Yolov5 yolo;
    double x_degress, y_degress;
    // PCA对视差图进行处理
    std::vector<cv::Vec4i> hireachy;
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(depth_masked, contours, hireachy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
    cv::Mat result = depth_masked.clone();
    for (size_t i = 0; i < contours.size(); i++)
    {
        double area = cv::contourArea(contours[i]);
        if (area < 1000 || 100000 < area)
            continue;
        cv::drawContours(result, contours, i, cv::Scalar(0, 0, 255), 2, 8);

        double theta = yolo.PCA_2d(contours[i], result, color_image, x_degress, y_degress);
    }
    cv::imshow("result", result);
}