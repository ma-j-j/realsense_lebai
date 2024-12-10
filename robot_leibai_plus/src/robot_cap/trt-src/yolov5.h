#pragma once
#include <iostream>
#include <chrono>
#include "cuda_utils.h"
#include "logging.h"
#include "common.hpp"
#include "utils.h"
#include "calibrator.h"
#include <opencv2/ximgproc.hpp> // OpenCV 额外模块

#define USE_FP16 // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0 // GPU id
#define NMS_THRESH 0.4
#define CONF_THRESH 0.5
#define BATCH_SIZE 1

// 修改为自己的类别
char *my_classes[] = {"Citrus"};

class Yolov5
{
public:
    // static int get_width(int x, float gw, int divisor);
    // static int get_depth(int x, float gd);
    ICudaEngine *build_engine(unsigned int maxBatchSize, IBuilder *builder, IBuilderConfig *config, nvinfer1::DataType dt, float &gd, float &gw, std::string &wts_name);
    ICudaEngine *build_engine_p6(unsigned int maxBatchSize, IBuilder *builder, IBuilderConfig *config, nvinfer1::DataType dt, float &gd, float &gw, std::string &wts_name);
    void APIToModel(unsigned int maxBatchSize, IHostMemory **modelStream, float &gd, float &gw, std::string &wts_name);
    void doInference(IExecutionContext &context, cudaStream_t &stream, void **buffers, float *input, float *output, int batchSize);
    bool parse_args(int argc, char **argv, std::string &engine);
    void yolo_ROI(cv::Mat inputImage, std::vector<Yolo::Detection> Input, cv::Mat &color_ROI, cv::Mat &depth_ROI, cv::Mat &black_color_image, cv::Mat &black_depth_image, cv::Mat in_depth, cv::Mat out_depth, std::vector<cv::Point2d> &out_Center);
    double PCA_2d(std::vector<cv::Point> &pts, cv::Mat &image, cv::Mat &color_image, double x_degress, double y_degress);
    void PCAdraw_2d(cv::Mat depth, cv::Mat color);

private:
};
