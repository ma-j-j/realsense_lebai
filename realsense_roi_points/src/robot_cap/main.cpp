#include "yolov5.h"
#include "sgbm.h"
#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>
#include "ros/ros.h"
#include <geometry_msgs/Pose.h>
#include <thread> // 包含 sleep_for 和 this_thread 相关内容
#include <chrono> // 包含时间单位的定义
#include <cv_bridge/cv_bridge.h>
#include "sensor_msgs/Image.h"
#include "sensor_msgs/CameraInfo.h"

/********************************************************************************************/
//------------------------------------------------------------------------------------------//
// 定义关于网络和输入/输出 bbox 的东西
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int CLASS_NUM = Yolo::CLASS_NUM;
// 我们假设 yololayer 输出不超过 MAX_OUTPUT_BBOX_COUNT 个框，conf >= 0.1
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;
const char *INPUT_BLOB_NAME = "data";
const char *OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;

const size_t inWidth = 300;
const size_t inHeight = 300;
const float WHRatio = inWidth / (float)inHeight;
const float inScaleFactor = 0.007843f;
const float meanVal = 127.5;
//------------------------------------------------------------------------------------------//
/********************************************************************************************/

static cv::Mat frame_to_mat(const rs2::frame &f)
{
    using namespace cv;
    using namespace rs2;

    auto vf = f.as<video_frame>();
    const int w = vf.get_width();
    const int h = vf.get_height();

    if (f.get_profile().format() == RS2_FORMAT_BGR8) // BGR 格式
    {
        return Mat(Size(w, h), CV_8UC3, (void *)f.get_data(), Mat::AUTO_STEP);
    }
    else if (f.get_profile().format() == RS2_FORMAT_RGB8) // RGB 格式
    {
        auto r_rgb = Mat(Size(w, h), CV_8UC3, (void *)f.get_data(), Mat::AUTO_STEP);
        Mat r_bgr;
        cvtColor(r_rgb, r_bgr, COLOR_RGB2BGR);
        return r_bgr;
    }
    else if (f.get_profile().format() == RS2_FORMAT_Z16) // 深度图格式
    {
        return Mat(Size(w, h), CV_16UC1, (void *)f.get_data(), Mat::AUTO_STEP);
    }
    else if (f.get_profile().format() == RS2_FORMAT_Y8) // 灰度图格式
    {
        return Mat(Size(w, h), CV_8UC1, (void *)f.get_data(), Mat::AUTO_STEP);
    }
    else if (f.get_profile().format() == RS2_FORMAT_DISPARITY32) // 视差图格式
    {
        return Mat(Size(w, h), CV_32FC1, (void *)f.get_data(), Mat::AUTO_STEP);
    }

    throw std::runtime_error("Frame format is not supported yet!");
}

static cv::Mat depth_frame_to_meters(const rs2::depth_frame &f)
{
    cv::Mat dm = frame_to_mat(f);
    dm.convertTo(dm, CV_64F);
    dm = dm * f.get_units();
    return dm;
}

int main(int argc, char **argv)
{
    int t = 0;
    ros::init(argc, argv, "cap_points_pub");
    ros::NodeHandle nh;
    ros::Rate r(100);
    ros::Publisher pub;
    ros::Publisher color_pub;
    ros::Publisher depth_pub;
    ros::Publisher color_ROI_pub;
    ros::Publisher depth_ROI_pub;
    pub = nh.advertise<geometry_msgs::Pose>("/cap_pose", 1);
    color_pub = nh.advertise<sensor_msgs::Image>("/color_img", 1);
    depth_pub = nh.advertise<sensor_msgs::Image>("/depth_img", 1);
    color_ROI_pub = nh.advertise<sensor_msgs::Image>("/color_ROI_img", 1);
    depth_ROI_pub = nh.advertise<sensor_msgs::Image>("/depth_ROI_img", 1);
    geometry_msgs::Pose cap_point;

    // 定义yolo-sgbm的两个类
    depth DP;
    Yolov5 YL;
    // ROI区域
    cv::Mat color_ROI, depth_ROI;
    cv::Mat black_image;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    // pcl::visualization::CloudViewer viewer("Cloud Viewer");
    /*********yolo容器*********************************/
    std::vector<std::vector<int>> OutInfoemation;

    // 读模型
    std::string engine_name = "src/robot_cap/weight/827v6.engine";
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good())
    {
        std::cerr << " read " << engine_name << " error! " << std::endl;
        return -1;
    }

    /*****************************初始化**************************************************************/
    //----------------------------------------------------------------------------------------------//
    char *trtModelStream{nullptr};
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();
    static float data_left[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    static float data_right[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    static float prob_left[BATCH_SIZE * OUTPUT_SIZE];
    static float prob_right[BATCH_SIZE * OUTPUT_SIZE];
    IRuntime *runtime = createInferRuntime(gLogger);

    assert(runtime != nullptr);
    ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext *context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);
    void *buffers[2];
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    //------------------------------------------------------------------------------------------//
    /********************************************************************************************/

    // realsense相机参数配置
    rs2::pipeline pipe;
    rs2::config cfg;
    rs2::pointcloud pc;
    // boost::shared_ptr<pcl::visualization::PCLVisualizer> openCloud;
    // cfg.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_Z16, 6);
    // cfg.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_BGR8, 6);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    rs2::pipeline_profile profile = pipe.start(cfg);

    // 创建深度流和彩色流对齐的实例化对象
    rs2::align align_to(RS2_STREAM_COLOR);

    int fcount = 0;

    while (ros::ok())
    {
        /*********************************************************************/
        rs2::frameset data = pipe.wait_for_frames();
        data = align_to.process(data); // 实际进行流对齐

        // 基于对齐的混合流获取深度流和彩色流,进而获取流对齐后的深度内参
        rs2::video_frame color_stream = data.get_color_frame();
        rs2::depth_frame aligned_depth_stream = data.get_depth_frame();
        rs2::video_stream_profile depth_stream_profile = aligned_depth_stream.get_profile().as<rs2::video_stream_profile>();
        const auto depth_intrinsics = depth_stream_profile.get_intrinsics(); // 获取对齐后的深度内参

        // 获取彩色图像宽
        const int w = color_stream.as<rs2::video_frame>().get_width();
        const int h = color_stream.as<rs2::video_frame>().get_height();

        // 获取原图及深度图
        rs2::frame color_frame = data.get_color_frame();
        rs2::depth_frame depth_frame = data.get_depth_frame();

        // cv::Mat color_image = frame_to_mat(color_frame);
        // cv::Mat depth_image = depth_frame_to_meters(depth_frame); // 深度图单位转换为米

        cv::Mat color_image(cv::Size(640, 480), CV_8UC3, (void *)color_frame.get_data(), cv::Mat::AUTO_STEP);
        cv::Mat depth_image(cv::Size(640, 480), CV_16U, (void *)depth_frame.get_data(), cv::Mat::AUTO_STEP);
        /*********************************************************************/

        /**************************原图与深度图发布*******************************************/
        sensor_msgs::ImagePtr Color_Img_msgs = cv_bridge::CvImage(std_msgs::Header(), "bgr8", color_image).toImageMsg();
        sensor_msgs::ImagePtr depth_Img_msgs = cv_bridge::CvImage(std_msgs::Header(), "mono16", depth_image).toImageMsg();
        Color_Img_msgs->header.stamp = ros::Time::now();
        depth_Img_msgs->header.frame_id = "images";
        Color_Img_msgs->header.stamp = ros::Time::now();
        depth_Img_msgs->header.frame_id = "images";
        color_pub.publish(Color_Img_msgs);
        depth_pub.publish(depth_Img_msgs);
        /*********************************************************************/

        /*********************************************************************/
        fcount++;
        // 定义左图数据
        cv::Mat pr_img_left = preprocess_img(color_image, INPUT_W, INPUT_H);
        int i = 0;
        for (int row = 0; row < INPUT_H; ++row)
        {
            uchar *uc_pixel = pr_img_left.data + row * pr_img_left.step;
            for (int col = 0; col < INPUT_W; ++col)
            {
                data_left[i] = (float)uc_pixel[2] / 255.0;
                data_left[i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
                data_left[i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
                uc_pixel += 3;
                ++i;
            }
        }

        // 跑左图
        auto start = std::chrono::system_clock::now();
        YL.doInference(*context, stream, buffers, data_left, prob_left, BATCH_SIZE);
        auto end = std::chrono::system_clock::now();
        int fps_left = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        // 检测的结果
        std::vector<Yolo::Detection> res_left;
        nms(res_left, prob_left, CONF_THRESH, NMS_THRESH);

        std::cout << "citrus=" << res_left.size() << "个" << std::endl;

        // 创建深度图遮罩，只显示检测区域
        cv::Mat depth_masked = cv::Mat::zeros(depth_image.size(), depth_image.type());

        std::vector<cv::Point2d> ROI_Center;
        // 出图
        YL.yolo_ROI(color_image, res_left, color_ROI, depth_ROI, black_image, depth_image, depth_masked, ROI_Center);
        /*********************************************************************/

        /*********************************************************************/
        // 拿像素坐标
        float pixe_center[2];
        for (size_t i = 0; i < ROI_Center.size(); i++)
        {
            pixe_center[0] = ROI_Center[i].x;
            pixe_center[1] = ROI_Center[i].y;
            // std::cout << "Center " << i + 1 << ": (" << pixe_center[0] << ", " << pixe_center[1] << ")" << std::endl;
        }
        // std::cout << "ROI_Center:" << ROI_Center.size() << std::endl;

        depth_image.convertTo(depth_image, CV_8UC1, 255, 0); // 保存视差图开这个函数
        depth_ROI.convertTo(depth_ROI, CV_8UC1, 255, 0);     // 保存视差图开这个函数

        depth_masked.convertTo(depth_masked, CV_8UC1, 255, 0); // 保存视差图开这个函数
        depth_image.convertTo(depth_image, CV_8UC1, 255, 0);   // 保存视差图开这个函数
        // cv::imshow("Masked Depth", depth_masked); // 显示深度图遮罩结果

        // 像素坐标系转换到相机坐标系
        float pixed_center_depth_value = aligned_depth_stream.get_distance(pixe_center[0], pixe_center[1]);
        // 计算三维坐标
        float point_in_color_coordinates[3];
        rs2_deproject_pixel_to_point(point_in_color_coordinates, &depth_intrinsics, pixe_center, pixed_center_depth_value);
        // 单位米
        std::cout << "X坐标:" << point_in_color_coordinates[0] << ""
                  << "Y坐标:" << point_in_color_coordinates[1] << ""
                  << "Z坐标:" << point_in_color_coordinates[2] << std::endl;
        /*********************************************************************/
        cap_point.position.x = point_in_color_coordinates[0];
        cap_point.position.y = point_in_color_coordinates[1];
        cap_point.position.z = point_in_color_coordinates[2];
        pub.publish(cap_point);

        /**************************ROI区域发布*******************************************/
        sensor_msgs::ImagePtr Color_ROI_Img_msgs = cv_bridge::CvImage(std_msgs::Header(), "bgr8", color_ROI).toImageMsg();
        sensor_msgs::ImagePtr depth_ROI_Img_msgs = cv_bridge::CvImage(std_msgs::Header(), "mono16", depth_ROI).toImageMsg();
        Color_ROI_Img_msgs->header.stamp = ros::Time::now();
        Color_ROI_Img_msgs->header.frame_id = "images";
        depth_ROI_Img_msgs->header.stamp = ros::Time::now();
        depth_ROI_Img_msgs->header.frame_id = "images";
        color_ROI_pub.publish(Color_ROI_Img_msgs);
        depth_ROI_pub.publish(depth_ROI_Img_msgs);
        /*********************************************************************/

        // PCA对视差图进行处理
        // YL.PCAdraw(depth_masked, color_image);

        // 防止数据残留，在循环的结尾清一次容器
        // YOLO部分的容器清除
        res_left.clear();

        // std::string L_fps = "FPS: " + std::to_string(fps_left);
        // cv::putText(color_image, L_fps, cv::Point(11, 80), cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);

        cv::imshow("color_image", color_image);
        cv::imshow("depth_image", depth_image);
        cv::imshow("depth_masked", depth_masked);

        cv::imwrite("color_image.png", color_image);
        cv::imwrite("depth_image.png", depth_image);
        cv::imwrite("depth_masked.png", depth_masked);
        if (!color_ROI.empty() && !depth_ROI.empty())
        {
            cv::imwrite("color_ROI.png", color_ROI);
            cv::imwrite("depth_ROI.png", depth_ROI);
        }
        cv::waitKey(1);

        fcount = 0;
        r.sleep();
    }

    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[outputIndex]));
    context->destroy();
    engine->destroy();
    runtime->destroy();
}