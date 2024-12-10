#include "ros/ros.h"
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <cv_bridge/cv_bridge.h>
#include "sensor_msgs/Image.h"
#include "sensor_msgs/CameraInfo.h"

#include "lebai/robot.hh"
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Pose.h>
#include "geometry_msgs/PoseStamped.h"
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <thread>
#include <fstream>
#include <string>
#include <map>

#include <filesystem> // C++17
#include <chrono>     // 时间戳
#include <iomanip>    // 格式化时间
#include <sstream>    // 字符串流

using namespace Eigen;

#define Pi 3.1415926

pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr out_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr ALL_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr ROI_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

double roll_obb = 89.0, pitch_obb = 6.0, yaw_obb = -4.0;
cv::Mat cv_color_image, cv_depth_image, cv_color_ROI_image, cv_depth_ROI_image;
ros::Publisher pub;
cv::Mat C2F(4, 4, CV_64F);
cv::Mat G2F(4, 4, CV_64F);
cv::Mat F2R(4, 4, CV_64F);
geometry_msgs::PoseStamped pose;
bool step_2 = false;
bool four_step = false;
lebai::l_master::Robot a("192.168.1.61");
double initPose[6] = {2, -32, 89, -142, 92, -182};
double endPose[6] = {-3, -54, 82, -107, -93, -184};
std::vector<double> joint_group_start_positions(6);
std::vector<double> joint_group_end_positions(6);
pcl::MomentOfInertiaEstimation<pcl::PointXYZRGB> feature_extractor; // 实例化一个对象

Eigen::Vector3f major_vector, middle_vector, minor_vector;
//******************************************************************************************************************************* */
// 从字符串解析浮点数
float parseFloat(const std::string &value)
{
    try
    {
        return std::stof(value);
    }
    catch (const std::exception &e)
    {
        std::cerr << "无法解析浮点数：" << value << " 错误：" << e.what() << std::endl;
        return 0.0f;
    }
}

// 从字符串解析整数
int parseInt(const std::string &value)
{
    try
    {
        return std::stoi(value);
    }
    catch (const std::exception &e)
    {
        std::cerr << "无法解析整数：" << value << " 错误：" << e.what() << std::endl;
        return 0;
    }
}

// 从TXT文件中读取参数
std::map<std::string, std::string> readParameters(const std::string &fileName)
{
    std::map<std::string, std::string> parameters;
    std::ifstream file(fileName);
    if (!file.is_open())
    {
        std::cerr << "无法打开文件：" << fileName << std::endl;
        return parameters;
    }

    std::string line;
    while (std::getline(file, line))
    {
        size_t pos = line.find('=');
        if (pos != std::string::npos)
        {
            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1);
            parameters[key] = value;
        }
    }
    file.close();
    return parameters;
}

// 获取当前时间戳字符串
namespace fs = std::filesystem;
std::string getCurrentTimestamp()
{
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time_t_now), "%Y%m%d_%H%M%S");
    return oss.str();
}

//********************************************************************************************************************* */
void open_gripper()
{
    a.set_claw(50.0, 100.0);
}

void close_gripper()
{
    a.set_claw(100.0, 0.0);
}
void init_robot()
{
    joint_group_start_positions[0] = initPose[0] * Pi / 180; // degreee measure
    joint_group_start_positions[1] = initPose[1] * Pi / 180;
    joint_group_start_positions[2] = initPose[2] * Pi / 180;
    joint_group_start_positions[3] = initPose[3] * Pi / 180;
    joint_group_start_positions[4] = initPose[4] * Pi / 180;
    joint_group_start_positions[5] = initPose[5] * Pi / 180;

    joint_group_end_positions[0] = endPose[0] * Pi / 180;
    joint_group_end_positions[1] = endPose[1] * Pi / 180;
    joint_group_end_positions[2] = endPose[2] * Pi / 180;
    joint_group_end_positions[3] = endPose[3] * Pi / 180;
    joint_group_end_positions[4] = endPose[4] * Pi / 180;
    joint_group_end_positions[5] = endPose[5] * Pi / 180;

    a.movej(joint_group_start_positions, 4.0, 1.0, 0.0, 0.0);
    open_gripper();

    ROS_INFO("Robot_init_finished!");
    a.wait_move();

    ROS_INFO("isFinish!");
    a.set_do("ROBOT", 0, 1);
    a.set_do("ROBOT", 1, 0);
}
void get_C2F(cv::Mat C2F)
{
    cv::FileStorage fs1("/home/ma/lebai_workspace/rubo_lebai/src/robot_cap/data/C2F.xml", cv::FileStorage::READ);
    if (!fs1.isOpened())
    {
        std::cout << "C2F参数丢失" << std::endl;
    }
    else
    {
        fs1["C2F"] >> C2F;
        // std::cout << "C2F" << C2F << std::endl;
    }
    fs1.release();
    // std::cout << "已经初始化C2F" << std::endl;
}
void get_G2F(cv::Mat G2F)
{
    cv::FileStorage fs2("/home/ma/lebai_workspace/rubo_lebai/src/robot_cap/data/G2F.xml", cv::FileStorage::READ);
    if (!fs2.isOpened())
    {
        std::cout << "G2F参数丢失" << std::endl;
    }
    else
    {
        fs2["G2F"] >> G2F;
        // std::cout << "G2F" << G2F << std::endl;
    }
    fs2.release();
    // std::cout << "已经初始化G2F" << std::endl;
}
cv::Mat get_F2R(double _X, double _Y, double _Z, double _A, double _B, double _C)
{
    double X = _X;
    double Y = _Y;
    double Z = _Z;
    double A = _C;
    double B = _B;
    double C = _A;

    cv::Mat T(4, 4, CV_64F);
    T.at<double>(0, 0) = cos(A) * cos(B);
    T.at<double>(1, 0) = sin(A) * cos(B);
    T.at<double>(2, 0) = -1.0 * sin(B);
    T.at<double>(3, 0) = 0.0;
    T.at<double>(0, 1) = cos(A) * sin(B) * sin(C) - sin(A) * cos(C);
    T.at<double>(1, 1) = sin(A) * sin(B) * sin(C) + cos(A) * cos(C);
    T.at<double>(2, 1) = cos(B) * sin(C);
    T.at<double>(3, 1) = 0.0;
    T.at<double>(0, 2) = cos(A) * sin(B) * cos(C) + sin(A) * sin(C);
    T.at<double>(1, 2) = sin(A) * sin(B) * cos(C) - cos(A) * sin(C);
    T.at<double>(2, 2) = cos(B) * cos(C);
    T.at<double>(3, 2) = 0.0;
    T.at<double>(0, 3) = X;
    T.at<double>(1, 3) = Y;
    T.at<double>(2, 3) = Z;
    T.at<double>(3, 3) = 1;
    // std::cout << "T(F2R):" << T << std::endl;

    return T;
}
void color_ROI_callback(const sensor_msgs::Image::ConstPtr &color_ROI_image)
{
    cv_bridge::CvImagePtr color_ROI_cv_ptr;
    color_ROI_cv_ptr = cv_bridge::toCvCopy(color_ROI_image, "bgr8");
    cv_color_ROI_image = color_ROI_cv_ptr->image;
    if (!cv_color_ROI_image.empty())
    {
        // std::cout << "1" << std::endl;
        cv::imshow("cv_color_ROI_image", cv_color_ROI_image);
    }
}
void depth_ROI_callback(const sensor_msgs::Image::ConstPtr &depth_ROI_image)
{
    cv_bridge::CvImagePtr depth_ROI_cv_ptr;
    depth_ROI_cv_ptr = cv_bridge::toCvCopy(depth_ROI_image, "16UC1");
    cv_depth_ROI_image = depth_ROI_cv_ptr->image;
    if (!cv_depth_ROI_image.empty())
    {
        // std::cout << "2" << std::endl;

        // cv::imshow("cv_depth_ROI_image", cv_depth_ROI_image);
    }
}
bool first_move = false;
bool second_move = false;
bool three_step = false;
int num_b = 0;
Eigen::Vector3f new_points;
geometry_msgs::Pose cap_pose1_temp;
geometry_msgs::Pose cap_pose1_temp_2;
float scale_factor_points = 0.22f;
void cap_pose_callback(const geometry_msgs::Pose::ConstPtr &cap_pose1)
{
    cap_pose1_temp.position.x = cap_pose1->position.x;
    cap_pose1_temp.position.y = cap_pose1->position.y;
    cap_pose1_temp.position.z = cap_pose1->position.z;
    // std::cout << "111: " << endl;

    // std::cout << "cap_pose1:(" << cap_pose1->position.x << "," << cap_pose1->position.y << "," << cap_pose1->position.z << ")" << std::endl;
    if (first_move && cap_pose1->position.y != 0 && cap_pose1->position.x != 0)
    {
        num_b++;
        get_C2F(C2F);
        get_G2F(G2F);
        std::map<std::string, double> cur_pose = a.get_actual_tcp_pose();
        int num;
        num = 0;
        double xyz_rpy_temp[6] = {0};
        double xyz_rpy[6] = {0};
        for (const auto &pair : cur_pose)
        {
            xyz_rpy_temp[num] = pair.second;
            // std::cout << "xyz_rpy_temp[" << num << "]" << " : " << xyz_rpy_temp[num] << "标签为" << pair.first << std::endl;
            num++;
        }
        for (size_t i = 0; i < 3; i++)
        {
            xyz_rpy[i] = xyz_rpy_temp[i + 3];
        }
        for (size_t i = 3; i < 6; i++)
        {
            xyz_rpy[i] = xyz_rpy_temp[i - 3];
        }
        // 将单位转从m换成mm
        for (size_t i = 0; i < 3; i++)
        {
            xyz_rpy[i] *= 1000;
        }
        F2R = get_F2R(xyz_rpy[0], xyz_rpy[1], xyz_rpy[2], xyz_rpy[3], xyz_rpy[4], xyz_rpy[5]);

        /***************************************************************** */

        double m[4][1] = {cap_pose1->position.x * 1000, cap_pose1->position.y * 1000, cap_pose1->position.z * 1000, 1.0};
        cv::Mat P_C = cv::Mat(4, 1, cv::DataType<double>::type, m);
        cv::Mat result(4, 1, CV_64F);
        cv::Mat result1(4, 1, CV_64F);
        result = F2R * C2F * P_C;
        result1 = G2F.inv() * result;
        pose.pose.position.x = result1.at<double>(0, 0) / 1000; // 0.32773; 0.2367; 0.17125
        pose.pose.position.y = result1.at<double>(1, 0) / 1000;
        pose.pose.position.z = result1.at<double>(2, 0) / 1000;
        pose.pose.orientation.x = 0.058107; // 0.84294; 0.48045; 0.15271; 0.18788
        pose.pose.orientation.y = 0.066996;
        pose.pose.orientation.z = 0.54846;
        pose.pose.orientation.w = 0.83146;
        std::cout << "要移动的坐标为（" << pose.pose.position.x << "," << pose.pose.position.y << "," << pose.pose.position.z << ")" << std::endl;
        a.movej({{"x", pose.pose.position.x},
                 {"y", pose.pose.position.y},
                 {"z", pose.pose.position.z - 0.13},
                 {"rz", 89.0 / 180.0 * Pi},
                 {"ry", 6.0 / 180.0 * Pi},
                 {"rx", -4.0 / 180.0 * Pi}},
                6.0, 1.0, 0.0, 0.0);
        a.wait_move();
        std::cout << "第一次移动完成" << std::endl;
        std::cout << "移动的坐标为（" << pose.pose.position.x << "," << pose.pose.position.y << "," << pose.pose.position.z << ")" << std::endl;
        first_move = false;
    }
    if (second_move)
    {
        // num_b++;
        // std::map<std::string, double> cur_pose = a.get_actual_tcp_pose();
        // int num1;
        // num1 = 0;
        // double xyz_rpy_temp[6] = {0};
        // double xyz_rpy[6] = {0};
        // for (const auto &pair : cur_pose)
        // {
        //     xyz_rpy_temp[num1] = pair.second;
        //     num1++;
        // }
        // for (size_t i = 0; i < 3; i++)
        // {
        //     xyz_rpy[i] = xyz_rpy_temp[i + 3];
        // }
        // for (size_t i = 3; i < 6; i++)
        // {
        //     xyz_rpy[i] = xyz_rpy_temp[i - 3];
        // }
        // // 将单位转从m换成mm
        // for (size_t i = 0; i < 3; i++)
        // {
        //     xyz_rpy[i] *= 1000;
        // }
        // F2R = get_F2R(xyz_rpy[0], xyz_rpy[1], xyz_rpy[2], xyz_rpy[3], xyz_rpy[4], xyz_rpy[5]);

        /***************************************************************** */

        // double m1[4][1] = {new_points.x() * 1000, (new_points.y() * 1000), (new_points.z() * 1000), 1.0};
        // cv::Mat P_C1 = cv::Mat(4, 1, cv::DataType<double>::type, m1);
        // cv::Mat result2(4, 1, CV_64F);
        // cv::Mat result3(4, 1, CV_64F);
        // result2 = F2R * C2F * P_C1;
        // result3 = G2F.inv() * result2;
        // pose.pose.position.x = result3.at<double>(0, 0) / 1000; // 0.32773; 0.2367; 0.17125
        // pose.pose.position.y = result3.at<double>(1, 0) / 1000;
        // pose.pose.position.z = result3.at<double>(2, 0) / 1000;
        // pose.pose.orientation.x = 0.058107; // 0.84294; 0.48045; 0.15271; 0.18788
        // pose.pose.orientation.y = 0.066996;
        // pose.pose.orientation.z = 0.54846;
        // pose.pose.orientation.w = 0.83146;
        // std::cout << "第二次要移动的坐标为（" << pose.pose.position.x << "," << pose.pose.position.y << "," << pose.pose.position.z - 0.13 << ")" << std::endl;
        // if (pose.pose.position.x < -0.6)
        // {
        //     pose.pose.position.x = -0.6;
        // }

        // a.movej({{"x", pose.pose.position.x},
        //          {"y", pose.pose.position.y},
        //          {"z", pose.pose.position.z - 0.13},
        //          {"rz", yaw_obb / 180.0 * Pi},
        //          {"ry", pitch_obb / 180.0 * Pi},
        //          {"rx", roll_obb / 180.0 * Pi}},
        //         4.0, 1.0, 0.0, 0.0);
        // a.wait_move();
        // std::cout << "第二次移动完成" << std::endl;
        // std::cout << "移动的坐标为（" << pose.pose.position.x << "," << pose.pose.position.y << "," << pose.pose.position.z - 0.13 << ")" << std::endl;
        // second_move = false;
        // three_step = true;
    }
}

void get_obb(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &in_cloud)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);

    // 滤波
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
    sor.setInputCloud(in_cloud);
    sor.setMeanK(200);
    sor.setStddevMulThresh(1.0);
    sor.filter(*cloud_filtered);

    /*************************************************/
    // 惯性矩估计
    pcl::MomentOfInertiaEstimation<pcl::PointXYZRGB> feature_extractor; // 实例化一个对象
    feature_extractor.setInputCloud(cloud_filtered);                    // 设置输入点云
    feature_extractor.compute();                                        // 开始特征计算

    // 获取惯性矩特征
    std::vector<float> moment_of_inertia; // 存放惯性距的特征向量
    std::vector<float> eccentricity;      // 存放偏心率的特征向量
    pcl::PointXYZRGB min_point_OBB;
    pcl::PointXYZRGB max_point_OBB;
    pcl::PointXYZRGB position_OBB;
    Eigen::Matrix3f rotational_matrix_OBB;
    float major_value, middle_value, minor_value;
    // Eigen::Vector3f major_vector, middle_vector, minor_vector;
    Eigen::Vector3f mass_center;

    // // 计算
    feature_extractor.getMomentOfInertia(moment_of_inertia);                                     // 计算出的惯性矩
    feature_extractor.getEccentricity(eccentricity);                                             // 计算出的偏心率
    feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB); // OBB对应的相关参数
    feature_extractor.getEigenValues(major_value, middle_value, minor_value);                    // 三个特征值
    feature_extractor.getEigenVectors(major_vector, middle_vector, minor_vector);                // 三个特征向量
    feature_extractor.getMassCenter(mass_center);                                                // 计算质心

    // // 画框
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("rect"));
    viewer->setBackgroundColor(0.0, 0.0, 0.0);
    viewer->addCoordinateSystem(0.1);
    viewer->initCameraParameters();
    viewer->addPointCloud<pcl::PointXYZRGB>(cloud, "sample cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "sample cloud");

    Eigen::Vector3f position(position_OBB.x, position_OBB.y, position_OBB.z);
    // std::cout << "position_OBB: " << position_OBB << endl;
    // std::cout << "mass_center: " << mass_center << endl; // 中心坐标
    Eigen::Quaternionf quat(rotational_matrix_OBB);
    viewer->addCube(position, quat, max_point_OBB.x - min_point_OBB.x, max_point_OBB.y - min_point_OBB.y, max_point_OBB.z - min_point_OBB.z, "OBB");
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 1, 1, "OBB");
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.1, "OBB");
    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 10, "OBB");
    viewer->setRepresentationToWireframeForAllActors(); // 将所有actor的可视化表示更改为线框表示Visualizer> openCloud;
    pcl::PointXYZRGB center(mass_center(0), mass_center(1), mass_center(2));

    float scale_factor = 0.1; // 缩放因子，控制线的长度
    // pcl::PointXYZRGB x_axis(major_vector(0) * scale_factor + mass_center(0), major_vector(1) * scale_factor + mass_center(1), major_vector(2) * scale_factor + mass_center(2));
    // pcl::PointXYZRGB y_axis(middle_vector(0) * scale_factor + mass_center(0), middle_vector(1) * scale_factor + mass_center(1), middle_vector(2) * scale_factor + mass_center(2));
    // 确保三个向量的分量为正
    if (major_vector(0) < mass_center(0)) //|| major_vector(1) > mass_center(1)
    {
        major_vector = -major_vector;
    }
    if (middle_vector(1) < mass_center(1))
    {
        middle_vector = -middle_vector;
    }
    if (minor_vector(2) < mass_center(2))
    {
        minor_vector = -minor_vector;
    }
    // 根据缩放因子重新计算坐标
    pcl::PointXYZRGB x_axis(major_vector(0) * scale_factor + mass_center(0), major_vector(1) * scale_factor + mass_center(1), major_vector(2) * scale_factor + mass_center(2));
    pcl::PointXYZRGB y_axis(middle_vector(0) * scale_factor + mass_center(0), middle_vector(1) * scale_factor + mass_center(1), middle_vector(2) * scale_factor + mass_center(2));
    pcl::PointXYZRGB z_axis(minor_vector(0) * scale_factor + mass_center(0), minor_vector(1) * scale_factor + mass_center(1), minor_vector(2) * scale_factor + mass_center(2));
    viewer->addLine(center, x_axis, 1.0f, 0.0f, 0.0f, "major eigen vector");  // 红线
    viewer->addLine(center, y_axis, 0.0f, 1.0f, 0.0f, "middle eigen vector"); // 绿线
    viewer->addLine(center, z_axis, 0.0f, 0.0f, 1.0f, "minor eigen vector");  // 蓝线
    std::cout << "红线: " << major_vector << endl;
    std::cout << "绿线: " << middle_vector << endl;
    std::cout << "蓝线: " << minor_vector << endl;
    Eigen::Matrix3f ROTMatix(3, 3);
    Eigen::Matrix3f ROTMatix_plus_temp(3, 3);
    Eigen::Matrix3f ROTMatix_plus(3, 3);
    get_C2F(C2F);
    Eigen::Vector3f middle_vector_normalized = (-middle_vector).normalized();

    std::string fileName = "src/robot_cap/param/config.txt";
    // 读取参数
    std::map<std::string, std::string> parameters = readParameters(fileName);
    if (parameters.find("scale_factor_points") != parameters.end())
    {
        scale_factor_points = parseFloat(parameters["scale_factor_points"]);
        std::cout << "scale_factor_points= " << scale_factor_points << std::endl;
    }
    new_points = mass_center + middle_vector_normalized * scale_factor_points;
    // 可视化新点
    pcl::PointXYZ new_point_pcl(new_points.x(), new_points.y(), new_points.z());
    viewer->addSphere(new_point_pcl, 0.005, 0.0, 1.0, 0.0, "new_point"); // 绿色球体表示移动后的点
    ROTMatix << -middle_vector, major_vector, minor_vector;
    // std::cout << "ROTMatix=" << ROTMatix << std::endl;
    cv::Mat C2F_sub = C2F(cv::Range(0, 3), cv::Range(0, 3));
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            ROTMatix_plus_temp(i, j) = C2F_sub.at<double>(i, j);
        }
    }
    ROTMatix_plus = ROTMatix_plus_temp * ROTMatix;
    float roll, pitch, yaw;
    if (ROTMatix_plus(0, 2) < 1)
    {
        if (ROTMatix_plus(0, 2) > -1)
        {
            pitch = std::asin(ROTMatix_plus(0, 2));
            roll = std::atan2(-ROTMatix_plus(1, 2), ROTMatix_plus(2, 2));
            yaw = std::atan2(-ROTMatix_plus(0, 1), ROTMatix_plus(0, 0));
        }
        else
        { // rotationMatrix(0, 2) == -1
            pitch = -M_PI / 2;
            roll = std::atan2(-ROTMatix_plus(1, 0), ROTMatix_plus(1, 1));
            yaw = 0;
        }
    }
    else
    { // rotationMatrix(0, 2) == 1
        pitch = M_PI / 2;
        roll = std::atan2(ROTMatix_plus(1, 0), ROTMatix_plus(1, 1));
        yaw = 0;
    }
    // 输出RPY角度
    // std::cout << "Roll (in radians): " << roll << ", Pitch (in radians): " << pitch << ", Yaw (in radians): " << yaw << std::endl;

    // 如果需要转换为度数
    // std::cout << "Roll (in degrees): " << roll * 180.0 / M_PI << ", Pitch (in degrees): " << pitch * 180.0 / M_PI << ", Yaw (in degrees): " << yaw * 180.0 / M_PI << std::endl;
    roll_obb = roll * 180.0 / M_PI;
    pitch_obb = pitch * 180.0 / M_PI;
    yaw_obb = yaw * 180.0 / M_PI;
    Eigen::Vector3f A1(cap_pose1_temp_2.position.x, cap_pose1_temp_2.position.y, cap_pose1_temp_2.position.z);
    Eigen::Vector3f A2 = ROTMatix * A1;
    // cap_pose1_temp_2.position.x = A2(0);
    // cap_pose1_temp_2.position.y = A2(1);
    // cap_pose1_temp_2.position.z = A2(2);
    // mass_center = ROTMatix * mass_center;
    // std::cout << "mass_center:(" << mass_center(0) << "," << mass_center(1) << "," << mass_center(2) << ")" << std::endl;
    // Eigen::Matrix3f ROTMatix_12(3, 3);
    // ROTMatix_12 << -middle_vector, -major_vector, minor_vector;
    // new_points = ROTMatix * new_points;
    // std::cout << "size of cloud :" << cloud_filtered->points.size() << endl;
    // std::cout << "moment_of_inertia :" << moment_of_inertia.size() << endl;
    // std::cout << "eccentricity :" << eccentricity.size() << endl;

    // float height = max_point_OBB.z - min_point_OBB.z;
    // float width = max_point_OBB.y - min_point_OBB.y;
    // float depth = max_point_OBB.x - min_point_OBB.x;
    // cout << "长：" << depth << endl;
    // cout << "宽：" << width << endl;
    // cout << "高：" << height << endl;
    /*************************************************/
}
void colorcallback(const sensor_msgs::Image::ConstPtr &color_image)
{
    cv_bridge::CvImagePtr color_cv_ptr;
    color_cv_ptr = cv_bridge::toCvCopy(color_image, "bgr8");
    cv_color_image = color_cv_ptr->image;
    // cv::imshow("color_sub", cv_color_image);
}
void depthcallback(const sensor_msgs::Image::ConstPtr &depth_image)
{
    cv_bridge::CvImagePtr depth_cv_ptr;
    depth_cv_ptr = cv_bridge::toCvCopy(depth_image, "mono16");
    cv_depth_image = depth_cv_ptr->image;
    // cv::imshow("depth_sub", cv_depth_image);
}
int key;
int num_sec = 0;
int main(int argc, char *argv[])
{
    ros::init(argc, argv, "cap_points_sub");
    ros::NodeHandle nh;
    ros::Rate r(100);
    ros::Subscriber subROIcolor;
    ros::Subscriber subROIdepth;
    ros::Subscriber subcolor;
    ros::Subscriber subdepth;
    ros::Subscriber sub;
    pub = nh.advertise<geometry_msgs::PoseStamped>("/pick_pose", 1);
    subROIcolor = nh.subscribe<sensor_msgs::Image>("/color_ROI_img", 1, &color_ROI_callback);
    subROIdepth = nh.subscribe<sensor_msgs::Image>("/depth_ROI_img", 1, &depth_ROI_callback);
    subcolor = nh.subscribe<sensor_msgs::Image>("/color_img", 1, &colorcallback);
    subdepth = nh.subscribe<sensor_msgs::Image>("/depth_img", 1, &depthcallback);
    sub = nh.subscribe<geometry_msgs::Pose>("/cap_pose", 1, &cap_pose_callback);
    a.start_sys(); // 启动机械臂
    init_robot();  // 移动到初始位置
    pcl::visualization::CloudViewer viewer("Cloud Viewer");
    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr out_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    while (ros::ok())
    {
        ros::spinOnce();

        /****************************************************************************************************************** */
        // 将深度图和彩色图转换为点云
        // 相机参数
        const float fx = 380.954;
        const float fy = 380.954;
        const float cx = 319.145;
        const float cy = 241.156;
        if (!cv_depth_ROI_image.empty() && !cv_color_ROI_image.empty())
        {
            // /************************************************************************************/
            // 根据视差和相机模型计算每一个点的三维坐标, 并添加到PCL点云中
            for (int i = 0; i < cv_depth_ROI_image.rows; ++i)
            {
                for (int j = 0; j < cv_depth_ROI_image.cols; ++j)
                {
                    // 获取 RGB 图像中的颜色值
                    cv::Vec3b color1 = cv_color_ROI_image.at<cv::Vec3b>(i, j);
                    ushort d = cv_depth_ROI_image.at<ushort>(i, j);

                    if (d == 0)
                    {
                        continue;
                    }
                    if (d < 500)
                    {
                        pcl::PointXYZRGB cap;
                        cap.z = d / 1000.f; // 距离由mm转为m
                        cap.x = (j - cx) * cap.z / fx;
                        cap.y = (i - cy) * cap.z / fy;
                        cap.b = color1[0];
                        cap.g = color1[1];
                        cap.r = color1[2];
                        cloud->points.push_back(cap);
                    }
                }
            }

            /************************************************************************************************************************** */

            if (!first_move && num_b == 0)
            {
                // std::cout << "111" << std::endl;
                // std::cout << "在图形窗口按c开始" << std::endl;
                if (key == 'c')
                {
                    first_move = true;
                    // std::cout << "222" << std::endl;
                    key = ' ';
                }
            }
            if (!first_move && num_b == 1 && !second_move) // 到达目标点前一点
            {
                // std::cout << "点云处理" << std::endl;

                // if (num_sec > 1)
                // {
                // num_sec = 0;
                // while (ros::ok()) // 点云处理完成后
                // {
                if (key == 'o')
                {
                    *ROI_cloud = *cloud;
                    cap_pose1_temp_2 = cap_pose1_temp;
                    get_obb(cloud);
                    // std::cout << "point" << ROI_cloud->points.size() << std::endl;

                    if (!cv_color_image.empty() && !cv_depth_image.empty())
                    {
                        // 根据视差和相机模型计算每一个点的三维坐标, 并添加到PCL点云中
                        for (int i = 0; i < cv_depth_image.rows; ++i)
                        {
                            for (int j = 0; j < cv_depth_image.cols; ++j)
                            {
                                // 获取 RGB 图像中的颜色值
                                cv::Vec3b color1 = cv_color_image.at<cv::Vec3b>(i, j);
                                ushort d = cv_depth_image.at<ushort>(i, j);

                                if (d == 0)
                                {
                                    continue;
                                }
                                if (d < 500)
                                {
                                    pcl::PointXYZRGB p;
                                    p.z = d / 1000.f; // 距离由mm转为m
                                    p.x = (j - cx) * p.z / fx;
                                    p.y = (i - cy) * p.z / fy;
                                    p.b = color1[0];
                                    p.g = color1[1];
                                    p.r = color1[2];
                                    ALL_cloud->points.push_back(p);
                                }
                            }
                        }
                        std::cout << "point" << ALL_cloud->points.size() << std::endl;
                    }

                    key = ' ';
                }
                if (key == 'm')
                {
                    key = ' ';
                    std::map<std::string, double> cur_pose = a.get_actual_tcp_pose();
                    int num1;
                    num1 = 0;
                    double xyz_rpy_temp[6] = {0};
                    double xyz_rpy[6] = {0};
                    for (const auto &pair : cur_pose)
                    {
                        xyz_rpy_temp[num1] = pair.second;
                        num1++;
                    }
                    for (size_t i = 0; i < 3; i++)
                    {
                        xyz_rpy[i] = xyz_rpy_temp[i + 3];
                    }
                    for (size_t i = 3; i < 6; i++)
                    {
                        xyz_rpy[i] = xyz_rpy_temp[i - 3];
                    }
                    // 将单位转从m换成mm
                    for (size_t i = 0; i < 3; i++)
                    {
                        xyz_rpy[i] *= 1000;
                    }
                    F2R = get_F2R(xyz_rpy[0], xyz_rpy[1], xyz_rpy[2], xyz_rpy[3], xyz_rpy[4], xyz_rpy[5]);

                    // double m1[4][1] = {cap_pose1_temp_2.position.x * 1000, ((cap_pose1_temp_2.position.y - 0.03) * 1000), (cap_pose1_temp_2.position.z * 1000), 1.0};
                    double m1[4][1] = {new_points(0) * 1000, ((new_points(1)) * 1000), (new_points(2) * 1000), 1.0};
                    cv::Mat P_C1 = cv::Mat(4, 1, cv::DataType<double>::type, m1);
                    cv::Mat result2(4, 1, CV_64F);
                    cv::Mat result3(4, 1, CV_64F);
                    result2 = F2R * C2F * P_C1;
                    result3 = G2F.inv() * result2;
                    pose.pose.position.x = result3.at<double>(0, 0) / 1000; // 0.32773; 0.2367; 0.17125
                    pose.pose.position.y = result3.at<double>(1, 0) / 1000;
                    pose.pose.position.z = result3.at<double>(2, 0) / 1000;
                    std::cout << "第二次要移动的坐标为（" << pose.pose.position.x << "," << pose.pose.position.y << "," << pose.pose.position.z - 0.13 << ")" << std::endl;
                    if (pose.pose.position.x < -0.6)
                    {
                        pose.pose.position.x = -0.6;
                    }
                    std::map<std::string, double> curr_pose;
                    curr_pose = a.get_actual_tcp_pose();
                    a.movej({{"x", curr_pose["x"]},
                             {"y", curr_pose["y"]},
                             {"z", curr_pose["z"]},
                             {"rz", yaw_obb / 180.0 * Pi},
                             {"ry", pitch_obb / 180.0 * Pi},
                             {"rx", roll_obb / 180.0 * Pi}},
                            4.0, 1.0, 0.0, 0.0);
                    std::cout << "末端姿态转换" << std::endl;
                    std::cout << "移动设置的坐标为（" << curr_pose["x"] << "," << curr_pose["y"] << "," << curr_pose["z"] << ")" << std::endl;
                    std::cout << "设置的RPY为" << roll_obb / 180.0 * Pi << "," << pitch_obb / 180.0 * Pi << "," << yaw_obb / 180.0 * Pi << ")" << std::endl;
                    a.wait_move();
                }
                if (key == 'n')
                {
                    key = ' ';

                    std::cout << "第二次要移动的坐标为（" << pose.pose.position.x << "," << pose.pose.position.y << "," << pose.pose.position.z - 0.13 << ")" << std::endl;
                    a.movej({{"x", pose.pose.position.x},
                             {"y", pose.pose.position.y},
                             {"z", pose.pose.position.z - 0.13},
                             {"rz", yaw_obb / 180.0 * Pi},
                             {"ry", pitch_obb / 180.0 * Pi},
                             {"rx", roll_obb / 180.0 * Pi}},
                            4.0, 1.0, 0.0, 0.0);
                    a.wait_move();
                    std::cout << "第二次移动完成" << std::endl;
                    std::cout << "设置的RPY为" << roll_obb / 180.0 * Pi << "," << pitch_obb / 180.0 * Pi << "," << yaw_obb / 180.0 * Pi << ")" << std::endl;
                    std::cout << "移动的坐标为（" << pose.pose.position.x << "," << pose.pose.position.y << "," << pose.pose.position.z - 0.13 << ")" << std::endl;
                    second_move = false;
                    three_step = true;
                }
            }

            /***************************************************************************************************************************** */
            /*******点云可视化***********/
            // if (cloud->points.size() != 0)
            // {
            //     // std::cout << "" << cloud->points.size() << std::endl;
            //     cloud->width = cloud->points.size();
            //     cloud->height = 1;
            //     cloud->is_dense = true;
            //     viewer.showCloud(cloud);
            // }
            cloud->clear();
        }

        if (three_step)
        {
            // close_gripper();                                        // 关爪子
            // a.wait_move();                                         // 等待关完
            std::cout << "第三步" << std::endl;
            std::map<std::string, double> curr_pose1;
            curr_pose1 = a.get_actual_tcp_pose();
            // std::cout << "curr_pose1[' x ']" << curr_pose1["x"] << std::endl;
            // std::cout << "curr_pose1['y']" << curr_pose1["y"] << std::endl;
            // std::cout << "curr_pose1[' z ']" << curr_pose1["z"] << std::endl;
            // std::cout << "前伸要移动设置的坐标为（" << curr_pose1["x"] << "," << curr_pose1["y"] << "," << curr_pose1["z"] + 0.18 << ")" << std::endl;
            // pose.pose.position.z = pose.pose.position.z + 0.18;
            curr_pose1["z"] = curr_pose1["z"] + 0.18;
            // if (pose.pose.position.z > 0.6)
            // {
            //     pose.pose.position.z = 0.6;
            // }
            std::cout << "前伸要移动设置的坐标为（" << curr_pose1["x"] << "," << curr_pose1["y"] << "," << curr_pose1["z"] << ")" << std::endl;
            std::cout << "设置的RPY为" << roll_obb / 180.0 * Pi << "," << pitch_obb / 180.0 * Pi << "," << yaw_obb / 180.0 * Pi << ")" << std::endl;
            a.movej({{"x", curr_pose1["x"]},
                     {"y", curr_pose1["y"]},
                     {"z", curr_pose1["z"]},
                     {"rz", yaw_obb / 180.0 * Pi},
                     {"ry", pitch_obb / 180.0 * Pi},
                     {"rx", roll_obb / 180.0 * Pi}},
                    4.0, 1.0, 0.0, 0.0);
            std::cout << "前伸移动设置的坐标为（" << curr_pose1["x"] << "," << curr_pose1["y"] << "," << curr_pose1["z"] << ")" << std::endl;
            std::cout << "设置的RPY为" << roll_obb / 180.0 * Pi << "," << pitch_obb / 180.0 * Pi << "," << yaw_obb / 180.0 * Pi << ")" << std::endl;
            // std::cout << "末端姿态转换" << std::endl;
            a.wait_move();
            // close_gripper(); // 关爪子
            // a.wait_move();   // 等待关完
            std::map<std::string, double> curr_pose;
            curr_pose = a.get_actual_tcp_pose();
            // std::cout << "curr_pose[' x ']" << curr_pose["x"] << std::endl;
            // std::cout << "curr_pose['y']" << curr_pose["y"] << std::endl;
            // std::cout << "curr_pose[' z ']" << curr_pose["z"] << std::endl;
            a.movej({{"x", curr_pose["x"]},
                     {"y", curr_pose["y"]},
                     {"z", curr_pose["z"] - 0.18},
                     {"rz", yaw_obb / 180.0 * Pi},
                     {"ry", pitch_obb / 180.0 * Pi},
                     {"rx", roll_obb / 180.0 * Pi}},
                    4.0, 1.0, 0.0, 0.0);
            // std::cout << "末端姿态转换" << std::endl;
            std::cout << "剪完伸出来移动设置的坐标为（" << curr_pose["x"] << "," << curr_pose["y"] << "," << curr_pose["z"] - 0.1 << ")" << std::endl;
            std::cout << "设置的RPY为" << roll_obb / 180.0 * Pi << "," << pitch_obb / 180.0 * Pi << "," << yaw_obb / 180.0 * Pi << ")" << std::endl;

            a.wait_move();

            a.movej(joint_group_end_positions, 7.0, 1.0, 0.0, 0.0); // 移动到放置位置
            a.wait_move();                                          // 等待完成
            std::cout << "到达放置位置" << std::endl;
            open_gripper(); // 开爪子
            a.wait_move();
            a.movej(joint_group_start_positions, 7.0, 1.0, 0.0, 0.0); // 移动到原始位置
            a.wait_move();                                            // 等待完成
            std::cout << "回到原始位置" << std::endl;

            three_step = false;
            four_step = true;
        }
        if (four_step)
        {

            std::cout << "成功请按s,失败请按f" << std::endl;
            if (key == 's')
            {
                // 生成时间戳命名
                std::string folder_name = "src/robot_cap/save_data/shiyan_" + getCurrentTimestamp();
                fs::create_directory(folder_name);
                std::ofstream feature_vector_file(folder_name + "/feature_vector.txt", std::ios::app);
                std::ofstream result_outfile("src/robot_cap/save_data/result.txt", std::ios::app);
                feature_vector_file << "红：" << std::endl
                                    << major_vector << std::endl
                                    << "绿：" << std::endl
                                    << middle_vector << std::endl
                                    << "蓝：" << std::endl
                                    << minor_vector << std::endl;
                key = ' ';
                four_step = false;
                result_outfile << "1" << std::endl;
                num_b = 0;
                first_move = false;

                ROI_cloud->width = ROI_cloud->points.size();
                ROI_cloud->height = 1;
                ROI_cloud->is_dense = true;

                ALL_cloud->width = ALL_cloud->points.size();
                ALL_cloud->height = 1;
                ALL_cloud->is_dense = true;
                std::cout << "roipoint" << ALL_cloud->points.size() << std::endl;

                pcl::io::savePCDFileASCII(folder_name + "/ROI_cloud.pcd", *ROI_cloud);
                pcl::io::savePCDFileASCII(folder_name + "/ALL_cloud.pcd", *ALL_cloud);

                std::cout << "成功,按C重新开始" << std::endl;
            }
            if (key == 'f')
            {
                // 生成时间戳命名
                std::string folder_name = "src/robot_cap/save_data/shiyan_" + getCurrentTimestamp();
                fs::create_directory(folder_name);
                std::ofstream feature_vector_file(folder_name + "/feature_vector.txt", std::ios::app);
                std::ofstream result_outfile("src/robot_cap/save_data/result.txt", std::ios::app);
                feature_vector_file << "红：" << std::endl
                                    << major_vector << std::endl
                                    << "绿：" << std::endl
                                    << middle_vector << std::endl
                                    << "蓝：" << std::endl
                                    << minor_vector << std::endl;
                key = ' ';
                four_step = false;
                result_outfile << "0" << std::endl;
                num_b = 0;
                first_move = false;
                std::cout << "失败,按C重新开始" << std::endl;
            }
        }

        r.sleep();
        key = cv::waitKey(1);
    }

    return 0;
}
