#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/extract_clusters.h> //欧式聚类
#include <pcl/kdtree/kdtree.h>                 //KDTree
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
using namespace Eigen;
int main()
{

    // 读取点云
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);

    pcl::PCDReader reader;
    if (reader.read<pcl::PointXYZRGB>("/home/ma/lebai_workspace/robot_leibai_plus/cloud.pcd", *cloud_filtered) < 0)
    {
        std::cerr << "Error loading point cloud!" << std::endl;
        return -1;
    }

    // 滤波
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
    sor.setInputCloud(cloud_filtered);
    sor.setMeanK(200);
    sor.setStddevMulThresh(1.0);
    sor.filter(*cloud);

    /*************************************************/
    // 惯性矩估计
    pcl::MomentOfInertiaEstimation<pcl::PointXYZRGB> feature_extractor; // 实例化一个对象
    feature_extractor.setInputCloud(cloud);                             // 设置输入点云
    feature_extractor.compute();                                        // 开始特征计算

    // 获取惯性矩特征
    std::vector<float> moment_of_inertia; // 存放惯性距的特征向量
    std::vector<float> eccentricity;      // 存放偏心率的特征向量
    pcl::PointXYZRGB min_point_OBB;
    pcl::PointXYZRGB max_point_OBB;
    pcl::PointXYZRGB position_OBB;
    Eigen::Matrix3f rotational_matrix_OBB;
    float major_value, middle_value, minor_value;
    Eigen::Vector3f major_vector, middle_vector, minor_vector;
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
    viewer->setRepresentationToWireframeForAllActors(); // 将所有actor的可视化表示更改为线框表示
    pcl::PointXYZRGB center(mass_center(0), mass_center(1), mass_center(2));

    float scale_factor = 0.1; // 缩放因子，控制线的长度
    // pcl::PointXYZRGB x_axis(major_vector(0) * scale_factor + mass_center(0), major_vector(1) * scale_factor + mass_center(1), major_vector(2) * scale_factor + mass_center(2));
    // pcl::PointXYZRGB y_axis(middle_vector(0) * scale_factor + mass_center(0), middle_vector(1) * scale_factor + mass_center(1), middle_vector(2) * scale_factor + mass_center(2));
    // 确保三个向量的分量为正
    if (major_vector(0) < mass_center(0) || major_vector(1) > mass_center(1))
    {
        major_vector = -major_vector;
    }
    if (middle_vector(1) > mass_center(1))
    {
        middle_vector = -middle_vector;
    }
    if (minor_vector(2) < 0)
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
    ROTMatix << -middle_vector, major_vector, minor_vector;
    std::cout << "ROTMatix=" << ROTMatix << std::endl;
    float roll, pitch, yaw;
    if (ROTMatix(0, 2) < 1)
    {
        if (ROTMatix(0, 2) > -1)
        {
            pitch = std::asin(ROTMatix(0, 2));
            roll = std::atan2(-ROTMatix(1, 2), ROTMatix(2, 2));
            yaw = std::atan2(-ROTMatix(0, 1), ROTMatix(0, 0));
        }
        else
        { // rotationMatrix(0, 2) == -1
            pitch = -M_PI / 2;
            roll = std::atan2(-ROTMatix(1, 0), ROTMatix(1, 1));
            yaw = 0;
        }
    }
    else
    { // rotationMatrix(0, 2) == 1
        pitch = M_PI / 2;
        roll = std::atan2(ROTMatix(1, 0), ROTMatix(1, 1));
        yaw = 0;
    }

    // 输出RPY角度
    std::cout << "Roll (in radians): " << roll << ", Pitch (in radians): " << pitch << ", Yaw (in radians): " << yaw << std::endl;

    // 如果需要转换为度数
    std::cout << "Roll (in degrees): " << roll * 180.0 / M_PI << ", Pitch (in degrees): " << pitch * 180.0 / M_PI << ", Yaw (in degrees): " << yaw * 180.0 / M_PI << std::endl;
    // std::cout << "size of cloud :" << cloud->points.size() << endl;
    // std::cout << "moment_of_inertia :" << moment_of_inertia.size() << endl;
    // std::cout << "eccentricity :" << eccentricity.size() << endl;

    float height = max_point_OBB.z - min_point_OBB.z;
    float width = max_point_OBB.y - min_point_OBB.y;
    float depth = max_point_OBB.x - min_point_OBB.x;
    // cout << "长：" << depth << endl;
    // cout << "宽：" << width << endl;
    // cout << "高：" << height << endl;
    // /*************************************************/

    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
    }

    // pcl::visualization::CloudViewer viewer("Cloud Viewer");
    // viewer.showCloud(cloud_filtered);

    // while (!viewer.wasStopped())
    // {
    // }
    return 0;
}