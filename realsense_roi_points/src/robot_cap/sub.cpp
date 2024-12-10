#include "ros/ros.h"
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include "sensor_msgs/Image.h"
#include "sensor_msgs/CameraInfo.h"

#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>

#include <Eigen/Dense>
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/common/pca.h>

#include <pcl/common/eigen.h>
#include <Eigen/Dense>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/features/moment_of_inertia_estimation.h>

cv::Mat cv_color_image, cv_depth_image, cv_color_ROI_image, cv_depth_ROI_image;
pcl::visualization::CloudViewer viewer("Cloud Viewer");

void colorcallback(const sensor_msgs::Image::ConstPtr &color_image)
{
    cv_bridge::CvImagePtr color_cv_ptr;
    color_cv_ptr = cv_bridge::toCvCopy(color_image, "bgr8");
    cv_color_image = color_cv_ptr->image;
    cv::imshow("color", cv_color_image);
}
void depthcallback(const sensor_msgs::Image::ConstPtr &depth_image)
{
    cv_bridge::CvImagePtr depth_cv_ptr;
    depth_cv_ptr = cv_bridge::toCvCopy(depth_image, "16UC1");
    cv_depth_image = depth_cv_ptr->image;
    // cv::imshow("cdepth", cv_depth_image);
}

void color_ROI_callback(const sensor_msgs::Image::ConstPtr &color_ROI_image)
{
    cv_bridge::CvImagePtr color_ROI_cv_ptr;
    color_ROI_cv_ptr = cv_bridge::toCvCopy(color_ROI_image, "bgr8");
    cv_color_ROI_image = color_ROI_cv_ptr->image;
    if (!cv_color_ROI_image.empty())
    {
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
        // cv::imshow("cv_depth_ROI_image", cv_depth_ROI_image);
    }
}

int main(int argc, char *argv[])
{
    // pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    // viewer->setBackgroundColor(0, 0, 0); // 黑色背景
    // viewer->addCoordinateSystem(0.1);
    // viewer->initCameraParameters();

    ros::init(argc, argv, "cap_points_sub");
    ros::NodeHandle nh;
    ros::Rate r(100);
    ros::Publisher pub;
    ros::Subscriber subcolor;
    ros::Subscriber subdepth;
    ros::Subscriber subROIcolor;
    ros::Subscriber subROIdepth;
    subcolor = nh.subscribe<sensor_msgs::Image>("/color_img", 1, &colorcallback);
    subdepth = nh.subscribe<sensor_msgs::Image>("/depth_img", 1, &depthcallback);
    subROIcolor = nh.subscribe<sensor_msgs::Image>("/color_ROI_img", 1, &color_ROI_callback);
    subROIdepth = nh.subscribe<sensor_msgs::Image>("/depth_ROI_img", 1, &depth_ROI_callback);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);

    pcl::MomentOfInertiaEstimation<pcl::PointXYZRGB> feature_extractor; // 实例化一个对象
    // std::shared_ptr<pcl::MomentOfInertiaEstimation<pcl::PointXYZRGB>> feature_extractor =
    //     std::make_shared<pcl::MomentOfInertiaEstimation<pcl::PointXYZRGB>>();

    // boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("rect"));
    // viewer->setBackgroundColor(0.0, 0.0, 0.0);

    while (ros::ok())
    {
        ros::spinOnce();
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
                    if (d < 1000)
                    {
                        pcl::PointXYZRGB point;
                        point.z = d / 1000.f; // 距离由mm转为m
                        point.x = (j - cx) * point.z / fx;
                        point.y = (i - cy) * point.z / fy;
                        point.b = color1[0];
                        point.g = color1[1];
                        point.r = color1[2];
                        cloud->points.push_back(point);
                    }
                }
            }

            /*******点云处理***********/
            if (cloud->points.size() != 0)
            {
                // 惯性矩估计
                // feature_extractor.setInputCloud(cloud); // 设置输入点云
                // feature_extractor.compute();

                // 显示点云
                cloud->width = cloud->points.size();
                cloud->height = 1;
                cloud->is_dense = true;
                viewer.showCloud(cloud);
                pcl::io::savePCDFileASCII("roi_cloud.pcd", *cloud);
            }
            cloud->clear();
            // cloud_filtered->clear();
        }

        r.sleep();
        cv::waitKey(1);
    }

    return 0;
}
