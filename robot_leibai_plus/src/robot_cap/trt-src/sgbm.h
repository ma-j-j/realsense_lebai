#pragma once
#include <iostream>
#include <vector>
#include <string>

/********opencv**************/
#include <opencv2/opencv.hpp>
// #include <opencv2/cudastereo.hpp>
// #include <opencv2/cudaimgproc.hpp>

/********pcl*****************/
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>

/********点云处理**************/
#include <pcl/common/centroid.h>     //质心计算
#include <pcl/filters/passthrough.h> //体素滤波
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h> //统计滤波
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h> //欧式聚类
#include <pcl/octree/octree.h>                 //OCTree
#include <pcl/kdtree/kdtree.h>                 //KDTree

using namespace std;
using namespace cv;
using namespace pcl;

class depth
{
private:
  // 图像
  Mat src, disp, R_L, R_R;
  // 容器
  // vector<int> OneImgInformation, Vec_D;
  // vector<Point2i> Vec_P;
  // vector<Mat> Vec_L, Vec_R, Vec_disp;
  // pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  // pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_prt(new pcl::PointCloud<pcl::PointXYZRGB>);
  // pcl::PointCloud<pcl::PointXYZRGB>::Ptr filter_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);  // 创建体素滤波器
  // pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZRGB>); // 创建统计滤波器
  // pcl::PointCloud<pcl::PointXYZ>::Ptr cldPtr(new pcl::PointCloud<pcl::PointXYZ>);              // 聚类
  // pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZRGB>);

public:
  /***********************相机标定参数********************************************************/
  double K_left[3][3] = {1043.1086744721986, 0.0, 657.2447557781136, 0.0, 1040.1539395836605, 537.6136528434018, 0.0, 0.0, 1.0};
  Mat K1 = cv::Mat(3, 3, cv::DataType<double>::type, K_left);
  double d_left[1][5] = {-0.005891768437156508, 0.06343321635754288, 0.00408542772213255, 0.003843733509744989, 0.0};
  Mat D1 = cv::Mat(1, 5, cv::DataType<double>::type, d_left);
  double K_right[3][3] = {1041.2807502229716, 0.0, 634.570868378219, 0.0, 1034.9189746543698, 527.3271848730394, 0.0, 0.0, 1.0};
  Mat K2 = cv::Mat(3, 3, cv::DataType<double>::type, K_right);
  double d_right[1][5] = {-0.01873856687777212, 0.05555557739227337, 0.0011500931421514662, -0.0003064168757133014, 0.0};
  Mat D2 = cv::Mat(1, 5, cv::DataType<double>::type, d_right);
  double R_stereo[3][3] = {0.9999612130832157, -0.0010789960474459136, 0.008741172499918267, 0.0010342497616122681, 0.9999863479887843, 0.005121929664170197, -0.008746579707196193, -0.005112690444736289, 0.9999486775529243};
  Mat R = cv::Mat(3, 3, cv::DataType<double>::type, R_stereo);
  Vec3d T = {-0.12049636420237257, 0.00010405880259926675, -0.0032857802391733585};
  /*******************************************************************************************/

  /****************视差获取函数**********************************************************/
  void calibration(Mat &left, Mat &right);
  void SGBM(Mat srcleft, Mat srcright, Mat &disp);
  // void SGBM_CUDA(Mat SL, Mat SR, Mat &DP);
  void get_disp_Img(vector<Mat> input_L, vector<Mat> input_R, vector<Mat> &disp);
  void ImgProcessing(Mat &left, Mat &right);
  void get_point_cloud(Mat L, vector<Point2i> input_P, vector<int> input_D, vector<Mat> disp, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &output_cloud);
  void Vec_connector(Mat In_L, Mat In_R, vector<vector<int>> Input, vector<Point2i> &output_P, vector<int> &output_D, vector<Mat> &output_L, vector<Mat> &output_R);
  /******************点云处理函数**********************************/
  void RANSAC(pcl::PointCloud<pcl::PointXYZRGB>::Ptr in_cloude, pcl::PointCloud<pcl::PointXYZRGB>::Ptr out_cloud);
  void cloud_out(pcl::PointCloud<pcl::PointXYZRGB>::Ptr in_cloude, pcl::PointCloud<pcl::PointXYZRGB>::Ptr out_cloud);
  void rs_point(cv::Mat color_image, cv::Mat depth_image, pcl::PointCloud<pcl::PointXYZRGB>::Ptr out_cloud);
  Mat P1, R1, P2, R2, Q, Lmapx, Lmapy, Rmapx, Rmapy, depth_masked, depth_image;

  enum mode_view
  {
    LEFT,
    RIGHT,
    WLS
  }; // 定义了一个枚举类  mode_view
  mode_view view; // 输出左视差图or右视差
};