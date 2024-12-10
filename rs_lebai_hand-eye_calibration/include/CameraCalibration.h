#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>

class Camera_Calibration
{
public:
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    cv::Mat R, T, R1, R2, P1, P2, E, F, Q;
    cv::Mat map_x, map_y;

    std::vector<std::vector<cv::Point2f>> imagePoints; // 图像坐标系下的角点位置
    std::vector<cv::Point2f> imageCorners;
    std::vector<cv::Mat> rvecs, tvecs;
    std::vector<std::vector<cv::Point3f>> objectPoints; // 世界坐标系下的角点位置
    std::vector<cv::Point3f> objectCorners;

    int chessBoardWidth;
    int chessBoardHeight;
    int boardWidth;
    int boardHeight;
    int grid_lenth;

    cv::Size imageSize;
    cv::Size boardSize;

    cv::Mat frame;

public:
    Camera_Calibration();

    /*1.单目标定。输入的摄像机坐标，世界坐标，求解方程组输出内外参*/
    void calibrate();
    /*2.立体标定。输入单目标定所得的内参，同时内部会重新计算一遍外参，由多组内外参运算、优化，输出最优的结构参数R、T*/ // R、T是左右摄像机坐标系的旋转、平移关系
    void Get_Structuring_Paremeter(std::vector<std::vector<cv::Point2f>> &L_imagePoints, std::vector<std::vector<cv::Point2f>> &R_imagePoints, cv::Mat &L_intrinsic, cv::Mat &R_intrinsic, cv::Mat &L_discoeff, cv::Mat &R_discoeff);

    // 将左右相机的内参数和畸变参数打印出来
    void print_calibration_parameters(cv::Mat &L_intrinsic, cv::Mat &R_intrinsic, cv::Mat &L_discoeff, cv::Mat &R_discoeff);

    // 对立体相机进行立体矫正
    void Polar_Rectification(cv::Mat &L_intrinsic, cv::Mat &R_intrinsic, cv::Mat &L_discoeff, cv::Mat &R_discoeff);

    void Remap(cv::Mat &image, cv::Mat &camearMatrix, cv::Mat &distCoeffs, cv::Mat &R, cv::Mat &P, cv::Mat &map_x, cv::Mat &map_y, cv::Mat &undistorted);

    void Show_rectified_image(cv::Mat rectifyImage_L, cv::Mat rectifyImage_R);

    void get_calibration_parameters(cv::Mat &L_intrinsic, cv::Mat &R_intrinsic, cv::Mat &L_discoeff, cv::Mat &R_discoeff);

    void get_Q();

    void print_Q();

    void iniWorldCoordinate();

    void sava_cameraParameters_xml(const std::string &parameters_path, cv::Mat &intrinsic, cv::Mat &discoeff);

    void sava_cameraParameters_xml(const std::string &parameters_path, cv::Mat &R, cv::Mat &T, cv::Mat &Q, cv::Mat &map_xL, cv::Mat &map_yL, cv::Mat &map_xR, cv::Mat &map_yR);
};