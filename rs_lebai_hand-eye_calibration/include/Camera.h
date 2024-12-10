#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>

using namespace cv;

class Camera_Calibration
{
public:
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    cv::Mat Q;
    // cv::Mat R, T, R1, R2, P1, P2, E, F, Q;

    /***********************相机标定参数********************************************************/
    double K_left[3][3] = {1069.3039574294553, 0.0, 665.2163059735186, 0.0, 1073.720513260743, 528.7356690754631, 0.0, 0.0, 1.0};
    Mat K1 = cv::Mat(3, 3, cv::DataType<double>::type, K_left);
    double d_left[1][5] = {-0.008306369653743267, 0.04258339001017961, 0.0023554580395859876, 0.004920878848336363, 0.0};
    Mat D1 = cv::Mat(1, 5, cv::DataType<double>::type, d_left);
    double K_right[3][3] = {1064.475499494731, 0.0, 632.4710201600866, 0.0, 1062.2203358670179, 501.63114003237433, 0.0, 0.0, 1.0};
    Mat K2 = cv::Mat(3, 3, cv::DataType<double>::type, K_right);
    double d_right[1][5] = {0.06476167062605563, -0.038084199014947, 0.0026171014732020146, 0.0067858261234519796, 0.0};
    Mat D2 = cv::Mat(1, 5, cv::DataType<double>::type, d_right);
    double R_stereo[3][3] = {0.9998366008092249, -0.002713108583850083, 0.017872065467283454, 0.00254656425063597, 0.999953186499124, 0.009334871223826091, -0.01789655533257755, -0.009287833550418121, 0.9997967040629648};
    Mat R = cv::Mat(3, 3, cv::DataType<double>::type, R_stereo);
    Vec3d T = {-0.11953709793677482, 0.00011448251606452993, -0.003274497612802587};
    /*******************************************************************************************/
    cv::Mat map_x, map_y;

    std::vector<std::vector<cv::Point2f>> imagePoints; // 图像坐标系下的角点位置
    std::vector<cv::Point2f> imageCorners;
    std::vector<cv::Mat> rvecs, tvecs;
    std::vector<std::vector<cv::Point3f>> objectPoints; // 世界坐标系下的角点位置
    std::vector<cv::Point3f> objectCorners;

    cv::Size imageSize;
    cv::Size boardSize;

    cv::Mat frame;
    void calibration(cv::Mat &left, cv::Mat &right);
    cv::Mat P1, R1, P2, R2, Lmapx, Lmapy, Rmapx, Rmapy;
};