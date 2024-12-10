#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>

class getB2C
{
public:
    getB2C(std::string);
    getB2C(cv::Mat &, cv::Mat &);

    void getBoardPoints();    // 将三维世界点按列排得到矩阵 4*88，三维空间点 （标定板坐标系下的角点坐标）
    int getImgPoints();       // 寻找左右相机角点，在图像坐标系下的角点
    int getImgPoints_plus();  // 寻找左右相机角点，在图像坐标系下的角点
    void getCamPoints_plus(); // 获取角点在图像坐标系对应的控件坐标系，（在相机坐标系下的角点坐标，由图像坐标系下的角点重投影得到）

    void getCamPoints(); // 获取角点在图像坐标系对应的控件坐标系，（在相机坐标系下的角点坐标，由图像坐标系下的角点重投影得到）
    void get_Q();
    void get_B2C();
    void transformCamPointsToWorld();
    void outPutB2C(cv::Mat &output);
    void outPutCamPoints(cv::Mat &output);
    std::vector<cv::Point2d> imageCorners1;
    cv::Size boardSize;
    cv::Mat camPoints;

private:
    cv::Mat boardPoints;
    cv::Mat worldPoints;
    cv::Mat image1;
    cv::Mat image2;
    cv::Mat Q;
    cv::Mat B2C;
    std::vector<cv::Point2d> imageCorners2;
    cv::Ptr<cv::FeatureDetector> blobDetector;
    cv::SimpleBlobDetector::Params parameters; // 监测角点
    int useCirclesGrid;
    int grid_lenth;
};