#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>

class getC2F
{
public:
    getC2F();
    cv::Mat getF2R_rpy(double _X, double _Y, double _Z, double _A, double _B, double _C);

public:
    cv::Mat B2C;
    cv::Mat G2F;
    cv::Mat B2R;
    cv::Mat F2R;
    cv::Mat C2F;
    cv::Mat sum_C2F; // 历次求得的C2F的和，用于求解C2F的平均值
    int num_of_C2F;
};

/*
如果是将成员变量设置为私有，只有类内成员函数才可以访问；外界参数先传递给成员函数的形参，再由形参传递给成员变量
*/