#include "getC2F.h"

getC2F::getC2F()
{
    sum_C2F = cv::Mat::zeros(4, 4, CV_64F);
    num_of_C2F = 0;
}

cv::Mat getC2F::getF2R_rpy(double _X, double _Y, double _Z, double _A, double _B, double _C)
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
    std::cout << "T(F2R):" << T << std::endl;

    return T;
}