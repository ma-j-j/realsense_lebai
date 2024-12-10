#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>

#define PI_RAD 0.001745329

class CoordinatesTransform
{
private:
    cv::Mat G2F; // ����to������
    cv::Mat G2F_Ideal;
    cv::Mat B2R; // �궨��to������
    cv::Mat B2C; // �궨��to���
    cv::Mat C2R; // ���to������

    // ������G2F������F2R�ĸ���任����4*4
    cv::Mat F2R_1;
    cv::Mat F2R_2;
    cv::Mat F2R_3;

    // ������B2R
    cv::Mat F2R_11;
    cv::Mat F2R_22;
    cv::Mat F2R_33;

    // ���ڲ���B2R
    cv::Mat F2R_TEST_B2R;

public:
    CoordinatesTransform(); // ���캯��

    void init_Matrix();

    // ��xyzrpyת���ɸ���任����4*4
    void getF2R(double _X, double _Y, double _Z, double _A, double _B, double _C, double _D, int flag);
    // ������Բ��佫xyz+��Ԫ��ת���ɸ������  �ؽڽ�ת���ɸ������
    void getF2R_rpy(double _X, double _Y, double _Z, double _A, double _B, double _C, int flag);

    void getG2F(double delta_X, double delta_Y, double delta_Z);

    void getB2R();

    void testB2R();

    void loadB2C(cv::Mat &the_B2C); // ����ⲿ���󸳸���ԱB2C

    void getC2R();
};