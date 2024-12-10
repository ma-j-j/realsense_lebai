#include "CameraCalibration.h"
using namespace cv;

Camera_Calibration::Camera_Calibration()
{
    cameraMatrix = Mat(3, 3, CV_64FC1);
    distCoeffs = Mat(1, 5, CV_64FC1);
    R = Mat(3, 3, CV_64FC1);
    T = Mat(3, 1, CV_64FC1);
    E = Mat(3, 3, CV_64FC1);
    F = Mat(3, 3, CV_64FC1);
    Q = Mat(4, 4, CV_64FC1);

    chessBoardWidth = 2580; // 这个参数跟随着相机参数进行变化
    chessBoardHeight = 720; // 1080分辨率 --对应30
    /*	chessBoardWidth = 2208;
        chessBoardHeight = 1242*/
    ; //*/

    imageSize = Size(chessBoardWidth, chessBoardHeight);
    grid_lenth = 25;

    boardWidth = 8;
    boardHeight = 5;
    boardSize = Size(boardWidth, boardHeight);
}

void Camera_Calibration::calibrate() // 单目标定
{
    calibrateCamera(
        objectPoints, // 世界坐标点
        imagePoints,  // 图像坐标点
        imageSize,    // 图像大小
        cameraMatrix, // 标定后输出的相机矩阵
        distCoeffs,   // 标定后输出的畸变矩阵
        rvecs,        // 标定后获得的多个旋转向量
        tvecs,        // 标定后获得的多个平移向量
        0             // 标定方式
    );
    // distCoeffs = Mat();
}

void Camera_Calibration::Get_Structuring_Paremeter(
    std::vector<std::vector<cv::Point2f>> &L_imagePoints,
    std::vector<std::vector<cv::Point2f>> &R_imagePoints,
    Mat &L_intrinsic,
    Mat &R_intrinsic,
    Mat &L_discoeff,
    Mat &R_discoeff) // 立体标定
{
    stereoCalibrate(
        objectPoints,  // 世界坐标
        L_imagePoints, // 相机的摄像机坐标
        R_imagePoints,
        L_intrinsic, // 相机纯内参矩阵
        L_discoeff,  // 相机畸变矩阵
        R_intrinsic,
        R_discoeff,
        imageSize,
        R, T, E, F, //*/结构参数、本征矩阵、基础矩阵
        cv::CALIB_SAME_FOCAL_LENGTH,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 1e-6)); // 选此flag，可把输入的内参矩阵和畸变矩阵作为初始值，进一步标定出更精确的内参和畸变参数，覆盖原值
}

void Camera_Calibration::print_calibration_parameters(Mat &L_intrinsic, Mat &R_intrinsic, Mat &L_discoeff, Mat &R_discoeff)
{
    /*创建txt文档*/
    std::ofstream outfile_L1("../data/标定参数/左内参矩阵.txt");
    std::ofstream outfile_L2("../data/标定参数/左畸变矩阵.txt");
    std::ofstream outfile_R1("../data/标定参数/右内参矩阵.txt");
    std::ofstream outfile_R2("../data/标定参数/右畸变矩阵.txt");
    std::ofstream outfile_S_R("../data/标定参数/旋转结构参数.txt");
    std::ofstream outfile_S_T("../data/标定参数/平移结构参数.txt");
    std::ofstream outfile_E("../data/标定参数/本征矩阵.txt");
    std::ofstream outfile_F("../data/标定参数/基础矩阵.txt");

    /*保存左右内参矩阵、旋转结构参数*/
    for (int y = 0; y < L_intrinsic.rows; y++)
    {
        for (int x = 0; x < L_intrinsic.cols; x++)
        {
            outfile_L1 << L_intrinsic.at<double>(y, x) << ' '; // calibratecamera()返回的cameraMatrix其元素为double(即CV64FC1)型数据
            outfile_R1 << R_intrinsic.at<double>(y, x) << ' ';
            outfile_S_R << R.at<double>(y, x) << ' ';
            outfile_E << E.at<double>(y, x) << ' ';
            outfile_F << F.at<double>(y, x) << ' ';
        }
        outfile_L1 << std::endl;
        outfile_R1 << std::endl;
        outfile_S_R << std::endl;
        outfile_E << std::endl;
        outfile_F << std::endl;
    }
    /*保存左右畸变矩阵*/
    for (int y = 0; y < L_discoeff.rows; y++)
    {
        for (int x = 0; x < L_discoeff.cols; x++)
        {
            outfile_L2 << L_discoeff.at<double>(y, x) << ' ';
            outfile_R2 << R_discoeff.at<double>(y, x) << ' ';
        }
    }
    /*保存平移结构参数*/
    for (int y = 0; y < T.rows; y++)
    {
        for (int x = 0; x < T.cols; x++)
        {
            outfile_S_T << T.at<double>(y, x) << ' ';
        }
    }
}

void Camera_Calibration::Polar_Rectification(Mat &L_intrinsic, Mat &R_intrinsic, Mat &L_discoeff, Mat &R_discoeff) // 双目校正参数获取
{
    stereoRectify(   // 立体矫正
        L_intrinsic, // 左相机纯内参矩阵
        L_discoeff,  // 左相机畸变矩阵
        R_intrinsic,
        L_discoeff,
        imageSize,
        R, // 结构参数R,T
        T,
        R1, // 两个摄像机坐标系按照R1,R2旋转，能实现基本共面、光轴平行、极线水平平行且尽可能共行，并能保证由于这种旋转而造成的视场改变最小（原理看learning OpenCV）
        R2,
        P1,                       // 获得3*4矩阵，前3*3为较正后的新内参矩阵，最后一列为0向量
        P2,                       // P1,P2为两个相机的投影矩阵，其作用是将3D点的坐标投影到公共成像平面上的2D点坐标 P*[X, Y, Z, W] =[x ,y,w]
        Q,                        // 重投影矩阵,矩阵Q可以把2维平面（图像平面）上的点投影到3维空间中
        cv::CALIB_ZERO_DISPARITY, //
        1                         // 用于剪裁的flag。flag=0，显示最保守的很小一块区域；flag=1,显示包括黑点的全部区域
    );

    ////////////这里不参与荔枝采摘实验//////////
    std::ofstream outfile_Q("../data/标定参数/重投影矩阵.txt");
    std::ofstream outfile_R1("../data/标定参数/R1.txt");
    std::ofstream outfile_R2("../data/标定参数/R2.txt");
    std::ofstream outfile_P1("../data/标定参数/P1.txt");
    std::ofstream outfile_P2("../data/标定参数/P2.txt");
    // 保存重投影矩阵
    for (int y = 0; y < Q.rows; y++)
    {
        for (int x = 0; x < Q.cols; x++)
        {
            outfile_Q << Q.at<double>(y, x) << ' ';
        }
        outfile_Q << std::endl;
    }
    // 保存R1和R2
    for (int y = 0; y < R1.rows; y++)
    {
        for (int x = 0; x < R1.cols; x++)
        {
            outfile_R1 << R1.at<double>(y, x) << ' ';
            outfile_R2 << R2.at<double>(y, x) << ' ';
        }
        outfile_R1 << std::endl;
        outfile_R2 << std::endl;
    }
    // 保存P1和P2
    for (int y = 0; y < P1.rows; y++)
    {
        for (int x = 0; x < P1.cols; x++)
        {
            outfile_P1 << P1.at<double>(y, x) << ' ';
            outfile_P2 << P2.at<double>(y, x) << ' ';
        }
        outfile_P1 << std::endl;
        outfile_P2 << std::endl;
    }
}

void Camera_Calibration::Remap(cv::Mat &image, cv::Mat &camearMatrix, cv::Mat &distCoeffs, cv::Mat &R, cv::Mat &P, cv::Mat &map_x, cv::Mat &map_y, cv::Mat &undistorted) // 重投影
{
    cv::initUndistortRectifyMap(camearMatrix, // 旧内参矩阵
                                distCoeffs,   // 畸变矩阵
                                R,            // 极线校准用的旋转变换矩阵
                                P,            // 带新内参的3*4矩阵，函数会自动取带内参的前三列，忽略最后一列0向量
                                image.size(),
                                CV_32FC1,
                                map_x,                         // 新旧x坐标的映射关系
                                map_y                          // 新旧y坐标的映射关系
    );                                                         //*/
                                                               /*根据新旧x,y坐标的映射关系，绘制双目校正后的图像*/
    remap(image, undistorted, map_x, map_y, cv::INTER_LINEAR); // 映射函数作用于image,生成校正图undistorted
}

void Camera_Calibration::Show_rectified_image(cv::Mat rectifyImage_L, cv::Mat rectifyImage_R)
{
    cv::Mat canvas;
    cv::Rect validROIL, validROIR;
    double sf;
    int w, h;
    sf = 600. / MAX(imageSize.width, imageSize.height);
    w = cvRound(imageSize.width * sf);
    h = cvRound(imageSize.height * sf);
    canvas.create(h, w * 2, CV_8UC1);
    /*左图像画到画布上*/
    cv::Mat canvasPart = canvas(cv::Rect(w * 0, 0, w, h));                         // 得到画布的一部分
    resize(rectifyImage_L, canvasPart, canvasPart.size(), 0, 0, cv::INTER_LINEAR); // 把图像缩放到跟canvasPart一样大小
    cv::Rect vroiL(cvRound(validROIL.x * sf), cvRound(validROIL.y * sf),           // 获得被截取的区域
                   cvRound(validROIL.width * sf), cvRound(validROIL.height * sf));
    rectangle(canvasPart, vroiL, cv::Scalar(255, 255, 255), 3, 8); // 画上一个矩形
    /*右图像画到画布上*/
    canvasPart = canvas(cv::Rect(w, 0, w, h)); // 获得画布的另一部分
    resize(rectifyImage_R, canvasPart, canvasPart.size(), 0, 0, cv::INTER_LINEAR);
    cv::Rect vroiR(cvRound(validROIR.x * sf), cvRound(validROIR.y * sf),
                   cvRound(validROIR.width * sf), cvRound(validROIR.height * sf));
    rectangle(canvasPart, vroiR, cv::Scalar(255, 255, 255), 3, 8);
    /*画上对应的线条*/
    cvtColor(canvas, canvas, COLOR_GRAY2BGR);
    for (int i = 0; i < canvas.rows; i += 16)
        line(canvas, cv::Point(0, i), cv::Point(canvas.cols, i), cv::Scalar(0, 255, 0), 1, 8);

    namedWindow("rectified_img", 0);
    imshow("rectified_img", canvas);
}

void Camera_Calibration::get_calibration_parameters(Mat &L_intrinsic, Mat &R_intrinsic, Mat &L_discoeff, Mat &R_discoeff)
{
    std::ifstream infile_L1("../data/标定参数/左内参矩阵.txt");
    std::ifstream infile_L2("../data/标定参数/左畸变矩阵.txt");
    std::ifstream infile_R1("../data/标定参数/右内参矩阵.txt");
    std::ifstream infile_R2("../data/标定参数/右畸变矩阵.txt");
    std::ifstream infile_S_R("../data/标定参数/旋转结构参数.txt");
    std::ifstream infile_S_T("../data/标定参数/平移结构参数.txt");

    for (int y = 0; y < L_intrinsic.rows; y++)
        for (int x = 0; x < L_intrinsic.cols; x++)
        {
            infile_L1 >> L_intrinsic.at<double>(y, x); // calibratecamera()返回的cameraMatrix其元素为double型数据
            infile_R1 >> R_intrinsic.at<double>(y, x);
            infile_S_R >> R.at<double>(y, x);
        }
    for (int y = 0; y < L_discoeff.rows; y++)
        for (int x = 0; x < L_discoeff.cols; x++)
        {
            infile_L2 >> L_discoeff.at<double>(y, x);
            infile_R2 >> R_discoeff.at<double>(y, x);
        }
    for (int y = 0; y < T.rows; y++)
        for (int x = 0; x < T.cols; x++)
        {
            infile_S_T >> T.at<double>(y, x);
        }
}

void Camera_Calibration::print_Q()
{
    std::ofstream outfile_Q("../data/标定参数/重投影矩阵.txt");
    /*保存重投影矩阵*/
    for (int y = 0; y < Q.rows; y++)
    {
        for (int x = 0; x < Q.cols; x++)
        {
            outfile_Q << Q.at<double>(y, x) << ' ';
        }
        outfile_Q << std::endl;
    }
}

void Camera_Calibration::get_Q()
{
    std::ifstream infile_Q("../data/标定参数/重投影矩阵.txt");

    for (int y = 0; y < Q.rows; y++)
        for (int x = 0; x < Q.cols; x++)
        {
            infile_Q >> Q.at<double>(y, x);
        }
}

void Camera_Calibration::iniWorldCoordinate()
{
    for (int i = 0; i < boardSize.height; i++)
        for (int j = 0; j < boardSize.width; j++) // 初始化世界坐标，x,y为格子坐标物理位置，z=0(单位:mm)
        {
            objectCorners.push_back(cv::Point3f(j * grid_lenth, i * grid_lenth, 0)); //*/
        }
}

void Camera_Calibration::sava_cameraParameters_xml(const std::string &parameters_path, cv::Mat &intrinsic, cv::Mat &discoeff)
{
    FileStorage fs(parameters_path, FileStorage::WRITE);
    fs << "intrinsic" << intrinsic;
    fs << "discoeff" << discoeff;
    fs.release();
}

void Camera_Calibration::sava_cameraParameters_xml(const std::string &parameters_path, cv::Mat &R, cv::Mat &T, cv::Mat &Q, cv::Mat &map_xL, cv::Mat &map_yL, cv::Mat &map_xR, cv::Mat &map_yR)
{
    FileStorage fs(parameters_path, FileStorage::WRITE);
    fs << "R" << R;
    fs << "T" << T;
    fs << "remapQ" << Q;
    fs << "remapX1" << map_xL;
    fs << "remapY1" << map_yL;
    fs << "remapX2" << map_xR;
    fs << "remapY2" << map_yR;
    fs.release();
}