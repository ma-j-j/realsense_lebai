
#include "getB2C.h"
#include <fstream>

// using namespace cv;

/*构造函数*/
getB2C::getB2C(std::string imgPathL)
{
    boardSize.width = 8;  // 标定版宽
    boardSize.height = 5; // 标定版长
    grid_lenth = 25;      // 网格长度

    boardPoints = cv::Mat(3, boardSize.width * boardSize.height, CV_64F);

    camPoints = cv::Mat(4, boardSize.width * boardSize.height, CV_64F);

    worldPoints = cv::Mat(4, boardSize.width * boardSize.height, CV_64F);
    Q = cv::Mat(4, 4, CV_64F);

    B2C = cv::Mat(4, 4, CV_64F);

    image1 = cv::imread(imgPathL);
    // cv::imshow("构造", image1);
    // cv::waitKey(1);
    // image2 = cv::imread(imgPathR);

    // 用于过滤面积小的blob，声明根据面积过滤，设置最大与最小面积
    parameters.maxArea = 10000;
    parameters.minArea = 150;
    parameters.filterByArea = true;
    // 初始化blob参数
    parameters.minDistBetweenBlobs = 100; // 分块最小距离
    parameters.filterByColor = true;      // 启用颜色过滤

    blobDetector = cv::SimpleBlobDetector::create(parameters);

    useCirclesGrid = 0;
}

getB2C::getB2C(cv::Mat &left, cv::Mat &right)
{
    boardSize.width = 11;
    boardSize.height = 8;
    grid_lenth = 10;
    boardPoints = cv::Mat(3, boardSize.width * boardSize.height, CV_64F);
    camPoints = cv::Mat(4, boardSize.width * boardSize.height, CV_64F);
    worldPoints = cv::Mat(4, boardSize.width * boardSize.height, CV_64F);
    Q = cv::Mat(4, 4, CV_64F);

    B2C = cv::Mat(4, 4, CV_64F);

    image1 = left;
    image2 = right;

    parameters.maxArea = 10000;
    parameters.minArea = 150;
    parameters.minDistBetweenBlobs = 100;
    parameters.filterByColor = true;
    parameters.filterByArea = true;
    blobDetector = cv::SimpleBlobDetector::create(parameters);

    useCirclesGrid = 0;
}

/*获得世界坐标按列合并的矩阵*/
void getB2C::getBoardPoints()
{
    int k = 0;
    for (int i = 0; i < boardSize.height; i++)
    {
        for (int j = 0; j < boardSize.width; j++)
        {
            boardPoints.at<double>(0, k) = double(j) * double(grid_lenth);
            boardPoints.at<double>(1, k) = double(i) * double(grid_lenth);
            boardPoints.at<double>(2, k++) = 1.0;
        }
    } //*/
}

int getB2C::getImgPoints_plus()
{

    bool found1 = false;

    cv::cvtColor(image1, image1, cv::COLOR_RGB2GRAY); // 彩色图转换成灰度图

    // 寻找角点
    if (!useCirclesGrid)
    {
        found1 = findChessboardCornersSB(255 - image1, boardSize, imageCorners1, cv::CALIB_CB_EXHAUSTIVE | cv::CALIB_CB_ACCURACY); // 这种方法的角点顺序与旧方法相反，在双目标定时可用，但这里涉及到机器人坐标转换，故不使用
    }

    if (useCirclesGrid)
    {
        found1 = findCirclesGrid(255 - image1, boardSize, imageCorners1, cv::CALIB_CB_SYMMETRIC_GRID, blobDetector);
    }

    int resp;
    (found1 == true) ? (resp = 1) : (resp = 0);

    if (resp == 1)
    {
        /*角点可视化*/
        for (int i = 0; i < imageCorners1.size(); i++)
        {
            circle(image1, imageCorners1[i], 10, cv::Scalar(0, 255, 0), -1);

            if (i % boardSize.width == 0) // 给每行第一个角点标序号
            {
                char sequence[50];
                sprintf(sequence, "%d", i + 1);
                std::string Sequence = sequence;
                putText(image1, Sequence, imageCorners1[i], cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(255, 255, 255), 3);
            }

            if (i >= 1) // 将角点连线
            {
                cv::line(image1, imageCorners1[double(i) - 1], imageCorners1[i], cv::Scalar(0, 0, 255), 2, 2);
            }
        }
        // cv::namedWindow("校正后的左图像", 0);
        // cv::namedWindow("校正后的右图像", 0);
        cv::imshow("校正后的左图像", image1);
        // cv::imshow("校正后的右图像", image2);
        cv::imwrite("../data/camero_picture/left.jpg", image1);
        // cv::imwrite("../data/camero_picture/right.jpg", image2);
        cv::waitKey(1);

        // cv::destroyWindow("校正后的左图像");
        // cv::destroyWindow("校正后的右图像");
    }

    return resp;
}

int getB2C::getImgPoints()
{
    if (image1.empty() || image2.empty())
        return -1;

    bool found1 = false, found2 = false;
    cv::cvtColor(image1, image1, cv::COLOR_RGB2GRAY); // 彩色图转换成灰度图
    cv::cvtColor(image2, image2, cv::COLOR_RGB2GRAY);

    // 寻找角点
    if (!useCirclesGrid)
    {
        found1 = findChessboardCornersSB(255 - image1, boardSize, imageCorners1, cv::CALIB_CB_EXHAUSTIVE | cv::CALIB_CB_ACCURACY); // 这种方法的角点顺序与旧方法相反，在双目标定时可用，但这里涉及到机器人坐标转换，故不使用
        found2 = findChessboardCornersSB(255 - image2, boardSize, imageCorners2, cv::CALIB_CB_EXHAUSTIVE | cv::CALIB_CB_ACCURACY); //*/
    }

    if (useCirclesGrid)
    {
        found1 = findCirclesGrid(255 - image1, boardSize, imageCorners1, cv::CALIB_CB_SYMMETRIC_GRID, blobDetector);
        found2 = findCirclesGrid(255 - image2, boardSize, imageCorners2, cv::CALIB_CB_SYMMETRIC_GRID, blobDetector);
    }

    int resp;
    (found1 == true && found2 == true) ? (resp = 1) : (resp = 0);

    if (resp == 1)
    {
        /*角点可视化*/
        for (int i = 0; i < imageCorners1.size(); i++)
        {
            circle(image1, imageCorners1[i], 10, cv::Scalar(0, 255, 0), -1);
            circle(image2, imageCorners2[i], 10, cv::Scalar(0, 255, 0), -1);

            if (i % boardSize.width == 0) // 给每行第一个角点标序号
            {
                char sequence[50];
                sprintf(sequence, "%d", i + 1);
                std::string Sequence = sequence;
                putText(image1, Sequence, imageCorners1[i], cv::FONT_HERSHEY_COMPLEX, 3, cv::Scalar(255, 255, 255), 3);
                putText(image2, Sequence, imageCorners2[i], cv::FONT_HERSHEY_COMPLEX, 3, cv::Scalar(255, 255, 255), 3);
            }

            if (i >= 1) // 将角点连线
            {
                cv::line(image1, imageCorners1[double(i) - 1], imageCorners1[i], cv::Scalar(0, 0, 255), 2, 2);
                cv::line(image2, imageCorners2[double(i) - 1], imageCorners2[i], cv::Scalar(0, 0, 255), 2, 2);
            }
        }
        cv::namedWindow("校正后的左图像", 0);
        cv::namedWindow("校正后的右图像", 0);
        cv::imshow("校正后的左图像", image1);
        cv::imshow("校正后的右图像", image2);
        cv::imwrite("../data/camero_picture/left.jpg", image1);
        cv::imwrite("../data/camero_picture/right.jpg", image2);
        cv::waitKey(1);

        cv::destroyWindow("校正后的左图像");
        cv::destroyWindow("校正后的右图像");
    }

    return resp;
}

void getB2C::get_Q()
{
    std::string Q_path = "../data/标定参数/stereo_cameraParameters.xml";
    cv::FileStorage fs(Q_path, cv::FileStorage::READ);
    fs["remapQ"] >> Q;
}

void getB2C::getCamPoints()
{
    /*获取三维点云数据*/
    cv::Mat srcVector(4, 1, CV_64F),
        camVector(4, 1, CV_64F);

    for (int i = 0; i < boardSize.height * boardSize.width; i++)
    {
        srcVector.at<double>(0, 0) = imageCorners1[i].x;
        srcVector.at<double>(1, 0) = imageCorners1[i].y;
        srcVector.at<double>(2, 0) = abs(imageCorners1[i].x - imageCorners2[i].x);
        srcVector.at<double>(3, 0) = 1.0;
        // std::cout << "srcVector=" << std::endl
        //           << srcVector << std::endl;

        camVector = Q * srcVector;

        camPoints.at<double>(0, i) = camVector.at<double>(0, 0) / camVector.at<double>(3, 0);
        camPoints.at<double>(1, i) = camVector.at<double>(1, 0) / camVector.at<double>(3, 0);
        camPoints.at<double>(2, i) = camVector.at<double>(2, 0) / camVector.at<double>(3, 0);
        camPoints.at<double>(3, i) = 1.0;

        // for循环结束后，得到角点相对于左摄像机坐标系的坐标camPoints大小为：4*角点数
    }
    // std::cout << "camPoints=" << std::endl
    //           << camPoints << std::endl;
} //*/

void getB2C::get_B2C()
{
    cv::Mat wTc(4, 3, CV_64F);
    wTc = camPoints * boardPoints.t() * (boardPoints * boardPoints.t()).inv();

    cv::Mat _wTc(4, 4, CV_64F);
    cv::Mat R1(3, 1, CV_64F);
    cv::Mat R2(3, 1, CV_64F);
    cv::Mat R3(3, 1, CV_64F);

    R1.at<double>(0, 0) = wTc.at<double>(0, 0);
    R1.at<double>(1, 0) = wTc.at<double>(1, 0);
    R1.at<double>(2, 0) = wTc.at<double>(2, 0);

    R2.at<double>(0, 0) = wTc.at<double>(0, 1);
    R2.at<double>(1, 0) = wTc.at<double>(1, 1);
    R2.at<double>(2, 0) = wTc.at<double>(2, 1);

    R3 = R1.cross(R2);

    _wTc.at<double>(0, 0) = wTc.at<double>(0, 0);
    _wTc.at<double>(0, 1) = wTc.at<double>(0, 1);
    _wTc.at<double>(0, 2) = R3.at<double>(0, 0);
    _wTc.at<double>(0, 3) = wTc.at<double>(0, 2);
    _wTc.at<double>(1, 0) = wTc.at<double>(1, 0);
    _wTc.at<double>(1, 1) = wTc.at<double>(1, 1);
    _wTc.at<double>(1, 2) = R3.at<double>(1, 0);
    _wTc.at<double>(1, 3) = wTc.at<double>(1, 2);
    _wTc.at<double>(2, 0) = wTc.at<double>(2, 0);
    _wTc.at<double>(2, 1) = wTc.at<double>(2, 1);
    _wTc.at<double>(2, 2) = R3.at<double>(2, 0);
    _wTc.at<double>(2, 3) = wTc.at<double>(2, 2);
    _wTc.at<double>(3, 0) = wTc.at<double>(3, 0);
    _wTc.at<double>(3, 1) = wTc.at<double>(3, 1);
    _wTc.at<double>(3, 2) = 0;
    _wTc.at<double>(3, 3) = wTc.at<double>(3, 2);

    std::ofstream outfile_wTc("../data/B2C.txt");
    /*保存B2C矩阵*/
    for (int y = 0; y < _wTc.rows; y++)
    {
        for (int x = 0; x < _wTc.cols; x++)
        {
            outfile_wTc << _wTc.at<double>(y, x) << ' ';
        }
        outfile_wTc << std::endl;
    }
    outfile_wTc.close();

    B2C = _wTc;

    cv::FileStorage fs("../data/B2C.xml", cv::FileStorage::WRITE);
    fs << "B2C" << B2C;
    fs.release();
}

void getB2C::transformCamPointsToWorld()
{
    worldPoints = B2C.inv() * camPoints;
    /*将角点世界坐标保存到xml*/
    cv::Mat temp = (worldPoints.t())(cv::Rect(0, 0, 3, boardSize.height * boardSize.width));
    cv::FileStorage fs("../data/角点世界坐标.xml", cv::FileStorage::WRITE);
    fs << "cornersInBoard" << temp;
    fs.release(); //*/
}

void getB2C::outPutB2C(cv::Mat &output)
{
    output = B2C;
}

void getB2C::outPutCamPoints(cv::Mat &output)
{
    output = camPoints;
}