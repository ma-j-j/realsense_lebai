#include "lebai/robot.hh"
#include "CoordinatesTransform.h"
#include <getC2F.h>
#include <getB2C.h>
// #include <Camera.h>
#include "CameraCalibration.h"
#include <librealsense2/rs.hpp>

#define ROBOT_IP "192.168.1.118" // 机器人IP地址
#define M_PI 3.14159265358979323846
Camera_Calibration camL, camR, stereo_Calibrator;
CoordinatesTransform proj;
getC2F proj_C2F;
bool Lebai_flag = false;     // Lebai_flag = 0代表没有登入机器人系统；Lebai_flag = 1则登入成功
bool lebai_arm_flag = false; // 用来表示遨博机械臂的状态 TRUE为已启动；FLASE为未启动
int lei_arm_mode;
int num;

static cv::Mat
frame_to_mat(const rs2::frame &f)
{
    using namespace cv;
    using namespace rs2;

    auto vf = f.as<video_frame>();
    const int w = vf.get_width();
    const int h = vf.get_height();

    if (f.get_profile().format() == RS2_FORMAT_BGR8) // BGR 格式
    {
        return Mat(Size(w, h), CV_8UC3, (void *)f.get_data(), Mat::AUTO_STEP);
    }
    else if (f.get_profile().format() == RS2_FORMAT_RGB8) // RGB 格式
    {
        auto r_rgb = Mat(Size(w, h), CV_8UC3, (void *)f.get_data(), Mat::AUTO_STEP);
        Mat r_bgr;
        cvtColor(r_rgb, r_bgr, COLOR_RGB2BGR);
        return r_bgr;
    }
    else if (f.get_profile().format() == RS2_FORMAT_Z16) // 深度图格式
    {
        return Mat(Size(w, h), CV_16UC1, (void *)f.get_data(), Mat::AUTO_STEP);
    }
    else if (f.get_profile().format() == RS2_FORMAT_Y8) // 灰度图格式
    {
        return Mat(Size(w, h), CV_8UC1, (void *)f.get_data(), Mat::AUTO_STEP);
    }
    else if (f.get_profile().format() == RS2_FORMAT_DISPARITY32) // 视差图格式
    {
        return Mat(Size(w, h), CV_32FC1, (void *)f.get_data(), Mat::AUTO_STEP);
    }

    throw std::runtime_error("Frame format is not supported yet!");
}

static cv::Mat depth_frame_to_meters(const rs2::depth_frame &f)
{
    cv::Mat dm = frame_to_mat(f);
    dm.convertTo(dm, CV_64F);
    dm = dm * f.get_units();
    return dm;
}

int main(int argc, char *argv[])
{
    lebai::l_master::Robot robot(ROBOT_IP);
    /***********************************************************************************************************/
    // robot.set_claw(50.0, 50.0);  // 设置夹爪的开合度，第一个参数是力度（0-100）第二个参数是开合度（0-100）
    // robot.start_sys();           // 启动机械臂
    // robot.stop_sys();            // 停止机械臂
    // robot.get_actual_tcp_pose(); // 获取机械臂末端在机械臂基坐标系下的实际位姿，CartesianPose = std::map<std::string,double>，应当包括键为x,y,z,rz,ry,rx的值.
    // robot.get_robot_mode();      // 获取机器人状态码(-1：控制系统故障 0：硬件通讯故障 1：已急停 2：初始化中 4：初始化完成 5：空闲 6：暂停
    // 7：运行中 8：更新中 9：启动中 10：正在停止 11：示教中 12：已停止)
    // robot.is_disconnected(); // 是否已与手臂断开连接
    /*******************************************************************************************************/
    cv::FileStorage fs("../data/标定参数/stereo_cameraParameters.xml", cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cout << "立体标定参数缺失，请进行标定" << std::endl;
        fs.release();
    }
    else
    {
        fs["remapX1"] >> camL.map_x; // 映射表读取是用来对获取的图像帧进行矫正的，使其达到标准立体相机模型，行对齐
        fs["remapY1"] >> camL.map_y;
        fs["remapX2"] >> camR.map_x;
        fs["remapY2"] >> camR.map_y;
        fs["remapQ"] >> stereo_Calibrator.Q;
        std::cout << "已经读取立体标定参数" << std::endl;
        fs.release();
    }
    proj.init_Matrix(); // 初始化手眼标定参数 G2F B2R B2C
    /******************************************相机相关参数*************************************************************************/
    // realsense相机参数配置
    rs2::pipeline pipe;
    rs2::config cfg;
    rs2::pointcloud pc;
    // boost::shared_ptr<pcl::visualization::PCLVisualizer> openCloud;
    // cfg.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_Z16, 6);
    // cfg.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_BGR8, 6);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    rs2::pipeline_profile profile = pipe.start(cfg);
    rs2::frameset data;

    // 创建深度流和彩色流对齐的实例化对象
    rs2::align align_to(RS2_STREAM_COLOR);

    cv::namedWindow("left", cv::WINDOW_NORMAL);
    cv::namedWindow("right", cv::WINDOW_NORMAL);
    cv::Mat imageL;
    cv::Mat imageR;
    /***********************************************************************************************************/
    int key; // 读按键
    int step_key;
    int lebai_login;  // 登陆乐白
    int camera_login; // 打开相机

    std::cout << "是否要启动USB相机,是--请输入1;不是--请输入0" << std::endl;
    std::cin >> camera_login;
    if (1 == camera_login)
    {
        // if (!cap.isOpened())
        // {
        //     std::cout << "usb相机没有打开！" << std::endl;
        //     return -1;
        // }
    }

    if (0 == camera_login)
    {
        std::string pathL = "../data/calibration_picture/left/left0.jpg";
        std::string pathR = "../data/calibration_picture/right/right0.jpg";

        imageL = cv::imread(pathL);
        imageR = cv::imread(pathR);
    }
    std::cout << "是否要启动遨博机器人，是--请输入1;不是--请输入0" << std::endl;
    std::cin >> lebai_login;
    if (1 == lebai_login)
    {
        if (!robot.is_disconnected())
        {
            // 如果登入成功就启动机械臂
            ////启动机械臂(必须连接真实机械臂）
            std::cout << "成功登入乐白机器人！" << std::endl;
            Lebai_flag = true;
            robot.start_sys();                     // 启动机械臂
            lei_arm_mode = robot.get_robot_mode(); // 获取机器人状态码(-1：控制系统故障 0：硬件通讯故障 1：已急停 2：初始化中 4：初始化完成 5：空闲 6：暂停
                                                   // 7：运行中 8：更新中 9：启动中 10：正在停止 11：示教中 12：已停止)
            if (lei_arm_mode == 5)
            {
                std::cout << "成功启动机械臂！" << std::endl;
                lebai_arm_flag = true;
            }
            else
            {
                std::cout << "启动机械臂失败！" << std::endl;
                return -1;
            }
        }
        else
        {
            lei_arm_mode = robot.get_robot_mode(); // 获取机器人状态码(-1：控制系统故障 0：硬件通讯故障 1：已急停 2：初始化中 4：初始化完成 5：空闲 6：暂停
                                                   // 7：运行中 8：更新中 9：启动中 10：正在停止 11：示教中 12：已停止)
            std::cout << "登入乐白机器人失败，请重试" << std::endl;
            std::cout << "lei_arm_mode=" << lei_arm_mode << std::endl;
            return -1;
        }
    }
    if (0 == lebai_login)
    {
        std::cout << "不登入乐白机器人,可使用相机" << std::endl;
    }

    int imagecount = 1; // 记录图像的个数
    std::cout << "按下b或B--双目标定; " << std::endl;
    std::cout << "按下g或G--获取G2F" << std::endl;
    std::cout << "按下r或R--获取B2R" << std::endl;
    std::cout << "求B2C的时候,先按下p或P--拍照，获取左右相机一帧图像" << std::endl;
    std::cout << "然后按下c或C--获取B2C" << std::endl;
    std::cout << "按下f,求C2F" << std::endl;
    std::cout << "按下q或Q--退出" << std::endl;
    while (true)
    {
        if (0 == camera_login)
        {
            cv::imshow("left", imageL);
            cv::imshow("right", imageR);
            key = cv::waitKey(1);
        }
        // 判断是否打开zed相机，获取图像
        if (1 == camera_login)
        {
            data = pipe.wait_for_frames();
            data = align_to.process(data); // 实际进行流对齐

            // 基于对齐的混合流获取深度流和彩色流,进而获取流对齐后的深度内参
            rs2::video_frame color_stream = data.get_color_frame();
            // rs2::depth_frame aligned_depth_stream = data.get_depth_frame();
            // rs2::video_stream_profile depth_stream_profile = aligned_depth_stream.get_profile().as<rs2::video_stream_profile>();
            // const auto depth_intrinsics = depth_stream_profile.get_intrinsics(); // 获取对齐后的深度内参

            // 获取彩色图像宽
            const int w = color_stream.as<rs2::video_frame>().get_width();
            const int h = color_stream.as<rs2::video_frame>().get_height();

            // 获取原图及深度图
            rs2::frame color_frame = data.get_color_frame();
            // rs2::depth_frame depth_frame = data.get_depth_frame();

            camL.frame = frame_to_mat(color_frame);
            // camR.frame = depth_frame_to_meters(depth_frame); // 深度图单位转换为米

            // // 将左右摄像头图像转换为 OpenCV 格式
            // camL.frame = cv::Mat(cv::Size(w, h), CV_8UC1, (void *)left_frame.get_data(), cv::Mat::AUTO_STEP);
            // camR.frame = cv::Mat(cv::Size(w, h), CV_8UC1, (void *)right_frame.get_data(), cv::Mat::AUTO_STEP);

            cv::imshow("left", camL.frame);
            // cv::imshow("right", camR.frame);
            key = cv::waitKey(1);
        }

        if (key == 'B' || key == 'b') // 双目标定
        {
            std::cout << "已经进入双目标定程序部分" << std::endl;
            bool found1 = false; // flag标志，用于判断角点检测
            bool found2 = false;
            if (imagecount == 1) // 对每张图片的角点进行世界坐标系坐标的计算
            {
                std::cout << "计算世界坐标" << std::endl;
                camL.iniWorldCoordinate();
                camR.iniWorldCoordinate();
                stereo_Calibrator.iniWorldCoordinate();
            }
            cv::Mat left_image, right_image;
            left_image = camL.frame;
            right_image = camR.frame;
            // cv::cvtColor(left_image, left_image, cv::COLOR_RGB2GRAY);
            // cv::cvtColor(right_image, right_image, cv::COLOR_RGB2GRAY);
            // cv::imshow("gray_left", left_image);
            // cv::imshow("gray_right", right_image);

            found1 = cv::findChessboardCornersSB(255 - left_image, camL.boardSize, camL.imageCorners, cv::CALIB_CB_EXHAUSTIVE | cv::CALIB_CB_ACCURACY);
            found2 = cv::findChessboardCornersSB(255 - right_image, camR.boardSize, camR.imageCorners, cv::CALIB_CB_EXHAUSTIVE | cv::CALIB_CB_ACCURACY);

            if (found1 && found2)
            {
                // 左右图像角点都检测成功

                // 将图像坐标系和世界坐标系中的角点坐标存放在最终存放容器中
                camL.imagePoints.push_back(camL.imageCorners); // 图像坐标系
                camR.imagePoints.push_back(camR.imageCorners);

                camL.objectPoints.push_back(camL.objectCorners); // 世界坐标系
                camR.objectPoints.push_back(camR.objectCorners);

                stereo_Calibrator.objectPoints.push_back(stereo_Calibrator.objectCorners);

                std::cout << "已拍摄" << imagecount << "对" << std::endl;
                cv::imwrite("../data/calibration_picture/left/" + std::to_string(imagecount) + ".jpg", left_image);
                cv::imwrite("../data/calibration_picture/right/" + std::to_string(imagecount) + ".jpg", left_image);
                imagecount++;
            }
            else
            {
                std::cout << "获取角点失败！" << std::endl;
            }

            if (imagecount == 20)
            {
                std::cout << "正在进行双目标定，请稍等..." << std::endl;
                // 先进行单目标定,获取左右相机的内参数和畸变矩阵
                camL.calibrate();
                camR.calibrate();

                // 将单目标定出来的内参和畸变放入立体标定中
                // 经过立体标定后，将输入的内参矩阵和畸变矩阵作为初始值，进一步标定出更精确的内参和畸变参数，覆盖原值
                // 同时得到R T E F，两个相机之间的旋转和平移，基础矩阵F和本征矩阵E
                stereo_Calibrator.Get_Structuring_Paremeter(camL.imagePoints, camR.imagePoints, camL.cameraMatrix, camR.cameraMatrix, camL.distCoeffs, camR.distCoeffs);

                // 保存左右相机内参和畸变、E F R T
                stereo_Calibrator.print_calibration_parameters(camL.cameraMatrix, camR.cameraMatrix, camL.distCoeffs, camR.distCoeffs);

                // 进行立体矫正，并且保存Q P1 P2 R1 R2（P1 P2可以用来求映射表
                stereo_Calibrator.Polar_Rectification(camL.cameraMatrix, camR.cameraMatrix, camL.distCoeffs, camR.distCoeffs);

                cv::Mat rectified_L, rectified_R; // 存储投影到公共成像平面的图像
                // remap函数利用R1 R2和P1 P2左右相机的map_x map_y
                stereo_Calibrator.Remap(left_image, camL.cameraMatrix, camL.distCoeffs, stereo_Calibrator.R1, stereo_Calibrator.P1, camL.map_x, camL.map_y, rectified_L);
                stereo_Calibrator.Remap(right_image, camR.cameraMatrix, camR.distCoeffs, stereo_Calibrator.R2, stereo_Calibrator.P2, camR.map_x, camR.map_y, rectified_R);

                // 显示矫正后的图像
                // stereo_Calibrator.Show_rectified_image(rectified_L, rectified_R);

                camL.sava_cameraParameters_xml("../data/标定参数/L_cameraParameters.xml", camL.cameraMatrix, camL.distCoeffs);
                camR.sava_cameraParameters_xml("../data/标定参数/R_cameraParameters.xml", camR.cameraMatrix, camR.distCoeffs);
                stereo_Calibrator.sava_cameraParameters_xml("../data/标定参数/stereo_cameraParameters.xml", stereo_Calibrator.R, stereo_Calibrator.T, stereo_Calibrator.Q, camL.map_x, camL.map_y, camR.map_x, camR.map_y);

                stereo_Calibrator.Show_rectified_image(rectified_L, rectified_R);

                // cv::destroyWindow("left");
                // cv::destroyWindow("right");
                key = ' ';
                imagecount = 1; // 重新将imagecount置1，便于下次进行双目标定
                std::cout << "双目标定已经完成！" << std::endl;
                std::cout << "返回循环" << std::endl;
                // break;
            }
        }

        //*----------工具标定，求G2F--------------*/
        if (key == 'G' || key == 'g')
        {
            std::cout << "已经进入工具标定程序，任务：求G2F" << std::endl;
            // 求G2F，需要三组数据，即利用工具点同一个空间点的F2R（旋转和平移数据）
            // 1. 获得当前机器人数据，xyzrpy
            if (lebai_arm_flag && Lebai_flag)
            {
                cv::Mat ToolDataMatrix(3, 6, CV_64F); // 每一行存储每一次的xyzrpy
                double xyz_rpy1[6] = {0};
                double xyz_rpy2[6] = {0};
                double xyz_rpy3[6] = {0};
                double xyz_rpy_temp[6] = {0};
                std::cout << "第一次将机器臂点在某个空间点,完成请在终端按  “1” 或 “1” " << std::endl;
                std::cin >> step_key;
                if (step_key == 1 || step_key == 1)
                {
                    std::map<std::string, double> one_pose = robot.get_actual_tcp_pose();
                    num = 0;
                    for (const auto &pair : one_pose)
                    {
                        xyz_rpy_temp[num] = pair.second;
                        std::cout << "xyz_rpy_temp[" << num << "]" << " : " << xyz_rpy_temp[num] << "标签为" << pair.first << std::endl;
                        num++;
                    }

                    for (size_t i = 0; i < 3; i++)
                    {
                        xyz_rpy1[i] = xyz_rpy_temp[i + 3];
                    }
                    for (size_t i = 3; i < 6; i++)
                    {
                        xyz_rpy1[i] = xyz_rpy_temp[i - 3];
                    }
                    // 将单位转从m换成mm
                    for (size_t i = 0; i < 3; i++)
                    {
                        xyz_rpy1[i] *= 1000;
                    }
                    for (size_t i = 0; i < 6; i++)
                    {
                        // std::cout << "xyz_rpy1[" << std::endl;
                        std::cout << "单位转换后xyz_rpy1[" << i << "]" << " : " << xyz_rpy1[i] << std::endl;
                    }

                    // 将获得的机器人当前位置的xyzrpy转换成刚体变换矩阵F2R
                    proj.getF2R_rpy(xyz_rpy1[0], xyz_rpy1[1], xyz_rpy1[2], xyz_rpy1[3], xyz_rpy1[4], xyz_rpy1[5], 1); // F2R_1
                    step_key = ' ';
                }
                std::cout << "第二次将机器臂点在某个空间点,完成请在终端按  “2” 或 “2” " << std::endl;
                // system("read");
                std::cin >> step_key;
                if (step_key == 2 || step_key == 2)
                {
                    std::map<std::string, double> two_pose = robot.get_actual_tcp_pose();
                    num = 0;
                    for (const auto &pair : two_pose)
                    {
                        xyz_rpy_temp[num] = pair.second;
                        std::cout << "xyz_rpy_temp[" << num << "]" << " : " << xyz_rpy_temp[num] << "标签为" << pair.first << std::endl;
                        num++;
                    }
                    for (size_t i = 0; i < 3; i++)
                    {
                        xyz_rpy2[i] = xyz_rpy_temp[i + 3];
                    }
                    for (size_t i = 3; i < 6; i++)
                    {
                        xyz_rpy2[i] = xyz_rpy_temp[i - 3];
                    }
                    // 将单位转从m换成mm
                    for (size_t i = 0; i < 3; i++)
                    {
                        xyz_rpy2[i] *= 1000;
                    }
                    for (size_t i = 0; i < 6; i++)
                    {
                        std::cout << "单位转换后xyz_rpy2[" << i << "]" << " : " << xyz_rpy2[i] << std::endl;
                    }
                    // 将获得的机器人当前位置的xyzrpy转换成刚体变换矩阵F2R
                    proj.getF2R_rpy(xyz_rpy2[0], xyz_rpy2[1], xyz_rpy2[2], xyz_rpy2[3], xyz_rpy2[4], xyz_rpy2[5], 2); // F2R_2
                    step_key = ' ';
                }
                std::cout << "第三次将机器臂点在某个空间点,完成请在终端按  “3” 或 “3” " << std::endl;
                // system("read");
                std::cin >> step_key;
                if (step_key == 3 || step_key == 3)
                {
                    std::map<std::string, double> three_pose = robot.get_actual_tcp_pose();
                    num = 0;
                    for (const auto &pair : three_pose)
                    {
                        xyz_rpy_temp[num] = pair.second;
                        std::cout << "xyz_rpy_temp[" << num << "]" << " : " << xyz_rpy_temp[num] << "标签为" << pair.first << std::endl;
                        num++;
                    }
                    for (size_t i = 0; i < 3; i++)
                    {
                        xyz_rpy3[i] = xyz_rpy_temp[i + 3];
                    }
                    for (size_t i = 3; i < 6; i++)
                    {
                        xyz_rpy3[i] = xyz_rpy_temp[i - 3];
                    }
                    // 将单位转从m换成mm
                    for (size_t i = 0; i < 3; i++)
                    {
                        xyz_rpy3[i] *= 1000;
                    }
                    for (size_t i = 0; i < 6; i++)
                    {
                        std::cout << "单位转换后xyz_rpy3[" << i << "]" << " : " << xyz_rpy3[i] << std::endl;
                    }
                    // 将获得的机器人当前位置的xyzrpy转换成刚体变换矩阵F2R
                    proj.getF2R_rpy(xyz_rpy3[0], xyz_rpy3[1], xyz_rpy3[2], xyz_rpy3[3], xyz_rpy3[4], xyz_rpy3[5], 3); // F2R_3
                    step_key = ' ';
                }
                // 保存三次空间点的zyxrpy坐标到txt文件
                for (int l = 0; l < 6; l++)
                {
                    ToolDataMatrix.at<double>(0, l) = xyz_rpy1[l]; // 第一次点的空间点坐标 xyzrpy
                    ToolDataMatrix.at<double>(1, l) = xyz_rpy2[l];
                    ToolDataMatrix.at<double>(2, l) = xyz_rpy3[l];
                }

                std::ofstream save_ToolData("../data/ToolData.txt");
                for (int i = 0; i < 3; i++)
                {
                    for (int j = 0; j < 6; j++)
                    {
                        save_ToolData << ToolDataMatrix.at<double>(i, j) << ' ';
                    }
                    save_ToolData << std::endl;
                }
                save_ToolData.close();
                // 获取G2F
                proj.getG2F(0, 0, 0);
                std::cout << "G2F已经获取" << std::endl;
                key = ' ';
                std::cout << "返回循环" << std::endl;
            }
            else
            {
                std::cout << "没有成功登入机器人系统" << std::endl;
                std::cout << "返回循环" << std::endl;
            }
        }
        /*-----------求B2R----------------*/
        if (key == 'R' || key == 'r')
        {
            std::cout << "已经进入工具标定程序，任务：求B2R" << std::endl;
            if (lebai_arm_flag && Lebai_flag)
            {
                cv::Mat UserDataMatrix(3, 6, CV_64F); // 每一行存储每一次的xyzrpy
                double b2r_xyz_rpy1[6] = {0};
                double b2r_xyz_rpy2[6] = {0};
                double b2r_xyz_rpy3[6] = {0};
                double b2r_xyz_rpy_temp[6] = {0};

                std::cout << "第一次将机器臂点在标定板的原点,完成请按‘4’" << std::endl;
                std::cin >> step_key;
                if (step_key == 4 || step_key == 4)
                {
                    std::map<std::string, double> b2r_one_pose = robot.get_actual_tcp_pose();
                    num = 0;
                    for (const auto &pair : b2r_one_pose)
                    {
                        b2r_xyz_rpy_temp[num] = pair.second;
                        std::cout << "b2r_xyz_rpy_temp[" << num << "]" << " : " << b2r_xyz_rpy_temp[num] << std::endl;
                        num++;
                    }
                    for (size_t i = 0; i < 3; i++)
                    {
                        b2r_xyz_rpy1[i] = b2r_xyz_rpy_temp[i + 3];
                    }
                    for (size_t i = 3; i < 6; i++)
                    {
                        b2r_xyz_rpy1[i] = b2r_xyz_rpy_temp[i - 3];
                    }
                    // 将单位转从m换成mm
                    for (size_t i = 0; i < 3; i++)
                    {
                        b2r_xyz_rpy1[i] *= 1000;
                    }
                    proj.getF2R_rpy(b2r_xyz_rpy1[0], b2r_xyz_rpy1[1], b2r_xyz_rpy1[2], b2r_xyz_rpy1[3], b2r_xyz_rpy1[4], b2r_xyz_rpy1[5], 11); // F2R_11
                    step_key = ' ';
                }
                std::cout << "第二次将机器臂点在标定板的x轴上,完成请按‘5’" << std::endl;
                std::cin >> step_key;
                if (step_key == 5 || step_key == 5)
                {
                    std::map<std::string, double> b2r_two_pose = robot.get_actual_tcp_pose();
                    num = 0;
                    for (const auto &pair : b2r_two_pose)
                    {
                        b2r_xyz_rpy_temp[num] = pair.second;
                        std::cout << "b2r_xyz_rpy_temp[" << num << "]" << " : " << b2r_xyz_rpy_temp[num] << std::endl;
                        num++;
                    }
                    for (size_t i = 0; i < 3; i++)
                    {
                        b2r_xyz_rpy2[i] = b2r_xyz_rpy_temp[i + 3];
                    }
                    for (size_t i = 3; i < 6; i++)
                    {
                        b2r_xyz_rpy2[i] = b2r_xyz_rpy_temp[i - 3];
                    }
                    // 将单位转从m换成mm
                    for (size_t i = 0; i < 3; i++)
                    {
                        b2r_xyz_rpy2[i] *= 1000;
                    }
                    proj.getF2R_rpy(b2r_xyz_rpy2[0], b2r_xyz_rpy2[1], b2r_xyz_rpy2[2], b2r_xyz_rpy2[3], b2r_xyz_rpy2[4], b2r_xyz_rpy2[5], 22); // F2R_22
                    step_key = ' ';
                }
                std::cout << "第三次将机器臂点在标定板的y轴上,完成请按‘6’" << std::endl;
                std::cin >> step_key;
                if (step_key == 6 || step_key == 6)
                {
                    std::map<std::string, double> b2r_three_pose = robot.get_actual_tcp_pose();
                    num = 0;
                    for (const auto &pair : b2r_three_pose)
                    {
                        b2r_xyz_rpy_temp[num] = pair.second;
                        std::cout << "b2r_xyz_rpy_temp[" << num << "]" << " : " << b2r_xyz_rpy_temp[num] << std::endl;
                        num++;
                    }
                    for (size_t i = 0; i < 3; i++)
                    {
                        b2r_xyz_rpy3[i] = b2r_xyz_rpy_temp[i + 3];
                    }
                    for (size_t i = 3; i < 6; i++)
                    {
                        b2r_xyz_rpy3[i] = b2r_xyz_rpy_temp[i - 3];
                    }
                    // 将单位转从m换成mm
                    for (size_t i = 0; i < 3; i++)
                    {
                        b2r_xyz_rpy3[i] *= 1000;
                    }
                    proj.getF2R_rpy(b2r_xyz_rpy3[0], b2r_xyz_rpy3[1], b2r_xyz_rpy3[2], b2r_xyz_rpy3[3], b2r_xyz_rpy3[4], b2r_xyz_rpy3[5], 33); // F2R_33
                    step_key = ' ';
                }
                // 保存三次空间点的zyxrpy坐标到UserData.txt文件
                for (int k = 0; k < 6; k++)
                {
                    UserDataMatrix.at<double>(0, k) = b2r_xyz_rpy1[k]; // 第一次点的空间点坐标 xyzrpy
                    UserDataMatrix.at<double>(1, k) = b2r_xyz_rpy2[k];
                    UserDataMatrix.at<double>(2, k) = b2r_xyz_rpy3[k];
                }
                std::ofstream save_UserData("../data/UserData.txt");
                for (int i = 0; i < 3; i++)
                {
                    for (int j = 0; j < 6; j++)
                    {
                        save_UserData << UserDataMatrix.at<double>(i, j) << ' ';
                    }
                    save_UserData << std::endl;
                }
                save_UserData.close();

                // 获取B2R
                proj.getB2R();

                std::cout << "B2R已经获取" << std::endl;
                key = ' ';
                std::cout << "返回循环" << std::endl;
            }
        }
        /*----求B2C----*/
        if (key == 'C' || key == 'c')
        {

            std::cout << "已经进入工具标定程序，任务：求B2C" << std::endl;
            if (lebai_arm_flag && Lebai_flag)
            {
                char path_L[100] = "../data/camero_picture/cam0/0.jpg";

                getB2C proj_B2C(path_L); // 修改过

                proj_B2C.getBoardPoints();
                // std::cout << "3" << std::endl;

                int resp = proj_B2C.getImgPoints_plus(); // 获取角点坐标，图像坐标系
                if (!resp)
                {
                    std::cout << "未找到角点，请检查标定板图像" << std::endl;
                }
                // proj_B2C.get_Q();

                // 拿像素坐标
                float pixe_center[2];
                for (size_t i = 0; i < proj_B2C.boardSize.width * proj_B2C.boardSize.height; i++)
                {
                    pixe_center[0] = proj_B2C.imageCorners1[i].x;
                    pixe_center[1] = proj_B2C.imageCorners1[i].y;
                    // std::cout << "Center " << i + 1 << ": (" << pixe_center[0] << ", " << pixe_center[1] << ")" << std::endl;
                    rs2::depth_frame aligned_depth_stream = data.get_depth_frame();
                    rs2::video_stream_profile depth_stream_profile = aligned_depth_stream.get_profile().as<rs2::video_stream_profile>();
                    const auto depth_intrinsics = depth_stream_profile.get_intrinsics(); // 获取对齐后的深度内参
                    rs2::depth_frame depth_frame = data.get_depth_frame();

                    camR.frame = depth_frame_to_meters(depth_frame); // 深度图单位转换为米
                    // 像素坐标系转换到相机坐标系
                    float pixed_center_depth_value = aligned_depth_stream.get_distance(pixe_center[0], pixe_center[1]);
                    // 计算三维坐标
                    float point_in_color_coordinates[3];
                    rs2_deproject_pixel_to_point(point_in_color_coordinates, &depth_intrinsics, pixe_center, pixed_center_depth_value);

                    std::cout << "X坐标:" << point_in_color_coordinates[0] * 1000 << ""
                              << "Y坐标:" << point_in_color_coordinates[1] * 1000 << ""
                              << "Z坐标:" << point_in_color_coordinates[2] * 1000 << std::endl; // 单位米
                    proj_B2C.camPoints.at<double>(0, i) = point_in_color_coordinates[0] * 1000;
                    proj_B2C.camPoints.at<double>(1, i) = point_in_color_coordinates[1] * 1000;
                    proj_B2C.camPoints.at<double>(2, i) = point_in_color_coordinates[2] * 1000;
                    proj_B2C.camPoints.at<double>(3, i) = 1.0;
                }

                // proj_B2C.getCamPoints(); // 得到角点相对于左相机坐标系的位置[X Y Z 1]
                proj_B2C.get_B2C();
                proj_B2C.transformCamPointsToWorld();

                std::cout << "B2C已经获取" << std::endl;
                key = ' ';
                std::cout << "返回循环" << std::endl;
            }
        }
        /*----拍照，获取左右相机一帧图像，用于求B2C-------*/
        if (key == 'P' || key == 'p')
        {
            std::cout << "进入拍照程序，请调整相机位置，拍摄一帧标定板图像，保存" << std::endl;
            std::cout << "拍摄的图像用于求B2C" << std::endl;

            cv::Mat temp_L, temp_R;
            cv::Mat undistorted_L, undistorted_R;
            // std::cout << "444" << std::endl;
            camL.frame.copyTo(temp_L);
            // camR.frame.copyTo(temp_R);
            // std::cout << "444" << std::endl;
            // cv::cvtColor(temp_L, temp_L, cv::COLOR_RGB2GRAY);
            // cv::cvtColor(temp_R, temp_R, cv::COLOR_RGB2GRAY);
            temp_L.copyTo(undistorted_L);
            // temp_R.copyTo(undistorted_R);
            std::cout << "444" << std::endl;

            // 要把获取的图像进行矫正，用映射表；不能是原图，要行对齐
            // cv::remap(temp_L, undistorted_L, camL.map_x, camL.map_y, cv::INTER_LINEAR); // 映射函数作用于image,生成校正图undistorted
            // cv::remap(temp_R, undistorted_R, camR.map_x, camR.map_y, cv::INTER_LINEAR); // 映射函数作用于image,生成校正图undistorted
            std::cout << "555" << std::endl;
            cv::imwrite("../data/camero_picture/cam0/0.jpg", camL.frame);
            // cv::imwrite("../data/camero_picture/cam1/0.jpg", camR.frame);
            // cv::imwrite("/home/zhou/lebai_hand_eye_calibration/data/camero_picture/cam0/0.jpg", temp_L);
            // cv::imwrite("/home/zhou/lebai_hand_eye_calibration/data/camero_picture/cam1/0.jpg", temp_R);
            std::cout << "图像已矫正，已经保存" << std::endl;
            key = ' ';
            std::cout << "返回循环" << std::endl;
        }
        /*-------求Eye in Hand， C2F-----------*/
        if (key == 'f' || key == 'F')
        {
            std::cout << "已经进入工具标定程序，任务：eye-in-hand,求C2F" << std::endl;
            if (lebai_arm_flag && Lebai_flag)
            {
                // 读入B2R B2C

                cv::FileStorage fs("../data/B2R.xml", cv::FileStorage::READ);
                cv::FileStorage fs1("../data/B2C.xml", cv::FileStorage::READ);
                if (!fs.isOpened() || !fs1.isOpened())
                {
                    std::cout << "B2R.xml或B2C.xml文件打开失败，请检查" << std::endl;
                    std::cout << "跳过以下程序，返回循环开始处" << std::endl;
                    continue;
                }
                fs["B2R"] >> proj_C2F.B2R;
                fs1["B2C"] >> proj_C2F.B2C;
                fs.release();
                fs1.release();

                // 求当前的F2R
                double cur_f2r_rpy[6] = {0};
                double cur_f2r_rpy_temp[6] = {0};
                std::cout << "按下7获得F2R" << std::endl;
                std::cin >> step_key;
                if (step_key == 7 || step_key == 7)
                {
                    std::map<std::string, double> cur_pose = robot.get_actual_tcp_pose();
                    num = 0;
                    for (const auto &pair : cur_pose)
                    {
                        cur_f2r_rpy_temp[num] = pair.second;
                        std::cout << "cur_f2r_rpy_temp[" << num << "]" << " : " << cur_f2r_rpy_temp[num] << std::endl;
                        num++;
                    }
                    for (size_t i = 0; i < 3; i++)
                    {
                        cur_f2r_rpy[i] = cur_f2r_rpy_temp[i + 3];
                    }
                    for (size_t i = 3; i < 6; i++)
                    {
                        cur_f2r_rpy[i] = cur_f2r_rpy_temp[i - 3];
                    }
                    // 将单位转从m换成mm
                    for (size_t i = 0; i < 3; i++)
                    {
                        cur_f2r_rpy[i] *= 1000;
                    }
                    proj_C2F.F2R = proj_C2F.getF2R_rpy(cur_f2r_rpy[0], cur_f2r_rpy[1], cur_f2r_rpy[2], cur_f2r_rpy[3], cur_f2r_rpy[4], cur_f2r_rpy[5]);
                    for (size_t i = 0; i < 6; i++)
                    {
                        std::cout << "单位转换后cur_f2r_rpy[" << i << "]" << " : " << cur_f2r_rpy[i] << std::endl;
                    }
                    step_key = ' ';
                }
                std::cout << "按下8计算C2F" << std::endl;
                std::cin >> step_key;
                if (step_key == 8 || step_key == 8)
                {
                    // 计算C2F
                    // proj_C2F.C2F = proj_C2F.F2R * proj_C2F.B2R.inv() * (proj_C2F.B2C);
                    proj_C2F.C2F = (proj_C2F.B2C * proj_C2F.B2R.inv() * proj_C2F.F2R).inv(); // proj_C2F.B2C=上标C下标B的T；  proj_C2F.B2R=上标R下标B的T  proj_C2F.F2R=上标R下标F的T
                    // std::cout << "C2F :" << proj_C2F.C2F << std::endl;
                    std::cout << "proj_C2F.F2R  :" << proj_C2F.F2R << std::endl;
                    std::cout << "proj_C2F.F2R.inv()  :" << proj_C2F.F2R.inv() << std::endl;
                    std::cout << "proj_C2F.B2R.inv()  :" << proj_C2F.B2R.inv() << std::endl;
                    std::cout << "(proj_C2F.B2C).inv()  :" << (proj_C2F.B2C).inv() << std::endl;
                    std::cout << " proj_C2F.C2F.inv()  :" << (proj_C2F.C2F).inv() << std::endl;
                    step_key = ' ';
                }

                // 计算已有手眼关系的平均值
                proj_C2F.num_of_C2F++;
                proj_C2F.sum_C2F += proj_C2F.C2F;

                cv::Mat C2F_average = proj_C2F.sum_C2F / proj_C2F.num_of_C2F;

                // 保存数据
                cv::FileStorage fs2("../data/C2F.xml", cv::FileStorage::WRITE);
                fs2 << "C2F" << proj_C2F.C2F;
                fs2.release();

                cv::FileStorage fs3(std::string("../data/C2F") + std::to_string(proj_C2F.num_of_C2F) + std::string(".xml"), cv::FileStorage::WRITE);
                fs3 << "C2F_average" << C2F_average;
                fs3.release();

                std::ofstream out_c2f("../data/C2F.txt");
                for (int y = 0; y < proj_C2F.C2F.rows; y++)
                {
                    for (int x = 0; x < proj_C2F.C2F.cols; x++)
                    {
                        out_c2f << proj_C2F.C2F.at<double>(y, x) << ' ';
                    }
                    out_c2f << std::endl;
                }
                out_c2f.close();

                std::cout << "C2F已经获取" << std::endl;
                key = ' ';
                std::cout << "返回循环" << std::endl;
            }
        }
        if (key == 'q' || key == 'Q') // 退出
        {
            break;
        }
    }
    cv::destroyAllWindows();
    robot.stop_sys(); // 停止机械臂
    std::cout << "已退出" << std::endl;
    return 0;
}