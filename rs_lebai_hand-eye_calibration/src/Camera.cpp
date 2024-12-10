#include <Camera.h>

/******************畸变矫正************************/
void Camera_Calibration::calibration(cv::Mat &left, cv::Mat &right)
{
    Q = Mat(4, 4, CV_64FC1);
    // R = Mat(3, 3, CV_64FC1);
    // T = Mat(3, 1, CV_64FC1);
    stereoRectify(K1, D1, K2, D2, left.size(), R, T, R1, R2, P1, P2, Q);
    // 进行矫正畸变与基线对齐
    cv::initUndistortRectifyMap(K1, D1, R1, P1, left.size(), CV_32F, Lmapx, Lmapy);
    cv::initUndistortRectifyMap(K2, D2, R2, P2, right.size(), CV_32F, Lmapx, Rmapy);
    cv::remap(left, left, Lmapx, Lmapy, cv::INTER_LINEAR);
    cv::remap(right, right, Lmapx, Rmapy, cv::INTER_LINEAR);
    cv::Rect roil(40, 10, 1200, 680);
    cv::Rect roir(40, 10, 1200, 680);
    left = left(roil);
    right = right(roir);
}

// header:
//   seq: 80
//   stamp:
//     secs: 1730723533
//     nsecs: 630712032
//   frame_id: "camera_color_optical_frame"
// height: 480
// width: 640
// distortion_model: "plumb_bob"
// D: [0.0, 0.0, 0.0, 0.0, 0.0]
// K: [612.8624877929688, 0.0, 324.2457580566406, 0.0, 613.0888671875, 236.00860595703125, 0.0, 0.0, 1.0]
// R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
// P: [612.8624877929688, 0.0, 324.2457580566406, 0.0, 0.0, 613.0888671875, 236.00860595703125, 0.0, 0.0, 0.0, 1.0, 0.0]
// binning_x: 0
// binning_y: 0
// roi:
//   x_offset: 0
//   y_offset: 0
//   height: 0
//   width: 0
//   do_rectify: False

// D = [0.12403259672525574, -0.23867680956566753, -0.0011018771913829763, 0.0001984289861116209, 0.0]
// K = [607.2768141895752, 0.0, 326.5702494804273, 0.0, 606.6472547099634, 233.10449328671555, 0.0, 0.0, 1.0]
// R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
// P = [617.1150166387998, 0.0, 326.6987900388782, 0.0, 0.0, 616.3680887167518, 232.70175547980116, 0.0, 0.0, 0.0, 1.0, 0.0]
