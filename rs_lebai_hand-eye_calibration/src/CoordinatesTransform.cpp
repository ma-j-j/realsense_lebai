#include "CoordinatesTransform.h"
using namespace cv;
using namespace std;
using namespace Eigen;

CoordinatesTransform::CoordinatesTransform()
{
    G2F = cv::Mat(4, 4, CV_64F);
    G2F_Ideal = cv::Mat(4, 4, CV_64F);
    B2R = cv::Mat(4, 4, CV_64F);
    C2R = cv::Mat(4, 4, CV_64F);

    /*机器人以3个不同姿态带动工具末端到达同一个空间点*/
    F2R_1 = cv::Mat(4, 4, CV_64F);
    F2R_2 = cv::Mat(4, 4, CV_64F);
    F2R_3 = cv::Mat(4, 4, CV_64F);

    /*戳标定板原点、x轴、y轴*/
    F2R_11 = cv::Mat(4, 4, CV_64F);
    F2R_22 = cv::Mat(4, 4, CV_64F);
    F2R_33 = cv::Mat(4, 4, CV_64F);
}

void CoordinatesTransform::init_Matrix()
{
    // 初始化 G2F B2R B2C
    cv::FileStorage fs1("../data/G2F.xml", cv::FileStorage::READ);
    if (!fs1.isOpened())
    {
        std::cout << "G2F参数丢失" << std::endl;
    }
    else
    {
        fs1["G2F"] >> G2F;
        std::cout << "G2F" << G2F << std::endl;
    }
    fs1.release();
    std::cout << "已经初始化G2F" << std::endl;

    cv::FileStorage fs2("../data/B2R.xml", cv::FileStorage::READ);
    if (!fs2.isOpened())
    {
        std::cout << "B2R参数丢失" << std::endl;
    }
    else
    {
        fs2["B2R"] >> B2R;
        std::cout << "B2R" << B2R << std::endl;
    }
    fs2.release();
    std::cout << "已经初始化B2R" << std::endl;

    cv::FileStorage fs3("../data/B2C.xml", cv::FileStorage::READ);
    if (!fs3.isOpened())
    {
        std::cout << "B2C参数丢失" << std::endl;
    }
    else
    {
        fs3["B2C"] >> B2C;
        std::cout << "B2C" << B2C << std::endl;
    }
    fs3.release();
    std::cout << "已经初始化B2C" << std::endl;
}

void CoordinatesTransform::getF2R(double _X, double _Y, double _Z, double _A, double _B, double _C, double _D, int flag)
{

    // 修改，用eigen将输入进来的四元素转换成旋转矩阵
    // 实践证明，eigen计算出来的旋转矩阵是正确的

    double X = _X;
    double Y = _Y;
    double Z = _Z;
    double A = _A;
    double B = _B;
    double C = _C;
    double D = _D;

    cv::Mat T(4, 4, CV_64F);
    cv::Mat T_eigen(4, 4, CV_64F);

    Eigen::Quaterniond q;
    q.x() = B;
    q.y() = C;
    q.z() = D;
    q.w() = A;
    Eigen::Matrix3d R = q.normalized().toRotationMatrix();
    // Eigen::Matrix3d R = q.toRotationMatrix();
    std::cout << "Quaternion2RotationMatrix result is:" << std::endl;
    std::cout << "R = " << std::endl
              << R << std::endl
              << std::endl;
    // std::cout << "R[0,0]=" << std::endl << R.block<2, 2>(0, 0);

    // cv::Mat R_(3, 3, CV_64F);

    T_eigen.at<double>(0, 0) = R(0, 0);
    T_eigen.at<double>(1, 0) = R(1, 0);
    T_eigen.at<double>(2, 0) = R(2, 0);
    T_eigen.at<double>(3, 0) = 0.0;

    T_eigen.at<double>(0, 1) = R(0, 1);
    T_eigen.at<double>(1, 1) = R(1, 1);
    T_eigen.at<double>(2, 1) = R(2, 1);
    T_eigen.at<double>(3, 1) = 0.0;

    T_eigen.at<double>(0, 2) = R(0, 2);
    T_eigen.at<double>(1, 2) = R(1, 2);
    T_eigen.at<double>(2, 2) = R(2, 2);
    T_eigen.at<double>(3, 2) = 0.0;

    T_eigen.at<double>(0, 3) = X;
    T_eigen.at<double>(1, 3) = Y;
    T_eigen.at<double>(2, 3) = Z;
    T_eigen.at<double>(3, 3) = 1;

    /*由公式 计算而来*/
    cv::Mat temp(3, 3, CV_64F);
    cv::Mat temp_(3, 3, CV_64F);
    temp.at<double>(0, 0) = 1 - 2 * C * C - 2 * D * D;
    temp.at<double>(1, 0) = 2 * B * C - 2 * A * D;
    temp.at<double>(2, 0) = 2 * B * D + 2 * A * C;

    temp.at<double>(0, 1) = 2 * B * C + 2 * A * D;
    temp.at<double>(1, 1) = 1 - 2 * B * B - 2 * D * D;
    temp.at<double>(2, 1) = 2 * C * D - 2 * A * B;

    temp.at<double>(0, 2) = 2 * B * D - 2 * A * C;
    temp.at<double>(1, 2) = 2 * C * D + 2 * A * B;
    temp.at<double>(2, 2) = 1 - 2 * B * B - 2 * C * C;

    temp_ = temp.inv();
    // temp_ = temp;
    T.at<double>(0, 0) = temp_.at<double>(0, 0);
    T.at<double>(1, 0) = temp_.at<double>(1, 0);
    T.at<double>(2, 0) = temp_.at<double>(2, 0);
    T.at<double>(3, 0) = 0.0;

    T.at<double>(0, 1) = temp_.at<double>(0, 1);
    T.at<double>(1, 1) = temp_.at<double>(1, 1);
    T.at<double>(2, 1) = temp_.at<double>(2, 1);
    T.at<double>(3, 1) = 0.0;

    T.at<double>(0, 2) = temp_.at<double>(0, 2);
    T.at<double>(1, 2) = temp_.at<double>(1, 2);
    T.at<double>(2, 2) = temp_.at<double>(2, 2);
    T.at<double>(3, 2) = 0.0;

    T.at<double>(0, 3) = X;
    T.at<double>(1, 3) = Y;
    T.at<double>(2, 3) = Z;
    T.at<double>(3, 3) = 1;

    switch (flag) // 根据flag来将T赋值
    {
    case 1:
        F2R_1 = T;
    case 2:
        F2R_2 = T;
    case 3:
        F2R_3 = T;
    case 11:
        F2R_11 = T;
    case 22:
        F2R_22 = T;
    case 33:
        F2R_33 = T;
    case 0:
        F2R_TEST_B2R = T;
    default:
        break;
    }
    std::cout << "当前T(公式计算)为：" << std::endl;
    std::cout << T << std::endl;

    std::cout << "当前T_eigen为:" << std::endl;
    std::cout << T_eigen << std::endl;
}

void CoordinatesTransform::getF2R_rpy(double _X, double _Y, double _Z, double _A, double _B, double _C, int flag)
{
    cv::Mat T_(4, 4, CV_64F);
    double X = _X;
    double Y = _Y;
    double Z = _Z;
    double A = _C;
    double B = _B;
    double C = _A;

    T_.at<double>(0, 0) = cos(A) * cos(B);
    T_.at<double>(1, 0) = sin(A) * cos(B);
    T_.at<double>(2, 0) = -1.0 * sin(B);
    T_.at<double>(3, 0) = 0.0;

    T_.at<double>(0, 1) = cos(A) * sin(B) * sin(C) - sin(A) * cos(C);
    T_.at<double>(1, 1) = sin(A) * sin(B) * sin(C) + cos(A) * cos(C);
    T_.at<double>(2, 1) = cos(B) * sin(C);
    T_.at<double>(3, 1) = 0.0;

    T_.at<double>(0, 2) = cos(A) * sin(B) * cos(C) + sin(A) * sin(C);
    T_.at<double>(1, 2) = sin(A) * sin(B) * cos(C) - cos(A) * sin(C);
    T_.at<double>(2, 2) = cos(B) * cos(C);
    T_.at<double>(3, 2) = 0.0;

    T_.at<double>(0, 3) = X;
    T_.at<double>(1, 3) = Y;
    T_.at<double>(2, 3) = Z;
    T_.at<double>(3, 3) = 1;

    switch (flag) // 根据flag来将T赋值
    {
    case 1:
        F2R_1 = T_;
    case 2:
        F2R_2 = T_;
    case 3:
        F2R_3 = T_;
    case 11:
        F2R_11 = T_;
    case 22:
        F2R_22 = T_;
    case 33:
        F2R_33 = T_;
    case 0:
        F2R_TEST_B2R = T_;
    default:
        break;
    }
    std::cout << "当前T_rpy为：" << std::endl;
    std::cout << T_ << std::endl;
}

void CoordinatesTransform::getG2F(double delta_X, double delta_Y, double delta_Z)
{
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // 最小二乘计算G2F的平移向量
    Mat A(6, 3, CV_64F), B(6, 1, CV_64F);
    A.at<double>(0, 0) = F2R_1.at<double>(0, 0) - F2R_2.at<double>(0, 0);
    A.at<double>(0, 1) = F2R_1.at<double>(0, 1) - F2R_2.at<double>(0, 1);
    A.at<double>(0, 2) = F2R_1.at<double>(0, 2) - F2R_2.at<double>(0, 2);
    A.at<double>(1, 0) = F2R_2.at<double>(0, 0) - F2R_3.at<double>(0, 0);
    A.at<double>(1, 1) = F2R_2.at<double>(0, 1) - F2R_3.at<double>(0, 1);
    A.at<double>(1, 2) = F2R_2.at<double>(0, 2) - F2R_3.at<double>(0, 2);

    A.at<double>(2, 0) = F2R_1.at<double>(1, 0) - F2R_2.at<double>(1, 0);
    A.at<double>(2, 1) = F2R_1.at<double>(1, 1) - F2R_2.at<double>(1, 1);
    A.at<double>(2, 2) = F2R_1.at<double>(1, 2) - F2R_2.at<double>(1, 2);
    A.at<double>(3, 0) = F2R_2.at<double>(1, 0) - F2R_3.at<double>(1, 0);
    A.at<double>(3, 1) = F2R_2.at<double>(1, 1) - F2R_3.at<double>(1, 1);
    A.at<double>(3, 2) = F2R_2.at<double>(1, 2) - F2R_3.at<double>(1, 2);

    A.at<double>(4, 0) = F2R_1.at<double>(2, 0) - F2R_2.at<double>(2, 0);
    A.at<double>(4, 1) = F2R_1.at<double>(2, 1) - F2R_2.at<double>(2, 1);
    A.at<double>(4, 2) = F2R_1.at<double>(2, 2) - F2R_2.at<double>(2, 2);
    A.at<double>(5, 0) = F2R_2.at<double>(2, 0) - F2R_3.at<double>(2, 0);
    A.at<double>(5, 1) = F2R_2.at<double>(2, 1) - F2R_3.at<double>(2, 1);
    A.at<double>(5, 2) = F2R_2.at<double>(2, 2) - F2R_3.at<double>(2, 2);

    B.at<double>(0, 0) = F2R_2.at<double>(0, 3) - F2R_1.at<double>(0, 3);
    B.at<double>(1, 0) = F2R_3.at<double>(0, 3) - F2R_2.at<double>(0, 3);
    B.at<double>(2, 0) = F2R_2.at<double>(1, 3) - F2R_1.at<double>(1, 3);
    B.at<double>(3, 0) = F2R_3.at<double>(1, 3) - F2R_2.at<double>(1, 3);
    B.at<double>(4, 0) = F2R_2.at<double>(2, 3) - F2R_1.at<double>(2, 3);
    B.at<double>(5, 0) = F2R_3.at<double>(2, 3) - F2R_2.at<double>(2, 3);

    Mat T(3, 1, CV_64F);
    T = (A.t() * A).inv() * A.t() * B; // T就是要求得G2F的平移部分

    // 整理出整个G2F
    Mat one = Mat::eye(Size(3, 3), CV_64F);
    Mat temp4(3, 4, CV_64F);
    Mat unit = (Mat_<double>(1, 4) << 0, 0, 0, 1);
    hconcat(one, T, temp4);    // 列方向拼接
    vconcat(temp4, unit, G2F); // 行方向拼接

    // 考虑理想作业端点与实际工具端点的偏移
    G2F.copyTo(G2F_Ideal);
    G2F_Ideal.at<double>(0, 3) = G2F_Ideal.at<double>(0, 3) + delta_X;
    G2F_Ideal.at<double>(1, 3) = G2F_Ideal.at<double>(1, 3) + delta_Y;
    G2F_Ideal.at<double>(2, 3) = G2F_Ideal.at<double>(2, 3) + delta_Z; // 末端执行器长度补偿,得到实际夹取点对应的G2F
                                                                       // （这个矩阵在最后计算真实拾取位姿时才会用到，前面的标定计算都使用G2F）
    FileStorage fs("../data/G2F.xml", FileStorage::WRITE);
    FileStorage fs_i("../data/G2F_Ideal.xml", FileStorage::WRITE);
    fs << "G2F" << G2F;
    fs_i << "G2F_Ideal" << G2F_Ideal;
    fs.release();
    fs_i.release(); //*/

    ofstream out_g2f("../data/G2F.txt");
    for (int y = 0; y < G2F.rows; y++)
    {
        for (int x = 0; x < G2F.cols; x++)
        {
            out_g2f << G2F.at<double>(y, x) << ' ';
        }
        out_g2f << std::endl;
    }
    out_g2f.close();
}

void CoordinatesTransform::getB2R()
{
    /*在标定板上点3次，对于每一次来说，工具坐标系原点在工具坐标系上的坐标值都是(0,0,0,1))*/
    double pg0[4] = {0, 0, 0, 1}, pgx[4] = {0, 0, 0, 1}, pgy[4] = {0, 0, 0, 1};
    Mat pg_g0 = Mat(4, 1, CV_64F, pg0).clone(),
        pg_gx = Mat(4, 1, CV_64F, pgx).clone(),
        pg_gy = Mat(4, 1, CV_64F, pgy).clone();

    /*将每一次工具坐标系原点相对于工具坐标系的坐标值(即(0,0,0,1))转化为每一次工具坐标系原点(也即标定板上那三个点)相对于基坐标系的坐标值*/
    Mat Pg0(4, 1, CV_64F), Pgx(4, 1, CV_64F), Pgy(4, 1, CV_64F);
    Pg0 = F2R_11 * G2F * pg_g0;
    Pgx = F2R_22 * G2F * pg_gx;
    Pgy = F2R_33 * G2F * pg_gy;
    std::cout << "Pg0=" << std::endl
              << Pg0 << std::endl
              << std::endl; //*/
    std::cout << "Pgx=" << std::endl
              << Pgx << std::endl
              << std::endl;          //*/
    Mat PG0 = Pg0(Rect(0, 0, 1, 3)); // 取齐次坐标前三行
    Mat PGx = Pgx(Rect(0, 0, 1, 3));
    Mat PGy = Pgy(Rect(0, 0, 1, 3));
    std::cout << "PG0=" << std::endl
              << PG0 << std::endl
              << std::endl; //*/
    std::cout << "PGx=" << std::endl
              << PGx << std::endl
              << std::endl;
    /*根据标定板上的三个点相对于基坐标系的坐标值，获得标定板坐标系与基坐标系的旋转平移关系*/
    Mat vx(3, 1, CV_64F), vy(3, 1, CV_64F), vz(3, 1, CV_64F);
    Mat B2R_R(3, 3, CV_64F), B2R_T(3, 1, CV_64F);
    Mat temp1(3, 2, CV_64F), temp2(3, 4, CV_64F);
    Mat unit = (Mat_<double>(1, 4) << 0, 0, 0, 1);

    vx = PGx - PG0; // 标定板坐标系三个轴在基坐标上的向量
    vy = PGy - PG0;
    vz = vx.cross(vy);
    std::cout << "VX=" << std::endl
              << vx << std::endl
              << std::endl; //*/

    Mat x(3, 1, CV_64F), y(3, 1, CV_64F), z(3, 1, CV_64F);
    x = vx / norm(vx); //	归一化得到旋转矩阵的三个列向量
    y = vy / norm(vy);
    // z = vz;
    z = vz / norm(vz);
    B2R_T = PG0; // 点标定板原点的时候，算出的工具坐标系原点(这时跟标定板坐标系原点重合)相对于基坐标的坐标值，就是所求平移向量

    /*整理得到4*4的变换矩阵*/
    hconcat(x, y, temp1);
    hconcat(temp1, z, B2R_R);
    hconcat(B2R_R, B2R_T, temp2);
    vconcat(temp2, unit, B2R);

    std::cout << "B2R=" << std::endl
              << B2R << std::endl
              << std::endl; //*/

    FileStorage fs("../data/B2R.xml", FileStorage::WRITE);

    fs << "B2R" << B2R;
    fs.release();

    ofstream out_b2r("../data/B2R.txt");
    for (int y = 0; y < B2R.rows; y++)
    {
        for (int x = 0; x < B2R.cols; x++)
        {
            out_b2r << B2R.at<double>(y, x) << ' ';
        }
        out_b2r << std::endl;
    }
    out_b2r.close();
}

void CoordinatesTransform::testB2R()
{
    double pB[4] = {60, 120, 0, 1}, // 用于测试的示教点相对于标定板坐标系的坐标
        pg[4] = {0, 0, 0, 1};
    Mat PB = Mat(4, 1, CV_64F, pB).clone(),
        Pg = Mat(4, 1, CV_64F, pg).clone();

    Mat P_real = B2R * PB;
    Mat P_calculate = F2R_TEST_B2R * G2F * Pg;

    std::cout << "P_real=" << std::endl
              << P_real << std::endl
              << std::endl;
    std::cout << "P_calculate=" << std::endl
              << P_calculate << std::endl
              << std::endl;

    std::ofstream out("../data/test_B2R.txt");
    out << "P_real:" << std::endl;
    for (int i = 0; i < 3; i++)
    {
        out << " " << P_real.at<double>(i, 0);
    }
    out << std::endl;
    out << "P_calculate:" << std::endl;
    for (int i = 0; i < 3; i++)
    {
        out << " " << P_calculate.at<double>(i, 0);
    }
    out.close();

    cv::Mat G2B(4, 4, CV_64F);
    G2B = (G2F.inv() * F2R_TEST_B2R.inv() * B2R).inv();
    std::cout << "G2B" << G2B << std::endl;

    ofstream out_g2b("../data/test_G2B.txt");
    for (int y = 0; y < G2B.rows; y++)
    {
        for (int x = 0; x < G2B.cols; x++)
        {
            out_g2b << G2B.at<double>(y, x) << ' ';
        }
        out_g2b << std::endl;
    }
    out_g2b.close();
}

void CoordinatesTransform::loadB2C(Mat &the_B2C)
{
    B2C = the_B2C;
    cout << "B2C=" << endl
         << B2C << endl
         << endl;

    FileStorage fs("../data/B2C.xml", FileStorage::READ);

    fs["B2C"] >> B2C;
    fs.release();
}

void CoordinatesTransform::getC2R()
{
    C2R = B2R * B2C.inv();
    cout << "C2R=" << endl
         << C2R << endl
         << endl;

    /*将C2R保存成txt*/
    ofstream outfile_C2R("../data/C2R.txt");
    for (int y = 0; y < C2R.rows; y++)
    {
        for (int x = 0; x < C2R.cols; x++)
        {
            outfile_C2R << C2R.at<double>(y, x) << "\t";
        }
        outfile_C2R << endl;
    }

    FileStorage fs("../data/C2R.xml", FileStorage::WRITE);
    fs << "C2R" << C2R;
    fs.release();
}