#ifndef seam_h
#define seam_h

#include <iostream> 
#include <iomanip>
#include <Eigen/Dense>
#include "common.h"
//关于seam_carving的复现
using namespace cv;
using namespace std;
using namespace Eigen;

enum pixel_type
{
    BLACK_PIXEL,//0 代表黑色像素
    SEAM_PIXEL//1 代表seam像素
};

enum direction
{
    UP,
    DOWN,
    LEFT,
    RIGHT
};

enum seam_mode
{
    VERTICAL,
    HORIZONTAL
};

class seam_carving
{
public:
    vector<vector<Geometry::point>> U;;//储存位移场 
    Mat img;//储存图片
    Mat gray;   
    vector<vector<double>> energy;//储存能量图   float类型
    Mat mask;//储存mask图
    seam_carving(Mat img);
    seam_mode mode;//模式
    direction dire;//方向
    int n,m;//原始图像的行列数
    int seg_x,seg_y;//每一次分割的区间
    void get_energy(int rx,int ry,int cx,int cy);
    void local_wrapping();//主函数
    bool find_bdSegment();//找到边界分割
    void dp_energy(Mat& sub_img,Mat& sub_mask);//动态规划计算最少的能量路线
    void get_seam(Mat& sub_imge);//得到seam
    void seam_insert();//插入seam
   

};
#endif // seam_h