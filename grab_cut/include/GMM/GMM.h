/*
*   实现GMM模型   前景与背景分别需要一个GMM模型
*   一个GMM的参数个数为13,1个权重+3个均值+9个协方差  协方差行列式
*   每个GMM模型使用5个高斯分量
*   彩色图片的通道数为3  GMM实际上是3维高斯分布的混合
*   参考OpenCV的GMM实现源码https://github.com/opencv/opencv/blob/master/modules/imgproc/src/grabcut.cpp#L465
*/
#ifndef GMM_H_
#define GMM_H_
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
using namespace cv;
class GMM
{
public:
    static const int K = 10;
    //整个数据集使用GMM混合高斯拟合，但是单个像素只使用一个高斯分布
    GMM( Mat& _model );
    //对相关变量初始化
    void initLearning();
    //增加样本 ci是K个高斯分布的索引 color是像素n
    void addSample( int ci, const Vec3d color );
    //计算该像素使用哪一个K会使得D更小，返回索引
    int ArgminKn( const Vec3d color ) const;
    //计算该像素属于第ci个高斯分布的概率
    double getSingleP( int ci, const Vec3d color ) const;
    double getP( const Vec3d color ) const;
    //更新GMM参数
    void endLearning();

private:
    void calcInverseCovAndDeterm(int ci, double singularFix);
    Mat model;//储存GMM参数
    double* pai;
    double* mean;
    double* cov;
    //协方差的逆
    double inverseCovs[K][3][3];
    //协方差行列式
    double covDeterms[K];

    //用于step2更新参数 mean cov pai
    double sums[K][3];
    //conv(i,j)=E(ij)-E(i)E(j) 故计算乘积
    double prods[K][3][3];
    //用于计算pai
    int sampleCounts[K];
    int totalSampleCount;
};


#endif // GMM_H_