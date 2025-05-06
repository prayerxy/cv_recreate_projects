
#ifndef GRABCUT_H_
#define GRABCUT_H_
#include"../GMM/GMM.h"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "../maxflow/graph.h"

using namespace cv;
using namespace std;

static double calcBeta(const Mat&image){
    //计算beta值  期望/均值
    //beta = 1/(2*E(sqr(||p1-p2||)))
    double beta = 0;
    Point p;
    for(p.x=0;p.x<image.cols;p.x++){
        for(p.y=0;p.y<image.rows;p.y++){
            Vec3d color = (Vec3d)image.at<Vec3b>(p);
            if(p.x>0){//左
                Point left(p.x-1,p.y);//左边的点
                Vec3d diff = color - (Vec3d)image.at<Vec3b>(left);
                beta += diff.dot(diff);//平方和
            }
            if(p.x>0&&p.y>0){//左上
                Point upleft(p.x-1,p.y-1);
                Vec3d diff = color - (Vec3d)image.at<Vec3b>(upleft);
                beta += diff.dot(diff);
            }
            if(p.y>0){//上
                Point up(p.x,p.y-1);
                Vec3d diff = color - (Vec3d)image.at<Vec3b>(up);
                beta += diff.dot(diff);
            }
            if(p.x<image.cols-1&&p.y>0){//右上
                Point upright(p.x+1,p.y-1);
                Vec3d diff = color - (Vec3d)image.at<Vec3b>(upright);
                beta += diff.dot(diff);
            }
        }
    }
    if( beta <= std::numeric_limits<double>::epsilon() )
        beta = 0;
    else//求期望  总和/个数  边的个数之和
        beta = 1.f / (2 * beta/(4*image.cols*image.rows - 3*image.cols - 3*image.rows + 2) );

    return beta;
}
//计算V项权重
static void calcNWeights( const Mat& img, Mat& leftW, Mat& upleftW, Mat& upW, Mat& uprightW, double beta, double gamma ){
    
    leftW.create( img.size(), CV_64FC1 );//64位浮点数 单通道
    upleftW.create( img.size(), CV_64FC1 );
    upW.create( img.size(), CV_64FC1 );
    uprightW.create( img.size(), CV_64FC1 );
    Point p;
    /*
    * 公式  w = gamma*exp(-beta*||I(p1)-I(p2)||^2)   注意写对角线要除以sqrt(2)
    */
    for(p.x=0;p.x<img.cols;p.x++){
        for(p.y=0;p.y<img.rows;p.y++){
            Vec3d color = (Vec3d)img.at<Vec3b>(p);
            if(p.x>0){ //左
                Point left(p.x-1,p.y);
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(left);
                leftW.at<double>(p) = gamma*exp(-beta*diff.dot(diff));
            }else{
                leftW.at<double>(p) = 0;
            }
            if(p.x>0&&p.y>0){//左上
                Point upleft(p.x-1,p.y-1);
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(upleft);
                //对角线邻居 所以权重为1/sqrt(2)
                upleftW.at<double>(p) = gamma/sqrt(2)*exp(-beta*diff.dot(diff));
            }else{
                upleftW.at<double>(p) = 0;
            }
            if(p.y>0){//上
                Point up(p.x,p.y-1);
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(up);
                upW.at<double>(p) = gamma*exp(-beta*diff.dot(diff));
            }else{
                upW.at<double>(p) = 0;
            }
            if(p.x+1<img.cols&&p.y-1>=0){
                //右上
                Point upright(p.x+1,p.y-1);
                Vec3d diff = color - (Vec3d)img.at<Vec3b>(upright);
                //对角线邻居 所以权重为1/sqrt(2)
                uprightW.at<double>(p) = gamma/sqrt(2)*exp(-beta*diff.dot(diff));
            }else{
                uprightW.at<double>(p) = 0;
            }
        }
    }
}

class GrabCut{
public:
    //主函数
     void grabCut( const Mat& img, Mat& mask, Rect rect, Mat& bgdModel, Mat& fgdModel, int iterCount, int mode );
    //step0 initGMMs
     void initGMMs( const Mat& img, const Mat& mask, GMM& bgdGMM, GMM& fgdGMM );
     void initMaskWithRect( Mat& mask, Size imgSize, Rect rect );
    //step1 assignGMMsComponents
     void assignGMMs( const Mat& img, const Mat& mask, const GMM& bgdGMM, const GMM& fgdGMM, Mat& compIdxs );
    //step2 learnGMMs
     void learnGMMs( const Mat& img, const Mat& mask, const Mat& compIdxs, GMM& bgdGMM, GMM& fgdGMM );
    //step3 buildGraph  最大流使用maxflow库
     void buildGraph( const Mat& img, const Mat& mask, const GMM& bgdGMM, const GMM& fgdGMM, 
        double lambda, const Mat& leftW, const Mat& upleftW, const Mat& upW, const Mat& uprightW, 
        Graph<double,double,double>& graph );
    //step4 estimate
     void updateMask( Graph<double,double,double>& graph, Mat& mask );
    // 评估能量值
     void getEnergy(Graph<double,double,double>& graph,Mat&mask,
            const Mat& leftW, const Mat& upleftW, const Mat& upW, const Mat& uprightW);

private:
//储存值，计算能量 p索引double值
    Mat W_SOUCRE;
    Mat W_SINK;
};
#endif // GRABCUT_H_