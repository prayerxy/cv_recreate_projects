
#ifndef GLOBAL_HPP
#define GLOBAL_HPP
#include "common.h"
#include "lsd.h"
#include <vector>
#include <iostream>
#include <chrono>
#include <cassert>
#include <fstream>
#include <array>
#include <optional>
#include <cmath>

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <Eigen/Eigen>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
using namespace std;
using namespace cv;


namespace Global_warp {

    int const MESH_ROW_CNT=20;
    int const MESH_COL_CNT=20;//网格顶点数
    int const BIN_NUM=50;
    int MIN_DIS=3;//最小线段长度
    //注意四边形数是顶点数-1
class Segment{
    // 存储线段，方便进行保线运算
public:
    cv::Vec2d a, b;//2维db
    double rad;
    int id;
    //构造函数自动计算线段的角度
    Segment(cv::Vec2d const & p1, cv::Vec2d const & p2):a(p1), b(p2){
        if(a[0]>b[0] || (a[0]==b[0]&&a[1]>b[1])) std::swap(a, b);
        cv::Vec2d v = b-a;//方向向量
        rad = v[0]/std::sqrt(v[0]*v[0]+v[1]*v[1]);//求出余弦值
        if(rad>1) rad = 1-1e-5;
        if(rad<-1) rad = -1+1e-5; // 避免NaN
        rad = acos(rad);
        id = -1;
    }
    Segment(double x1, double y1, double x2, double y2):Segment(cv::Vec2d(x1,y1), cv::Vec2d(x2,y2)){}
};
      using LinesInQuadType=std::array<std::array<std::vector<Segment>,MESH_COL_CNT-1>,MESH_ROW_CNT-1>;

void PlaceMesh(cv::Mat &orimg,cv::Mat& img,vector<vector<Geometry::point>>& U,vector<vector<Geometry::point>>&Grid){
    //对局部变形之后的Img进行网格放置
    //U是位移场，Grid是放置的网格
    int n=img.rows;
    int m=img.cols;
    int mesh_x=n/(MESH_ROW_CNT-1);//每个网格的大小
    int mesh_y=m/(MESH_COL_CNT-1);
    //保存所有点的位置
    Grid.resize(MESH_ROW_CNT,vector<Geometry::point>(MESH_COL_CNT));
    for(int i=0;i<MESH_ROW_CNT;i++){
        int x=i*mesh_x;
        if(i==MESH_ROW_CNT-1)x=n-1;
        for(int j=0;j<MESH_COL_CNT;j++){
            int y=j*mesh_y;
            if(j==MESH_COL_CNT-1)y=m-1;
            Grid[i][j]=Geometry::point(x,y);
        }
    }
    // cout<<"PlaceMesh"<<endl;
    //绘制图像+绿色网格
    cv::Mat img1=img.clone();
    for(int i=0;i<MESH_ROW_CNT;i++){
        for(int j=0;j<MESH_COL_CNT;j++){
            int x=Grid[i][j].x;
            int y=Grid[i][j].y;
            //绘制一个网格  注意cv::Point y是行，x是列
            if(i!=0)line(img1,cv::Point(Grid[i][j].y,Grid[i][j].x),cv::Point(Grid[i-1][j].y,Grid[i-1][j].x),cv::Scalar(0,255,0),2);
            if(j!=0)line(img1,cv::Point(Grid[i][j].y,Grid[i][j].x),cv::Point(Grid[i][j-1].y,Grid[i][j-1].x),cv::Scalar(0,255,0),2);


        }
    }
    imshow("Grid",img1);
    //将网格进行位移
    for(int i=0;i<MESH_ROW_CNT;i++){
        for(int j=0;j<MESH_COL_CNT;j++){
            int x=Grid[i][j].x;
            int y=Grid[i][j].y;
            //绘制一个网格  注意point的x是行，y是列
            Grid[i][j].x+=U[x][y].x;
            Grid[i][j].y+=U[x][y].y;
        }
    }
    //重新绘制
    cv::Mat img2=orimg.clone();
    for(int i=0;i<MESH_ROW_CNT;i++){
        for(int j=0;j<MESH_COL_CNT;j++){
            //绘制一个网格  注意point的x是行，y是列
            if(i!=0)line(img2,cv::Point(Grid[i][j].y,Grid[i][j].x),cv::Point(Grid[i-1][j].y,Grid[i-1][j].x),cv::Scalar(0,255,0),1);
            if(j!=0)line(img2,cv::Point(Grid[i][j].y,Grid[i][j].x),cv::Point(Grid[i][j-1].y,Grid[i][j-1].x),cv::Scalar(0,255,0),1);
        }
    }
    imshow("Grid2",img2);
    
}

void drawMesh(cv::Mat img,vector<vector<Geometry::point>> const& grid){
    for(size_t i=0;i<MESH_ROW_CNT;i++){
        for(size_t j=0;j<MESH_COL_CNT;j++){
            if(i!=0)line(img,cv::Point(grid[i][j].y,grid[i][j].x),cv::Point(grid[i-1][j].y,grid[i-1][j].x),cv::Scalar(0,255,0),1);
            if(j!=0)line(img,cv::Point(grid[i][j].y,grid[i][j].x),cv::Point(grid[i][j-1].y,grid[i][j-1].x),cv::Scalar(0,255,0),1);
        }
    }
    imshow("Mesh",img);

   
}



//线段L=HW-1Vq  双线性插值使用四边形的变化求得线段的变化   quad是原图四边形的整数坐标
Eigen::MatrixXd getBil_interpolationMat(std::vector<cv::Vec2i>const&quad,Segment const&line){
    Eigen::MatrixXd H(4,8);
    H<<line.a[0],line.a[1],line.a[0]*line.a[1],1,0,0,0,0,
        0,0,0,0,line.a[0],line.a[1],line.a[0]*line.a[1],1,
        line.b[0],line.b[1],line.b[0]*line.b[1],1,0,0,0,0,
        0,0,0,0,line.b[0],line.b[1],line.b[0]*line.b[1],1;
    Eigen::MatrixXd W=Eigen::MatrixXd::Zero(8,8);
    for(int i=0;i<4;i++){
        W(i*2,0)=quad[i][0];
        W(i*2,1)=quad[i][1];
        W(i*2,2)=quad[i][0]*quad[i][1];
        W(i*2,3)=1;

        W(i*2+1,4)=quad[i][0];
        W(i*2+1,5)=quad[i][1];
        W(i*2+1,6)=quad[i][0]*quad[i][1];
        W(i*2+1,7)=1;
    }
    Eigen::MatrixXd ret=H*(W.inverse());
    return ret;
}

//竖直方向拼接矩阵
Eigen::SparseMatrix<double> vconcat(Eigen::SparseMatrix<double> const &A, Eigen::SparseMatrix<double> const &B){
    Eigen::SparseMatrix<double> ret(A.rows()+B.rows(), A.cols());
    for(size_t k=0; k<A.outerSize(); ++k){
        for(Eigen::SparseMatrix<double>::InnerIterator it(A,k); it; ++it){
            ret.insert(it.row(), it.col()) = it.value();
        }
    }
    for(size_t k=0; k<B.outerSize(); ++k){
        for(Eigen::SparseMatrix<double>::InnerIterator it(B,k); it; ++it){
            ret.insert(it.row()+A.rows(), it.col()) = it.value();
        }
    }
    ret.makeCompressed();
    return ret;

}

//竖直方向拼接向量
Eigen::VectorXd vconcat(Eigen::VectorXd const &A, Eigen::VectorXd const &B){
    Eigen::VectorXd ret(A.rows()+B.rows());
    ret<<A,B;
    return ret;

}

//给定quad,line进行求交点
std::pair<bool,cv::Vec4f> quadCutSeg(std::array<cv::Vec2f,4>const&quad,cv::Vec4f const&seg){

    std::vector<Geometry::point>poly;
    //插入四边形的四个点 逆时针
    poly.emplace_back(quad[0][0], quad[0][1]);
    poly.emplace_back(quad[2][0], quad[2][1]);
    poly.emplace_back(quad[3][0], quad[3][1]);
    poly.emplace_back(quad[1][0], quad[1][1]);//按顶点逆时针给出的多边形

    //线段两个端点
    Geometry::point p1{seg[0],seg[1]};
    Geometry::point p2{seg[2],seg[3]};
    if(Geometry::len(p1-p2)<MIN_DIS) return {false,seg};

    //线段在四边形内部
    if(Geometry::inpoly(poly,p1)&&Geometry::inpoly(poly,p2)) return {true,seg};

    //求交点
    std::vector<Geometry::point>interP;
    std::vector<Geometry::Seg>segs;
    //四边形四条边
    segs.emplace_back(Geometry::point(quad[0][0],quad[0][1]),Geometry::point(quad[1][0],quad[1][1]));
    segs.emplace_back(Geometry::point(quad[1][0],quad[1][1]),Geometry::point(quad[3][0],quad[3][1]));
    segs.emplace_back(Geometry::point(quad[2][0], quad[2][1]), Geometry::point(quad[3][0], quad[3][1]));
    segs.emplace_back(Geometry::point(quad[0][0], quad[0][1]), Geometry::point(quad[2][0], quad[2][1]));
    Geometry::Seg seg2{Geometry::point(seg[0],seg[1]),Geometry::point(seg[2],seg[3])};

    for(auto const&seg1:segs){
        auto tmp=Geometry::inter(seg1,seg2);
        for(auto const&p:tmp){
            interP.push_back(p);
        }
    }
    //没有交点
    if(interP.empty()) return {false,cv::Vec4f()};
    if(interP.size()==1){
        if(Geometry::inpoly(poly,p1))interP.push_back(p1);
        else interP.push_back(p2);
    }
    cv::Vec4f ret;
    if(Geometry::len(interP[0]-interP[1])<MIN_DIS) return {false,seg};
    ret[0]=interP[0].x;
    ret[1]=interP[0].y;
    ret[2]=interP[1].x;
    ret[3]=interP[1].y;
    return {true,ret};

}

void drawLines(cv::Mat& img,vector<vector<Geometry::point>>grid, LinesInQuadType const & lines){
    // 在图像上绘制各个quad里的线段，主要是用于检查实现正确性
    for(size_t i=0;i<MESH_ROW_CNT-1;i++){
        for(size_t j=0;j<MESH_COL_CNT-1;j++){
            for(auto const & l:lines[i][j]){
                cv::Point p1(l.a[1], l.a[0]);
                cv::Point p2(l.b[1], l.b[0]);
                cv::line(img, p1, p2, cv::Scalar(0, 0, 255));
            }
        }
    }
    for(size_t i=0;i<MESH_ROW_CNT;i++){
        for(size_t j=0;j<MESH_COL_CNT;j++){
            if(i!=0)line(img,cv::Point(grid[i][j].y,grid[i][j].x),cv::Point(grid[i-1][j].y,grid[i-1][j].x),cv::Scalar(0,255,0),1);
            if(j!=0)line(img,cv::Point(grid[i][j].y,grid[i][j].x),cv::Point(grid[i][j-1].y,grid[i][j-1].x),cv::Scalar(0,255,0),1);
        }
    }

    imshow("Lines", img);
    waitKey(0);
}

// 将返回的线段数组转换为 std::vector<cv::Vec4f>
std::vector<cv::Vec4f> convertToVec4f(double* lines, int n_out) {
    std::vector<cv::Vec4f> result;
    for (int i = 0; i < n_out; ++i) {
        double y1 = lines[7 * i];
        double x1 = lines[7 * i + 1];
        double y2 = lines[7 * i + 2];
        double x2 = lines[7 * i + 3];
        result.push_back(cv::Vec4f(x1, y1, x2, y2));
    }
    return result;
}
//lsd线段检测，将线段切割后放置各个quad中
void getlines(cv::Mat const&img,vector<vector<Geometry::point>>&gridInit,LinesInQuadType&res){
    cv::Mat img_gray;
    cv::cvtColor(img,img_gray,cv::COLOR_BGR2GRAY);
    // cv::Ptr<cv::LineSegmentDetector> lsd = cv::createLineSegmentDetector(cv::LSD_REFINE_STD);
    // std::vector<cv::Vec4f> lines_std;
    // //opencv自带的lsd检测，得到线段
    // lsd->detect(img_gray, lines_std);
    // 将图像数据转换为双精度浮点数数组
    int X = img_gray.cols;
    int Y = img_gray.rows;
    std::vector<double> img_data(X * Y);
    for (int i = 0; i < Y; ++i) {
        for (int j = 0; j < X; ++j) {
            img_data[i * X + j] = static_cast<double>(img_gray.at<uchar>(i, j));
        }
    }

    // 调用 lsd 函数
    int n_out;
    double* lines = lsd(&n_out, img_data.data(), X, Y);

    // 将返回的线段数组转换为 std::vector<cv::Vec4f>
    std::vector<cv::Vec4f> lines_std = convertToVec4f(lines, n_out);
    // 绘制检测到的线段
    // cv::Mat img_color;
    // cv::cvtColor(img_gray, img_color, cv::COLOR_GRAY2BGR);
    // for (const auto& line : lines_std) {
    //     cv::line(img_color, cv::Point(line[1], line[0]), cv::Point(line[3], line[2]), cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    // }

    // // // 显示结果
    // cv::imshow("Detected Lines", img_color);
    // cv::waitKey(0);

    for(size_t i=0;i<MESH_ROW_CNT-1;i++){
        for(size_t j=0;j<MESH_COL_CNT-1;j++){
            //遍历每一个四边形
            res[i][j].clear();
            std::array<cv::Vec2f,4>quad;
            quad[0]=cv::Vec2f(gridInit[i][j].x,gridInit[i][j].y);
            quad[1]=cv::Vec2f(gridInit[i][j+1].x,gridInit[i][j+1].y);
            quad[2]=cv::Vec2f(gridInit[i+1][j].x,gridInit[i+1][j].y);
            quad[3]=cv::Vec2f(gridInit[i+1][j+1].x,gridInit[i+1][j+1].y);
            for(auto const&line:lines_std){
                auto ans=quadCutSeg(quad,line);
                if(ans.first) res[i][j].emplace_back(ans.second[0], ans.second[1], ans.second[2], ans.second[3]);
            }

        }
    }
}

//得到形状保持的矩阵A    AVq=0
void getEsMat(std::vector<std::vector<Geometry::point>>const&grid,Eigen::SparseMatrix<double>&esM,double lambda=1.0){
    size_t rows=grid.size();
    size_t cols=grid[0].size();
    //四边形个数*8   点数量*2
    Eigen::SparseMatrix<double>ret((rows-1)*(cols-1)*8,rows*cols*2);
    for(size_t i=0;i<rows-1;i++){
        for(size_t j=0;j<cols-1;j++){
            Eigen::MatrixXd Aq(8,4);
            Geometry::point v=grid[i][j];
            Aq(0,0)=v.x;Aq(0,1)=-v.y;Aq(0,2)=1.0;Aq(0,3)=0.0;
            Aq(1,0)=v.y;Aq(1,1)=v.x;Aq(1,2)=0.0;Aq(1,3)=1.0;

            v=grid[i][j+1];
            Aq(2,0)=v.x;Aq(2,1)=-v.y;Aq(2,2)=1.0;Aq(2,3)=0.0;
            Aq(3,0)=v.y;Aq(3,1)=v.x;Aq(3,2)=0.0;Aq(3,3)=1.0;

            v=grid[i+1][j];
            Aq(4,0)=v.x;Aq(4,1)=-v.y;Aq(4,2)=1.0;Aq(4,3)=0.0;
            Aq(5,0)=v.y;Aq(5,1)=v.x;Aq(5,2)=0.0;Aq(5,3)=1.0;

            v=grid[i+1][j+1];
            Aq(6,0)=v.x;Aq(6,1)=-v.y;Aq(6,2)=1.0;Aq(6,3)=0.0;
            Aq(7,0)=v.y;Aq(7,1)=v.x;Aq(7,2)=0.0;Aq(7,3)=1.0;

            //Aq(Aq^T*Aq)^-1*Aq^T-I
            Aq=Aq*(Aq.transpose()*Aq).inverse()*Aq.transpose()-Eigen::MatrixXd::Identity(8,8);

            for(int k=0;k<8;k++){
                int x=(i*(cols-1)+j)*8+k;
                int y=(i*cols+j)*2;

                ret.insert(x,y)=Aq(k,0);
                ret.insert(x,y+1)=Aq(k,1);
                ret.insert(x,y+2)=Aq(k,2);
                ret.insert(x,y+3)=Aq(k,3);

                y=((i+1)*cols+j)*2;
                ret.insert(x,y)=Aq(k,4);
                ret.insert(x,y+1)=Aq(k,5);
                ret.insert(x,y+2)=Aq(k,6);
                ret.insert(x,y+3)=Aq(k,7);
            }
        }
    }
    ret.makeCompressed();
    ret=ret*lambda/((rows-1)*(cols-1));
    esM=std::move(ret);
}

void initBins(Global_warp::LinesInQuadType& lines,std::vector<int>&binsCntOut,std::vector<double>&binsRadOut,int M=BIN_NUM){
    binsCntOut.clear();
    binsRadOut.clear();
    binsCntOut.resize(M);
    binsRadOut.resize(M);
    for(int i=0;i<M;i++){
        binsCntOut[i]=0;
        binsRadOut[i]=0;//旋转角度相对值，故这里初始化为0
    }
    for(size_t i=0;i<MESH_ROW_CNT-1;i++){
        for(size_t j=0;j<MESH_COL_CNT-1;j++){
            for(auto &line:lines[i][j]){
                Geometry::point p1(line.a[0],line.a[1]);
                Geometry::point p2(line.b[0],line.b[1]);
                //保证p2的x大于p1的x  或者两者x相同时，p2的y大于p1的y
                if(p2.x<p1.x||(p2.x==p1.x&&p2.y<p1.y)) std::swap(p1,p2);

                Geometry::Vec v=p2-p1;//方向向量;
                double const pi=Geometry::PI;

                double rad=v.x/std::sqrt(v.x*v.x+v.y*v.y);//求出余弦值
               
                if(rad>1) rad=1-1e-5;
                if(rad<-1) rad=-1+1e-5;
                rad=acos(rad);//弧度值  0-pi
                double tmp=rad/pi*M;
                int id=int(tmp);
                id=std::min(std::max(0,id),M-1);
              
                binsCntOut[id]++;

                line.id=id;//每个线段落到一个bin中
            }
        }
    }
}

void getEbMat(cv::Size const&rectSz,vector<vector<Geometry::point>>&grid,Eigen::SparseMatrix<double>&out_ebM,Eigen::VectorXd&out_ebV,double lambda=1e8){
    size_t rows=grid.size();
    size_t cols=grid[0].size();
    size_t boundVertexCnt=(rows+cols)*2-4;//边界约束的点数
    Eigen::SparseMatrix<double>retM(boundVertexCnt*2,rows*cols*2);
    Eigen::VectorXd retV(boundVertexCnt*2);

    size_t cnt=0;

    //第一行与最后一行  固定
    for(size_t i=1;i<cols-1;i++){
        retM.insert(cnt,i*2)=1;//x是行
        retV(cnt)=0;
        cnt++;
        retM.insert(cnt,i*2+1)=0;
        retV(cnt)=0;
        cnt++;

        retM.insert(cnt,((rows-1)*cols+i)*2)=1;
        retV(cnt)=rectSz.height-1;
        cnt++;
        retM.insert(cnt,((rows-1)*cols+i)*2+1)=0;
        retV(cnt)=0;
        cnt++;
    }

    //第一列与最后一列
    for(size_t j=1;j<rows-1;j++){
        retM.insert(cnt,j*cols*2)=0;
        retV(cnt)=0;
        cnt++;
        retM.insert(cnt,j*cols*2+1)=1;
        retV(cnt)=0;
        cnt++;

        retM.insert(cnt,(j*cols+cols-1)*2)=0;
        retV(cnt)=0;
        cnt++;
        retM.insert(cnt,(j*cols+cols-1)*2+1)=1;
        retV(cnt)=rectSz.width-1;//y是列
        cnt++;

    }
    //四个角
    //左上角
    retM.insert(cnt,0*2)=1;
    retV(cnt)=0;
    cnt++;
    retM.insert(cnt,0*2+1)=1;
    retV(cnt)=0;
    cnt++;

    //右上角
    retM.insert(cnt,(cols-1)*2)=1;
    retV(cnt)=0;
    cnt++;
    retM.insert(cnt,(cols-1)*2+1)=1;
    retV(cnt)=rectSz.width-1;
    cnt++;

    //左下角
    retM.insert(cnt,(rows-1)*cols*2)=1;
    retV(cnt)=rectSz.height-1;
    cnt++;
    retM.insert(cnt,(rows-1)*cols*2+1)=1;
    retV(cnt)=0;
    cnt++;

    //右下角
    retM.insert(cnt,((rows-1)*cols+cols-1)*2)=1;
    retV(cnt)=rectSz.height-1;
    cnt++;
    retM.insert(cnt,((rows-1)*cols+cols-1)*2+1)=1;
    retV(cnt)=rectSz.width-1;
    cnt++;

    retV=retV*lambda;
    retM=retM*lambda;
    out_ebM=std::move(retM);
    out_ebV=std::move(retV);
}

//线条保持
void getElMat(vector<vector<Geometry::point>>&grid,Eigen::SparseMatrix<double>&out_elM,
                LinesInQuadType const&lines,vector<int>&binsCnt,
                vector<double>&binsRad,double lambda=100.0){


    size_t LineCnt=0;
    size_t rows=grid.size();
    size_t cols=grid[0].size();
    for(auto const&bin:binsCnt) LineCnt+=bin;

    Eigen::SparseMatrix<double>retM(LineCnt*2,rows*cols*2);
    std::vector<cv::Vec2i>quad;
    int cnt=0;
    for(size_t i=0;i<rows-1;i++){
        for(size_t j=0;j<cols-1;j++){
            quad.clear();
            quad.push_back(cv::Vec2i(grid[i][j].x,grid[i][j].y));
            quad.push_back(cv::Vec2i(grid[i][j+1].x,grid[i][j+1].y));
            quad.push_back(cv::Vec2i(grid[i+1][j].x,grid[i+1][j].y));
            quad.push_back(cv::Vec2i(grid[i+1][j+1].x,grid[i+1][j+1].y));
            //该四边形内部的所有线段
            for(auto const&l:lines[i][j]){
                //角度R矩阵
                Eigen::MatrixXd R(2,2);
                double theta=binsRad[l.id];
                R<<cos(theta),-sin(theta),
                    sin(theta),cos(theta);
                Eigen::MatrixXd e(2,1);//输入线段
                e<<l.b[0]-l.a[0],
                    l.b[1]-l.a[1];
                //C=Re(e^Te)^-1e^TR^T-I
                Eigen::MatrixXd C=R*e*(e.transpose()*e).inverse()*e.transpose()*R.transpose()-Eigen::MatrixXd::Identity(2,2);
                Eigen::MatrixXd HW_=getBil_interpolationMat(quad,l);
                //计算L的向量e使用两个端点相减
                Eigen::MatrixXd D(2,4);
                D<< -1,0,1,0,
                   0,-1,0,1;
                Eigen::MatrixXd result=C*D*HW_;
                for(int t=0;t<2;t++){
                    int y=(i*cols+j)*2;//列是根据Vq的点的组织形式来的
                    retM.insert(cnt,y)=result(t,0);
                    retM.insert(cnt,y+1)=result(t,1);
                    retM.insert(cnt,y+2)=result(t,2);
                    retM.insert(cnt,y+3)=result(t,3);

                    //四边形下一行两个点  同一行
                    y=((i+1)*cols+j)*2;
                    retM.insert(cnt,y)=result(t,4);
                    retM.insert(cnt,y+1)=result(t,5);
                    retM.insert(cnt,y+2)=result(t,6);
                    retM.insert(cnt,y+3)=result(t,7);

                    cnt++;
                    
                }

            }
        }
    }
    retM.makeCompressed();
    retM=retM*lambda/LineCnt;
    out_elM=std::move(retM);

}

//将线性约束求解器得到的V转化为网格
void V2Grid(cv::Size const&rectSz,vector<vector<Geometry::point>>&grid_in,
        Eigen::VectorXd const&V,vector<vector<Geometry::point>>&grid_out){
    size_t rows=grid_in.size();
    size_t cols=grid_in[0].size();
    grid_out.resize(rows,vector<Geometry::point>(cols));
    for(size_t i=0;i<rows;i++){
        for(size_t j=0;j<cols;j++){
            int x=V(i*cols*2+j*2);//行坐标
            int y=V(i*cols*2+j*2+1);//列坐标
            x=std::min(std::max(0,x),rectSz.height-1);
            y=std::min(std::max(0,y),rectSz.width-1);
            grid_out[i][j]=Geometry::point(x,y);
        }
    }
}

void updateBinsRad(Global_warp::LinesInQuadType const&lines,vector<vector<Geometry::point>>&gridIn,
                    vector<vector<Geometry::point>>&gridOut,vector<int>const&binsCnt,vector<double>&binsRad,int M=BIN_NUM){
    
    //根据第一部分的grid结果更新theta，采用平均值方式

    //先将相对角度赋值0
    for(size_t i=0;i<M;i++){
        binsRad[i]=0.0;
    }
    size_t rows=gridIn.size();
    size_t cols=gridOut[0].size();

    vector<cv::Vec2i>quadIn;
    int cnt=0;
    for(size_t i=0;i<rows-1;i++){
        for(size_t j=0;j<cols-1;j++){
            quadIn.clear();
            quadIn.push_back(cv::Vec2i(gridIn[i][j].x,gridIn[i][j].y));
            quadIn.push_back(cv::Vec2i(gridIn[i][j+1].x,gridIn[i][j+1].y));
            quadIn.push_back(cv::Vec2i(gridIn[i+1][j].x,gridIn[i+1][j].y));
            quadIn.push_back(cv::Vec2i(gridIn[i+1][j+1].x,gridIn[i+1][j+1].y));

            for(auto const&line:lines[i][j]){
                Eigen::MatrixXd HW_=Global_warp::getBil_interpolationMat(quadIn,line);
                Eigen::MatrixXd V(8,1);
                V<<gridOut[i][j].x,gridOut[i][j].y,gridOut[i][j+1].x,gridOut[i][j+1].y,
                    gridOut[i+1][j].x,gridOut[i+1][j].y,gridOut[i+1][j+1].x,gridOut[i+1][j+1].y;
                
                auto e=HW_*V;//(2,1)  quad变化得到变化后的线段
                Segment line2(e(0,0),e(1,0),e(2,0),e(3,0));//计算线段前后的相对旋转角度
                binsRad[line.id]+=line2.rad-line.rad;

            }
        }
    }
    for(size_t i=0;i<M;i++){
        if(binsCnt[i]>0) binsRad[i]/=binsCnt[i];
    }
}

/*
减少图片拉伸
公式：S_x=(xmax1-xmin1)/(xmax0-xmin0)
*/
void reduceStretch(vector<vector<Geometry::point>>const&grid_in,vector<vector<Geometry::point>>const&grid_out,
                    cv::Size&rectSz)
{
    double avgsx=0.0,avgsy=0.0;
    //计算每个四边形的拉伸 然后取平均值
    for(size_t i=0;i<MESH_ROW_CNT-1;i++){
        for(size_t j=0;j<MESH_COL_CNT-1;j++){
            double xmin0=1e9,xmax0=-1e9,ymin0=1e9,ymax0=-1e9;

            //四个顶点
            xmin0=std::min(std::min(grid_in[i][j].x,grid_in[i][j+1].x),std::min(grid_in[i+1][j].x,grid_in[i+1][j+1].x));
            xmax0=std::max(std::max(grid_in[i][j].x,grid_in[i][j+1].x),std::max(grid_in[i+1][j].x,grid_in[i+1][j+1].x));
            ymin0=std::min(std::min(grid_in[i][j].y,grid_in[i][j+1].y),std::min(grid_in[i+1][j].y,grid_in[i+1][j+1].y));
            ymax0=std::max(std::max(grid_in[i][j].y,grid_in[i][j+1].y),std::max(grid_in[i+1][j].y,grid_in[i+1][j+1].y));

            double xmin1=1e9,xmax1=-1e9,ymin1=1e9,ymax1=-1e9;
            xmin1=std::min(std::min(grid_out[i][j].x,grid_out[i][j+1].x),std::min(grid_out[i+1][j].x,grid_out[i+1][j+1].x));
            xmax1=std::max(std::max(grid_out[i][j].x,grid_out[i][j+1].x),std::max(grid_out[i+1][j].x,grid_out[i+1][j+1].x));
            ymin1=std::min(std::min(grid_out[i][j].y,grid_out[i][j+1].y),std::min(grid_out[i+1][j].y,grid_out[i+1][j+1].y));
            ymax1=std::max(std::max(grid_out[i][j].y,grid_out[i][j+1].y),std::max(grid_out[i+1][j].y,grid_out[i+1][j+1].y));
            if(xmax0-xmin0>1) avgsx+=(xmax1-xmin1)/(xmax0-xmin0);
            if(ymax0-ymin0>1) avgsy+=(ymax1-ymin1)/(ymax0-ymin0);
        }
    }
    avgsx/=(MESH_ROW_CNT-1)*(MESH_COL_CNT-1);
    avgsy/=(MESH_ROW_CNT-1)*(MESH_COL_CNT-1);

    rectSz.width=rectSz.width/avgsy;
    rectSz.height=rectSz.height/avgsx;


}

//全局变形
void globalWarp(cv::Mat const&img,cv::Size const&rectSz,vector<vector<Geometry::point>>&grid,vector<vector<Geometry::point>>&grid_out,int iterCnt=10){
    grid_out.resize(MESH_ROW_CNT,vector<Geometry::point>(MESH_COL_CNT));
    
    Eigen::SparseMatrix<double>esM;
    getEsMat(grid,esM,4);
  
    Global_warp::LinesInQuadType lines;
    Global_warp::getlines(img,grid,lines);
    // Mat img1=img.clone();
    // drawLines(img1,grid,lines);
   

    std::vector<int>binsCnt;
    std::vector<double>binsRad;//旋转相对角度
    initBins(lines,binsCnt,binsRad);
   
    Eigen::SparseMatrix<double>ebM;
    Eigen::VectorXd ebV;
    getEbMat(rectSz,grid,ebM,ebV);
   
    for(int T=0;T<iterCnt;T++){
        //step1:固定theta，求解V
        //线条保持
        // cout<<"iter:"<<T<<endl;
        Eigen::SparseMatrix<double>elM;
        getElMat(grid,elM,lines,binsCnt,binsRad,100);

        //AV=B，求解V   B是列向量
        Eigen::VectorXd B1=Eigen::VectorXd::Zero(esM.rows()+elM.rows());
        Eigen::VectorXd B=vconcat(B1,ebV);
        
        Eigen::SparseMatrix<double> A=vconcat(vconcat(esM,elM),ebM);
        //A的行，列
        // cout<<A.rows()<<" "<<A.cols()<<endl;

        Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int> > solver;
        solver.compute(A);

        Eigen::VectorXd V=solver.solve(B);

        //储存V结果至grid_out
        V2Grid(rectSz,grid,V,grid_out);

        //step2:固定V，求解theta
        updateBinsRad(lines,grid,grid_out,binsCnt,binsRad);
    }
    // cout<<"end\n";
}

}//namespace Global_warp


//显示最后变形的图像，用quad的变形插值计算像素的变形
namespace GL{
class Shader{
    // from https://learnopengl.com/code_viewer_gh.php?code=includes/learnopengl/shader_s.h
public:
    unsigned int ID;
    Shader(const char* vertexPath, const char* fragmentPath){
        std::string vertexCode;
        std::string fragmentCode;
        std::ifstream vShaderFile;
        std::ifstream fShaderFile;

        vShaderFile.exceptions (std::ifstream::failbit | std::ifstream::badbit);
        fShaderFile.exceptions (std::ifstream::failbit | std::ifstream::badbit);
        try{
            vShaderFile.open(vertexPath);
            fShaderFile.open(fragmentPath);
            std::stringstream vShaderStream, fShaderStream;

            vShaderStream << vShaderFile.rdbuf();
            fShaderStream << fShaderFile.rdbuf();

            vShaderFile.close();
            fShaderFile.close();

            vertexCode   = vShaderStream.str();
            fragmentCode = fShaderStream.str();
        }
        catch (std::ifstream::failure& e)
        {
            std::cout << "ERROR::SHADER::FILE_NOT_SUCCESSFULLY_READ: " << e.what() << std::endl;
        }
        const char* vShaderCode = vertexCode.c_str();
        const char* fShaderCode = fragmentCode.c_str();

        unsigned int vertex, fragment;

        vertex = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertex, 1, &vShaderCode, NULL);
        glCompileShader(vertex);
        checkCompileErrors(vertex, "VERTEX");

        fragment = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragment, 1, &fShaderCode, NULL);
        glCompileShader(fragment);
        checkCompileErrors(fragment, "FRAGMENT");

        ID = glCreateProgram();
        glAttachShader(ID, vertex);
        glAttachShader(ID, fragment);
        glLinkProgram(ID);
        checkCompileErrors(ID, "PROGRAM");

        glDeleteShader(vertex);
        glDeleteShader(fragment);
    }
    // activate the shader
    // ------------------------------------------------------------------------
    void use()
    {
        glUseProgram(ID);
    }
    // utility uniform functions
    // ------------------------------------------------------------------------
    void setBool(const std::string &name, bool value) const
    {
        glUniform1i(glGetUniformLocation(ID, name.c_str()), (int)value);
    }
    // ------------------------------------------------------------------------
    void setInt(const std::string &name, int value) const
    {
        glUniform1i(glGetUniformLocation(ID, name.c_str()), value);
    }
    // ------------------------------------------------------------------------
    void setFloat(const std::string &name, float value) const
    {
        glUniform1f(glGetUniformLocation(ID, name.c_str()), value);
    }

private:
    // utility function for checking shader compilation/linking errors.
    // ------------------------------------------------------------------------
    void checkCompileErrors(unsigned int shader, std::string type)
    {
        int success;
        char infoLog[1024];
        if (type != "PROGRAM")
        {
            glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
            if (!success)
            {
                glGetShaderInfoLog(shader, 1024, NULL, infoLog);
                std::cout << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
            }
        }
        else
        {
            glGetProgramiv(shader, GL_LINK_STATUS, &success);
            if (!success)
            {
                glGetProgramInfoLog(shader, 1024, NULL, infoLog);
                std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
            }
        }
    }
};

void processInput(GLFWwindow *window)
{
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

void GlShow(cv::Mat const&img,vector<vector<Geometry::point>>&gridInit,vector<vector<Geometry::point>>&gridOut,size_t width,size_t height,cv::Mat&output,string pathimg){
    
    //opengl进行纹理采样，显示图像
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(width, height, "Final_result", NULL, NULL);
    if (window == NULL){
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        std::terminate();
    }

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)){
        std::cerr << "Failed to initialize GLAD" << std::endl;
        std::terminate();
    }

    Shader shader("D:/workin2024/Panoramic_rectangling/func/texture/img.vs", "D:/workin2024/Panoramic_rectangling/func/texture/img.fs");
    size_t rows=gridInit.size();
    size_t cols=gridInit[0].size();
    auto vertices = std::unique_ptr<float[]>(new float[cols * rows * 5]);

    //顶点坐标和纹理坐标  贴图
    for(size_t i=0;i<rows;i++){
        for(size_t j=0;j<cols;j++){
            int x=gridOut[i][j].x;
            int y=gridOut[i][j].y;
            //位置信息计算（屏幕坐标系）
            vertices[(i*cols+j)*5]=std::min(std::max(float(2.0*y/width-1), -1.f), 1.f);
            vertices[(i*cols+j)*5+1]=std::min(std::max(-float(2.0*x/height-1), -1.f), 1.f);
            vertices[(i*cols+j)*5+2]=0.f;
            //纹理坐标计算（图像坐标系）
            x=gridInit[i][j].x;
            y=gridInit[i][j].y;
            vertices[(i*cols+j)*5+3] = std::min(std::max(float(1.0*y/img.cols), 0.f), 1.f);
            vertices[(i*cols+j)*5+4] = std::min(std::max(float(1-1.0*x/img.rows), 0.f), 1.f);



        }
    }
    // auto indices = std::make_unique<unsigned int[]>((cols-1)*(rows-1)*6);
    auto indices = std::unique_ptr<unsigned int[]>(new unsigned int[(cols - 1) * (rows - 1) * 6]);
    
    //用于绘制图元（如三角形或四边形）的顶点索引
    for(size_t i=0;i<rows-1;i++){
        for(size_t j=0;j<cols-1;j++){
            //一个quad有两个三角形
            indices[(i*(cols-1)+j)*6] = i*cols+j;
            indices[(i*(cols-1)+j)*6+1] = i*cols+j+1;
            indices[(i*(cols-1)+j)*6+2] = (i+1)*cols+j;
            indices[(i*(cols-1)+j)*6+3] = i*cols+j+1;
            indices[(i*(cols-1)+j)*6+4] = (i+1)*cols+j+1;
            indices[(i*(cols-1)+j)*6+5] = (i+1)*cols+j;
        }
    }

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    unsigned int VAO, VBO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float)*cols*rows*5, vertices.get(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(float)*(cols-1)*(rows-1)*6, indices.get(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    GLuint texture;
    glGenTextures(1, &texture);

    glBindTexture(GL_TEXTURE_2D, texture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    cv::Mat tempMat = img.clone();
    cv::cvtColor(tempMat, tempMat, cv::COLOR_BGR2RGB);
    cv::flip(tempMat, tempMat, 0);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.cols, img.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, tempMat.data);
    glGenerateMipmap(GL_TEXTURE_2D);

    while(!glfwWindowShouldClose(window))
    {
        processInput(window);

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glBindTexture(GL_TEXTURE_2D, texture);

        shader.use();
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6*(rows-1)*(cols-1) , GL_UNSIGNED_INT, 0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    output = cv::Mat::zeros(height, width, CV_8UC3);
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, output.data);
    cv::cvtColor(output, output, cv::COLOR_RGB2BGR);
    cv::flip(output, output, 0);

    //将此窗口的图片保存
    cv::imwrite("D:/workin2024/Panoramic_rectangling/data/"+pathimg+"_output.jpg", output);

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);

    glfwTerminate();

}
}
#endif // GLOBAL_HPP