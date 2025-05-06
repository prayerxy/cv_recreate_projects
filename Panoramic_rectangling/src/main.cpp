#include "seam.h"
#include "common.h"
#include "global.hpp"
#include <chrono>


//输入参数
int main(int argc, char *argv[])
{
    if(argc != 2)
    {
        cout << "Usage: please input the img label!" << endl;
        return -1;
    }
    string path="../../input/"+string(argv[1])+string("_input.jpg");
    cout<<path<<endl;
    cv::Mat image = cv::imread(path);
    cout<<"image size:"<<image.size()<<endl;
    auto start = std::chrono::high_resolution_clock::now();
    seam_carving sc(image);
    sc.local_wrapping();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> time_span = end - start;
    double time1 = time_span.count();
    cout<<"seam_insert time:"<<time1/1000<<"s" <<endl;//时间单位为ms

    vector<vector<Geometry::point>> GridIn;
    Global_warp::PlaceMesh(image,sc.img,sc.U,GridIn);
    
    vector<vector<Geometry::point>> GridOut;
    cv::Size rectSz;
    rectSz.width = image.cols;
    rectSz.height = image.rows;

    start =std::chrono::high_resolution_clock::now();
    Global_warp::globalWarp(image,rectSz,GridIn,GridOut,1);
    //减少拉伸，然后再全局变形
    Global_warp::reduceStretch(GridIn,GridOut,rectSz);
    Global_warp::globalWarp(image,rectSz,GridIn,GridOut,4);
    
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> time_span2 = end - start;
    double time2 = time_span2.count();
    cout<<"global_warp time:"<<time2/1000<<"s"<<endl;//时间单位为ms

    Mat output;
    GL::GlShow(image,GridIn,GridOut,rectSz.width,rectSz.height,output,string(argv[1]));
    return 0;
}