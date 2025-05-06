#include <iostream>
#include <vector>
#include <GLAD/glad.h>
#include <GLFW/glfw3.h>
#include <Eigen/Dense>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
using namespace cv;
using namespace std;
cv::Mat fillHole(const cv::Mat srcBw)
{
	cv::Size m_Size = srcBw.size();
	cv::Mat Temp = cv::Mat::zeros(m_Size.height + 2, m_Size.width + 2, srcBw.type());//延展图像,宽度加2，高度加2
	cv::Mat roi(Temp, cv::Rect(1, 1, srcBw.cols, srcBw.rows));
    srcBw.copyTo(roi);

	cv::floodFill(Temp, cv::Point(0, 0), cv::Scalar(255)); //使用泛洪填充
	cv::Mat cutImg;//裁剪延展的图像
	cv::Rect roi_rect(1, 1, m_Size.width, m_Size.height);

// Extract the ROI from Temp
cv::Mat roi1(Temp, roi_rect);

// Copy the extracted ROI to cutImg
roi1.copyTo(cutImg);
    cout<<2;
	return srcBw | (~cutImg);
}
int main() {
    // 读取彩色图像
    cv::Mat src = cv::imread("../../data/1_input.jpg");
    
    // 检查图像是否成功加载
    if (src.empty()) {
        std::cerr << "Error loading image" << std::endl;
        return -1;
    }

    // 打印转换前的通道数
    std::cout << "Channels before conversion: " << src.channels() << std::endl;

    cv::Mat bw;
    cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);

	uchar thr = 252;
    int n=src.rows;
    int m=src.cols;
	Mat mask = Mat :: zeros(n,m,CV_8UC1);
    for(int i = 0;i < n;++ i)
        for(int j = 0;j < m;++ j)
            mask.at<uchar>(i,j) = abs(src.at<uchar>(i,j) - 255) < 3 ? 255 : 0;
     int sx[n],sy[m],tx[n] = {},ty[m] = {};
    memset(sx,0x3f,sizeof sx);
    memset(sy,0x3f,sizeof sy);
    for(int i = 0;i < n; ++ i)
        for(int j = 0;j < m;++ j) if(mask.at<uchar>(i,j) == 0)
        {
            sx[i] = min(sx[i],j);
            sy[j] = min(sy[j],i);

            tx[i] = max(tx[i],j);
            ty[j] = max(ty[j],i);
        }
    
    for(int i = 0;i < n; ++ i)
        for(int j = 0;j < m;++ j) if(mask.at<uchar>(i,j) == 255)
            if(j < tx[i] && j > sx[i] && i < ty[j] && i > sy[j])
                mask.at<uchar>(i,j) = 0;
     
    Mat erode_out,dilate_out;//腐蚀,膨胀

    Mat element = getStructuringElement(MORPH_ELLIPSE,Size(5, 5));
    imshow("sk",element);
	cv::dilate(mask, dilate_out, element);
	cv::dilate(dilate_out, dilate_out, element);
	cv::dilate(dilate_out, dilate_out, element);
    erode(dilate_out,erode_out,getStructuringElement(MORPH_ELLIPSE,Size(5, 5)));    
    // namedWindow("erode_out");
    // imshow("erode_out",erode_out);
    
    mask = ~erode_out;
   imshow("mask",mask);
    waitKey(0);
    return 0;
}