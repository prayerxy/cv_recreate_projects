
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include "GMM/GMM.h"
#include "GrabCut/GrabCut.h"
#include "BorderMatting/BorderMatting.h"
using namespace std;
using namespace cv;


//颜色
//红色  背景刷
const Scalar RED = Scalar(0, 0, 255);
//白色  前景刷
const Scalar WHITE = Scalar(255, 255, 255);
bool hasConverged(const cv::Mat& currentMask, const cv::Mat& previousMask, double threshold) {
    cv::Mat diff;
    cv::absdiff(currentMask, previousMask, diff);

    double totalDiff = cv::sum(diff)[0]; // 计算掩码差异的总和
    double totalPixels = currentMask.rows * currentMask.cols; // 计算像素总数

    double meanDiff = totalDiff / totalPixels; // 计算平均差异
    printf("meanDiff=%f\n", meanDiff);
    return meanDiff < threshold; // 如果平均差异小于阈值，则认为收敛
}

//定义GrabCut类
class GrabCutAPI {
private:
    const double threshold = 0.1;
    const string* winName;
    const Mat* image;
    Mat mask;
    Rect rect;
    Mat bgdModel;
    Mat fgdModel;
    int iter_Count;
    bool isInitialized;
    //处理状态 0:未处理 1:正在处理 2:处理完成 用于OnMouse函数
    uchar rectState, bgdState, fgdState;
    //标记的像素点 每update一次要清空,用于用户编辑
    vector<Point> bgdPxls, fgdPxls;
    GrabCut gc;
    BorderMatting bm;
    Mat alphamask;
public:
    //构造函数
    void init(Mat& image, string& winName) {
        this->image = &image;
        this->winName = &winName;
        this->mask.create(image.size(), CV_8UC1);
        //fgdmodel,bgdmodel会自动初始化在GMM中
        iter_Count = 0;
        isInitialized = false;
        resizeWindow(winName, 600, 600);
    }
    int getIterCount() {
        return iter_Count;
    }

    void setRectInMask() {
        CV_Assert(rectState == 2);
        mask.setTo(GC_BGD);
        rect.x = MAX(0, rect.x);
        rect.y = MAX(0, rect.y);
        rect.width = MIN(rect.width, image->cols - rect.x);
        rect.height = MIN(rect.height, image->rows - rect.y);
        printf("rect.x=%d,rect.y=%d,rect.width=%d,rect.height=%d\n", rect.x, rect.y, rect.width, rect.height);
        //设置Tu区域为GC_PR_FGD
        (mask(rect)).setTo(Scalar(GC_PR_FGD));
    }
    void setLabsInMask(Point p, int flags) {

        if (flags & EVENT_FLAG_CTRLKEY) {
            bgdPxls.push_back(p);
            //标记为背景像素点
            circle(mask, p, 1, GC_BGD, -1);
        }
        else if (flags & EVENT_FLAG_SHIFTKEY) {
            fgdPxls.push_back(p);
            //标记为前景像素点
            circle(mask, p, 1, GC_FGD, -1);
        }
    }
    void reset() {
        bgdPxls.clear();
        fgdPxls.clear();
        rectState = 0;
        bgdState = 0;
        fgdState = 0;
        iter_Count = 0;
        isInitialized = false;
        if (!mask.empty())
            mask.setTo(Scalar::all(GC_BGD));//黑色掩码
        //关闭Result与Mask窗口
        destroyWindow("Result");
        destroyWindow("Mask");
    }
    //更新模型
    void update() {
        if (!isInitialized) {
            //grabcut直至收敛
            Mat previousMask = mask.clone();
            while (1) {
                gc.grabCut(*image, mask, rect, bgdModel, fgdModel, 1, GC_INIT_WITH_RECT);
                iter_Count++;
                printf("itercount:%d\n", iter_Count);
                if (iter_Count > 1 && hasConverged(mask, previousMask, threshold)) {
                    break;
                }
                previousMask = mask.clone();
            }
            isInitialized = true;

        }
        else {
            //用户编辑，只执行一次grabcut
            gc.grabCut(*image, mask, rect, bgdModel, fgdModel, 1, GC_INIT_WITH_MASK);
            iter_Count++;
        }
        //清空之前标记的像素点
        bgdPxls.clear();
        fgdPxls.clear();
    }
    //显示结果
    void show() {
        Mat res;
        Mat binMask;
        binMask = mask & 1;
        image->copyTo(res, binMask);
        imshow("Result", res);
        imshow("Mask", mask * 80);//*80为了显示效果
        showImage();
    }
    void showImage() {
        Mat res;
        image->copyTo(res);
        //画出标记的像素点
        for (int i = 0; i < bgdPxls.size(); i++) {
            circle(res, bgdPxls[i], 1, RED, -1);
        }
        for (int i = 0; i < fgdPxls.size(); i++) {
            circle(res, fgdPxls[i], 1, WHITE, -1);
        }
        //画出矩形框
        if (rectState == 1 || rectState == 2) {
            rectangle(res, Point(rect.x, rect.y), Point(rect.x + rect.width, rect.y + rect.height), RED, 2);
        }
        imshow((*winName), res);
    }
    //鼠标回调函数
    void OnMouse(int event, int x, int y, int flags, void* param) {
        bool isBgd;
        bool isFgd;
        switch (event) {
            //bgdState=1表示正在标记背景像素点 crtl+左键
            //fgdState=1表示正在标记前景像素点 shift+左键
        case EVENT_LBUTTONDOWN:
            isBgd = (flags & EVENT_FLAG_CTRLKEY) != 0;
            isFgd = (flags & EVENT_FLAG_SHIFTKEY) != 0;
            if (rectState == 0 && !isBgd && !isFgd) {
                rectState = 1; rect = Rect(x, y, 1, 1);
            }
            if ((isBgd || isFgd) && rectState == 2) {
                if (isBgd) {
                    bgdState = 1;
                }
                else if (isFgd) {
                    fgdState = 1;
                }
            }
            break;
        case EVENT_LBUTTONUP:
            if (rectState == 1) {
                rect = Rect(Point(rect.x, rect.y), Point(x, y));
                rectState = 2;
                setRectInMask();//设置TU,TB区域
                CV_Assert(bgdPxls.empty() && fgdPxls.empty());
                showImage();
            }
            if (bgdState == 1 || fgdState == 1) {
                setLabsInMask(Point(x, y), flags);
                if (bgdState == 1) {
                    bgdState = 2;
                }
                else if (fgdState == 1) {
                    fgdState = 2;
                }
                showImage();
            }
            break;
        case EVENT_MOUSEMOVE:
            //鼠标左键按下并移动
            if (rectState == 1) {
                rect = Rect(Point(rect.x, rect.y), Point(x, y));
                CV_Assert(bgdPxls.empty() && fgdPxls.empty());
                showImage();
            }
            else if (bgdState == 1) {
                setLabsInMask(Point(x, y), flags);
                showImage();
            }
            else if (fgdState == 1) {
                setLabsInMask(Point(x, y), flags);
                showImage();
            }
            break;
        }
    }
};
