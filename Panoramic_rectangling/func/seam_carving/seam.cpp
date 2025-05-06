
#include"seam.h"
#include <algorithm>
#include <omp.h>
#define INF 1e8
/*
初始化  得到Mask gray img
*/

vector<vector<double>>M_energy;//dp时用到
vector<vector<int>>M_prev;//储存前继
vector<int>pos;//存储seam
Mat maskx;//副本  避免每一次选择同样的seam
Mat cimg;//带红线的图像
vector<vector<double>>sub_energy;
seam_carving::seam_carving(Mat src)
{
    this->img=src.clone();
    cimg=src.clone();
    n=src.rows;
    m=src.cols;
    int rows=src.rows;
    int cols=src.cols;
    cv::cvtColor(img,gray, cv::COLOR_BGR2GRAY);
    //初始化位移场
    U.resize(rows,vector<Geometry::point>(cols,Geometry::point(0,0)));
    //初始化mask
    mask=Mat::zeros(rows,cols,CV_8UC1);
    int thrs=252;
    for(int i=0;i<img.rows;i++)
        for(int j=0;j<img.cols;j++){
            if(gray.at<uchar>(i,j)>thrs)
                mask.at<uchar>(i,j)=255;//图像区域为黑色0   边缘部分白色255
    }
    //对Mask进行预处理
    int sx[rows],sy[cols],tx[rows]={},ty[cols]={};
    memset(sx,0x3f,sizeof sx);
    memset(sy,0x3f,sizeof sy);
    for(int i=0;i<rows;i++)
        for(int j=0;j<cols;j++)
            if(mask.at<uchar>(i,j)==0){
                sx[i]=min(sx[i],j);
                sy[j]=min(sy[j],i);
                tx[i]=max(tx[i],j);
                ty[j]=max(ty[j],i);
            }
    //把图像内部区域的白块填补为0
    for(int i=0;i<rows;i++)
        for(int j=0;j<cols;j++)
            if(mask.at<uchar>(i,j)==255)
                if(j<tx[i]&&j>sx[i]&&i<ty[j]&&i>sy[j])
                    mask.at<uchar>(i,j)=0;
    //腐蚀膨胀操作  利用mask防止穿透
    Mat erode_out,dilate_out;
    Mat element=getStructuringElement(MORPH_ELLIPSE,Size(5,5));
    cv::dilate(mask,dilate_out,element);
    cv::dilate(dilate_out,dilate_out,element);
    cv::dilate(dilate_out,dilate_out,element);
    cv::erode(dilate_out,erode_out,getStructuringElement(MORPH_ELLIPSE,Size(5,5)));
    mask=~erode_out;//此时图像区域为白色255   边缘部分黑色0

    //初始化energy
    energy=vector<vector<double>>(rows,vector<double>(cols,0));

    Mat dx,dy;
    cv::Sobel(gray,dx,CV_32FC1,1,0);
    cv::Sobel(gray,dy,CV_32FC1,0,1);
    #pragma omp parallel for
    for(int i=0;i<img.rows;i++)
        for(int j=0;j<img.cols;j++)
        {
            //梯度之和为能量
            energy[i][j]=0.5*abs(dx.at<float>(i,j))+0.5*abs(dy.at<float>(i,j));
            if(mask.at<uchar>(i,j)==0)
                energy[i][j]=INF;
            
        }
    mask.copyTo(maskx);
    // imshow("mask",mask);
    // waitKey(0);
}

/*
找到最长的  设置m,d
*/
bool seam_carving::find_bdSegment(){
    //返回为false，说明已经无missing segment 结束迭代
    //第一行
    int first=0;
    seg_x=0;
    seg_y=0;
    for(int i=0;i<m;i++){
        if(mask.at<uchar>(0,i)==0){
            if(i-first>seg_y-seg_x){
                seg_x=first;
                seg_y=i;
                mode=seam_mode::HORIZONTAL;
                dire=direction::UP;
            }
        }else first=i+1;//记录第一个黑块
    }
    first=0;
    //最后一行
    for(int i=0;i<m;i++){
        if(mask.at<uchar>(n-1,i)==0){
            if(i-first>seg_y-seg_x){
                seg_x=first;
                seg_y=i;
                mode=seam_mode::HORIZONTAL;
                dire=direction::DOWN;
            }
        }else first=i+1;
    }
    first=0;
    //第一列
    for(int i=0;i<n;i++){
        if(mask.at<uchar>(i,0)==0){
            if(i-first>seg_y-seg_x){
                seg_x=first;
                seg_y=i;
                mode=seam_mode::VERTICAL;
                dire=direction::LEFT;
            }
        }else first=i+1;
    }
    first=0;
    //最后一列
    for(int i=0;i<n;i++){
        if(mask.at<uchar>(i,m-1)==0){
            if(i-first>seg_y-seg_x){
                seg_x=first;
                seg_y=i;
                mode=seam_mode::VERTICAL;
                dire=direction::RIGHT;
            }
        }else first=i+1;
    }
    if(seg_y-seg_x==1)
        return false;
    else
        return true;
}

/*
M(i,j)=min(M(i-1,j-1),M(i-1,j),M(i-1,j+1))+e(i,j)
*/
void seam_carving::dp_energy(Mat& sub_img,Mat& sub_mask){
    CV_Assert(sub_img.rows==sub_energy.size()&&sub_img.cols==sub_energy[0].size());
    M_energy.clear();
    M_energy.resize(sub_img.rows,vector<double>(sub_img.cols,0));
    M_prev.clear();
    M_prev.resize(sub_img.rows,vector<int>(sub_img.cols,0));
    int sx=sub_img.rows;
    int sy=sub_img.cols;
    switch (mode)
    {
        case seam_mode::HORIZONTAL:
            //水平方向 行   每一列选择最小的
            for(int i=0;i<sx;i++)
                M_energy[i][0]=sub_energy[i][0];
            for(int i=1;i<sy;i++)
                for(int j=0;j<sx;j++){
                    if(j==0){
                        M_energy[j][i]=min(M_energy[j][i-1],M_energy[j+1][i-1]);
                        //储存前继
                        if(M_energy[j][i]==M_energy[j][i-1])
                            M_prev[j][i]=j;
                        else
                            M_prev[j][i]=j+1;
                    }
                    else if(j==sx-1){
                        M_energy[j][i]=min(M_energy[j-1][i-1],M_energy[j][i-1]);
                        if(M_energy[j][i]==M_energy[j-1][i-1])
                            M_prev[j][i]=j-1;
                        else
                            M_prev[j][i]=j;
                    }
                    else{
                        double C1=M_energy[j-1][i-1]+abs(sub_energy[j-1][i]-sub_energy[j][i-1])+abs(sub_energy[j-1][i]-sub_energy[j+1][i]);
                        double C2=M_energy[j][i-1]+abs(sub_energy[j-1][i]-sub_energy[j+1][i]);
                        double C3=M_energy[j+1][i-1]+abs(sub_energy[j+1][i]-sub_energy[j-1][i])+abs(sub_energy[j][i-1]-sub_energy[j+1][i]);
                        M_energy[j][i]=min(min(C1,C2),C3);
                        if(M_energy[j][i]==C1)
                            M_prev[j][i]=j-1;
                        else if(M_energy[j][i]==C2)
                            M_prev[j][i]=j;
                        else
                            M_prev[j][i]=j+1;
                    }
                    M_energy[j][i]+=sub_energy[j][i];
                }
            break;
        case seam_mode::VERTICAL:
            //垂直方向 列   每一行选择最小的
            for(int i=0;i<sy;i++){
                M_energy[0][i]=sub_energy[0][i];
            }
            for(int i=1;i<sx;i++)
                for(int j=0;j<sy;j++){
                    if(j==0){
                        M_energy[i][j]=min(M_energy[i-1][j],M_energy[i-1][j+1]);
                        //前继
                        if(M_energy[i][j]==M_energy[i-1][j])
                            M_prev[i][j]=j;
                        else
                            M_prev[i][j]=j+1;
                    }
                    else if(j==sy-1){
                        M_energy[i][j]=min(M_energy[i-1][j-1],M_energy[i-1][j]);
                        if(M_energy[i][j]==M_energy[i-1][j-1])
                            M_prev[i][j]=j-1;
                        else
                            M_prev[i][j]=j;
                    }
                    else{
                        double C1=M_energy[i-1][j-1]+abs(sub_energy[i-1][j]-sub_energy[i][j-1])+abs(sub_energy[i][j-1]-sub_energy[i][j+1]);
                        double C2=M_energy[i-1][j]+abs(sub_energy[i][j-1]-sub_energy[i][j+1]);
                        double C3=M_energy[i-1][j+1]+abs(sub_energy[i][j+1]-sub_energy[i][j-1])+abs(sub_energy[i][j+1]-sub_energy[i-1][j]);
                        M_energy[i][j]=min(min(C1,C2),C3);
                        if(M_energy[i][j]==C1)
                            M_prev[i][j]=j-1;
                        else if(M_energy[i][j]==C2)
                            M_prev[i][j]=j;
                        else
                            M_prev[i][j]=j+1;
                    }
                    M_energy[i][j]+=sub_energy[i][j];
                }
            break;
            
    }
}
void seam_carving::get_seam(Mat& sub_imge){
    pos.clear();
    double tmp_energy=0;
    switch(mode){
        case seam_mode::HORIZONTAL:
            //水平方向  从最后一列开始
            for(int i=sub_imge.cols-1;i>=0;i--){
                if(i==sub_imge.cols-1){
                    double minn=1e18;
                    int min_pos=-1;
                    for(int j=0;j<sub_imge.rows;j++){
                        if(M_energy[j][i]<=minn){
                            minn=M_energy[j][i];
                            tmp_energy=minn-sub_energy[j][i];//上一层的能量
                            min_pos=j;
                        }
                    }
                    pos.push_back(min_pos);
                }
                else{
                    int prev=M_prev[pos.back()][i+1];
                    pos.push_back(prev);

                }
            }
            reverse(pos.begin(),pos.end());
            break;
        case seam_mode::VERTICAL:
            //垂直方向  从最后一行开始
            for(int i=sub_imge.rows-1;i>=0;i--){
                if(i==sub_imge.rows-1){
                    double minn=1e18;
                    int min_pos=-1;
                    for(int j=0;j<sub_imge.cols;j++){
                        if(M_energy[i][j]<=minn){
                            minn=M_energy[i][j];
                            tmp_energy=minn-sub_energy[i][j];//上一层的能量
                            min_pos=j;
                        }
                    }
                    pos.push_back(min_pos);
                }
                else{
                    int prev=M_prev[i+1][pos.back()];
                    pos.push_back(prev);

                }
            }
            reverse(pos.begin(),pos.end());
            break;
        
    }
    // CV_Assert(seg_y-seg_x+1==pos.size());
}



void seam_carving::seam_insert(){
    //得到pos

    int len=seg_y-seg_x+1;
    switch(dire){
        case direction::UP:
            //向上移动
            for(int j=0;j<len;j++){
                maskx.at<uchar>(pos[j],j+seg_x)=SEAM_PIXEL;
                for(int i=0;i<pos[j];i++){
                    U[i][j+seg_x]=U[i+1][j+seg_x];//因为往上移动 所以坐标变化  但是像素不变
                    U[i][j+seg_x].x+=1;
                    img.at<Vec3b>(i,j+seg_x)=img.at<Vec3b>(i+1,j+seg_x);
                    mask.at<uchar>(i,j+seg_x)=mask.at<uchar>(i+1,j+seg_x);
                    cimg.at<Vec3b>(i,j+seg_x)=cimg.at<Vec3b>(i+1,j+seg_x);
                    maskx.at<uchar>(i,j+seg_x)=maskx.at<uchar>(i+1,j+seg_x);
                    energy[i][j+seg_x]=energy[i+1][j+seg_x];
                }
                if(pos[j]-1>=0&&pos[j]+1<n)
                    img.at<Vec3b>(pos[j],j+seg_x)=img.at<Vec3b>(pos[j]-1,j+seg_x)*0.5+img.at<Vec3b>(pos[j]+1,j+seg_x)*0.5;
                cimg.at<Vec3b>(pos[j],j+seg_x)=Vec3b(0,0,255);
            }
            break;
        case direction::DOWN:
            //向下移动
            for(int j=0;j<len;j++){
                maskx.at<uchar>(pos[j],j+seg_x)=SEAM_PIXEL;
                for(int i=n-1;i>pos[j];i--){
                    U[i][j+seg_x]=U[i-1][j+seg_x];//因为往下移动 所以坐标变化  但是像素不变
                    U[i][j+seg_x].x-=1;
                    img.at<Vec3b>(i,j+seg_x)=img.at<Vec3b>(i-1,j+seg_x);
                    mask.at<uchar>(i,j+seg_x)=mask.at<uchar>(i-1,j+seg_x);
                    cimg.at<Vec3b>(i,j+seg_x)=cimg.at<Vec3b>(i-1,j+seg_x);
                    maskx.at<uchar>(i,j+seg_x)=maskx.at<uchar>(i-1,j+seg_x);
                    energy[i][j+seg_x]=energy[i-1][j+seg_x];
                }
                if(pos[j]-1>=0&&pos[j]+1<n)
                    img.at<Vec3b>(pos[j],j+seg_x)=img.at<Vec3b>(pos[j]-1,j+seg_x)*0.5+img.at<Vec3b>(pos[j]+1,j+seg_x)*0.5;
                cimg.at<Vec3b>(pos[j],j+seg_x)=Vec3b(0,0,255);
            }
            break;
        case direction::LEFT:
            //向左移动
            for(int i=0;i<len;i++){
                maskx.at<uchar>(i+seg_x,pos[i])=SEAM_PIXEL; 
                for(int j=0;j<pos[i];j++){
                    U[i+seg_x][j]=U[i+seg_x][j+1];//因为往左移动 所以坐标变化  但是像素不变
                    U[i+seg_x][j].y+=1;//列+1
                    img.at<Vec3b>(i+seg_x,j)=img.at<Vec3b>(i+seg_x,j+1);
                    mask.at<uchar>(i+seg_x,j)=mask.at<uchar>(i+seg_x,j+1);
                    cimg.at<Vec3b>(i+seg_x,j)=cimg.at<Vec3b>(i+seg_x,j+1);
                    maskx.at<uchar>(i+seg_x,j)=maskx.at<uchar>(i+seg_x,j+1);
                    energy[i+seg_x][j]=energy[i+seg_x][j+1];
                    
                }
                if(pos[i]-1>=0&&pos[i]+1<m)
                    img.at<Vec3b>(i+seg_x,pos[i])=img.at<Vec3b>(i+seg_x,pos[i]-1)*0.5+img.at<Vec3b>(i+seg_x,pos[i]+1)*0.5;
                cimg.at<Vec3b>(i+seg_x,pos[i])=Vec3b(0,0,255);
            }
            break;
        case direction::RIGHT:
            //向右移动
            for(int i=0;i<len;i++){
                maskx.at<uchar>(i+seg_x,pos[i])=SEAM_PIXEL;
                for(int j=m-1;j>pos[i];j--){
                    U[i+seg_x][j]=U[i+seg_x][j-1];//因为往右移动 所以坐标变化  但是像素不变
                    U[i+seg_x][j].y-=1;//列-1
                    img.at<Vec3b>(i+seg_x,j)=img.at<Vec3b>(i+seg_x,j-1);
                    mask.at<uchar>(i+seg_x,j)=mask.at<uchar>(i+seg_x,j-1);
                    cimg.at<Vec3b>(i+seg_x,j)=cimg.at<Vec3b>(i+seg_x,j-1);
                    maskx.at<uchar>(i+seg_x,j)=maskx.at<uchar>(i+seg_x,j-1);
                    energy[i+seg_x][j]=energy[i+seg_x][j-1];
                }
                if(pos[i]-1>=0&&pos[i]+1<m)
                    img.at<Vec3b>(i+seg_x,pos[i])=img.at<Vec3b>(i+seg_x,pos[i]-1)*0.5+img.at<Vec3b>(i+seg_x,pos[i]+1)*0.5;
                cimg.at<Vec3b>(i+seg_x,pos[i])=Vec3b(0,0,255);
            }
            break;
    }
}
void seam_carving::get_energy(int rx,int ry,int cx,int cy){
    sub_energy.clear();
    //相当于原图上选择n条然后移动，形成矩形子图
    sub_energy.resize(ry-rx+1,vector<double>(cy-cx+1,0));
    #pragma omp parallel for
    for(int i=rx;i<=ry;i++){
        for(int j=cx;j<=cy;j++){
            if(maskx.at<uchar>(i,j)==BLACK_PIXEL)
                sub_energy[i-rx][j-cx]=INF;
            else if(maskx.at<uchar>(i,j)==SEAM_PIXEL)
                sub_energy[i-rx][j-cx]=1e5;
            else
                sub_energy[i-rx][j-cx]=energy[i][j];
        }
    }
}

/*
主函数，执行seam_carving算法
*/
void seam_carving::local_wrapping(){
    bool missing_flag=1;
    while(1){
        //step1: 找到最长的边界分割
        missing_flag=find_bdSegment();
        if(!missing_flag)
            break;
        //step1-2：得到子图像sub_img 注意有一方长度为seg_y-seg_x+1
        Mat sub_img;
        Mat sub_mask;
        if(mode==seam_mode::HORIZONTAL){
            //水平方向 行不变n
            sub_img=img(Range(0,n),Range(seg_x,seg_y+1));
            sub_mask=mask(Range(0,n),Range(seg_x,seg_y+1));
            get_energy(0,n-1,seg_x,seg_y);
            
        }else{
            //垂直方向 子图是行
            sub_img=img(Range(seg_x,seg_y+1),Range(0,m));
            sub_mask=mask(Range(seg_x,seg_y+1),Range(0,m));
            get_energy(seg_x,seg_y,0,m-1);
        }
        //step2-1: 运用动态规划算法计算最佳接缝路径
        dp_energy(sub_img,sub_mask);
        //step2-2: 根据M_energy得到最佳接缝路径 
        get_seam(sub_img);
        //step2-3: 完成接缝插入  更新
        seam_insert();
    }
    //显示结果
    imshow("img",img);
    imshow("cimg",cimg);
    //根据位移场U把图像转回去
   
}

