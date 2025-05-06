#include"GrabCut.h"



void GrabCut::grabCut( const Mat& img, Mat& mask, Rect rect, Mat& bgdModel, Mat& fgdModel, int iterCount, int mode )
{
    //step0
    W_SOUCRE.release();
    W_SINK.release();
    GMM bgdGMM(bgdModel), fgdGMM(fgdModel);

    if( mode == GC_INIT_WITH_RECT){
        initMaskWithRect( mask, img.size(), rect );
    }
    initGMMs( img, mask, bgdGMM, fgdGMM );
    if( iterCount <= 0 )
        return;
    const double gamma = 150;
    const double lambda = 9*gamma;
    //计算beta值
    const double beta = calcBeta( img );
    //计算V项smooth值 V不会变化 所以只需要计算一次
    Mat leftW, upleftW, upW, uprightW;
    calcNWeights( img, leftW, upleftW, upW, uprightW, beta, gamma );
    for(int i=0;i<iterCount;i++){
         //step1
        int vtxCount = img.cols*img.rows;
        //边数  左 左上 上 右上   8个邻居由于对称性可以减少到4个
        int edgeCount = 2*(4*vtxCount-3*img.cols-3*img.rows+2);
        //使用maxflow的图模型
        Graph<double,double,double> graph(vtxCount, edgeCount);
        Mat compIdxs( img.size(), CV_32SC1 );//32位有符号整型
        assignGMMs( img, mask, bgdGMM, fgdGMM, compIdxs );
        //step2
        learnGMMs( img, mask, compIdxs, bgdGMM, fgdGMM );
        //step3
        buildGraph( img, mask, bgdGMM, fgdGMM, lambda, leftW, upleftW, upW, uprightW, graph);
        //step4
        updateMask( graph, mask );
        //评估能量值
        getEnergy(graph,mask,leftW,upleftW,upW,uprightW);
    }
   
    
}
/*
* 将所有像素先根据mask的TB,TU分成两类
* 然后将前景与背景的两类中的像素聚类为5类
* 然后再根据聚类结果将像素添加进GMM中
*/
void GrabCut::initGMMs( const Mat& img, const Mat& mask, GMM& bgdGMM, GMM& fgdGMM )
{
    Mat bgdLabels, fgdLabels;
    std::vector<Vec3f> bgdSamples, fgdSamples;

    Point p;
    for(p.y = 0; p.y < img.rows; p.y++){
        for(p.x = 0; p.x < img.cols; p.x++){
            if(mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD)
                bgdSamples.push_back(img.at<Vec3b>(p));
            else
                fgdSamples.push_back(img.at<Vec3b>(p));
        }
    }
    // 构造行数为bgdSamples.size()，列数为3的矩阵，每一行存储一个样本，初始化为bgdSamples[0][0]
    // CV_32FC1: Vec3f = vector<float, 3>  RGB
	Mat _bgdSamples((int)bgdSamples.size(), 3, CV_32FC1, &bgdSamples[0][0]);
	Mat _fgdSamples((int)fgdSamples.size(), 3, CV_32FC1, &fgdSamples[0][0]);

    const int Kmean_itercount=10;
    kmeans(_bgdSamples,GMM::K,bgdLabels, TermCriteria(TermCriteria::MAX_ITER, Kmean_itercount, 0.0), 0, KMEANS_PP_CENTERS);
    kmeans(_fgdSamples,GMM::K,fgdLabels, TermCriteria(TermCriteria::MAX_ITER, Kmean_itercount, 0.0), 0, KMEANS_PP_CENTERS);
    //聚类完的标签在bgdLabels和fgdLabels中

    //往bgdGMM和fgdGMM中添加样本
    bgdGMM.initLearning();
    for(int i=0;i<bgdSamples.size();i++){
        bgdGMM.addSample(bgdLabels.at<int>(i,0), bgdSamples[i]);
    }
    //计算GMM参数
    bgdGMM.endLearning();

    fgdGMM.initLearning();
    for(int i=0;i<fgdSamples.size();i++){
        fgdGMM.addSample(fgdLabels.at<int>(i,0), fgdSamples[i]);
    }
    //计算GMM参数
    fgdGMM.endLearning();

}

//矩形框内的设置为TU，其他设置为TB，TF为空集
void GrabCut::initMaskWithRect( Mat& mask, Size imgSize, Rect rect )
{
    mask.create( imgSize, CV_8UC1 );
    mask.setTo(GC_BGD);
    rect.x = std::max(0, rect.x);
    rect.y = std::max(0, rect.y);
    rect.width = std::min(rect.width, imgSize.width-rect.x);
    rect.height = std::min(rect.height, imgSize.height-rect.y);
    (mask(rect)).setTo(Scalar(GC_PR_FGD));
}

void GrabCut::assignGMMs( const Mat& img, const Mat& mask, const GMM& bgdGMM, const GMM& fgdGMM, Mat& compIdxs )
{
    compIdxs.create( img.size(), CV_32SC1 );
    Point p;
    for(p.y = 0; p.y < img.rows; p.y++){
        for(p.x = 0; p.x < img.cols; p.x++){
            //对每一个像素计算出使D最小的类别K
            Vec3d color = (Vec3d)img.at<Vec3b>(p);
            if(mask.at<uchar>(p)==GC_BGD || mask.at<uchar>(p)==GC_PR_BGD)
                compIdxs.at<int>(p) = bgdGMM.ArgminKn(color);
            else
                compIdxs.at<int>(p) = fgdGMM.ArgminKn(color);
        }
    }
}


void GrabCut::learnGMMs( const Mat& img, const Mat& mask, const Mat& compIdxs, GMM& bgdGMM, GMM& fgdGMM )
{
    //对之前的GMM清空
    bgdGMM.initLearning();
    fgdGMM.initLearning();
    Point p;
    for(p.y = 0; p.y < img.rows; p.y++){
        for(p.x = 0; p.x < img.cols; p.x++){
            Vec3d color = (Vec3d)img.at<Vec3b>(p);
            if(mask.at<uchar>(p)==GC_BGD || mask.at<uchar>(p)==GC_PR_BGD)
                bgdGMM.addSample(compIdxs.at<int>(p), color);
            else
                fgdGMM.addSample(compIdxs.at<int>(p), color);
        }
    }
    //更新GMM的pai,mean,cov参数，之前初始化的参数作废
    bgdGMM.endLearning();
    fgdGMM.endLearning();
}

void GrabCut::buildGraph( const Mat& img, const Mat& mask, const GMM& bgdGMM, const GMM& fgdGMM, double lambda, const Mat& leftW, const Mat& upleftW, const Mat& upW, const Mat& uprightW, Graph<double,double,double>& graph )
{
    //构建s-t图  利用maxflow库
    W_SOUCRE.create(img.size(),CV_64FC1);
    W_SINK.create(img.size(),CV_64FC1);
    Point p;
    for(p.y=0;p.y<img.rows;p.y++){
        for(p.x=0;p.x<img.cols;p.x++){
            //在graph添加点
            Vec3d color = (Vec3d)img.at<Vec3b>(p);
            int v=graph.add_node();
            //判断对应Mask  生成s,t的权重
            double fromsrc, totar;
            if(mask.at<uchar>(p)==GC_PR_BGD || mask.at<uchar>(p)==GC_PR_FGD){
                //TU，添加两条边  log内部对调是因为最小割原理
                fromsrc=-log(bgdGMM.getP(color));
                totar=-log(fgdGMM.getP(color));
                W_SOUCRE.at<double>(p)=totar;
                W_SINK.at<double>(p)=fromsrc;

            }
            else if(mask.at<uchar>(p)==GC_BGD){
                //TB，添加一条边  最小割，所以s->这个点的权重0
                fromsrc=0;
                totar=lambda;

            }
            else{
                //TF，添加一条边
                fromsrc=lambda;
                totar=0;
            }
            //用于在图中的特定节点上添加来自源点和汇点的权重
            graph.add_tweights(v,fromsrc,totar);

            //添加邻居之间的边
            //注意高对比度的邻居之间v会小，根据最小割算法会割除一个区域外流的流量最小
            if(p.x>0){ //左
                Point left(p.x-1,p.y);
                double w = leftW.at<double>(p);
                graph.add_edge(v, v-1, w, w);
            }
            if(p.x>0&&p.y>0){//左上
                Point upleft(p.x-1,p.y-1);
                double w = upleftW.at<double>(p);
                graph.add_edge(v, v-img.cols-1, w, w);
            }
            if(p.y>0){//上
                Point up(p.x,p.y-1);
                double w = upW.at<double>(p);
                graph.add_edge(v, v-img.cols, w, w);
            }
            if(p.x+1<img.cols&&p.y>0){//右上
                Point upright(p.x+1,p.y-1);
                double w = uprightW.at<double>(p);
                graph.add_edge(v, v-img.cols+1, w, w);
            }

        }
    }
}

void GrabCut::updateMask( Graph<double,double,double>& graph, Mat& mask )
{
    //最大流计算
    graph.maxflow();
    //划分S-T  对掩码更新
    Point p;
    //注意迭代方式p.y在外层是行坐标

    for(p.y=0;p.y<mask.rows;p.y++){
        for(p.x=0;p.x<mask.cols;p.x++){
            //对于不确定的前景后景更新
            if(mask.at<uchar>(p)==GC_PR_BGD || mask.at<uchar>(p)==GC_PR_FGD){
                int v = p.y*mask.cols+p.x;
                if(graph.what_segment(v) == Graph<double,double,double>::SOURCE)
                    mask.at<uchar>(p) = GC_PR_FGD;
                else
                    mask.at<uchar>(p) = GC_PR_BGD;
            }
        }
    }
}


void GrabCut::getEnergy(Graph<double,double,double>& graph,Mat&mask,const Mat& leftW, const Mat& upleftW, const Mat& upW, const Mat& uprightW){
    Point p;
    double U;
    double V;
    for(p.y=0;p.y<mask.rows;p.y++){
        for(p.x=0;p.x<mask.cols;p.x++){
            int v = p.y*mask.cols+p.x;
            //注意U只对TU区域计算，TB和TF中的数据拟合误差为0
            if(mask.at<uchar>(p)==GC_PR_BGD || mask.at<uchar>(p)==GC_PR_FGD){
                if(graph.what_segment(v) == Graph<double,double,double>::SOURCE){
                    U+=W_SOUCRE.at<double>(p);
                   //最小割是总和最小，而不是单个最小
                }
                else{
                    U+=W_SINK.at<double>(p);
                }
            }
            //邻居之间
            if(p.x>0){ //左
                Point left(p.x-1,p.y);
                if(graph.what_segment(p.y*mask.cols+p.x) != graph.what_segment(left.y*mask.cols+left.x))
                    V+=leftW.at<double>(p);
            }
            if(p.x>0&&p.y>0){//左上
                Point upleft(p.x-1,p.y-1);
                if(graph.what_segment(p.y*mask.cols+p.x) != graph.what_segment(upleft.y*mask.cols+upleft.x))
                    V+=upleftW.at<double>(p);
            }
            if(p.y>0){//上
                Point up(p.x,p.y-1);
                if(graph.what_segment(p.y*mask.cols+p.x) != graph.what_segment(up.y*mask.cols+up.x))
                    V+=upW.at<double>(p);
            }
            if(p.x+1<mask.cols&&p.y>0){//右上
                Point upright(p.x+1,p.y-1);
                if(graph.what_segment(p.y*mask.cols+p.x) != graph.what_segment(upright.y*mask.cols+upright.x))
                    V+=uprightW.at<double>(p);
            }
        }

    }
    printf("U:%f,V:%f\n",U,V);
    printf("Energy:%f\n",U+V);
}