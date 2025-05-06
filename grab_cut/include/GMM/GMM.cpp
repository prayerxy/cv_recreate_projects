
#include"GMM.h"
#include<iostream>
using namespace std;


GMM::GMM(Mat& _model) {
    if(_model.empty()){
        _model.create(1, 13*K, CV_64FC1);
        _model.setTo(Scalar(0));
    }
    model = _model;
    pai = model.ptr<double>(0);
    mean = pai+K;
    cov = mean+3*K;//协方差矩阵的起始位置
    for (int ci = 0; ci < K; ci++) {
        //权重不为0，计算逆矩阵和行列式
        if(pai[ci]>0){
            calcInverseCovAndDeterm(ci, 0.0);
        }
    }
}

void GMM::initLearning(){
    for( int ci = 0; ci < K; ci++)
    {
        sums[ci][0] = sums[ci][1] = sums[ci][2] = 0;
        prods[ci][0][0] = prods[ci][0][1] = prods[ci][0][2] = 0;
        prods[ci][1][0] = prods[ci][1][1] = prods[ci][1][2] = 0;
        prods[ci][2][0] = prods[ci][2][1] = prods[ci][2][2] = 0;
        sampleCounts[ci] = 0;
    }
    totalSampleCount = 0;
}

void GMM::addSample(int ci, const Vec3d color) {
    //计算均值
    sums[ci][0] += color[0];
    sums[ci][1] += color[1];
    sums[ci][2] += color[2];
    //为计算E(xy)准备，求xy的和 x,y代表通道  3个维度(3个通道)，像素是样本点，3维向量
    prods[ci][0][0] += color[0] * color[0];
    prods[ci][0][1] += color[0] * color[1];
    prods[ci][0][2] += color[0] * color[2];
    prods[ci][1][0] += color[1] * color[0];
    prods[ci][1][1] += color[1] * color[1];
    prods[ci][1][2] += color[1] * color[2];
    prods[ci][2][0] += color[2] * color[0];
    prods[ci][2][1] += color[2] * color[1];
    prods[ci][2][2] += color[2] * color[2];
    sampleCounts[ci]++;
    totalSampleCount++;
}

int GMM::ArgminKn(const Vec3d color) const{
    double minD = DBL_MAX;
    int result = -1;
    for (int ci = 0; ci < K; ci++) {
        if (pai[ci] > 0) {
            double D = getSingleP(ci, color);
            if (D < minD) {
                minD = D;
                result = ci;
            }
        }
    }
    //返回最小的D对应的K
    return result;
}

double GMM::getSingleP(int ci, const Vec3d color)const  {
    //计算概率
    //计算(x-mu)^T * sigma^(-1) * (x-mu)
    double* m =mean+3*ci;
    double* c =cov+9*ci;
    double res=0;
    if(pai[ci]>0){
        Vec3d diff = color;
        diff[0] -= m[0];
        diff[1] -= m[1];
        diff[2] -= m[2]; //(x-mu)
        /* (X - \mu)^T * cov^{-1} * (X - \mu)
           = [diff{0}, diff{1}, diff{2}] *  [covInv[k][0][0], covInv[k][0][1], covInv[k][0][2]] * [diff{0}]
                                            [covInv[k][1][0], covInv[k][1][1], covInv[k][1][2]]   |diff{1}|
                                            [covInv[k][2][0], covInv[k][2][1], covInv[k][2][2]]   [diff{2}]
        */
        double mult = diff[0]*(diff[0]*inverseCovs[ci][0][0] + diff[1]*inverseCovs[ci][1][0] + diff[2]*inverseCovs[ci][2][0])
                   + diff[1]*(diff[0]*inverseCovs[ci][0][1] + diff[1]*inverseCovs[ci][1][1] + diff[2]*inverseCovs[ci][2][1])
                   + diff[2]*(diff[0]*inverseCovs[ci][0][2] + diff[1]*inverseCovs[ci][1][2] + diff[2]*inverseCovs[ci][2][2]);
        res = 1.0f/sqrt(covDeterms[ci]) * exp(-0.5f*mult);
    }
    return res;

}

double GMM::getP(const Vec3d color) const{
    double res = 0;
    //GMM概率公式
    for (int ci = 0; ci < K; ci++) {
        res += pai[ci] * getSingleP(ci, color);
    }
    return res;
}

void GMM::endLearning(){
    //更新参数
    for(int ci=0;ci<K;ci++){
        //更新mu
        int n=sampleCounts[ci];
        if(n==0)pai[ci]=0;
        else{
            CV_Assert(totalSampleCount > 0);
            pai[ci] = (double)n/totalSampleCount;
            for(int i=0;i<3;i++){
                mean[3*ci+i] = sums[ci][i]/n;//该通道的均值
            }
            //更新协方差
            //conv(i,j)=E(ij)-E(i)E(j) 故计算乘积
            for(int i=0;i<3;i++){
                for(int j=0;j<3;j++){
                    cov[3*3*ci+3*i+j] = prods[ci][i][j]/n - mean[3*ci+i]*mean[3*ci+j];
                }
            }
            //计算逆矩阵和行列式   避免行列式为0
            calcInverseCovAndDeterm(ci, 0.01);
        }
    }
}

//(x-mu)^T * sigma^(-1) * (x-mu)
void GMM::calcInverseCovAndDeterm(int ci, double singularFix) {
    /*
    *  计算协方差的逆、行列式
    *  逆矩阵的计算公式
    * 1.计算伴随矩阵
    * 2.计算行列式
    * 3.计算逆矩阵 A^(-1) = adj(A)/det(A)
    */

    if (pai[ci] > 0) {
        //计算行列式  第ci个分布的协方差行列式
        /*
        * 行列式
        det(C) =| c[0] c[1] c[2] | = c[0]*M_{00} - c[1]*M_{01} + c[2]*M_{02}
                | c[3] c[4] c[5] |
                | c[6] c[7] c[8] |
        M是余子式
        */
        double* c = cov + 9 * ci;
        double dtrm = c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) + c[2] * (c[3] * c[7] - c[4] * c[6]);
        if (dtrm <= 1e-6 && singularFix > 0)
        {
            // Adds the white noise to avoid singular covariance matrix.
            //防止行列式为0
            c[0] += singularFix;
            c[4] += singularFix;
            c[8] += singularFix;
            dtrm = c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) + c[2] * (c[3] * c[7] - c[4] * c[6]);
        }
        covDeterms[ci] = dtrm;
        CV_Assert(dtrm > std::numeric_limits<double>::epsilon());
        //除法转为乘法更快
        double inv_dtrm = 1.0 / dtrm;
        //伴随矩阵的每一项是代数余子式
        /*
        *伴随矩阵: A* =  | A_{00} A_{10} A_{20} | 
                        | A_{01} A_{11} A_{21} |
                        | A_{02} A_{12} A_{22} |
        */

        inverseCovs[ci][0][0] = (c[4] * c[8] - c[5] * c[7]) * inv_dtrm;
        inverseCovs[ci][1][0] = -(c[3] * c[8] - c[5] * c[6]) * inv_dtrm;
        inverseCovs[ci][2][0] = (c[3] * c[7] - c[4] * c[6]) * inv_dtrm;
        inverseCovs[ci][0][1] = -(c[1] * c[8] - c[2] * c[7]) * inv_dtrm;
        inverseCovs[ci][1][1] = (c[0] * c[8] - c[2] * c[6]) * inv_dtrm;
        inverseCovs[ci][2][1] = -(c[0] * c[7] - c[1] * c[6]) * inv_dtrm;
        inverseCovs[ci][0][2] = (c[1] * c[5] - c[2] * c[4]) * inv_dtrm;
        inverseCovs[ci][1][2] = -(c[0] * c[5] - c[2] * c[3]) * inv_dtrm;
        inverseCovs[ci][2][2] = (c[0] * c[4] - c[1] * c[3]) * inv_dtrm;

    }
}
