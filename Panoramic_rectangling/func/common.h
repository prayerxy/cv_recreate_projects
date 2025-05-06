#ifndef common_h
#define common_h
#include <iostream>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <Eigen/Eigen>
#include <iomanip>
using namespace std;


namespace Geometry{
typedef double db;
struct point{
    db x,y;
    point() : x(0), y(0) {} // 默认构造函数
    point(double x_, double y_):x(x_),y(y_){}
    
};
typedef point Vec;//向量
struct Line{
    point p; Vec v;
    Line(point p_, Vec v_):p(p_.x, p_.y),v(v_.x, v_.y){}
};//点向式直线，方向向量不一定单位向量
struct Seg{
    point a,b;
    Seg(point a_, point b_):a(a_.x, a_.y),b(b_.x, b_.y){}
};//线段

db const EPS = 1e-9;
db const PI = acos(-1);

bool eq(db a, db b)  {return std::abs(a - b)< EPS;}//等于
bool ge(db a, db b)  {return a - b          > EPS;}//大于
bool le(db a, db b)  {return a - b          < -EPS;}//小于
bool geq(db a, db b) {return a - b          > -EPS;}//大于等于
bool leq(db a, db b) {return a - b          < EPS;}//小于等于
int sgn(db x) {
    if (std::abs(x) < EPS) return 0;
    if (x < 0) return -1;
    return 1;
} // 符号，等于零返回0，大于零返回1，小于零返回-1

Vec operator+(Vec a, Vec b){return {a.x+b.x, a.y+b.y};}
Vec operator-(Vec a, Vec b){return {a.x-b.x, a.y-b.y};}
Vec operator*(db k, Vec v){return {k*v.x, k*v.y};}
Vec operator*(Vec v, db k){return {v.x*k, v.y*k};}
db operator*(Vec a, Vec b){return a.x*b.x+a.y*b.y;}
db operator^(Vec a, Vec b){return a.x*b.y-a.y*b.x;}//叉积
db len2(Vec v){return v.x*v.x+v.y*v.y;}//长度平方
db len(Vec v){return std::sqrt(len2(v));}//向量长度

Line line(point a, point b){return {a,b-a};}//两点式直线
Line line(db k, db b){return {{0,b},{1,k}};}//斜截式直线y=kx+b
Line line(point p, db k){return {p,{1,k}};}//点斜式直线
Line line(Seg l){return {l.a, l.b-l.a};}//线段所在直线

bool on(point p, Line l){return eq((p.x-l.p.x)*l.v.y, (p.y-l.p.y)*l.v.x);}//点是否在直线上
bool on(point p, Seg l){return eq(len(p-l.a)+len(p-l.b),len(l.a-l.b));}//点是否在线段上

bool operator==(point a, point b){return eq(a.x,b.x)&&eq(a.y,b.y);}//点重合
bool operator==(Line a, Line b){return on(a.p,b)&&on(a.p+a.v,b);}//直线重合
bool operator==(Seg a, Seg b){return ((a.a==b.a&&a.b==b.b)||(a.a==b.b&&a.b==b.a));}//线段（完全）重合

point rotate(point p, db rad){return {cos(rad)*p.x-sin(rad)*p.y,sin(rad)*p.x+cos(rad)*p.y};}

std::vector<point> inter(Line a, Line b){
    //两直线的交点，没有交点返回空vector，否则返回一个大小为1的vector
    // 不能重叠
    db c = a.v^b.v;
    std::vector<point> ret;
    if(eq(c,0.0)) return ret;
    Vec v = 1/c*Vec{a.p^(a.p+a.v), b.p^(b.p+b.v)};
    ret.push_back({v*Vec{-b.v.x, a.v.x},v*Vec{-b.v.y, a.v.y}});
    return ret;
}

std::vector<point> inter(Seg s1, Seg s2) {
    // 两线段的交点，没有交点返回空vector，否则返回一个大小为1的vector
    // 这里特别规定，如果两条线段有重叠线段，会返回第一条线段的两个端点
    std::vector<point> ret;
    using std::max;
    using std::min;
    bool check = true;
    //首先检查两条线段的包围盒是否相交
    check = check && geq(max(s1.a.x, s1.b.x), min(s2.a.x, s2.b.x));
    check = check && geq(max(s2.a.x, s2.b.x), min(s1.a.x, s1.b.x));
    check = check && geq(max(s1.a.y, s1.b.y), min(s2.a.y, s2.b.y));
    check = check && geq(max(s2.a.y, s2.b.y), min(s1.a.y, s1.b.y));
    if (!check) return ret;

    db pd1 = (s2.a - s1.a) ^ (s1.b - s1.a);
    db pd2 = (s2.b - s1.a) ^ (s1.b - s1.a);
    if (sgn(pd1 * pd2) == 1) return ret;
    std::swap(s1, s2);  // 双方都要跨立实验
    pd1 = (s2.a - s1.a) ^ (s1.b - s1.a);
    pd2 = (s2.b - s1.a) ^ (s1.b - s1.a);
    if (sgn(pd1 * pd2) == 1) return ret;

    if (sgn(pd1) == 0 && sgn(pd2) == 0) {
        ret.push_back(s2.a);
        ret.push_back(s2.a);
        return ret;
    }
    return inter(line(s2), line(s1));
}
//一共点p是否在一个多边形内部
int inpoly(std::vector<point> const & poly, point p){
    // 一个点是否在多边形内？
    // 0外部，1内部，2边上，3顶点上
    int n=poly.size();
    for(int i=0;i<n;i++){
        if(poly[i]==p) return 3;
    }
    for(int i=0;i<n;i++){
        if(on(p, Seg{poly[(i+1)%n],poly[i]})) return 2;
    }
    int cnt = 0;
    for(int i=0;i<n;i++){
        int j = (i+1)%n;
        int k = sgn((p-poly[j])^(poly[i]-poly[j]));
        int u = sgn(poly[i].y-p.y);
        int v = sgn(poly[j].y-p.y);
        if(k>0 && u<0 && v>=0) cnt++;
        if(k<0 && v<0 && u>=0) cnt--;
    }
    return cnt != 0;
}
};
/*
采取20*20=400的网格
*/

#endif // utils_h