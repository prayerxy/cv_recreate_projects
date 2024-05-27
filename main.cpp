#include<iomanip>
#include <time.h>
#include "include/GrabCutAPI.h"
using namespace std;
using namespace cv;
GrabCutAPI api;
static void onMouse(int event, int x, int y, int flags, void* param) {
    api.OnMouse(event, x, y, flags, param);
}

int main(int argc, char** argv) {
    printf("########GrabCut API########\n");
    printf("Usage: %s <Input image>\n", argv[0]);
    printf("Hot keys: \n"
        "\tESC - quit the program\n"
        "\tr - reset the segmentation\n"
        "\tn - segment the image\n"
        "\tleft mouse button - set rectangle\n"
        "\tctrl+left mouse button - set GC_BGD pixels\n"
        "\tshift+left mouse button - set GC_FGD pixels\n");
    string filename = argc > 2 ? argv[1] : "../../data/sheep.jpg";
    //加载图像 RGB
    Mat src = imread(filename, IMREAD_COLOR);
    if (src.empty()) {
        cout << "Could not open or find the image!\n" << endl;
        cout << "Usage: " << argv[0] << " <Input image>" << endl;
        return -1;
    }
    string winName = filename.substr(filename.find_last_of("/\\") + 1);
    winName = winName.substr(0, winName.find_last_of("."));
    namedWindow(winName, WINDOW_AUTOSIZE);
    setMouseCallback(winName, onMouse, &api);
    //生成grabcut对象
    api.init(src, winName);
    //设置鼠标回调函数
    setMouseCallback(winName, onMouse, &api);

    //显示原图
    imshow(winName, src);
    //开始处理

    while (1) {
        char c = (char)waitKey(0);
        switch (c)
        {
        case 27:
            cout << "Exit" << endl;
            goto exit_main;
        case 'r':
            cout << "Reset" << endl;
            //注意不能销毁窗口再创建，否则会导致鼠标回调函数失效
            api.reset();
            api.showImage();
            break;
        case 'n':
            cout << "Update" << endl;
            //计算处理时间
            clock_t start = clock();
            api.update();
            clock_t end = clock();
            cout << "Time: " << setprecision(3) << (double)(end - start) / CLOCKS_PER_SEC << "s" << endl;
            printf("Iteration count: %d\n", api.getIterCount());
            api.show();
            break;
        }
    }
exit_main:
    destroyAllWindows();
    return 0;
}
