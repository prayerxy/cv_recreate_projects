cmake_minimum_required(VERSION 3.0)
project(MyGrabCut)
set(CMAKE_CXX_STANDARD 11)
#输出 gcc位置
message("CMAKE_C_COMPILER:${CMAKE_C_COMPILER}")

# 设置可执行文件输出路径
SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin/debug)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/bin/release)

# Debug 模式
if(CMAKE_BUILD_TYPE AND (CMAKE_BUILD_TYPE STREQUAL "Debug"))
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wall -O0")
    message("Debug mode:${CMAKE_C_FLAGS_DEBUG}")
# Release 模式
elseif(CMAKE_BUILD_TYPE AND (CMAKE_BUILD_TYPE STREQUAL "Release"))
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -Wall -O3")
    message("Release mode:${CMAKE_C_FLAGS_RELEASE}")
# 默认模式
else()
    message("else:${CMAKE_BUILD_TYPE}")
    message("else:${CMAKE_C_FLAGS_RELEASE}")
endif()

# 设置 OpenCV 路径


# 包含 OpenCV 包
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 包含项目的头文件目录
include_directories(include)

# 添加 graph 库
add_library(graph SHARED include/maxflow/graph.cpp)
# 添加 maxflow 库，并链接到 graph 库
add_library(maxflow SHARED include/maxflow/maxflow.cpp)
target_link_libraries(maxflow graph)

# 添加 GMM 库，并链接到 OpenCV 库
add_library(GMM SHARED include/GMM/GMM.cpp)
target_link_libraries(GMM ${OpenCV_LIBS})

#添加BorderMatting库
add_library(BorderMatting SHARED include/BorderMatting/BorderMatting.cpp)
target_link_libraries(BorderMatting ${OpenCV_LIBS})

# 添加 GrabCut 库，并链接到 GMM、graph、maxflow 库
add_library(GrabCut SHARED include/GrabCut/GrabCut.cpp)
target_link_libraries(GrabCut GMM graph maxflow BorderMatting)

# 添加 main 可执行文件，并链接到 GrabCut、OpenCV 库，其中GrabCut 依赖 GMM、graph、maxflow 库,已经在上面链接
add_executable(main main.cpp)
target_link_libraries(main GrabCut ${OpenCV_LIBS})
