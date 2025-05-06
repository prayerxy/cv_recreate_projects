cmake_minimum_required(VERSION 3.0)
project(MyWarping)
set(CMAKE_CXX_STANDARD 11)

# 输出 gcc位置
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

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
#输出${OpenCV_LIBS}
message("OpenCV_LIBS:${OpenCV_LIBS}")
find_package(OpenMP REQUIRED)
# 包含项目的头文件目录
include_directories(
    include
    func
    func/seam_carving
    func/lsd
    func/global_warp
)

# 链接库文件目录
link_directories(${CMAKE_SOURCE_DIR}/lib)

add_library(SEAM SHARED func/seam_carving/seam.cpp)
target_link_libraries(SEAM ${OpenCV_LIBS} OpenMP::OpenMP_CXX
            ${CMAKE_SOURCE_DIR}/lib/libglfw3.a
            ${CMAKE_SOURCE_DIR}/lib/libglfw3dll.a
            ${CMAKE_SOURCE_DIR}/lib/libglad.a )


add_library(LSD SHARED func/lsd/lsd.cpp)
target_link_libraries(LSD )

# 添加 main 可执行文件，并链接到 GrabCut、OpenCV 库，其中GrabCut 依赖 GMM、graph、maxflow 库,已经在上面链接
add_executable(main src/main.cpp)
target_link_libraries(main SEAM LSD ${OpenCV_LIBS}
                                    ${CMAKE_SOURCE_DIR}/lib/libglfw3.a
                                    ${CMAKE_SOURCE_DIR}/lib/libglfw3dll.a
                                    ${CMAKE_SOURCE_DIR}/lib/libglad.a)


