cmake_minimum_required(VERSION 3.10)

# 设置 C++ 标准为 C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 设置项目名称和版本
project(ITK_Segmentation_Tutorial VERSION 1.0)

# 追加 LibTorch 的路径到 CMAKE_PREFIX_PATH
list(APPEND CMAKE_PREFIX_PATH /database/home/tangchi/Deployments/libtorch)
# 添加头文件
include_directories("/usr/include/eigen3")


# 打印 CMAKE_PREFIX_PATH 的值
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# 查找 PyTorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# 查找 ITK
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

# 添加执行文件
add_executable(${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/main_1.cpp)

# 链接 ITK 和 PyTorch 库
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${ITK_LIBRARIES}
    ${TORCH_LIBRARIES}
)

# 设置 PyTorch 头文件路径
include_directories(${TORCH_INCLUDE_DIRS})