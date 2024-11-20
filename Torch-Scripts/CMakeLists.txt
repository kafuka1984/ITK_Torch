cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(Torch-Scripts)


# 追加 LibTorch 的路径到 CMAKE_PREFIX_PATH
list(APPEND CMAKE_PREFIX_PATH /database/home/tangchi/Deployments/libtorch)

# 打印 CMAKE_PREFIX_PATH 的值
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")


find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(Torch-Scripts main.cpp)
target_link_libraries(Torch-Scripts "${TORCH_LIBRARIES}")
set_property(TARGET Torch-Scripts PROPERTY CXX_STANDARD 17)

# # The following code block is suggested to be used on Windows.
# # According to https://github.com/pytorch/pytorch/issues/25457,
# # the DLLs need to be copied to avoid memory errors.
# if (MSVC)
#   file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
#   add_custom_command(TARGET Torch-Scripts
#                      POST_BUILD
#                      COMMAND ${CMAKE_COMMAND} -E copy_if_different
#                      ${TORCH_DLLS}
#                      $<TARGET_FILE_DIR:Torch-Scripts>)
# endif (MSVC)