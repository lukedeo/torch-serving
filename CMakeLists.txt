cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(pytorch_serving)

set(JSON_BuildTests OFF CACHE INTERNAL "")

find_package(Torch REQUIRED)

add_subdirectory(extern/spdlog)

add_executable(torch-serving serving.cpp ServableManager.h ModelServer.h)

target_link_libraries(torch-serving PRIVATE "${TORCH_LIBRARIES}" spdlog::spdlog)

set_property(TARGET torch-serving PROPERTY CXX_STANDARD 11)
