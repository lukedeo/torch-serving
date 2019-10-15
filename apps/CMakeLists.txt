add_executable(torch-serving serving.cpp)
target_compile_features(torch-serving PRIVATE cxx_std_11)
target_link_libraries(torch-serving PRIVATE torch_serving "${TORCH_LIBRARIES}" spdlog::spdlog)
