add_executable(test-torch-serving test_torch_serving.cpp)
target_include_directories(test-torch-serving PUBLIC ./include)
target_compile_features(test-torch-serving PRIVATE cxx_std_14)
target_link_libraries(test-torch-serving PRIVATE ${PROJECT_NAME} "${TORCH_LIBRARIES}" spdlog::spdlog)
