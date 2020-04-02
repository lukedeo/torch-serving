add_executable(torch-serving serving.cpp)
target_compile_features(torch-serving PRIVATE cxx_std_14)
target_link_libraries(torch-serving PRIVATE ${PROJECT_NAME} "${TORCH_LIBRARIES}" spdlog::spdlog)
