cmake_minimum_required(VERSION 3.11...3.15)

project(
    torch_serving
    VERSION 0.1.0
    DESCRIPTION "An optimized C++ serving layer for Torch"
    LANGUAGES CXX
)

# Only do these if this is the main project, and not if it is included through add_subdirectory
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)

    # Optionally set things like CMAKE_CXX_STANDARD, CMAKE_POSITION_INDEPENDENT_CODE here

    # Let's ensure -std=c++xx instead of -std=g++xx
    set(CMAKE_CXX_EXTENSIONS OFF)

    # Let's nicely support folders in IDE's
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)

    # Set up testing
#    include(CTest)
#
#    add_subdirectory(tests)
endif()

find_package(Torch REQUIRED)

include_directories(${TORCH_INCLUDE_DIRS})

include_directories(".")

# The compiled library code is here
add_subdirectory(src)

# The executable code is here
add_subdirectory(apps)

set(JSON_BuildTests OFF CACHE INTERNAL "")

add_subdirectory(extern/spdlog)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    # Set up testing
    include(CTest)

    add_subdirectory(tests)
    add_test(NAME test_tensor_io COMMAND tests/test-torch-serving)
endif()