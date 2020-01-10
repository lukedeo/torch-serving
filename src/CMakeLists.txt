# Note that headers are optional, and do not affect add_library, but they will not
# show up in IDEs unless they are listed in add_library.

file(GLOB HEADER_LIST CONFIGURE_DEPENDS "../include/torch_serving/*.h")

# Make an automatic library - will be static or dynamic based on user setting
#add_library(torch_serving model_server.cpp servable_manager.cpp tensor_io.cpp ${HEADER_LIST} )
add_library(torch_serving tensor_io.cpp ${HEADER_LIST} )

target_include_directories(torch_serving PUBLIC ../include)
# We need this directory, and users of our library will need it too

# This depends on (header only) boost
target_link_libraries(torch_serving PRIVATE "${TORCH_LIBRARIES}" spdlog::spdlog)

# All users of this library will need at least C++11
target_compile_features(torch_serving PUBLIC cxx_std_14)

# IDEs should put the headers in a nice place
source_group(TREE "${PROJECT_SOURCE_DIR}/include" PREFIX "Header Files" FILES ${HEADER_LIST})
