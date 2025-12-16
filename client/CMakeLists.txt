cmake_minimum_required(VERSION 3.16)
project(ocr_client VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Find Qt6
find_package(Qt6 REQUIRED COMPONENTS Widgets)

# Find packages using pkg-config
find_package(PkgConfig REQUIRED)
pkg_check_modules(PROTOBUF REQUIRED protobuf)
pkg_check_modules(GRPC REQUIRED grpc++)

# Find protoc and grpc_cpp_plugin
find_program(PROTOC protoc REQUIRED)
find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin REQUIRED)

# Generate protobuf and gRPC files
set(PROTO_FILES ocr_service.proto)
set(PROTO_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

get_filename_component(PROTO_FILE_ABS ${PROTO_FILES} ABSOLUTE)
get_filename_component(PROTO_FILE_WE ${PROTO_FILES} NAME_WE)

set(PROTO_SRCS "${CMAKE_CURRENT_BINARY_DIR}/${PROTO_FILE_WE}.pb.cc")
set(PROTO_HDRS "${CMAKE_CURRENT_BINARY_DIR}/${PROTO_FILE_WE}.pb.h")
set(GRPC_SRCS "${CMAKE_CURRENT_BINARY_DIR}/${PROTO_FILE_WE}.grpc.pb.cc")
set(GRPC_HDRS "${CMAKE_CURRENT_BINARY_DIR}/${PROTO_FILE_WE}.grpc.pb.h")

add_custom_command(
    OUTPUT ${PROTO_SRCS} ${PROTO_HDRS} ${GRPC_SRCS} ${GRPC_HDRS}
    COMMAND ${PROTOC}
    ARGS --grpc_out="${CMAKE_CURRENT_BINARY_DIR}"
         --cpp_out="${CMAKE_CURRENT_BINARY_DIR}"
         -I"${PROTO_PATH}"
         --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
         "${PROTO_FILE_ABS}"
    DEPENDS ${PROTO_FILES}
)

# Add executable
add_executable(ocr_client 
    client.cpp
    ${PROTO_SRCS}
    ${GRPC_SRCS}
)

# Include directories
target_include_directories(ocr_client PRIVATE 
    ${CMAKE_CURRENT_BINARY_DIR}
    ${PROTOBUF_INCLUDE_DIRS}
    ${GRPC_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(ocr_client PRIVATE 
    Qt6::Widgets
    ${GRPC_LIBRARIES}
    ${PROTOBUF_LIBRARIES}
)

# Threading support
find_package(Threads REQUIRED)
target_link_libraries(ocr_client PRIVATE Threads::Threads)