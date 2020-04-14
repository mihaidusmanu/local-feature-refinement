cmake_minimum_required(VERSION 3.6)
project(MultiViewRefinement)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g")

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

find_package(Boost REQUIRED COMPONENTS program_options)
include_directories(${Boost_INCLUDE_DIRS})

find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})

find_package(COLMAP REQUIRED)
include_directories(${COLMAP_INCLUDE_DIRS})
link_directories(${COLMAP_LINK_DIRS})

add_library(Cost cost.cc)
target_link_libraries(Cost ceres)

add_library(Graph graph.cc)
target_link_libraries(Graph ceres)

add_library(Types types.pb.cc)
target_link_libraries(Types ${Protobuf_LIBRARIES})

set(LIBS ${CERES_LIBRARIES} ${Boost_LIBRARIES} ${Protobuf_LIBRARIES} Graph Types Cost ${COLMAP_LIBRARIES})

add_executable(solve solve.cc)
target_link_libraries(solve ${LIBS})