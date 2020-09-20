cmake_minimum_required(VERSION 3.17)
project(KmeansOpenMp)

set(CMAKE_CXX_STANDARD 14)

add_executable(KmeansOpenMp KmeansOpenMP.h DataFrame.h SaveToCSV.h SaveToCSV.cpp DataFrame.cpp KmeansOpenMP.cpp)
