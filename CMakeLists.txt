cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(teat_tracking)
find_package( OpenCV REQUIRED )
add_executable(${PROJECT_NAME} "main.cpp")
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS})
