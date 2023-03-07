cmake_minimum_required(VERSION 3.0.0)
project(imFlipCPU VERSION 0.1.0)

add_executable(imFlipCPU imflip.c imageStuff.c)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})

