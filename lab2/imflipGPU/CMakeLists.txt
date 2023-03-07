cmake_minimum_required(VERSION 3.0.0)
project(imgflipGPU LANGUAGES CUDA CXX)

include(CheckLanguage)
check_language(CUDA)

add_executable(imgflipGPU imFlipGPU.cu)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
