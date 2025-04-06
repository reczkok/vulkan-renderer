cmake_minimum_required(VERSION 4.0)
project(RayMarcher VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(cmake/CPM.cmake)

find_package(Vulkan)
add_executable(RayMarcher main.cpp)
target_link_libraries(RayMarcher Vulkan::Vulkan)

CPMAddPackage("gh:glfw/glfw#3.4")
target_link_libraries(RayMarcher glfw)

CPMAddPackage("gh:g-truc/glm#1.0.1")
target_link_libraries(RayMarcher glm)