cmake_minimum_required(VERSION 3.25)

project(BST_XR)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
file(GLOB_RECURSE SOURCE_LIST
   CONFIGURE_DEPENDS "BST_XR-Source/*.cpp"
   CONFIGURE_DEPENDS "BST_XR-Source/*.hpp"
)
add_executable(${PROJECT_NAME} ${SOURCE_LIST})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCE_LIST})
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${PROJECT_NAME} PRIVATE BST_XR-Source)
set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "/PROFILE")

find_package(OpenCV REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS})

find_package(depthai CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE depthai::opencv)

