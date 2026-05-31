cmake_minimum_required(VERSION 3.8)
project(m3_kinematic_mecanum)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

find_package(gz-cmake3 REQUIRED)
find_package(gz-plugin2 REQUIRED COMPONENTS register)
find_package(gz-sim8 REQUIRED)
find_package(gz-transport13 REQUIRED)
find_package(gz-msgs10 REQUIRED)
find_package(gz-math7 REQUIRED)

add_library(kinematic_mecanum_system SHARED
  src/kinematic_mecanum_system.cpp
)

target_link_libraries(kinematic_mecanum_system
  gz-plugin2::gz-plugin2
  gz-sim8::gz-sim8
  gz-transport13::gz-transport13
  gz-msgs10::gz-msgs10
  gz-math7::gz-math7
)

install(
  TARGETS kinematic_mecanum_system
  LIBRARY DESTINATION lib
)



add_library(mecanum_pose_drive_system SHARED
  src/kinematic_mecanum_pose_system.cpp
)

target_link_libraries(mecanum_pose_drive_system
  gz-plugin2::gz-plugin2
  gz-sim8::gz-sim8
  gz-transport13::gz-transport13
  gz-msgs10::gz-msgs10
  gz-math7::gz-math7
)

install(
  TARGETS mecanum_pose_drive_system
  LIBRARY DESTINATION lib
)

ament_package()
