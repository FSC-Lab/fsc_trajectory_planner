cmake_minimum_required(VERSION 3.16)
project(fsc_trajectory_planner LANGUAGES CXX)

# The planner is numeric Eigen code: an unoptimised build is ~100x slower
# (the repo-wide 2026-08-27 finding). Default to Release when nothing is set.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(fsc_autopilot_ros2_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

# ---------------------------------------------------------------------------
# The planning library: pure Eigen, no ROS. Exported so another package can
# plan with exactly this code.
# ---------------------------------------------------------------------------
add_library(planner_lib STATIC
  src/wb_model.cpp
  src/kinematics.cpp
  src/vehicle_model.cpp
  src/flat_planner.cpp
  src/transition_planner.cpp
  src/trajectory.cpp)
set_target_properties(planner_lib PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME fsc_trajectory_planner)
target_include_directories(planner_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(planner_lib PUBLIC Eigen3::Eigen)

# ---------------------------------------------------------------------------
# The ROS 2 node.
# ---------------------------------------------------------------------------
add_executable(whole_body_trajectory_planner
  src/whole_body_trajectory_planner_node.cpp)
target_link_libraries(whole_body_trajectory_planner planner_lib)
ament_target_dependencies(whole_body_trajectory_planner
  rclcpp std_msgs std_srvs geometry_msgs nav_msgs sensor_msgs
  trajectory_msgs px4_msgs fsc_autopilot_ros2_msgs)

install(TARGETS planner_lib
  EXPORT export_planner_lib
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
install(TARGETS whole_body_trajectory_planner
  DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})
install(PROGRAMS
  test/test_planner_loopback.py
  test/test_go_home.py
  test/test_replan_stream.py
  test/test_traj_viz.py
  scripts/dump_python_fixtures.py
  DESTINATION lib/${PROJECT_NAME})

ament_export_targets(export_planner_lib HAS_LIBRARY_TARGET)
ament_export_dependencies(Eigen3)

# ---------------------------------------------------------------------------
# Tests: parity against the Python planners (committed fixtures) and the
# planner self-checks.
# ---------------------------------------------------------------------------
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_kinematics test/test_kinematics.cpp
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test)
  target_link_libraries(test_kinematics planner_lib)
  ament_add_gtest(test_transition_planner test/test_transition_planner.cpp
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test)
  target_link_libraries(test_transition_planner planner_lib)
  ament_add_gtest(test_flat_planner test/test_flat_planner.cpp
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test)
  target_link_libraries(test_flat_planner planner_lib)
endif()

ament_package()
