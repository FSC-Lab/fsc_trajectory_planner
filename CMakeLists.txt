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
# The whole-body model and the flat B-spline planner are the flight stack's
# exported wb_law library (fsc_autopilot_ros2::wb_law) -- linked, never copied.
find_package(fsc_autopilot_ros2 REQUIRED)

# ---------------------------------------------------------------------------
# The planning library: pure Eigen, no ROS, on top of wb_law. Exported so
# another package can plan with exactly this code.
# ---------------------------------------------------------------------------
add_library(planner_lib STATIC
  src/kinematics.cpp
  src/vehicle_model.cpp
  src/transition_planner.cpp
  src/bspline_fit.cpp
  src/ee_trajectory_planner.cpp
  src/trajectory.cpp)
set_target_properties(planner_lib PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME fsc_trajectory_planner)
target_include_directories(planner_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(planner_lib PUBLIC fsc_autopilot_ros2::wb_law Eigen3::Eigen)

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
  test/test_ee_trajectory_loopback.py
  test/ee_trajectory_sim_driver.py
  test/ee_trajectory_sim_cycle.sh
  scripts/dump_python_fixtures.py
  DESTINATION lib/${PROJECT_NAME})

ament_export_targets(export_planner_lib HAS_LIBRARY_TARGET)
ament_export_dependencies(fsc_autopilot_ros2 Eigen3)

# ---------------------------------------------------------------------------
# Tests: parity against the Python planners (committed fixtures for the ported
# code, the flight stack's fixture for wb_law) and the planner self-checks.
# ---------------------------------------------------------------------------
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_kinematics test/test_kinematics.cpp
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test)
  target_link_libraries(test_kinematics planner_lib)
  ament_add_gtest(test_flat_planner test/test_flat_planner.cpp
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test)
  target_link_libraries(test_flat_planner planner_lib)
  ament_add_gtest(test_ee_trajectory_planner test/test_ee_trajectory_planner.cpp
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test)
  target_link_libraries(test_ee_trajectory_planner planner_lib)
  # The parity fixture is the flight stack's own (installed beside its
  # flat_planner_selftest), so the two tests can never disagree on the truth.
  get_filename_component(_wb_share "${fsc_autopilot_ros2_DIR}/.." ABSOLUTE)
  target_compile_definitions(test_flat_planner PRIVATE
    WB_FLAT_FIXTURE="${_wb_share}/test_data/flat_plan_t650.txt")
endif()

ament_package()
