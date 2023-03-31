cmake_minimum_required(VERSION 2.8.3)
project(lane_follow)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  nav_msgs
  std_msgs
  message_generation
  duckietown_msgs # Every duckietown packages should use this.
)


add_service_files(
  FILES
  img.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  duckietown_msgs
)

catkin_package(
   CATKIN_DEPENDS
   message_runtime
   duckietown_msgs
)

add_dependencies(${catkin_EXPORTED_TARGETS})



include_directories(
  ${catkin_INCLUDE_DIRS}
)
