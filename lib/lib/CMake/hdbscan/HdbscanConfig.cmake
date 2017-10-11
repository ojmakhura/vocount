# - Config file for the hdbscan package
# It defines the following variables
#  HDBSCAN_INCLUDE_DIRS - include directories for hdbscan
#  HDBSCAN_LIBRARIES    - libraries to link against
 
# Compute paths
get_filename_component(HDBSCAN_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
set(HDBSCAN_INCLUDE_DIRS "${HDBSCAN_CMAKE_DIR}/../../../include")
 
# Our library dependencies (contains definitions for IMPORTED targets)
if(NOT TARGET hdbscan AND NOT hdbscan_BINARY_DIR)
  include("${HDBSCAN_CMAKE_DIR}/HdbscanTargets.cmake")
endif()
 
# These are IMPORTED targets created by HdbscanTargets.cmake
set(HDBSCAN_LIBRARIES hdbscan)
set(HDBSCAN_EXECUTABLE hdbscan_csample hdbscan_cppsample)
