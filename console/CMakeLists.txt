cmake_minimum_required (VERSION 2.8)

project(vocount_cli)

add_executable(vocount_cli vocount_cli.cpp)
target_link_libraries(vocount_cli ${VOCOUNT_LIB})
include_directories( ${VOCOUNT_INCLUDE_DIR} ${HDBSCAN_INCLUDE_DIRS} ${GLIB2_INCLUDE_DIRS} ${LISTLIB_INCLUDE_DIR} ${GNULIB_INCLUDE_DIR})
