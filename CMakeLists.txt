cmake_minimum_required (VERSION 2.8)

project(vocount)

# The version number.
set (VOCOUNT_MAJOR_VERSION 1)
set (VOCOUNT_MINOR_VERSION 0)
set (VOCOUNT_PATCH_VERSION 0)
set (VOCOUNT_VERSION ${VOCOUNT_VERSION_MAJOR}.${VOCOUNT_VERSION_MINOR}.${VOCOUNT_PATCH_VERSION})

set(INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries")
set(INSTALL_BIN_DIR bin CACHE PATH "Installation directory for executables")
set(INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files")
set(HDBSCAN_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(HDBSCAN_INCLUDE_DIR ${HDBSCAN_LIB_DIR}/include)
set(VOCOUNT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set( VOCOUNT_LIB ${CMAKE_PROJECT_NAME})

#ensure we have pkg-config in the system
find_package(PkgConfig)

# lookup gli-2.0 in pkg-config
pkg_check_modules(GLIB2 REQUIRED glib-2.0>=2.36.0)

# Find QT 5 Core module

find_package(Qt5 COMPONENTS Core REQUIRED)

add_definitions(${Qt5Core_DEFINITIONS})
#include_directories(${Qt5Core_INCLUDES})

OPTION (USE_OpenMP "Use OpenMP" ON)
IF(USE_OpenMP)
  FIND_PACKAGE(OpenMP)
  IF(OPENMP_FOUND)
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  ENDIF()
ENDIF()

SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g3 -Wall -fmessage-length=0 -fPIC")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -Wall -fmessage-length=0 -fPIC")

# find the necessary libraries
find_package( OpenCV REQUIRED )
find_package( Hdbscan REQUIRED )
find_package(GSL REQUIRED) 

add_subdirectory(src)
add_subdirectory(console)
add_subdirectory(vui)
 
