cmake_minimum_required(VERSION 2.8.11)

project(vui)

set(VUI_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(VUI_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(VUI_FORMS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/forms)
set(VUI_ICONS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/icons)

# Tell CMake to compile with C++11
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(Qt5 COMPONENTS Widgets Core REQUIRED)

add_definitions(${Qt5Widgets_DEFINITIONS})
include_directories(${Qt5Widgets_INCLUDES})

#
# project main
#
set(TARGET ${PROJECT_NAME})

set(VUI_CPPSRC 
	${VUI_SRC_DIR}/main.cpp
	${VUI_SRC_DIR}/vuiwindow.cpp
	${VUI_SRC_DIR}/vuiplayer.cpp
)

set(CMAKE_AUTOUIC_SEARCH_PATHS ${VUI_FORMS_DIR})

set ( RESOURCES icons.qrc)

set ( MOC_HEADERS
	${VUI_INCLUDE_DIR}/vuiwindow.h
)

QT5_ADD_RESOURCES( RES_SOURCES ${RESOURCES} )
QT5_WRAP_CPP( MOC_SRCS ${MOC_HEADERS} )

add_executable(${TARGET} ${VUI_CPPSRC} ${MOC_SRCS} ${RES_SOURCES})
target_link_libraries(${TARGET} ${VOCOUNT_LIB} ${Qt5Widgets_LIBRARIES})
include_directories( ${VUI_INCLUDE_DIR} ${CMAKE_BINARY_DIR} ${HDBSCAN_INCLUDE_DIR} ${GLIB2_INCLUDE_DIRS} ${LISTLIB_INCLUDE_DIR} ${GNULIB_INCLUDE_DIR})
