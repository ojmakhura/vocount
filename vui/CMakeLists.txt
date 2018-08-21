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

find_package(Qt5 COMPONENTS Widgets REQUIRED)

add_definitions(${Qt5Widgets_DEFINITIONS})

#
# project main
#
set(TARGET ${PROJECT_NAME})

set(VUI_CPPSRC 
	${VUI_SRC_DIR}/main.cpp
	${VUI_SRC_DIR}/vuiwindow.cpp
	${VUI_SRC_DIR}/vuiplayer.cpp
	${VUI_SRC_DIR}/previewdialog.cpp
	${VUI_SRC_DIR}/previewtablemodel.cpp
	${VUI_SRC_DIR}/clusterpreviewitem.cpp
)

set(CMAKE_AUTOUIC_SEARCH_PATHS ${VUI_FORMS_DIR})

set ( RESOURCES icons.qrc)

set ( MOC_HEADERS
	${VUI_INCLUDE_DIR}/vuiwindow.h
	${VUI_INCLUDE_DIR}/previewdialog.h
	${VUI_INCLUDE_DIR}/previewtablemodel.h
)

QT5_ADD_RESOURCES( RES_SOURCES ${RESOURCES} )
QT5_WRAP_CPP( MOC_SRCS ${MOC_HEADERS} )

add_executable(${TARGET} ${VUI_CPPSRC} ${MOC_SRCS} ${RES_SOURCES})
target_link_libraries(${TARGET} ${VOCOUNT_LIB} )
include_directories( ${VOCOUNT_INCLUDE_DIR} ${VUI_INCLUDE_DIR} ${CMAKE_BINARY_DIR} ${HDBSCAN_INCLUDE_DIRS} ${GLIB2_INCLUDE_DIRS} ${LISTLIB_INCLUDE_DIR} ${GNULIB_INCLUDE_DIR})
qt5_use_modules(${TARGET} Widgets)
