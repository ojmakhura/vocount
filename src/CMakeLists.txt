project(vocount_LIB)

file(GLOB VOCOUNT_CPPSRC "*.cpp")

#ensure we have pkg-config in the system
#find_package(PkgConfig)

# lookup gli-2.0 in pkg-config
#pkg_check_modules(GLIB2 REQUIRED glib-2.0>=2.36.0)

add_library(${VOCOUNT_LIB} SHARED ${VOCOUNT_CPPSRC} )
target_link_libraries(${VOCOUNT_LIB} LINK_PUBLIC ${GLIB2_LIBRARIES} ${HDBSCAN_LIBRARIES} ${OpenCV_LIBS} ${GSL_LIBRARIES})

include_directories( ${GLIB2_INCLUDE_DIRS} ${HDBSCAN_INCLUDE_DIRS} ${VOCOUNT_INCLUDE_DIR})
