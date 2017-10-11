project(vocount)

file(GLOB VOCOUNT_CPPSRC "*.cpp")

add_executable(vocount ${VOCOUNT_CPPSRC})
target_link_libraries(vocount ${GLIB2_LIBRARIES} ${HDBSCAN_LIBRARIES} ${OpenCV_LIBS} ${GSL_LIBRARIES})

include_directories( ${GLIB2_INCLUDE_DIRS} ${HDBSCAN_INCLUDE_DIRS} ${VOCOUNT_INCLUDE_DIR})
