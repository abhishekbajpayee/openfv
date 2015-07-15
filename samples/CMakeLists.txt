# prototyping executable
add_executable (sa-ptv ${PROJECT_SOURCE_DIR}/samples/sa-ptv.cpp)

# release files
add_executable (refocus ${PROJECT_SOURCE_DIR}/samples/refocus.cpp)
add_executable (jobFind ${PROJECT_SOURCE_DIR}/samples/jobFind.cpp)
add_executable (track ${PROJECT_SOURCE_DIR}/samples/track.cpp)
add_executable (calibrate ${PROJECT_SOURCE_DIR}/samples/calibrate.cpp)
add_executable (safeRefocus ${PROJECT_SOURCE_DIR}/samples/safeRefocus.cpp)

target_link_libraries (sa-ptv ${LIBS})
target_link_libraries (jobFind ${LIBS})
target_link_libraries (refocus ${LIBS})
target_link_libraries (track ${LIBS})
target_link_libraries (calibrate ${LIBS})
target_link_libraries (safeRefocus ${LIBS})
