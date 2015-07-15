add_library(rendering SHARED ${PROJECT_SOURCE_DIR}/src/modules/rendering.cpp)
set_target_properties(rendering PROPERTIES ${PYTHON_PROPS} PREFIX "")
target_link_libraries(rendering ${LIBS})

add_library(refocusing SHARED ${PROJECT_SOURCE_DIR}/src/modules/refocusing.cpp)
set_target_properties(refocusing PROPERTIES ${PYTHON_PROPS} PREFIX "")
target_link_libraries(refocusing ${LIBS})

add_library(tools SHARED ${PROJECT_SOURCE_DIR}/src/modules/tools.cpp)
set_target_properties(tools PROPERTIES ${PYTHON_PROPS} PREFIX "")
target_link_libraries(tools ${LIBS})

add_library(tracking SHARED ${PROJECT_SOURCE_DIR}/src/modules/tracking.cpp)
set_target_properties(tracking PROPERTIES ${PYTHON_PROPS} PREFIX "")
target_link_libraries(tracking ${LIBS})

add_library(calibration SHARED ${PROJECT_SOURCE_DIR}/src/modules/calibration.cpp)
set_target_properties(calibration PROPERTIES ${PYTHON_PROPS} PREFIX "")
target_link_libraries(calibration ${LIBS})

# add_library(localization SHARED ${PROJECT_SOURCE_DIR}/src/modules/pLoc.cpp)
# set_target_properties(localization PROPERTIES ${PYTHON_PROPS} PREFIX "")
# target_link_libraries(localization ${LIBS})
