add_library (lib optimization.cpp
                 calibration.cpp
		 refocusing.cpp
                 pLoc.cpp
		 tools.cpp)

find_package (OpenCV REQUIRED)

target_link_libraries (lib ${OpenCV_LIBS})
