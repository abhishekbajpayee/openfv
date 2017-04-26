file(GLOB hdrs "openfv/*.h*")
install(FILES ${hdrs}
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include/openfv)