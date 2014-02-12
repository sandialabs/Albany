 
add_subdirectory(Sphere10)
add_subdirectory(Sphere15)
add_subdirectory(Planar)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/TestCase2.xml
               ${CMAKE_CURRENT_BINARY_DIR}/TestCase2.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/TestCase5.xml
               ${CMAKE_CURRENT_BINARY_DIR}/TestCase5.xml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/TestCase6.xml
               ${CMAKE_CURRENT_BINARY_DIR}/TestCase6.xml COPYONLY)

