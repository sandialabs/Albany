cmake_policy(SET CMP0003 NEW)
cmake_policy(SET CMP0060 NEW)
cmake_minimum_required(VERSION 3.15)

project (LinkAgainstAlbany C CXX Fortran)

add_executable (dummy main.cpp)
target_link_libraries (dummy PUBLIC ${ALBANY_LIBRARIES} ${TPL_LIBRARIES})
