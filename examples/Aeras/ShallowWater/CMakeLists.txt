 
add_subdirectory(Sphere10)
add_subdirectory(Sphere15)
add_subdirectory(Planar)
add_subdirectory(TC2)
add_subdirectory(TC5)
add_subdirectory(TC6)
add_subdirectory(TC3)
add_subdirectory(TCGalewsky)
#add_subdirectory(TC4)
add_subdirectory(TC7)

#The Spectral example runs with the AlbanyT executable only.
if (NOT ALBANY_EPETRA) 
add_subdirectory(Spectral)
endif()
