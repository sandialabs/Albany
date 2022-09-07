# Read link.txt
file (READ ${ALBANY_BINARY_DIR}/dummy/CMakeFiles/dummy.dir/link.txt LINK_FILE_CONTENT)

# Remove multiple spaces, as well as trailing spaces/newline
string (REGEX REPLACE " +" " " ALBANY_LINK_LIBS ${LINK_FILE_CONTENT})
string (STRIP ${ALBANY_LINK_LIBS} ALBANY_LINK_LIBS)

# Turn into a list, so we can iterate over it
string (REPLACE " " ";" ALBANY_LINK_LIBS ${ALBANY_LINK_LIBS})

cmake_policy (SET CMP0007 NEW)

# Find the name of the executable.
list (FIND ALBANY_LINK_LIBS "dummy" DUMMY_INDEX)
if (DUMMY_INDEX EQUAL -1)
  message (FATAL_ERROR "Somethinkg went wrong while generating export_albany.in")
endif()

# Remove any entry coming before the executable name
foreach(index RANGE ${DUMMY_INDEX})
  list (REMOVE_AT ALBANY_LINK_LIBS 0)
endforeach()

# Also, remove rpath, since we  will link ALL libs using full path
foreach (item IN ITEMS ${ALBANY_LINK_LIBS})
  if (item MATCHES "-rpath")
    list(REMOVE_ITEM ALBANY_LINK_LIBS ${item})
    break()
  endif()
endforeach()

# Replace ../src/<albLib> with /path/to/install/lib/<albLib>
macro(replace_list_item LIST INDEX NEWVALUE)
  list (GET ${LIST} ${INDEX} ITEM)
  list(INSERT ${LIST} ${INDEX} ${NEWVALUE})
  math(EXPR __INDEX "${INDEX} + 1")
  list (REMOVE_AT ${LIST} ${__INDEX})
endmacro(replace_list_item)

string (REPLACE " " ";" ALBANY_LIBRARIES ${ALBANY_LIBRARIES})
foreach (lib IN ITEMS ${ALBANY_LIBRARIES})
  foreach (item IN ITEMS ${ALBANY_LINK_LIBS})
    if ("lib${item}" MATCHES "${lib}")
      get_filename_component(fname ${item} NAME)
      list (FIND ALBANY_LINK_LIBS ${item} itemIdx)
      replace_list_item (ALBANY_LINK_LIBS ${itemIdx} "${ALBANY_INSTALL_PREFIX}/${ALBANY_INSTALL_LIBDIR}/${fname}")
      break()
    endif()
  endforeach()
endforeach()

# Configure the export_albany.in file
configure_file(${ALBANY_SOURCE_DIR}/cmake/export_albany.in ${ALBANY_BINARY_DIR}/export_albany.in)
