macro(move_xml_file pattern filename)

# Save a copy of the Trilinos configure to post to the CDash site.

#EXECUTE_PROCESS( COMMAND ${CTEST_SCP_COMMAND} ${CTEST_DROP_SITE}:${CTEST_DROP_LOCATION}/Configure.xml 
#                 ${CTEST_DROP_SITE}:${CTEST_DROP_LOCATION}/Build_Trilinos.xml
#               )

# Note: CTest will store files in ${CTEST_BINARY_DIRECTORY}/${CTEST_DROP_LOCATION} - ending in Update.xml.
# Not sure what the first part of that filename is, so glob all possibilities into UPDATE_FILES. Then
# grab the last one and rename it to Update_Trilinos.xml in the drop location. We assume there is only a
# single update file, if there are more skip this.

FILE(GLOB BUILD_FILES "${CTEST_BINARY_DIRECTORY}/${CTEST_DROP_LOCATION}/${pattern}")
LIST(LENGTH BUILD_FILES BU_LIST_LEN)
IF(BU_LIST_LEN EQUAL 1)
  LIST(GET BUILD_FILES -1 SINGLE_BUILD_FILE)
  FILE(RENAME "${SINGLE_BUILD_FILE}" "${CTEST_BINARY_DIRECTORY}/${CTEST_DROP_LOCATION}/${filename}")
ENDIF()

endmacro(move_xml_file)
