# This script uses the following env vars:
#
#  - TRILINOS_DIR: the path to trilinos install dir (REQUIRED)
#  - CACHE_FILE  : the path to the cmake cache file with the desired albany settings (REQUIRED)
#  - SOURCE_DIR  : the path to the Albany source dir (REQUIRED)
#  - INSTALL_DIR : the path to the folder where to install Albany (OPTIONAL)
#

for var in "CACHE_FILE" "SOURCE_DIR" "INSTALL_DIR" "TRILINOS_DIR"; do
  if [[ "${!var}" == "" ]]; then
    echo "Error! ${var} env var is not set."
    exit 1
  else
    echo "${var}: ${!var}"
  fi
done

rm -rf CMakeFiles.txt
rm -f  CMakeCache.txt

cmake \
  -C ${CACHE_FILE}                              \
  -D CMAKE_INSTALL_PREFIX:PATH=${INSTALL_DIR}   \
  -D ALBANY_TRILINOS_DIR:PATH=${TRILINOS_DIR}   \
  \
  -S ${SOURCE_DIR}
