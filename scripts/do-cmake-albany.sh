# This script uses the following env vars:
#
# TRILINOS_DIR: the path to trilinos install dir (REQUIRED)
# SOURCE_DIR  : the path to the Albany source dir (REQUIRED)
# CACHE_FILE  : the path to this machine/build cmake var settings (OPTIONAL: defaults to an empty file)
# INSTALL_DIR: the path where to install albany (OPTIONAL: defaults to $(pwd)/install)
#

echo "Configuring albany ...\n"
for var in "SOURCE_DIR" "TRILINOS_DIR"; do
  if [[ "${!var}" == "" ]]; then
    echo "Error! ${var} env var is not set."
    exit 1
  else
    echo "${var}: ${!var}"
  fi
done

if [[ "${CACHE_FILE}" == "" ]]; then
  echo "Warning! CACHE_FILE env var is not set. Using an empty cache file"
  touch $(pwd)/empty.cmake
  export CACHE_FILE=$(pwd)/empty.cmake
fi
if [[ "${INSTALL_DIR}" == "" ]]; then
  echo "Warning! INSTALL_DIR env var is not set. Installing in $(pwd)/install"
  export CACHE_FILE=$(pwd)/install
fi

rm -rf CMakeFiles.txt
rm -f  CMakeCache.txt

cmake \
  -C ${CACHE_FILE}                              \
  -D CMAKE_INSTALL_PREFIX:PATH=${INSTALL_DIR}   \
  -D ALBANY_TRILINOS_DIR:PATH=${TRILINOS_DIR}   \
  \
  -S ${SOURCE_DIR}
