##=============================================================================
##
##  Copyright (c) Kitware, Inc.
##  All rights reserved.
##  See LICENSE.txt for details.
##
##  This software is distributed WITHOUT ANY WARRANTY; without even
##  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
##  PURPOSE.  See the above copyright notice for more information.
##
##=============================================================================
# - Try to find rabbitmq-c headers and libraries
#   (see https://github.com/alanxz/rabbitmq-c)
#
# Usage of this module as follows:
#
#     find_package(RabbitMQ)
#
# Variables used by this module, they can change the default behaviour and need
# to be set before calling find_package:
#
#  RabbitMQ_ROOT_DIR  Set this variable to the root installation of
#                     RabbitMQ if the module has problems finding
#                     the proper installation path.
#
# Variables defined by this module:
#
#  RabbitMQ_FOUND              System has RabbitMQ libs/headers
#  RabbitMQ_LIBRARIES          The RabbitMQ libraries
#  RabbitMQ_INCLUDE_DIR        The location of RabbitMQ headers
#  RabbitMQ_VERSION            The version of RabbitMQ

find_path(RabbitMQ_ROOT_DIR
  NAMES include/amqp.h
  )

if(MSVC)
  #try to find the release and debug version
  find_library(RabbitMQ_LIBRARY_RELEASE
    NAMES rabbitmq librabbitmq
    HINTS ${RabbitMQ_ROOT_DIR}/bin
          ${RabbitMQ_ROOT_DIR}/lib
    )

  find_library(RabbitMQ_LIBRARY_DEBUG
    NAMES rabbitmq librabbitmq
    HINTS ${RabbitMQ_ROOT_DIR}/bin
          ${RabbitMQ_ROOT_DIR}/lib
    )

  if(RabbitMQ_LIBRARY_RELEASE AND RabbitMQ_LIBRARY_DEBUG)
    set(RabbitMQ_LIBRARY
        debug ${RabbitMQ_LIBRARY_DEBUG}
        optimized ${RabbitMQ_LIBRARY_RELEASE}
        )
  elseif(RabbitMQ_LIBRARY_RELEASE)
    set(RabbitMQ_LIBRARY ${RabbitMQ_LIBRARY_RELEASE})
  elseif(RabbitMQ_LIBRARY_DEBUG)
    set(RabbitMQ_LIBRARY ${RabbitMQ_LIBRARY_DEBUG})
  endif()

else()
  find_library(RabbitMQ_LIBRARY
    NAMES rabbitmq librabbitmq
    HINTS ${RabbitMQ_ROOT_DIR}/lib
    )
endif()

find_path(RabbitMQ_INCLUDE_DIR
  NAMES amqp.h
  HINTS ${RabbitMQ_ROOT_DIR}/include
  )

function(extract_version_value value_name file_name value)
  file(STRINGS ${file_name} val REGEX "${value_name} .")
  string(FIND ${val} " " last REVERSE)
  string(SUBSTRING ${val} ${last} -1 val)
  string(STRIP ${val} val)
  set(${value} ${val} PARENT_SCOPE)
endfunction(extract_version_value)

extract_version_value("AMQP_VERSION_MAJOR" ${RabbitMQ_INCLUDE_DIR}/amqp.h MAJOR)
extract_version_value("AMQP_VERSION_MINOR" ${RabbitMQ_INCLUDE_DIR}/amqp.h MINOR)
extract_version_value("AMQP_VERSION_PATCH" ${RabbitMQ_INCLUDE_DIR}/amqp.h PATCH)

set(RabbitMQ_VER "${MAJOR}.${MINOR}.${PATCH}")

#We are using the 2.8.10 signature of find_package_handle_standard_args,
#as that is the version that ParaView 5.1 && VTK 6/7 ship, and inject
#into the CMake module path. This allows our FindModule to work with
#projects that include VTK/ParaView before searching for this application
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  RabbitMQ
  REQUIRED_VARS RabbitMQ_LIBRARY RabbitMQ_INCLUDE_DIR
  VERSION_VAR RabbitMQ_VER
  )

set(RabbitMQ_FOUND ${ZEROMQ_FOUND})
set(RabbitMQ_INCLUDE_DIRS ${RabbitMQ_INCLUDE_DIR})
set(RabbitMQ_LIBRARIES ${RabbitMQ_LIBRARY})
set(RabbitMQ_VERSION ${RabbitMQ_VER})

mark_as_advanced(
  RabbitMQ_ROOT_DIR
  RabbitMQ_LIBRARY
  RabbitMQ_LIBRARY_DEBUG
  RabbitMQ_LIBRARY_RELEASE
  RabbitMQ_INCLUDE_DIR
  RabbitMQ_VERSION
  )
