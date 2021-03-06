cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
#cmake_policy(SET CMP0022 NEW)
#cmake_policy(SET CMP0023 NEW)

# ---[ Project and semantic versioning.
project(Caffe2 CXX C)

set(CAFFE2_VERSION_MAJOR 0)
set(CAFFE2_VERSION_MINOR 8)
set(CAFFE2_VERSION_PATCH 1)
set(CAFFE2_VERSION
    "${CAFFE2_VERSION_MAJOR}.${CAFFE2_VERSION_MINOR}.${CAFFE2_VERSION_PATCH}")

# One variable that determines whether the current cmake process is being run
# with the main Caffe2 library. This is useful for building modules - if
# modules are built with the main Caffe2 library then one does not need to do
# find caffe2 in the cmake script. One can usually guard it in some way like
#    if (NOT CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO)
#      find_package(Caffe2 REQUIRED)
#    endif()
set(CAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO ON)

# ---[ Options.
# Note to developers: if you add an option below, make sure you also add it to
# cmake/Summary.cmake so that the summary prints out the option values.
include(CMakeDependentOption)
option(BUILD_BINARY "Build C++ binaries" ON)
option(BUILD_SHARED_LIBS "Build libcaffe2.so" ON)
cmake_dependent_option(
    CAFFE2_USE_MSVC_STATIC_RUNTIME "Using MSVC static runtime libraries" ON
    "NOT BUILD_SHARED_LIBS" OFF)
option(BUILD_TEST "Build C++ test binaries (need gtest and gbenchmark)" ON)
option(USE_ACL "Use ARM Compute Library" OFF)
option(USE_ASAN "Use Address Sanitizer" OFF)
option(USE_ATEN "Use ATen" OFF)
option(USE_NNPACK "Use NNPACK" ON)
option(USE_OBSERVERS "Use Observer Library" OFF)
option(USE_OPENMP "Use OpenMP for parallel code" OFF)
option(USE_PROF "Use profiling" OFF)
option(USE_THREADS "Use Threads" ON)

# ---[ CMake scripts + modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

if (MSVC AND ${BUILD_SHARED_LIBS})
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

# ---[ CMake build directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

enable_testing()

# ---[ Misc checks to cope with various compiler modes
include(cmake/MiscCheck.cmake)
include(cmake/BuildVariables.cmake)

# External projects
include(ExternalProject)

# TODO: merge the following 3 files into cmake/public/utils.cmake.
include(cmake/Utils.cmake)
include(cmake/public/utils.cmake)

set(CAFFE2_WHITELIST "" CACHE STRING "A whitelist file of files that one should build.")

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Build type not set - defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build from: Debug Release RelWithDebInfo MinSizeRel Coverage." FORCE)
endif()

# ---[ Dependencies
include(cmake/Dependencies.cmake)

# ---[ Whitelist file if whitelist is specified
include(cmake/Whitelist.cmake)

# ---[ Set link flag, handle additional deps for gcc 4.8 and above
if(CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.8.0 AND NOT ANDROID)
  message(STATUS "GCC ${CMAKE_CXX_COMPILER_VERSION}: Adding gcc and gcc_s libs to link line")
  list(APPEND Caffe2_DEPENDENCY_LIBS gcc_s gcc)
endif()

# ---[ Build flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11")
if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2 -fPIC")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-narrowing")
  # Eigen fails to build with some versions, so convert this to a warning
  # Details at http://eigen.tuxfamily.org/bz/show_bug.cgi?id=1459
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-invalid-partial-specialization")
else()
  foreach(flag_var
      CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
      CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
    if (${CAFFE2_USE_MSVC_STATIC_RUNTIME})
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES "/MD")
    else()
      if(${flag_var} MATCHES "/MT")
        string(REGEX REPLACE "/MT" "/MD" ${flag_var} "${${flag_var}}")
      endif()
    endif()
    set(${flag_var} "${${flag_var}} /MP /bigobj")
  endforeach(flag_var)
endif()

if(ANDROID)
  if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s")
  else()
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s")
  endif()
endif()

if(NOT APPLE AND UNIX)
  list(APPEND Caffe2_DEPENDENCY_LIBS dl)
endif()

# Prefix path to Caffe2 headers.
# If a directory containing installed Caffe2 headers was inadvertently
# added to the list of include directories, prefixing
# PROJECT_SOURCE_DIR means this source tree always takes precedence.
include_directories(BEFORE ${PROJECT_SOURCE_DIR})

# Prefix path to generated Caffe2 headers.
# These need to take precedence over their empty counterparts located
# in PROJECT_SOURCE_DIR.
include_directories(BEFORE ${PROJECT_BINARY_DIR})

# ---[ Main build
add_subdirectory(caffe2)

# ---[ CMake related files
# Uninistall option.
if(NOT TARGET caffe2_uninstall)
  configure_file(
      ${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in
      ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake
      IMMEDIATE @ONLY)

  add_custom_target(caffe2_uninstall
      COMMAND ${CMAKE_COMMAND} -P
      ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()

# Note(jiayq): when building static libraries, all PRIVATE dependencies
# will also become interface libraries, and as a result if there are any
# dependency libraries that are not exported, the following install export
# script will fail. As a result, we will only provide the targets cmake
# files for shared lib installation. For more info, read:
# https://cmake.org/pipermail/cmake/2016-May/063400.html
if (BUILD_SHARED_LIBS)
  configure_file(
      ${PROJECT_SOURCE_DIR}/cmake/Caffe2ConfigVersion.cmake.in
      ${PROJECT_BINARY_DIR}/Caffe2ConfigVersion.cmake
      @ONLY)
  configure_file(
      ${PROJECT_SOURCE_DIR}/cmake/Caffe2Config.cmake.in
      ${PROJECT_BINARY_DIR}/Caffe2Config.cmake
      @ONLY)
  install(FILES
      ${PROJECT_BINARY_DIR}/Caffe2ConfigVersion.cmake
      ${PROJECT_BINARY_DIR}/Caffe2Config.cmake
      DESTINATION share/cmake/Caffe2
      COMPONENT dev)
  install(FILES
      DESTINATION share/cmake/Caffe2/public
      COMPONENT dev)
  install(EXPORT Caffe2Targets DESTINATION share/cmake/Caffe2
      FILE Caffe2Targets.cmake
      COMPONENT dev)
else()
  message(WARNING
      "Generated cmake files are only available when building "
      "shared libs.")
endif()

include(cmake/Summary.cmake)
caffe2_print_configuration_summary()
