# ---[ Custom Protobuf
include("cmake/ProtoBuf.cmake")

# ---[ Threads
if(USE_THREADS)
  find_package(Threads REQUIRED)
  list(APPEND Caffe2_DEPENDENCY_LIBS ${CMAKE_THREAD_LIBS_INIT})
endif()

# ---[ git: used to generate git build string.
find_package(Git)
if(GIT_FOUND)
  execute_process(COMMAND ${GIT_EXECUTABLE} describe --tags --always --dirty
                  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
                  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
                  OUTPUT_VARIABLE CAFFE2_GIT_VERSION
                  RESULT_VARIABLE __git_result)
  if(NOT ${__git_result} EQUAL 0)
    set(CAFFE2_GIT_VERSION "unknown")
  endif()
else()
  message(
      WARNING
      "Cannot find git, so Caffe2 won't have any git build info available")
endif()


# ---[ BLAS
set(BLAS "Eigen" CACHE STRING "Selected BLAS library")
set_property(CACHE BLAS PROPERTY STRINGS "Eigen;ATLAS;OpenBLAS;MKL;vecLib")
message(STATUS "The BLAS backend of choice:" ${BLAS})

if(BLAS STREQUAL "Eigen")
  # Eigen is header-only and we do not have any dependent libraries
  set(CAFFE2_USE_EIGEN_FOR_BLAS 1)
elseif(BLAS STREQUAL "ATLAS")
  find_package(Atlas REQUIRED)
  include_directories(${ATLAS_INCLUDE_DIRS})
  list(APPEND Caffe2_PUBLIC_DEPENDENCY_LIBS ${ATLAS_LIBRARIES})
  list(APPEND Caffe2_PUBLIC_DEPENDENCY_LIBS cblas)
elseif(BLAS STREQUAL "OpenBLAS")
  find_package(OpenBLAS REQUIRED)
  include_directories(${OpenBLAS_INCLUDE_DIR})
  list(APPEND Caffe2_PUBLIC_DEPENDENCY_LIBS ${OpenBLAS_LIB})
elseif(BLAS STREQUAL "MKL")
  find_package(MKL REQUIRED)
  include_directories(${MKL_INCLUDE_DIR})
  list(APPEND Caffe2_PUBLIC_DEPENDENCY_LIBS ${MKL_LIBRARIES})
  set(CAFFE2_USE_MKL 1)
elseif(BLAS STREQUAL "vecLib")
  find_package(vecLib REQUIRED)
  include_directories(${vecLib_INCLUDE_DIR})
  list(APPEND Caffe2_PUBLIC_DEPENDENCY_LIBS ${vecLib_LINKER_LIBS})
else()
  message(FATAL_ERROR "Unrecognized blas option:" ${BLAS})
endif()

# ---[ On Android, Caffe2 uses cpufeatures library in the thread pool
if (ANDROID)
  if (NOT TARGET cpufeatures)
    add_library(cpufeatures STATIC
        "${ANDROID_NDK}/sources/android/cpufeatures/cpu-features.c")
    target_include_directories(cpufeatures
        PUBLIC "${ANDROID_NDK}/sources/android/cpufeatures")
    target_link_libraries(cpufeatures PUBLIC dl)
  endif()
  list(APPEND Caffe2_DEPENDENCY_LIBS cpufeatures)
endif()

# ---[ Googletest and benchmark
if(BUILD_TEST)
  set(TEMP_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
  # We will build gtest as static libs and embed it directly into the binary.
  set(BUILD_SHARED_LIBS OFF)
  # For gtest, we will simply embed it into our test binaries, so we won't
  # need to install it.
  set(BUILD_GTEST ON)
  set(INSTALL_GTEST OFF)
  # We currently don't need gmock right now.
  set(BUILD_GMOCK OFF)
  # For Windows, we will check the runtime used is correctly passed in.
  if (NOT CAFFE2_USE_MSVC_STATIC_RUNTIME)
    set(gtest_force_shared_crt ON)
  endif()
  add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/googletest)
  include_directories(${PROJECT_SOURCE_DIR}/third_party/googletest/googletest/include)

  # Recover the build shared libs option.
  set(BUILD_SHARED_LIBS ${TEMP_BUILD_SHARED_LIBS})
endif()

# ---[ EIGEN
# Due to license considerations, we will only use the MPL2 parts of Eigen.
set(EIGEN_MPL2_ONLY 1)
find_package(Eigen3)
if(EIGEN3_FOUND)
  message(STATUS "Found system Eigen at " ${EIGEN3_INCLUDE_DIR})
  include_directories(${EIGEN3_INCLUDE_DIR})
else()
  message(STATUS "Did not find system Eigen. Using third party subdirectory.")
  include_directories(${PROJECT_SOURCE_DIR}/third_party/eigen)
endif()

# ---[ OpenMP
if(USE_OPENMP)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    message(STATUS "Adding " ${OpenMP_CXX_FLAGS})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
  else()
    message(WARNING "Not compiling with OpenMP. Suppress this warning with -DUSE_OPENMP=OFF")
    set(USE_OPENMP OFF)
  endif()
endif()


# ---[ Android specific ones
if(ANDROID)
  list(APPEND Caffe2_DEPENDENCY_LIBS log)
endif()

# ---[ profiling
if(USE_PROF)
  find_package(htrace)
  if(htrace_FOUND)
    set(USE_PROF_HTRACE ON)
  else()
    message(WARNING "htrace not found. Caffe2 will build without htrace prof")
  endif()
endif()

if (USE_ATEN)
  list(APPEND Caffe2_DEPENDENCY_LIBS aten_op_header_gen ATen)
  include_directories(${PROJECT_BINARY_DIR}/caffe2/contrib/aten/aten/src/ATen)
  include_directories(${PROJECT_SOURCE_DIR}/third_party/aten/src)
  include_directories(${PROJECT_BINARY_DIR}/caffe2/contrib/aten)
endif()
