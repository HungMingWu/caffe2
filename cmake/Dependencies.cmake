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

# ---[ NNPACK
if(USE_NNPACK)
  include("cmake/External/nnpack.cmake")
  if(NNPACK_FOUND)
    if(TARGET nnpack)
      # ---[ NNPACK is being built together with Caffe2: explicitly specify dependency
      list(APPEND Caffe2_DEPENDENCY_LIBS nnpack)
    else()
      include_directories(${NNPACK_INCLUDE_DIRS})
      list(APPEND Caffe2_DEPENDENCY_LIBS ${NNPACK_LIBRARIES})
    endif()
  else()
    message(WARNING "Not compiling with NNPACK. Suppress this warning with -DUSE_NNPACK=OFF")
    set(USE_NNPACK OFF)
  endif()
endif()

if(USE_OBSERVERS)
  list(APPEND Caffe2_DEPENDENCY_LIBS Caffe2_CPU_OBSERVER)
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

  # We will not need to test benchmark lib itself.
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark testing as we don't need it.")
  # We will not need to install benchmark since we link it statically.
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "Disable benchmark install to avoid overwriting vendor install.")
  add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/benchmark)
  include_directories(${PROJECT_SOURCE_DIR}/third_party/benchmark/include)

  # Recover the build shared libs option.
  set(BUILD_SHARED_LIBS ${TEMP_BUILD_SHARED_LIBS})
endif()

# ---[ LMDB
if(USE_LMDB)
  find_package(LMDB)
  if(LMDB_FOUND)
    include_directories(${LMDB_INCLUDE_DIR})
    list(APPEND Caffe2_DEPENDENCY_LIBS ${LMDB_LIBRARIES})
  else()
    message(WARNING "Not compiling with LMDB. Suppress this warning with -DUSE_LMDB=OFF")
    set(USE_LMDB OFF)
  endif()
endif()

# ---[ Rocksdb
if(USE_ROCKSDB)
  find_package(RocksDB)
  if(ROCKSDB_FOUND)
    include_directories(${RocksDB_INCLUDE_DIR})
    list(APPEND Caffe2_DEPENDENCY_LIBS ${RocksDB_LIBRARIES})
  else()
    message(WARNING "Not compiling with RocksDB. Suppress this warning with -DUSE_ROCKSDB=OFF")
    set(USE_ROCKSDB OFF)
  endif()
endif()

# ---[ ZMQ
if(USE_ZMQ)
  find_package(ZMQ)
  if(ZMQ_FOUND)
    include_directories(${ZMQ_INCLUDE_DIR})
    list(APPEND Caffe2_DEPENDENCY_LIBS ${ZMQ_LIBRARIES})
  else()
    message(WARNING "Not compiling with ZMQ. Suppress this warning with -DUSE_ZMQ=OFF")
    set(USE_ZMQ OFF)
  endif()
endif()

# ---[ Redis
if(USE_REDIS)
  find_package(Hiredis)
  if(HIREDIS_FOUND)
    include_directories(${Hiredis_INCLUDE})
    list(APPEND Caffe2_DEPENDENCY_LIBS ${Hiredis_LIBRARIES})
  else()
    message(WARNING "Not compiling with Redis. Suppress this warning with -DUSE_REDIS=OFF")
    set(USE_REDIS OFF)
  endif()
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

# ---[ CUDA
if(USE_CUDA)
  include(cmake/public/cuda.cmake)
  if(CAFFE2_FOUND_CUDA)
    # A helper variable recording the list of Caffe2 dependent librareis
    # caffe2::cudart is dealt with separately, due to CUDA_ADD_LIBRARY
    # design reason (it adds CUDA_LIBRARIES itself).
    set(Caffe2_PUBLIC_CUDA_DEPENDENCY_LIBS
        caffe2::cuda caffe2::curand caffe2::cublas caffe2::cudnn caffe2::nvrtc)
  else()
    message(WARNING
        "Not compiling with CUDA. Suppress this warning with "
        "-DUSE_CUDA=OFF.")
    set(USE_CUDA OFF)
  endif()
endif()

# ---[ NCCL
if(USE_NCCL)
  if(NOT USE_CUDA)
    message(WARNING "If not using cuda, one should not use NCCL either.")
    set(USE_NCCL OFF)
  elseif(NOT ${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    message(WARNING "NCCL is currently only supported under Linux.")
    set(USE_NCCL OFF)
  else()
    include("cmake/External/nccl.cmake")
    list(APPEND Caffe2_CUDA_DEPENDENCY_LIBS __caffe2_nccl)
  endif()
endif()

# ---[ CUB
if(USE_CUDA)
  find_package(CUB)
  if(CUB_FOUND)
    include_directories(${CUB_INCLUDE_DIRS})
  else()
    include_directories(${PROJECT_SOURCE_DIR}/third_party/cub)
  endif()
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

if (USE_MOBILE_OPENGL)
  if (ANDROID)
    list(APPEND Caffe2_DEPENDENCY_LIBS EGL GLESv2)
  elseif (IOS)
    message(STATUS "TODO item for adding ios opengl dependency")
  else()
    message(WARNING "mobile opengl is only used in android or ios builds.")
    set(USE_MOBILE_OPENGL OFF)
  endif()
endif()

# ---[ ARM Compute Library: check compatibility.
if (USE_ACL)
  if (NOT ANDROID)
    message(WARNING "ARM Compute Library is only supported for Android builds.")
    set(USE_ACL OFF)
  else()
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "^armv")
      # 32-bit ARM (armv7, armv7-a, armv7l, etc)
      set(ACL_ARCH "armv7a")
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
      # 64-bit ARM
      set(ACL_ARCH "arm64-v8a")
    else()
      message(WARNING "ARM Compute Library is only supported for ARM/ARM64 builds.")
      set(USE_ACL OFF)
    endif()
  endif()
endif()

# ---[ ARM Compute Library: build the target.
if (USE_ACL)
  list(APPEND ARM_COMPUTE_INCLUDE_DIRS "third_party/ComputeLibrary/")
  list(APPEND ARM_COMPUTE_INCLUDE_DIRS "third_party/ComputeLibrary/include")
  include_directories(${ARM_COMPUTE_INCLUDE_DIRS})
  string (REPLACE ";" " -I" ANDROID_STL_INCLUDE_FLAGS "-I${ANDROID_STL_INCLUDE_DIRS}")
  set (ARM_COMPUTE_SRC_DIR "${PROJECT_SOURCE_DIR}/third_party/ComputeLibrary/")
  set (ARM_COMPUTE_LIB "${CMAKE_CURRENT_BINARY_DIR}/libarm_compute.a")
  set (ARM_COMPUTE_CORE_LIB "${CMAKE_CURRENT_BINARY_DIR}/libarm_compute_core.a")
  set (ARM_COMPUTE_LIBS ${ARM_COMPUTE_LIB} ${ARM_COMPUTE_CORE_LIB})

  add_custom_command(
      OUTPUT ${ARM_COMPUTE_LIBS}
      COMMAND
        /bin/sh -c "export PATH=\"$PATH:$(dirname ${CMAKE_CXX_COMPILER})\" && \
        scons -C \"${ARM_COMPUTE_SRC_DIR}\" -Q \
          examples=no validation_tests=no benchmark_tests=no standalone=yes \
          embed_kernels=yes opencl=no gles_compute=yes \
          os=android arch=${ACL_ARCH} \
          extra_cxx_flags=\"${ANDROID_CXX_FLAGS} ${ANDROID_STL_INCLUDE_FLAGS}\"" &&
        /bin/sh -c "cp ${ARM_COMPUTE_SRC_DIR}/build/libarm_compute-static.a ${CMAKE_CURRENT_BINARY_DIR}/libarm_compute.a" &&
        /bin/sh -c "cp ${ARM_COMPUTE_SRC_DIR}/build/libarm_compute_core-static.a ${CMAKE_CURRENT_BINARY_DIR}/libarm_compute_core.a" &&
        /bin/sh -c "rm -r ${ARM_COMPUTE_SRC_DIR}/build"
      COMMENT "Building ARM compute library" VERBATIM)
  add_custom_target(arm_compute_build ALL DEPENDS ${ARM_COMPUTE_LIBS})

  add_library(arm_compute_core STATIC IMPORTED)
  add_dependencies(arm_compute_core arm_compute_build)
  set_property(TARGET arm_compute_core PROPERTY IMPORTED_LOCATION ${ARM_COMPUTE_CORE_LIB})

  add_library(arm_compute STATIC IMPORTED)
  add_dependencies(arm_compute arm_compute_build)
  set_property(TARGET arm_compute PROPERTY IMPORTED_LOCATION ${ARM_COMPUTE_LIB})

  list(APPEND Caffe2_DEPENDENCY_LIBS arm_compute arm_compute_core)
endif()

if (USE_SNPE AND ANDROID)
  if (SNPE_LOCATION AND SNPE_HEADERS)
    message(STATUS "Using SNPE location specified by -DSNPE_LOCATION: " ${SNPE_LOCATION})
    message(STATUS "Using SNPE headers specified by -DSNPE_HEADERS: " ${SNPE_HEADERS})
    include_directories(SYSTEM ${SNPE_HEADERS})
    add_library(snpe SHARED IMPORTED)
    set_property(TARGET snpe PROPERTY IMPORTED_LOCATION ${SNPE_LOCATION})
    list(APPEND Caffe2_DEPENDENCY_LIBS snpe)
  else()
    set(USE_SNPE OFF)
  endif()
endif()

if (USE_METAL)
  if (NOT IOS)
    message(WARNING "Metal is only used in ios builds.")
    set(USE_METAL OFF)
  endif()
endif()

if (USE_ATEN)
  list(APPEND Caffe2_DEPENDENCY_LIBS aten_op_header_gen ATen)
  include_directories(${PROJECT_BINARY_DIR}/caffe2/contrib/aten/aten/src/ATen)
  include_directories(${PROJECT_SOURCE_DIR}/third_party/aten/src)
  include_directories(${PROJECT_BINARY_DIR}/caffe2/contrib/aten)
endif()

if (USE_ZSTD)
  list(APPEND Caffe2_DEPENDENCY_LIBS libzstd_static)
  include_directories(${PROJECT_SOURCE_DIR}/third_party/zstd/lib)
  add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/zstd/build/cmake)
  set_property(TARGET libzstd_static PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()
