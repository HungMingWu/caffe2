# ---[ Declare source file lists

# ---[ Add respective subdirectories
# Note: the folders that are being commented out have not been properly
# addressed yet.

add_subdirectory(proto)

add_subdirectory(contrib)
add_subdirectory(core)
add_subdirectory(cuda_rtc)
add_subdirectory(db)
add_subdirectory(distributed)
# add_subdirectory(experiments) # note, we may remove this folder at some point
add_subdirectory(image)
add_subdirectory(video)
add_subdirectory(mkl)
add_subdirectory(mobile)
add_subdirectory(observers)
add_subdirectory(operators)
add_subdirectory(perfkernels)
add_subdirectory(queue)
add_subdirectory(sgd)
add_subdirectory(share)
# add_subdirectory(test) # todo: use caffe2_gtest_main instead of gtest_main because we will need to call GlobalInit
add_subdirectory(transforms)
add_subdirectory(utils)

# Advanced: if we have white list specified, we will do intersections for all
# main lib srcs.
if (CAFFE2_WHITELISTED_FILES)
  caffe2_do_whitelist(Caffe2_CPU_SRCS CAFFE2_WHITELISTED_FILES)
  caffe2_do_whitelist(Caffe2_GPU_SRCS CAFFE2_WHITELISTED_FILES)
endif()

# Debug messages - if you want to get a list of source files, enable the
# following.
if (FALSE)
  message(STATUS "CPU sources: ")
  foreach(tmp ${Caffe2_CPU_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "GPU sources: ")
  foreach(tmp ${Caffe2_GPU_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "CPU test sources: ")
  foreach(tmp ${Caffe2_CPU_TEST_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "GPU test sources: ")
  foreach(tmp ${Caffe2_GPU_TEST_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()
endif()

# ---[ Generate and install header files.

# Write the macros file.
configure_file(
    ${PROJECT_SOURCE_DIR}/caffe2/core/macros.h.in
    ${PROJECT_BINARY_DIR}/caffe2/core/macros.h)

# Installing the header files
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
        DESTINATION include
        FILES_MATCHING PATTERN "*.h")
install(FILES ${PROJECT_BINARY_DIR}/caffe2/core/macros.h
        DESTINATION include/caffe2/core)


# ---[ List of libraries to link with

# Compile exposed libraries.
add_library(caffe2 ${Caffe2_CPU_SRCS} $<TARGET_OBJECTS:Caffe_PROTO> $<TARGET_OBJECTS:Caffe2_PROTO>)
target_link_libraries(caffe2 PUBLIC ${Caffe2_PUBLIC_DEPENDENCY_LIBS})
target_link_libraries(caffe2 PRIVATE ${Caffe2_DEPENDENCY_LIBS})
target_include_directories(caffe2 INTERFACE $<INSTALL_INTERFACE:include>)
target_compile_options(caffe2 INTERFACE "-std=c++11")
target_compile_options(caffe2 PRIVATE "-DCAFFE2_BUILD_MAIN_LIB")
install(TARGETS caffe2 EXPORT Caffe2Targets DESTINATION lib)
caffe2_interface_library(caffe2 caffe2_library)
list(APPEND Caffe2_MAIN_LIBS caffe2_library)

# ---[ CUDA library.
if(USE_CUDA)
  # A hack to deal with cuda library dependencies and modern CMake: the
  # CUDA_ADD_LIBRARY includes a target_link_libraries, and as a result,
  # one cannot use PUBLIC/PRIVATE/INTERFACE for the target anymore. This
  # hack adds the PRIVATE keywords to CUDA_LIBRARIES so we can deal with
  # it. We will then manually add the cudart library as interface libs.
  set(__tmp ${CUDA_LIBRARIES})
  set(CUDA_LIBRARIES PRIVATE ${CUDA_LIBRARIES})
  CUDA_ADD_LIBRARY(caffe2_gpu ${Caffe2_GPU_SRCS})
  set(CUDA_LIBRARIES ${__tmp})
  target_link_libraries(caffe2_gpu INTERFACE caffe2::cudart)

  target_include_directories(
      caffe2_gpu INTERFACE $<INSTALL_INTERFACE:include>)
  target_link_libraries(
      caffe2_gpu PUBLIC caffe2 ${Caffe2_PUBLIC_CUDA_DEPENDENCY_LIBS})
  target_link_libraries(
      caffe2_gpu PRIVATE ${Caffe2_CUDA_DEPENDENCY_LIBS})
  caffe2_interface_library(caffe2_gpu caffe2_gpu_library)
  list(APPEND Caffe2_MAIN_LIBS caffe2_gpu_library)
  install(TARGETS caffe2_gpu EXPORT Caffe2Targets DESTINATION lib)
endif()

# ---[ Test binaries.
if (BUILD_TEST)
  set(Caffe2_ALL_TEST_SRCS ${Caffe2_CPU_TEST_SRCS})
  if (USE_CUDA)
    list(APPEND Caffe2_ALL_TEST_SRCS ${Caffe2_GPU_TEST_SRCS})
  endif()

  foreach(test_src ${Caffe2_ALL_TEST_SRCS})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} "${test_src}")
    # For tests, some of the test code actually directly call the dependent
    # libraries even if they are not part of the public dependency libs. As a
    # result, we will explicitly link the test against the Caffe2 dependency
    # libs.
    target_link_libraries(${test_name} ${Caffe2_MAIN_LIBS} gtest_main)
    if (USE_CUDA)
      target_link_libraries(${test_name} ${Caffe2_CUDA_DEPENDENCY_LIBS})
    endif()
    if (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} GREATER 3.0)
      target_compile_features(${test_name} PRIVATE cxx_range_for)
    endif()
    add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
    install(TARGETS ${test_name} DESTINATION test)
  endforeach()
endif()

# ---[ Any builds that should happen after the main targets should be added here.
# Binaries
if (BUILD_BINARY)
  add_subdirectory(binaries)
  # Benchmarking binaries require observers included in the build
  # There is a linking issue that happens in some of the Windows builds.
  # TODO(Yangqing): after the module redesign, enable this back.
  if (BUILD_OBSERVERS AND NOT MSVC)
    add_subdirectory(share/contrib/binaries)
  endif()
endif()
