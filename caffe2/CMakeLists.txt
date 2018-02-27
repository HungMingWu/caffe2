# ---[ Declare source file lists

# ---[ Add respective subdirectories
# Note: the folders that are being commented out have not been properly
# addressed yet.

add_subdirectory(proto)

add_subdirectory(contrib)
add_subdirectory(core)
# add_subdirectory(experiments) # note, we may remove this folder at some point
add_subdirectory(observers)
add_subdirectory(operators)
add_subdirectory(perfkernels)
add_subdirectory(share)
# add_subdirectory(test) # todo: use caffe2_gtest_main instead of gtest_main because we will need to call GlobalInit
add_subdirectory(utils)

# Advanced: if we have white list specified, we will do intersections for all
# main lib srcs.
if (CAFFE2_WHITELISTED_FILES)
  caffe2_do_whitelist(Caffe2_CPU_SRCS CAFFE2_WHITELISTED_FILES)
endif()

# Debug messages - if you want to get a list of source files, enable the
# following.
if (FALSE)
  message(STATUS "CPU sources: ")
  foreach(tmp ${Caffe2_CPU_SRCS})
    message(STATUS "  " ${tmp})
  endforeach()

  message(STATUS "CPU test sources: ")
  foreach(tmp ${Caffe2_CPU_TEST_SRCS})
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

# ---[ Test binaries.
if (BUILD_TEST)
  set(Caffe2_ALL_TEST_SRCS ${Caffe2_CPU_TEST_SRCS})

  foreach(test_src ${Caffe2_ALL_TEST_SRCS})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} "${test_src}")
    # For tests, some of the test code actually directly call the dependent
    # libraries even if they are not part of the public dependency libs. As a
    # result, we will explicitly link the test against the Caffe2 dependency
    # libs.
    target_link_libraries(${test_name} ${Caffe2_MAIN_LIBS} gtest_main)
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
