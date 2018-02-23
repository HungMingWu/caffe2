# ---[ CPU files.
file(GLOB_RECURSE tmp *.cc)
set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} ${tmp})
# exclude test files and gpu files
file(GLOB_RECURSE tmp *_test.cc)
exclude(Caffe2_CPU_SRCS "${Caffe2_CPU_SRCS}" ${tmp})

# ---[ threadpool/pthreadpool* is a local modification of the NNPACK
# pthreadpool with a very similar interface. Neither NNPACK, nor this
# thread pool supports Windows.
if (MSVC)
  file(GLOB_RECURSE tmp pthreadpool*)
  exclude(Caffe2_CPU_SRCS "${Caffe2_CPU_SRCS}" ${tmp})
endif()

# ---[ CPU test files
file(GLOB_RECURSE tmp *_test.cc)
set(Caffe2_CPU_TEST_SRCS ${Caffe2_CPU_TEST_SRCS} ${tmp})

# ---[ Send the lists to the parent scope.
set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} PARENT_SCOPE)
set(Caffe2_CPU_TEST_SRCS ${Caffe2_CPU_TEST_SRCS} PARENT_SCOPE)
