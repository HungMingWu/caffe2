if (USE_PROF)
  set(Caffe2_CONTRIB_PROF_CPU_SRCS
      "${CMAKE_CURRENT_SOURCE_DIR}/prof_dag_net.cc"
      "${CMAKE_CURRENT_SOURCE_DIR}/prof_dag_stats_op.cc"
  )

  if (USE_PROF_HTRACE)
    set(Caffe2_CONTRIB_PROF_CPU_SRCS ${Caffe2_CONTRIB_PROF_CPU_SRCS}
        "${CMAKE_CURRENT_SOURCE_DIR}/htrace_conf.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/htrace_dag_net.cc"
    )
  endif()

  set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} ${Caffe2_CONTRIB_PROF_CPU_SRCS} PARENT_SCOPE)
endif()
