caffe2_binary_target("convert_caffe_image_db.cc")
caffe2_binary_target("convert_db.cc")
caffe2_binary_target("make_cifar_db.cc")
caffe2_binary_target("make_mnist_db.cc")
caffe2_binary_target("predictor_verifier.cc")
caffe2_binary_target("print_registered_core_operators.cc")
caffe2_binary_target("run_plan.cc")
caffe2_binary_target("speed_benchmark.cc")
caffe2_binary_target("split_db.cc")

if (USE_ZMQ)
  caffe2_binary_target("zmq_feeder.cc")
  target_link_libraries(zmq_feeder ${ZMQ_LIBRARIES})
endif()

# ---[ tutorials
caffe2_binary_target("tutorial_blob.cc")
