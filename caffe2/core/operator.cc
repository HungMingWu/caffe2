/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "caffe2/core/operator.h"

#include <algorithm>

#include "caffe2/core/logging.h"
#include "caffe2/core/net.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/types.h"
#include "caffe2/core/workspace.h"

#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/utils/string_utils.h"

CAFFE2_DEFINE_int(
    caffe2_operator_max_engine_name_length,
    10,
    "Maximum engine name length to be stored");
CAFFE2_DEFINE_bool(
    caffe2_disable_implicit_engine_preference,
    false,
    "If set, disable implicit engine preferences. This is useful for unit "
    "testing and debugging cases.");

namespace caffe2 {

OperatorBase::OperatorBase(const OperatorDef& operator_def, Workspace* ws)
    : operator_ws_(ws),
      operator_def_(std::make_shared<OperatorDef>(operator_def)),
      device_option_(
          operator_def.has_device_option() ? operator_def.device_option()
                                           : DeviceOption()),
      event_(caffe2::make_unique<Event>(device_option_)) {
  for (const string& input_str : operator_def.input()) {
    auto* blob = ws->GetBlob(input_str);
    CAFFE_ENFORCE(
        blob != nullptr,
        "op ",
        operator_def.type(),
        ": Encountered a non-existing input blob: ",
        input_str);
    inputs_.push_back(blob);
  }

  GetOperatorLogger()(operator_def);

  for (const string& output_str : operator_def.output()) {
    outputs_.push_back(CHECK_NOTNULL(ws->CreateBlob(output_str)));
  }
}

namespace {

PerOpEnginePrefType& g_per_op_engine_pref() {
  static auto* g_per_op_engine_pref_ = new PerOpEnginePrefType();
  return *g_per_op_engine_pref_;
}

GlobalEnginePrefType& g_global_engine_pref() {
  static auto* g_global_engine_pref_ =
      new GlobalEnginePrefType{{DeviceType::CUDA, {"CUDNN"}}};
  return *g_global_engine_pref_;
}

unique_ptr<OperatorBase> TryCreateOperator(
    const string& key, const OperatorDef& operator_def, Workspace* ws) {
  auto type = operator_def.device_option().device_type();
  CAFFE_ENFORCE(
      gDeviceTypeRegistry()->count(type),
      "Device type ",
      type,
      " not registered.");
  OperatorRegistry* registry = gDeviceTypeRegistry()->at(type);
  VLOG(1) << "Creating operator with device type " << type;
  try {
    return registry->Create(key, operator_def, ws);
  } catch (const UnsupportedOperatorFeature& err) {
    LOG(WARNING) << "Operator " << operator_def.type()
                 << " does not support the requested feature. Msg: "
                 << err.what()
                 << ". Proto is: " << ProtoDebugString(operator_def);
    return nullptr;
  }
}

unique_ptr<OperatorBase> _CreateOperator(
    const OperatorDef& operator_def,
    Workspace* ws) {
  static StaticLinkingProtector g_protector;
  const auto op_type = operator_def.type();
  const auto device_type = operator_def.device_option().device_type();

#ifndef CAFFE2_NO_OPERATOR_SCHEMA
  // first, check with OpSchema if the operator is legal.
  auto* schema = OpSchemaRegistry::Schema(op_type);
  if (schema) {
    CAFFE_ENFORCE(
        schema->Verify(operator_def),
        "Operator def did not pass schema checking: ",
        ProtoDebugString(operator_def));
  } else {
    // We would like to recommend every op to register its schema, so if there
    // is not one, we print a LOG_ERROR. But we will still allow the operator
    // to be constructed.
    LOG(ERROR) << "Cannot find operator schema for " << op_type
               << ". Will skip schema checking.";
  }
#endif

  // second try engines specified in the operator_def and preferred engines
  std::vector<std::string> engines{};
  if (operator_def.engine().size()) {
    const auto op_def_engines = split(',', operator_def.engine());
    engines.insert(engines.end(), op_def_engines.begin(), op_def_engines.end());
  }
  if (!FLAGS_caffe2_disable_implicit_engine_preference &&
      g_per_op_engine_pref().count(device_type) &&
      g_per_op_engine_pref()[device_type].count(op_type)) {
    const auto& preferred_engines =
        g_per_op_engine_pref()[device_type][op_type];
    VLOG(2) << "Inserting per-op engine preference: " << preferred_engines;
    engines.insert(
        engines.end(), preferred_engines.begin(), preferred_engines.end());
  }
  if (!FLAGS_caffe2_disable_implicit_engine_preference &&
      g_global_engine_pref().count(device_type)) {
    const auto& preferred_engines = g_global_engine_pref()[device_type];
    VLOG(2) << "Inserting global engine preference: " << preferred_engines;
    engines.insert(
        engines.end(), preferred_engines.begin(), preferred_engines.end());
  }
  for (const auto& engine : engines) {
    const std::string key = OpRegistryKey(op_type, engine);
    VLOG(1) << "Trying to create operator " << op_type << " with engine "
            << engine;
    auto op = TryCreateOperator(key, operator_def, ws);
    if (op) {
      if (engine.size() <= FLAGS_caffe2_operator_max_engine_name_length) {
        op->annotate_engine(engine);
      } else {
        op->annotate_engine(
            engine.substr(0, FLAGS_caffe2_operator_max_engine_name_length));
      }
      return op;
    } else {
      // If the above fails, we will just return the normal case with the
      // default implementation.
      LOG(INFO) << "Operator with engine " << engine
                << " is not available for operator " << op_type << ".";
    }
  }
  VLOG(1) << "Using default implementation.";

  // Lastly, if the engine does not work here, try using the default engine.
  auto op = TryCreateOperator(op_type, operator_def, ws);
  CAFFE_ENFORCE(
      op,
      "Cannot create operator of type '",
      op_type,
      "' on the device '",
      DeviceTypeName(device_type),
      "'. Verify that implementation for the corresponding device exist. It "
      "might also happen if the binary is not linked with the operator "
      "implementation code. If Python frontend is used it might happen if "
      "dyndep.InitOpsLibrary call is missing. Operator def: ",
      ProtoDebugString(operator_def));
  return op;
}

} // namespace

const std::string OpRegistryKey(
    const std::string& op_type,
    const std::string& engine) {
  if (engine == "" || engine == "DEFAULT") {
    return op_type;
  } else {
    return op_type + "_ENGINE_" + engine;
  }
}

unique_ptr<OperatorBase> CreateOperator(
    const OperatorDef& operator_def,
    Workspace* ws,
    int net_position) {
  try {
    auto op = _CreateOperator(operator_def, ws);
    op->set_net_position(net_position);
    return op;
  } catch (...) {
    if (net_position != 0) {
      VLOG(1) << "Operator constructor with net position " << net_position
              << " failed";
      ws->last_failed_op_net_position = net_position;
    } else {
      VLOG(1) << "Failed operator constructor doesn't have an id set";
    }
    throw;
  }
}

std::map<int32_t, OperatorRegistry*>* gDeviceTypeRegistry() {
  static std::map<int32_t, OperatorRegistry*> g_device_type_registry;
  return &g_device_type_registry;
}

CAFFE_DEFINE_REGISTRY(
    CPUOperatorRegistry,
    OperatorBase,
    const OperatorDef&,
    Workspace*);
CAFFE_REGISTER_DEVICE_TYPE(DeviceType::CPU, CPUOperatorRegistry);

CAFFE_DEFINE_REGISTRY(
    CUDAOperatorRegistry,
    OperatorBase,
    const OperatorDef&,
    Workspace*);
CAFFE_REGISTER_DEVICE_TYPE(DeviceType::CUDA, CUDAOperatorRegistry);

}  // namespace caffe2
