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

#include "caffe2/core/net_simple.h"
#include "caffe2/core/net.h"

#include <set>
#include <unordered_map>
#include <unordered_set>

#include "caffe2/core/operator.h"
#include "caffe2/core/static_tracepoint.h"
#include "caffe2/core/timer.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/utils/proto_utils.h"

namespace caffe2 {

SimpleNet::SimpleNet(
    const std::shared_ptr<const NetDef>& net_def,
    Workspace* ws)
    : NetBase(net_def, ws) {
  VLOG(1) << "Constructing SimpleNet " << net_def->name();
  const bool net_def_has_device_option = net_def->has_device_option();
  // Initialize the operators
  for (int idx = 0; idx < net_def->op_size(); ++idx) {
    const auto& operator_def = net_def->op(idx);
    VLOG(1) << "Creating operator " << operator_def.name() << ": "
            << operator_def.type();
    std::unique_ptr<OperatorBase> op{nullptr};
    if (!operator_def.has_device_option() && net_def_has_device_option) {
      // In the case that the operator def does not specify a device option but
      // the net def has a default option, we copy the device option over to the
      // operator def.
      OperatorDef temp_def(operator_def);
      temp_def.mutable_device_option()->CopyFrom(net_def->device_option());
      op = CreateOperator(temp_def, ws, idx);
    } else {
      op = CreateOperator(operator_def, ws, idx);
      op->set_debug_def(
          std::shared_ptr<const OperatorDef>{net_def, &(net_def->op(idx))});
    }
    operators_.emplace_back(std::move(op));
  }
}

bool SimpleNet::Run() {
  StartAllObservers();
  VLOG(1) << "Running net " << name_;
  for (auto& op : operators_) {
    VLOG(1) << "Running operator " << op->debug_def().name() << "("
            << op->debug_def().type() << ").";
#ifdef CAFFE2_ENABLE_SDT
    const auto& op_name = op->debug_def().name().c_str();
    const auto& op_type = op->debug_def().type().c_str();
    auto* op_ptr = op.get();
    const auto& net_name = name_.c_str();
    CAFFE_SDT(operator_start, net_name, op_name, op_type, op_ptr);
#endif
    bool res = op->Run();
#ifdef CAFFE2_ENABLE_SDT
    CAFFE_SDT(operator_done, net_name, op_name, op_type, op_ptr);
#endif
    if (!res) {
      LOG(ERROR) << "Operator failed: " << ProtoDebugString(op->debug_def());
      return false;
    }
  }
  StopAllObservers();
  return true;
}

bool SimpleNet::RunAsync() {
  return Run();
}

namespace {
template <typename A, typename B>
bool PairLargerThan(const std::pair<A, B>& x, const std::pair<A, B>& y) {
  return x.second > y.second;
}
}

REGISTER_NET(simple, SimpleNet);

} // namespace caffe2
