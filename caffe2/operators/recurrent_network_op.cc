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

#include "caffe2/operators/recurrent_network_op.h"
#include "caffe2/core/workspace.h"
#include "caffe2/utils/proto_utils.h"

#ifndef CAFFE2_RNN_NO_TEXT_FORMAT
#include <google/protobuf/text_format.h>
#endif

CAFFE2_DEFINE_bool(
    caffe2_rnn_executor,
    true,
    "If set, uses special RNN executor for executing RecurrentNetworkOp");

namespace caffe2 {
CAFFE_KNOWN_TYPE(detail::ScratchWorkspaces);

REGISTER_CPU_OPERATOR(RecurrentNetwork, RecurrentNetworkOp<CPUContext>);
OPERATOR_SCHEMA(RecurrentNetwork)
    .NumInputs(1, INT_MAX)
    .NumOutputs(2, INT_MAX)
    .SetDoc(R"DOC(
Run the input network in a recurrent fashion. This can be used to
implement fairly general recurrent neural networks (RNNs).

The operator proceeds as follows.

- First, initialized the states from the input recurrent states
- For each timestep T, apply the links (that map offsets from input/output
tensors into the inputs/outputs for the `step` network)
- Finally, alias the recurrent states to the specified output blobs.

This is a fairly special-case meta-operator, and so the implementation
is somewhat complex. It trades of generality (and frankly usability)
against performance and control (compared to e.g. TF
dynamic_rnn, Theano scan, etc).

See the usage examples for a flavor of how to use it.
)DOC");


REGISTER_CPU_OPERATOR(
    rnn_internal_apply_link,
    RNNApplyLinkOp<CPUContext>);
OPERATOR_SCHEMA(rnn_internal_apply_link)
    .NumInputs(2)
    .NumOutputs(2)
    .EnforceInplace({{1, 1}})
    .Private()
    .SetDoc(R"DOC(
Internal RNN operator.
)DOC");

namespace detail {

std::map<string, string> GetRecurrentMapping(
    const std::vector<detail::Link>& links,
    bool backward) {
  std::map<string, string> mappings;
  for (auto it = links.begin(); it != links.end(); ++it) {
    const auto& l1 = *it;

    // In backward op we expect to see offset 1 before offset 0 and
    // vice versa.
    const int offset_l1 = backward ? 1 : 0;
    const int offset_l2 = 1 - offset_l1;
    if (l1.offset == offset_l1) {
      // Find offset = 1 from links. We could probaby rely on order, but
      // since the number of links is links small, O(n^2) algo is ok
      for (auto it2 = it + 1; it2 != links.end(); ++it2) {
        const auto& l2 = *it2;
        if (l2.offset == offset_l2 && l2.external == l1.external) {
          mappings[l2.internal] = l1.internal;
          break;
        }
      }
    }
  }
  return mappings;
}

void PrependOps(std::vector<OperatorDef> ops, NetDef* netdef) {
  for (auto& o : netdef->op()) {
    ops.push_back(o);
  }
  netdef->mutable_op()->Clear();
  for (auto& o : ops) {
    auto* ao = netdef->add_op();
    ao->CopyFrom(o);
  }
}

void AddApplyLinkOps(
    const vector<Link>& links,
    std::string timestep,
    const DeviceOption& device_option,
    NetDef* netdef) {
  std::vector<OperatorDef> ops;
  for (auto& link : links) {
    OperatorDef opdef;
    opdef.set_type("rnn_internal_apply_link");
    opdef.add_input(timestep);
    opdef.add_input(link.external);
    opdef.add_output(link.internal);
    opdef.add_output(link.external);
    opdef.mutable_device_option()->CopyFrom(device_option);

    Argument* offset_arg = opdef.add_arg();
    offset_arg->set_name("offset");
    offset_arg->set_i(link.offset);

    Argument* window_arg = opdef.add_arg();
    window_arg->set_name("window");
    window_arg->set_i(link.window);

    // Find out if the linked blob is used first as an output: then we need
    // to add control_input to that op
    for (auto& op : *netdef->mutable_op()) {
      if (HasInput(op, link.internal)) {
        // First appears as an input, no need to do antyhing
        continue;
      }
      if (HasOutput(op, link.internal)) {
        op.add_control_input(link.internal);
        break;
      }
    }

    ops.push_back(opdef);

    netdef->add_external_input(link.internal);
    netdef->add_external_input(link.external);
  }

  detail::PrependOps(ops, netdef);
}

void extractLinks(
    OperatorBase* op,
    const std::string& internalArg,
    const std::string& externalArg,
    const std::string& offsetArg,
    const std::string& windowArg,
    std::vector<detail::Link>* links) {
  const auto& internal = op->GetRepeatedArgument<std::string>(internalArg);
  const auto& external = op->GetRepeatedArgument<std::string>(externalArg);
  const auto& offset = op->GetRepeatedArgument<int32_t>(offsetArg);
  const auto& window = op->GetRepeatedArgument<int32_t>(
      windowArg, vector<int32_t>(offset.size(), 1));
  CAFFE_ENFORCE_EQ(
      internal.size(),
      offset.size(),
      "internal/offset mismatch: ",
      internalArg,
      " ",
      externalArg);
  CAFFE_ENFORCE_EQ(
      external.size(),
      offset.size(),
      "external/offset mismatch: ",
      externalArg,
      " ",
      offsetArg);
  CAFFE_ENFORCE_EQ(
      external.size(),
      window.size(),
      "external/window mismatch: ",
      externalArg,
      " ",
      windowArg);
  for (auto i = 0; i < internal.size(); ++i) {
    detail::Link l;
    l.internal = internal[i];
    l.external = external[i];
    l.offset = offset[i];
    l.window = window[i];
    links->push_back(l);
  }
}

NetDef extractNetDef(const OperatorDef& op, const std::string& argName) {
  if (ArgumentHelper::HasSingleArgumentOfType<OperatorDef, NetDef>(
          op, argName)) {
    return ArgumentHelper::GetSingleArgument<OperatorDef, NetDef>(
        op, argName, NetDef());
  } else {
#ifndef CAFFE2_RNN_NO_TEXT_FORMAT
    NetDef result;
    const auto netString =
        ArgumentHelper::GetSingleArgument<OperatorDef, string>(op, argName, "");
    CAFFE_ENFORCE(
        google::protobuf::TextFormat::ParseFromString(netString, &result),
        "Invalid NetDef");
    return result;
#else
    CAFFE_THROW("No valid NetDef for argument ", argName);
#endif
  }
}
} // namespace detail
}
