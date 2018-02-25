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

#include "caffe2/operators/concat_split_op.h"

namespace caffe2 {
REGISTER_CPU_OPERATOR(Split, SplitOp<CPUContext>);
REGISTER_CPU_OPERATOR(Concat, ConcatOp<CPUContext>);
OPERATOR_SCHEMA(Split)
    .NumInputs(1, 2)
    .NumOutputs(1, INT_MAX)
    .Input(0, "input", "The tensor to split")
    .Input(1, "split", "Optional list of output lengths (see also arg 'split')")
    .Arg("axis", "Which axis to split on")
    .Arg("split", "length of each output")
    .Arg("order", "Either NHWC or NCWH, will split on C axis, defaults to NCHW")
    .SetDoc(R"DOC(
Split a tensor into a list of tensors, along the specified
'axis'. The lengths of the split can be specified using argument 'axis' or
optional second input blob to the operator. Otherwise, the tensor is split
to equal sized parts.
)DOC");

namespace {
OpSchema::Cost CostInferenceForConcat(
    const OperatorDef& def,
    const vector<TensorShape>& in) {
  ArgumentHelper helper(def);
  const int axis = helper.HasArgument("axis")
      ? helper.GetSingleArgument<int>("axis", -1)
      : GetDimFromOrderString(
            helper.GetSingleArgument<string>("order", "NCHW"));
  bool add_axis = helper.GetSingleArgument<int>("add_axis", 0) != 0;
  const int canonical_axis = canonical_axis_index_(axis, in[0].dims_size());
  CAFFE_ENFORCE_GT(in.size(), 0);
  vector<int> out_shape(in[0].dims().begin(), in[0].dims().end());
  if (add_axis) {
    out_shape.insert(out_shape.begin() + canonical_axis, in.size());
  } else {
    for (int i = 1; i < in.size(); ++i) {
      out_shape[canonical_axis] += in[i].dims(canonical_axis);
    }
  }
  int size = 1;
  for (auto& s : out_shape) {
    size *= s;
  }

  struct OpSchema::Cost cost;
  cost.flops = size;
  cost.bytes_moved = size * sizeof(float);
  cost.params_bytes = 0;
  return cost;
}
} // namespace

OPERATOR_SCHEMA(Concat)
    .NumInputs(1, INT_MAX)
    .NumOutputs(2)
    .Arg("axis", "Which axis to concat on")
    .Arg("order", "Either NHWC or NCHW, will concat on C axis, defaults to NCHW")
    .Arg(
        "add_axis",
        "Pass 1 to add the axis specified in arg 'axis' to all "
        "input tensors")
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      ArgumentHelper helper(def);
      const int axis = helper.HasArgument("axis")
          ? helper.GetSingleArgument<int>("axis", -1)
          : GetDimFromOrderString(
                helper.GetSingleArgument<string>("order", "NCHW"));
      bool add_axis = helper.GetSingleArgument<int>("add_axis", 0) != 0;
      const int canonical_axis = canonical_axis_index_(axis, in[0].dims_size());
      CAFFE_ENFORCE_GT(in.size(), 0);
      vector<int> split_shape(1, in.size());
      vector<int> out_shape(in[0].dims().begin(), in[0].dims().end());
      if (add_axis) {
        out_shape.insert(out_shape.begin() + canonical_axis, in.size());
      } else {
        for (int i = 1; i < in.size(); ++i) {
          out_shape[canonical_axis] += in[i].dims(canonical_axis);
        }
      }
      if (def.output_size() == 1) {
        return vector<TensorShape>{
            CreateTensorShape(out_shape, in[0].data_type())};
      }
      return vector<TensorShape>{
          CreateTensorShape(out_shape, in[0].data_type()),
          CreateTensorShape(split_shape, TensorProto::INT32)};
    })
    .CostInferenceFunction(CostInferenceForConcat)
    .SetDoc("Concatenate a list of tensors into a single tensor")
    .Output(0, "concat_result", "Concatenated tensor")
    .Output(1, "split_info", "The dimensions of the inputs.");

}  // namespace caffe2
