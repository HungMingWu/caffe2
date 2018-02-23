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

#include "caffe2/operators/matmul_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(MatMul, MatMulOp<float, CPUContext>);

OPERATOR_SCHEMA(MatMul)
    .NumInputs(2, 3)
    .NumOutputs(1)
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out(1);
      out[0].set_data_type(in[0].data_type());
      ArgumentHelper arg_helper(def);
      int axis_a = arg_helper.GetSingleArgument<int>("axis_a", 1);
      int axis_b = arg_helper.GetSingleArgument<int>("axis_b", 1);
      int trans_a = arg_helper.GetSingleArgument<bool>("trans_a", false);
      int trans_b = arg_helper.GetSingleArgument<bool>("trans_b", false);
      int canonical_axis_a = canonical_axis_index_(axis_a, in[0].dims().size());
      int canonical_axis_b = canonical_axis_index_(axis_b, in[0].dims().size());

      int M = size_to_dim_(canonical_axis_a, GetDimsVector(in[0]));
      int N = size_from_dim_(canonical_axis_b, GetDimsVector(in[1]));
      if (trans_a) {
        M = size_from_dim_(canonical_axis_a, GetDimsVector(in[0]));
      }
      if (trans_b) {
        N = size_to_dim_(canonical_axis_b, GetDimsVector(in[1]));
      }

      out[0].add_dims(M);
      out[0].add_dims(N);

      return out;
    })
    .SetDoc(R"DOC(
Matrix multiplication Y = A * B, where A has size (M x K), B has size (K x N),
and Y will have a size (M x N).
)DOC")
    .Input(0, "A", "2D matrix of size (M x K)")
    .Input(1, "B", "2D matrix of size (K x N)")
    .Output(0, "Y", "2D matrix of size (M x N)")
    .Arg(
        "axis_a",
        "Exclusive axis that divides the first and second dimension \
of matrix A, default to 1")
    .Arg(
        "axis_b",
        "Exclusive axis that divides the first and second dimension \
of matrix B, default to 1")
    .Arg(
        "trans_a",
        "Pass 1 to transpose A before multiplication and after the \
dimension adjustment using axis_a")
    .Arg(
        "trans_b",
        "Pass 1 to transpose B before multiplication and after the \
dimension adjustment using axis_b");


} // namespace caffe2
