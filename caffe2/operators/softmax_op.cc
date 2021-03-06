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

#include "caffe2/operators/softmax_op.h"
#include "caffe2/operators/softmax_shared.h"

namespace caffe2 {

// Implementation for the CPU context.
template <>
bool SoftmaxOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  const auto canonical_axis = X.canonical_axis_index(axis_);
  const int N = X.size_to_dim(canonical_axis);
  const int D = X.size_from_dim(canonical_axis);
  Y->ResizeLike(X);
  float* Ydata = Y->mutable_data<float>();
  // First, get scales
  if (scale_.size() != N) {
    scale_.Resize(N);
  }
  if (rowmax_.size() != N) {
    rowmax_.Resize(N);
  }
  if (sum_multiplier_.size() != D) {
    sum_multiplier_.Resize(D);
    math::Set<float, CPUContext>(D, 1.f, sum_multiplier_.mutable_data<float>(),
                                 &context_);
  }

  SoftmaxCPU(
      context_,
      N,
      D,
      X.data<float>(),
      Ydata,
      scale_.mutable_data<float>(),
      sum_multiplier_.data<float>(),
      false,
      rowmax_.mutable_data<float>());
  return true;
}

REGISTER_CPU_OPERATOR(Softmax, SoftmaxOp<float, CPUContext>);

OPERATOR_SCHEMA(Softmax)
  .NumInputs(1)
  .NumOutputs(1)
  .IdenticalTypeAndShape()
  .SetDoc(R"DOC(
The operator computes the softmax normalized values for each layer in the batch
 of the given input. The input is a 2-D tensor (Tensor<float>) of size
(batch_size x input_feature_dimensions). The output tensor has the same shape
and contains the softmax normalized values of the corresponding input.

X does not need to explicitly be a 2D vector; rather, it will be
coerced into one. For an arbitrary n-dimensional tensor
X \in [a_0, a_1, ..., a_{k-1}, a_k, ..., a_{n-1}] and k is
the axis provided, then X will be coerced into a 2-dimensional tensor with
dimensions [a_0 * ... * a_{k-1}, a_k * ... * a_{n-1}]. For the default
case where axis=1, this means the X tensor will be coerced into a 2D tensor
of dimensions [a_0, a_1 * ... * a_{n-1}], where a_0 is often the batch size.
In this situation, we must have a_0 = N and a_1 * ... * a_{n-1} = D.
Each of these dimensions must be matched correctly, or else the operator
will throw errors.
)DOC")
  .Arg("axis",
       "(int) default to 1; describes the axis of the inputs when coerced "
       "to 2D; defaults to one because the 0th axis most likely describes "
       "the batch_size")
  .Input(0, "input",
         "The input tensor that's coerced into a 2D matrix of size (NxD) "
         "as described above.")
  .Output(0, "output", "The softmax normalized output values with the same "
          "shape as input tensor.");



}  // namespace caffe2
