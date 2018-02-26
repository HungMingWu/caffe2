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

#include "caffe2/operators/lpnorm_op.h"

namespace caffe2 {

template <>
bool LpNormOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(X_IN);
  auto* norm = Output(OUT);
  norm->Resize(1);
  const float* X_data = X.data<float>();
  if (p_ == 1) {
    *(norm->mutable_data<float>()) =
        (ConstEigenVectorMap<float>(X_data, X.size()).array()).abs().sum();
    // L1(x) = sum(|x|)
  } else if (p_ == 2) {
    *(norm->mutable_data<float>()) =
        (ConstEigenVectorMap<float>(X_data, X.size()).array()).square().sum();
    // L2(x) = (sum(|x|^2))
  }
  return true;
}

namespace {
// LpNorm
REGISTER_CPU_OPERATOR(LpNorm, LpNormOp<float, CPUContext>);

OPERATOR_SCHEMA(LpNorm)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Given one input float tensor X, and produces one output float tensor
of the Lp norm of tensor X, computed as Lp(x) = sum over |x^p|,
in which p is either 1 or 2(currently only supports l1 and l2 norm),
determined by the argument p.
)DOC")
    .Input(0, "X", "1D input tensor")
    .Output(0, "Z", "1D output tensor")
    .Arg("p", "Order of the norm in p-norm");

} // namespace

} // namespace caffe2
