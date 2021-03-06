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

#include <cmath>

#include "caffe2/operators/elementwise_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

struct TanhCPUFunctor {
  template <typename T>
  inline void
  operator()(const int n, const T* x, T* y, CPUContext* /*device_context*/) {
#ifdef CAFFE2_USE_ACCELERATE
    vvtanhf(y, x, &n);
#else
    ConstEigenVectorArrayMap<T> x_arr(x, n);
    EigenVectorMap<T>(y, n) = 1 - 2 * ((x_arr * 2).exp() + 1).inverse();
#endif
  }
};

REGISTER_CPU_OPERATOR(
    Tanh, UnaryElementwiseOp<TensorTypes<float>, CPUContext, TanhCPUFunctor>);

OPERATOR_SCHEMA(Tanh)
  .NumInputs(1)
  .NumOutputs(1)
  .AllowInplace({{0, 0}})
  .IdenticalTypeAndShape()
  .SetDoc(R"DOC(
Calculates the hyperbolic tangent of the given input tensor element-wise. This
operation can be done in an in-place fashion too, by providing the same input
and output blobs.
)DOC")
  .Input(0, "input", "1-D input tensor")
  .Output(0, "output", "The hyperbolic tangent values of the input tensor "
          "computed element-wise");


}  // namespace caffe2
