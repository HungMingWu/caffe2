/**
 * Copyright (c) 2018-present, Facebook, Inc.
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

#include "caffe2/operators/pow_op.h"
#include "caffe2/utils/math.h"
// definition of NumericTypes and SameTypeAsInput is in below header file
//#include "caffe2/operators/elementwise_op.h"
#include <Eigen/Core>

namespace caffe2 {

#define EIGEN_POW(x, y) (x.pow(y))

struct EigenPowFunctor {
  template <int b_is_scalar, typename T1, typename T2, typename R>
  inline void
  Run(size_t n, const T1* a, const T2* b, T2 e, R* out, CPUContext*) {
    if (b == NULL) {
      EigenVectorArrayMap<R>(out, n) =
          EIGEN_POW((ConstEigenVectorArrayMap<T1>(a, n)), (e));
    } else {
      if (b_is_scalar) {
        EigenVectorArrayMap<R>(out, n) =
            EIGEN_POW((ConstEigenVectorArrayMap<T1>(a, n)), (b[0]));
      } else {
        EigenVectorArrayMap<R>(out, n) = EIGEN_POW(
            (ConstEigenVectorArrayMap<T1>(a, n)),
            (ConstEigenVectorArrayMap<T2>(b, n)));
      }
    }
  }
  template <typename T1, typename T2, typename R>
  void RunWithBroadcast(
      const T1* a,
      const T2* b,
      R* out,
      size_t pre,
      size_t n,
      CPUContext*) {
    EigenArrayMap<R>(out, n, pre) = EIGEN_POW(
        (ConstEigenArrayMap<T1>(a, n, pre)),
        (ConstEigenVectorArrayMap<T2>(b, n)).rowwise().replicate(pre));
    /*
    //below code only allows elementary ops, such as +, -, * and /,
    //and does not allow operations, such as pow, exp and log
    EIGEN_POW(
       (ConstEigenArrayMap<T>(a, n, pre).colwise()),
       (ConstEigenVectorArrayMap<T>(b, n)));
     */
  }
  template <typename T1, typename T2, typename R>
  void RunWithBroadcast2(
      const T1* a,
      const T2* b,
      R* out,
      size_t pre,
      size_t n,
      size_t post,
      CPUContext*) {
    for (int i = 0; i < pre; ++i) {
      EigenArrayMap<R>(out + i * n * post, post, n) = EIGEN_POW(
          (ConstEigenArrayMap<T1>(a + i * n * post, post, n)),
          (Eigen::Map<const Eigen::Array<T2, 1, Eigen::Dynamic>>(b, n))
              .colwise()
              .replicate(post));
      /*
      //below code only allows elementary ops, such as +, -, * and /,
      //and does not allow for operations, such as pow, exp and log
      EIEGN_POW(
        (ConstEigenArrayMap<T>(a + i * n * post, post, n).rowwise()),
        (Eigen::Map<const Eigen::Array<T, 1, Eigen::Dynamic>>(b, n)));
      */
    }
  }
};

REGISTER_CPU_OPERATOR(
    Pow,
    PowOp<
        TensorTypes<float> /*NumericTypes*/,
        CPUContext,
        EigenPowFunctor,
        SameTypeAsInput>)

OPERATOR_SCHEMA(Pow)
    .NumInputs(1, 2)
    .NumOutputs(1)
    .Arg("exponent", "The exponent of the power function.")
    .AllowInplace({{0, 0}, {1, 0}})
    .SetDoc(R"DOC(
Pow takes input data (Tensor<T>) and an argument exponent, which can be a
scalar or another tensor. It produces one output data (Tensor<T>), where
the function `f(x) = x^exponent` is applied to the data tensor elementwise.
)DOC")
    .Input(0, "X", "Input tensor of any shape")
    .Input(1, "exponent", "The exponent of the power function.")
    .Output(0, "Y", "Output tensor (same size as X)");

} // namespace caffe2
