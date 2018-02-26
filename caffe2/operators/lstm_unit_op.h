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

#ifndef CAFFE2_OPERATORS_LSTM_UNIT_OP_H_
#define CAFFE2_OPERATORS_LSTM_UNIT_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/conversions.h"

namespace caffe2 {
namespace detail {
template <typename T>
inline T sigmoid(T x) {
  return 1. / (1. + exp(-x));
}

template <typename T>
inline T host_tanh(T x) {
  return 2. * sigmoid(2. * x) - 1.;
}

template <typename T, typename Context>
void LSTMUnit(
    int N,
    int D,
    int t,
    const T* H_prev,
    const T* C_prev,
    const T* X,
    const int32_t* seqLengths,
    bool drop_states,
    T* C,
    T* H,
    const float forget_bias,
    Context* /*context*/) {
  for (int n = 0; n < N; ++n) {
    const bool valid = seqLengths == nullptr || t < seqLengths[n];

    for (int d = 0; d < D; ++d) {
      if (!valid) {
        if (drop_states) {
          H[d] = 0;
          C[d] = 0;
        } else {
          H[d] = H_prev[d];
          C[d] = C_prev[d];
        }
      } else {
        const T i = sigmoid(X[d]);
        const T f = sigmoid(X[1 * D + d] + convert::To<float, T>(forget_bias));
        const T o = sigmoid(X[2 * D + d]);
        const T g = host_tanh(X[3 * D + d]);
        const T c_prev = C_prev[d];
        const T c = f * c_prev + i * g;
        C[d] = c;
        const T host_tanh_c = host_tanh(c);
        H[d] = o * host_tanh_c;
      }
    }
    H_prev += D;
    C_prev += D;
    X += 4 * D;
    C += D;
    H += D;
  }
}

} // namespace detail

template <typename Context>
class LSTMUnitOp : public Operator<Context> {
 public:
  LSTMUnitOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        forget_bias_(
            static_cast<float>(OperatorBase::template GetSingleArgument<float>(
                "forget_bias",
                0.0))),
        sequence_lengths_(OperatorBase::template GetSingleArgument<bool>(
            "sequence_lengths",
            true)),
        drop_states_(OperatorBase::template GetSingleArgument<bool>(
            "drop_states",
            false)) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  using Operator<Context>::Operator;

  template <typename T>
  bool DoRunWithType() {
    // handle potentially-missing sequence lengths input
    const size_t TIMESTEP = SEQ_LENGTHS + (sequence_lengths_ ? 1 : 0);

    // Extract N
    const auto N = Input(CELL_T_M_1).dim(1);

    // Gates: 1xNxG
    const auto G = Input(GATES).dim(2);
    const auto D = Input(CELL_T_M_1).dim(2);

    CAFFE_ENFORCE_EQ(4 * D, G);
    const auto* H_prev = Input(HIDDEN_T_M_1).template data<T>();
    const auto* C_prev = Input(CELL_T_M_1).template data<T>();
    const auto* X = Input(GATES).template data<T>();

    const int32_t* seqLengths = nullptr;
    if (sequence_lengths_) {
      CAFFE_ENFORCE_EQ(Input(SEQ_LENGTHS).size(), N);
      seqLengths = Input(SEQ_LENGTHS).template data<int32_t>();
    }

    const auto t = static_cast<OperatorBase*>(this)
                       ->Input<Tensor<CPUContext>>(TIMESTEP)
                       .template data<int32_t>()[0];
    Output(CELL_T)->ResizeLike(Input(CELL_T_M_1));
    auto* C = Output(CELL_T)->template mutable_data<T>();
    Output(HIDDEN_T)->ResizeLike(Input(CELL_T_M_1));
    auto* H = Output(HIDDEN_T)->template mutable_data<T>();
    detail::LSTMUnit<T, Context>(
        N,
        D,
        t,
        H_prev,
        C_prev,
        X,
        seqLengths,
        drop_states_,
        C,
        H,
        forget_bias_,
        &context_);
    return true;
  }

  bool RunOnDevice() override {
    return DoRunWithType<float>();
  }

 protected:
  INPUT_TAGS(HIDDEN_T_M_1, CELL_T_M_1, GATES, SEQ_LENGTHS);
  // additional input tags are determined dynamically based on whether
  // sequence_lengths is present.
  OUTPUT_TAGS(HIDDEN_T, CELL_T);

  float forget_bias_;
  bool sequence_lengths_;

 private:
  bool drop_states_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_LSTM_UNIT_OP_H_
