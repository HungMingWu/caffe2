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

#ifndef CAFFE2_OPERATORS_GRU_UNIT_OP_H_
#define CAFFE2_OPERATORS_GRU_UNIT_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {
namespace detail {

template <typename T>
inline T sigmoid(T x) {
  return 1.0f / (1.0f + exp(-x));
}

template <typename T>
inline T host_tanh(T x) {
  return 2.0f * sigmoid(2.0f * x) - 1.0f;
}

template <typename T, typename Context>
void GRUUnit(
    int N,
    int D,
    int t,
    const T* H_prev,
    const T* X,
    const int32_t* seqLengths,
    bool drop_states,
    T* H,
    Context* /*context*/) {
  for (int n = 0; n < N; ++n) {
    const bool valid = seqLengths == nullptr || t < seqLengths[n];

    for (int d = 0; d < D; ++d) {
      if (!valid) {
        if (drop_states) {
          H[d] = 0;
        } else {
          H[d] = H_prev[d];
        }
      } else {
        const T update = X[1 * D + d];
        const T output = X[2 * D + d];
        T sigmoid_update = sigmoid(update);
        H[d] = H_prev[d] * sigmoid_update +
            host_tanh(output) * (1.0f - sigmoid_update);
      }
    }

    H_prev += D;
    X += 3 * D;
    H += D;
  }
}

} // namespace detail

template <typename T, typename Context>
class GRUUnitOp : public Operator<Context> {
 public:
  GRUUnitOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        drop_states_(OperatorBase::template GetSingleArgument<bool>(
            "drop_states",
            false)),
        sequence_lengths_(OperatorBase::template GetSingleArgument<bool>(
            "sequence_lengths",
            true)) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // handle potentially-missing sequence lengths input
    const size_t TIMESTEP = SEQ_LENGTHS + (sequence_lengths_ ? 1 : 0);

    // Extract N
    const auto N = Input(HIDDEN_T_M_1).dim(1);

    // Gates: 1xNxG
    const auto G = Input(GATES).dim(2);
    const auto D = Input(HIDDEN_T_M_1).dim(2);

    CAFFE_ENFORCE_EQ(3 * D, G);
    const auto* H_prev = Input(HIDDEN_T_M_1).template data<T>();
    const auto* X = Input(GATES).template data<T>();

    const int32_t* seqLengths = nullptr;
    if (sequence_lengths_) {
      CAFFE_ENFORCE_EQ(Input(SEQ_LENGTHS).size(), N);
      seqLengths = Input(SEQ_LENGTHS).template data<int32_t>();
    }

    const auto t = static_cast<OperatorBase*>(this)->
      Input<Tensor<CPUContext>>(TIMESTEP).template data<int32_t>()[0];
    Output(HIDDEN_T)->ResizeLike(Input(HIDDEN_T_M_1));
    auto* H = Output(HIDDEN_T)->template mutable_data<T>();

    detail::GRUUnit<T, Context>(
        N, D, t, H_prev, X, seqLengths, drop_states_, H, &context_);
    return true;
  }

 protected:
  INPUT_TAGS(HIDDEN_T_M_1, GATES, SEQ_LENGTHS);
  // additional input tags are determined dynamically based on whether
  // sequence_lengths is present.
  OUTPUT_TAGS(HIDDEN_T);

 private:
  bool drop_states_;
  bool sequence_lengths_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_GRU_UNIT_OP_H_
