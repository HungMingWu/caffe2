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

#include "caffe2/operators/channel_stats_op.h"

namespace caffe2 {

template <>
bool ChannelStatsOp<CPUContext>::RunOnDevice() {
  const auto& X = Input(INPUT);
  CAFFE_ENFORCE(X.ndim() >= 3 && X.ndim() <= 5);
  const int N = X.dim32(0);
  const int C = X.dim32(1);
  const int H = X.dim32(2);
  const int W = X.ndim() > 3 ? X.dim32(3) : 1;
  const int D = X.ndim() > 4 ? X.dim32(4) : 1;

  const int sampleSize = H * W * D;

  Output(SUM)->Resize(C);
  Output(SUMSQ)->Resize(C);
  EigenVectorArrayMap<float> sum(Output(SUM)->mutable_data<float>(), C);
  EigenVectorArrayMap<float> sumsq(Output(SUMSQ)->mutable_data<float>(), C);

  sum.setZero();
  sumsq.setZero();
  ConstEigenArrayMap<float> X_arr(X.data<float>(), sampleSize, N * C);
  auto index = 0;
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      sum(c) += X_arr.col(index).sum();
      sumsq(c) += X_arr.col(index).matrix().squaredNorm();
      index++;
    }
  }

  return true;
}

REGISTER_CPU_OPERATOR(ChannelStats, ChannelStatsOp<CPUContext>);

OPERATOR_SCHEMA(ChannelStats)
    .NumInputs(1)
    .NumOutputs(2)
    .SetDoc(R"DOC(
Given an input tensor in NCHW format, computes the sum of all elements per
channel and the sum of all elements squared per channel. These values can be
reduced across multiple batches and used to obtain the mean and variance across
the full set of batches. Using the new mean and variance as input to SpatialBN
has the effect of changing the batch size over which SpatialBN is applied.
)DOC")

    .Input(0, "X", "The input 4-dimensional tensor of shape NCHW")
    .Output(
        0,
        "sum",
        "The output 1-dimensional tensor of size C containing the sum of "
        "elements of X per channel.")
    .Output(
        1,
        "sumsq",
        "The output 1-dimensional tensor of size C containing the sum of "
        "elements squared per channel.");
} // namespace caffe2
