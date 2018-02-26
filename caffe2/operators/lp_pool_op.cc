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

// TODO: reduce the apparent redundancy of all the code below.
#include "caffe2/operators/pool_op.h"

namespace caffe2 {

using std::min;
using std::max;

class LpPool {};

template <>
bool PoolOp<float, CPUContext, LpPool>::RunOnDeviceWithOrderNCHW() {
  auto& X = Input(0);
  auto* Y = Output(0);
  ConvPoolOpBase::SetOutputSize(X, Y, X.dim32(1));
  const auto p = OperatorBase::GetSingleArgument<float>("p", 2.0);
  const auto inv_p = 1.0 / p;

  const float* Xdata = X.data<float>();
  float* Ydata = Y->mutable_data<float>();
  math::Set<float, CPUContext>(Y->size(), 0, Ydata, &context_);
  // The main loop
  int channels = X.dim32(1);
  int height = X.dim32(2);
  int width = X.dim32(3);
  int pooled_height = Y->dim32(2);
  int pooled_width = Y->dim32(3);

  for (int n = 0; n < X.dim32(0); ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
          int hstart = ph * stride_[0] - pads_[0];
          int wstart = pw * stride_[1] - pads_[1];
          int hend = min(hstart + kernel_[0], height);
          int wend = min(wstart + kernel_[1], width);
          hstart = max(hstart, 0);
          wstart = max(wstart, 0);
          const int pool_index = ph * pooled_width + pw;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              const int input_index = h * width + w;
              Ydata[pool_index] += std::pow(std::abs(Xdata[input_index]), p);
            }
          }
          Ydata[pool_index] = std::pow(Ydata[pool_index], inv_p);
        }
      }
      // Do offset.
      Xdata += height * width;
      Ydata += pooled_height * pooled_width;
    }
  }
  return true;
}

template <>
bool PoolOp<float, CPUContext, LpPool>::RunOnDeviceWithOrderNHWC() {
  auto& X = Input(0);
  auto* Y = Output(0);
  int height = X.dim32(1);
  int width = X.dim32(2);
  int channels = X.dim32(3);
  ConvPoolOpBase::SetOutputSize(X, Y, channels);

  const auto p = OperatorBase::GetSingleArgument<float>("p", 2.0);
  const auto inv_p = 1.0 / p;

  const float* Xdata = X.data<float>();
  float* Ydata = Y->mutable_data<float>();
  math::Set<float, CPUContext>(Y->size(), 0, Ydata, &context_);
  // The main loop
  int pooled_height = Y->dim32(1);
  int pooled_width = Y->dim32(2);
  for (int n = 0; n < X.dim32(0); ++n) {
    for (int ph = 0; ph < pooled_height; ++ph) {
      for (int pw = 0; pw < pooled_width; ++pw) {
        int hstart = ph * stride_[0] - pads_[0];
        int wstart = pw * stride_[1] - pads_[1];
        int hend = min(hstart + kernel_[0], height);
        int wend = min(wstart + kernel_[1], width);
        hstart = max(hstart, 0);
        wstart = max(wstart, 0);
        const int pool_index = (ph * pooled_width + pw) * channels;
        for (int h = hstart; h < hend; ++h) {
          for (int w = wstart; w < wend; ++w) {
            const int input_index = (h * width + w) * channels;
            for (int c = 0; c < channels; ++c) {
              Ydata[pool_index + c] +=
                  std::pow(std::abs(Xdata[input_index + c]), p);
            }
          }
        }
        for (int c = 0; c < channels; ++c) {
          Ydata[pool_index + c] = std::pow(Ydata[pool_index + c], inv_p);
        }
      }
    }
    // Do offset.
    Xdata += X.size() / X.dim32(0);
    Ydata += Y->size() / Y->dim32(0);
  }
  return true;
}

REGISTER_CPU_OPERATOR(LpPool, PoolOp<float, CPUContext, LpPool>);

OPERATOR_SCHEMA(LpPool)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
LpPool consumes an input blob X and applies L-p pooling across the
the blob according to kernel sizes, stride sizes, and pad lengths defined by the
ConvPoolOpBase operator. L-p pooling consisting of taking the L-p norm of a
subset of the input tensor according to the kernel size and downsampling the
data into the output blob Y for further processing.
)DOC")
    .Input(
        0,
        "X",
        "Input data tensor from the previous operator; dimensions "
        "depend on whether the NCHW or NHWC operators are being used. For example, "
        "in the former, the input has size (N x C x H x W), where N is the batch "
        "size, C is the number of channels, and H and W are the height and the width "
        "of the data. The corresponding permutation of dimensions is used in the "
        "latter case. ")
    .Output(
        0,
        "Y",
        "Output data tensor from L-p pooling across the input "
        "tensor. Dimensions will vary based on various kernel, stride, and pad "
        "sizes.");


}
