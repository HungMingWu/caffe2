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

#include "caffe2/operators/local_response_normalization_op.h"

namespace caffe2 {

template<>
bool LRNOp<float, CPUContext>::RunOnDeviceWithOrderNCHW() {
  // Note(Yangqing): this one is copied from my Caffe implementation.
  auto& X = Input(0);
  auto* Y = Output(0);
  DCHECK_EQ(X.ndim(), 4);
  const int N = X.dim32(0);
  const int C = X.dim32(1);
  const int H = X.dim32(2);
  const int W = X.dim32(3);
  const int image_size = C * H * W;
  const float* Xdata = X.data<float>();
  Y->ResizeLike(X);
  float* Ydata = Y->mutable_data<float>();

  if (OutputSize() > 1) {
    scale_ = Output(1);
  } else {
    if (!scale_) {
      scale_ = &local_scale_tensor_;
    }
  }
  scale_->ResizeLike(X);
  float* scale_data = scale_->mutable_data<float>();
  math::Set<float, CPUContext>(X.size(), bias_, scale_data, &context_);
  TensorCPU padded_square(
      vector<TIndex>{C + size_ - 1, H, W});
  float* padded_square_data = padded_square.mutable_data<float>();
  math::Set<float, CPUContext>(padded_square.size(), 0., padded_square_data,
                               &context_);
  const float alpha_over_size = alpha_ / size_;
  // go through the images
  for (int n = 0; n < N; ++n) {
    // compute the padded square
    math::Sqr<float, CPUContext>(image_size, Xdata + image_size * n,
                                 padded_square_data + pre_pad_ * H * W,
                                 &context_);
    // Create the first channel scale
    for (int c = 0; c < size_; ++c) {
      math::Axpy<float, CPUContext>(
          H * W, alpha_over_size, padded_square_data + c * H * W,
          scale_data + image_size * n, &context_);
    }
    for (int c = 1; c < C; ++c) {
      float* this_scale_slice = scale_data + n * image_size + c * H * W;
      // copy previous scale
      context_.Copy<float, CPUContext, CPUContext>(
          H * W, this_scale_slice - H * W, this_scale_slice);
      // add head
      math::Axpy<float, CPUContext>(
          H * W, alpha_over_size, padded_square_data + (c + size_ - 1) * H * W,
          this_scale_slice, &context_);
      // subtract tail
      math::Axpy<float, CPUContext>(
          H * W, -alpha_over_size, padded_square_data + (c - 1) * H * W,
          this_scale_slice, &context_);
    }
  }
  math::Powx<float, CPUContext>(
      X.size(), scale_data, -beta_, Ydata, &context_);
  math::Mul<float, CPUContext>(X.size(), Ydata, Xdata, Ydata, &context_);
  return true;
}

template<>
bool LRNOp<float, CPUContext>::RunOnDeviceWithOrderNHWC() {
  // Note(Yangqing): This one is copied from my Decaf implementation. How many
  // variants have I written...?
  auto& X = Input(0);
  auto* Y = Output(0);
  DCHECK_EQ(X.ndim(), 4);
  const int N = X.dim32(0);
  const int H = X.dim32(1);
  const int W = X.dim32(2);
  const int C = X.dim32(3);
  const int num_rows = N * H * W;
  const float* Xdata = X.data<float>();
  Y->ResizeLike(X);
  float* Ydata = Y->mutable_data<float>();

  if (OutputSize() > 1) {
    scale_ = Output(1);
  } else {
    if (!scale_) {
      scale_ = &local_scale_tensor_;
    }
  }
  scale_->ResizeLike(X);
  float* scale_data = scale_->mutable_data<float>();

  TensorCPU padded_square(vector<TIndex>(1, C + size_ - 1));
  float* padded_square_data = padded_square.mutable_data<float>();
  math::Set<float, CPUContext>(padded_square.size(), 0., padded_square_data,
                               &context_);
  const float alpha_over_size = alpha_ / size_;

  for (int n = 0; n < num_rows; ++n) {
    for (int c = 0; c < C; ++c) {
      padded_square_data[c + pre_pad_] =
          Xdata[n * C + c] * Xdata[n * C + c] * alpha_over_size;
    }
    float accum_scale = 0.;
    for (int i = 0; i < size_ - 1; ++i) {
      accum_scale += padded_square_data[i];
    }
    for (int c = 0; c < C; ++c) {
      accum_scale += padded_square_data[c + size_ - 1];
      scale_data[n * C + c] = bias_ + accum_scale;
      accum_scale -= padded_square_data[c];
    }
  }
  math::Powx<float, CPUContext>(
      X.size(), scale_data, -beta_, Ydata, &context_);
  math::Mul<float, CPUContext>(X.size(), Ydata, Xdata, Ydata, &context_);
  return true;
}

REGISTER_CPU_OPERATOR(LRN, LRNOp<float, CPUContext>);

OPERATOR_SCHEMA(LRN).NumInputs(1).NumOutputs(1,2);

}  // namespace caffe2
