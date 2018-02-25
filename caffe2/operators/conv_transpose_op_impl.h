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

// conv_transpose_op_impl.h is the templated implementation of the
// conv_transpose_op.h file.
#ifndef CAFFE2_OPERATORS_CONV_TRANSPOSE_OP_IMPL_H_
#define CAFFE2_OPERATORS_CONV_TRANSPOSE_OP_IMPL_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/conv_op_shared.h"
#include "caffe2/operators/conv_transpose_op.h"
#include "caffe2/operators/conv_transpose_unpool_op_base.h"
#include "caffe2/utils/math.h"

CAFFE2_DECLARE_bool(caffe2_force_shared_col_buffer);

namespace caffe2 {

template <typename T, class Context>
bool ConvTransposeOp<T, Context>::RunOnDeviceWithOrderNCHW() {
  const Tensor<Context>& X = Input(INPUT);
  auto& filter = Input(FILTER);
  Tensor<Context>* Y = Output(0);
  const int N = X.dim32(0), M = X.dim32(1), H = X.dim32(2), W = X.dim32(3);
  CAFFE_ENFORCE(filter.ndim() == 4, "filter must be 4D tensor");
  CAFFE_ENFORCE(
      filter.dim32(0) == M,
      "filter number must be equal to input channel number");
  const int C = filter.dim32(1);
  CAFFE_ENFORCE(
      filter.dim32(2) == this->kernel_h(),
      "filter height must be equal to kernel height");
  CAFFE_ENFORCE(
      filter.dim32(3) == this->kernel_w(),
      "filter width must be equal to kernel width");
  ConvTransposeUnpoolBase<Context>::SetOutputSize(X, Y, C);

  const int kernel_dim = C * this->kernel_h() * this->kernel_w();
  const int input_image_size = H * W;
  const int output_image_size = Y->dim32(2) * Y->dim32(3);

#ifndef __ARM_NEON__
  if (InputSize() == 3) {
    auto& bias = Input(BIAS);
    CAFFE_ENFORCE(bias.ndim() == 1, "bias must be 1D tensor");
    CAFFE_ENFORCE(
        bias.dim32(0) == C,
        "bias dimension must be equal to output channel number");
    if (bias_multiplier_.size() != output_image_size) {
      bias_multiplier_.Resize(vector<TIndex>(1, output_image_size));
      T* bm_data = bias_multiplier_.template mutable_data<T>();
      math::Set<T, Context>(
          output_image_size,
          static_cast<T>(1),
          bm_data,
          &context_);
    }
  }
#endif // !__ARM_NEON__

  const T* Xdata = X.template data<T>();
  const T* filter_data = filter.template data<T>();
  T* Ydata = Y->template mutable_data<T>();

  auto f = [&](Tensor<Context>* col_buffer) {
    col_buffer->Resize(
        vector<TIndex>{C, this->kernel_h(), this->kernel_w(), H, W});
    T* col_buffer_data = col_buffer->template mutable_data<T>();
    for (auto image_id = 0; image_id < N; ++image_id) {
      // Weight term
      math::Gemm<T, Context>(
          CblasTrans,
          CblasNoTrans,
          kernel_dim,
          input_image_size,
          M,
          1,
          filter_data,
          Xdata,
          0,
          col_buffer_data,
          &context_);

      // Col2im
      math::Col2im<T, Context, StorageOrder::NCHW>(
          col_buffer_data,
          C,
          Y->dim32(2),
          Y->dim32(3),
          this->kernel_h(),
          this->kernel_w(),
          1,
          1,
          this->pad_t(),
          this->pad_l(),
          this->pad_b(),
          this->pad_r(),
          this->stride_h(),
          this->stride_w(),
          Ydata,
          &context_);

      // Bias term
      if (InputSize() == 3) {
        const T* bias_data = Input(BIAS).template data<T>();
#ifndef __ARM_NEON__
        const T* bm_data = bias_multiplier_.template data<T>();
        math::Gemm<T, Context>(
            CblasNoTrans,
            CblasNoTrans,
            C,
            output_image_size,
            1,
            1,
            bias_data,
            bm_data,
            1,
            Ydata,
            &context_);
#else
        math::BiasCHW<T, Context>(
            bias_data,
            C,
            output_image_size,
            Ydata,
            &context_);
#endif // !__ARM_NEON__
      }

      Xdata += M * H * W;
      Ydata += Y->size() / Y->dim32(0);
    }
  };
  if (FLAGS_caffe2_force_shared_col_buffer || shared_buffer_) {
    runWithSharedBuffer<Context>(ws_, f);
  } else {
    f(&col_buffer_);
  }
  return true;
}

template <typename T, class Context>
bool ConvTransposeOp<T, Context>::RunOnDeviceWithOrderNHWC() {
  const Tensor<Context>& X = Input(INPUT);
  auto& filter = Input(FILTER);
  Tensor<Context>* Y = Output(0);
  const auto N = X.dim32(0), H = X.dim32(1), W = X.dim32(2), M = X.dim32(3);
  CAFFE_ENFORCE(filter.ndim() == 4, "filter must be 4D tensor");
  CAFFE_ENFORCE(
      filter.dim32(0) == M,
      "filter number must be equal to input channel number");
  CAFFE_ENFORCE(
      filter.dim32(1) == this->kernel_h(),
      "filter height must be equal to kernel height");
  CAFFE_ENFORCE(
      filter.dim32(2) == this->kernel_w(),
      "filter width must be equal to kernel width");
  const int C = filter.dim32(3);
  ConvTransposeUnpoolBase<Context>::SetOutputSize(X, Y, C);

  const auto kernel_dim = C * this->kernel_h() * this->kernel_w();
  const auto input_image_size = H * W;
  const auto output_image_size = Y->dim32(1) * Y->dim32(2);

  if (InputSize() == 3) {
    auto& bias = Input(BIAS);
    CAFFE_ENFORCE(bias.ndim() == 1, "bias must be 1D tensor");
    CAFFE_ENFORCE(
        bias.dim32(0) == C,
        "bias dimension must be equal to output channel number");
    if (bias_multiplier_.size() != output_image_size) {
      bias_multiplier_.Resize(vector<TIndex>(1, output_image_size));
      T* bm_data = bias_multiplier_.template mutable_data<T>();
      math::Set<T, Context>(
          output_image_size,
          static_cast<T>(1),
          bm_data,
          &context_);
    }
  }
  const T* Xdata = X.template data<T>();
  const T* filter_data = filter.template data<T>();
  T* Ydata = Y->template mutable_data<T>();

  auto f = [&](Tensor<Context>* /*col_buffer*/) {
    col_buffer_.Resize(
        vector<TIndex>{H, W, this->kernel_h(), this->kernel_w(), C});
    T* col_buffer_data = col_buffer_.template mutable_data<T>();
    for (auto image_id = 0; image_id < N; ++image_id) {
      // Weight term
      math::Gemm<T, Context>(
          CblasNoTrans,
          CblasNoTrans,
          input_image_size,
          kernel_dim,
          M,
          1,
          Xdata,
          filter_data,
          0,
          col_buffer_data,
          &context_);
      // Col2im
      math::Col2im<T, Context, StorageOrder::NHWC>(
          col_buffer_data,
          C,
          Y->dim32(1),
          Y->dim32(2),
          this->kernel_h(),
          this->kernel_w(),
          1,
          1,
          this->pad_t(),
          this->pad_l(),
          this->pad_b(),
          this->pad_r(),
          this->stride_h(),
          this->stride_w(),
          Ydata,
          &context_);
      // Bias term
      if (InputSize() == 3) {
        const T* bm_data = bias_multiplier_.template data<T>();
        const T* bias_data = Input(BIAS).template data<T>();
        math::Gemm<T, Context>(
            CblasNoTrans,
            CblasNoTrans,
            output_image_size,
            C,
            1,
            1,
            bm_data,
            bias_data,
            1,
            Ydata,
            &context_);
      }
      Xdata += M * H * W;
      Ydata += Y->size() / Y->dim32(0);
    }
  };
  if (FLAGS_caffe2_force_shared_col_buffer || shared_buffer_) {
    runWithSharedBuffer<Context>(ws_, f);
  } else {
    f(&col_buffer_);
  }
  return true;
}

} // namespace caffe2
#endif // CAFFE2_OPERATORS_CONV_TRANSPOSE_OP_IMPL_H_
