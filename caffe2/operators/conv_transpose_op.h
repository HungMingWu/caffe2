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

#ifndef CAFFE2_OPERATORS_CONV_TRANSPOSE_OP_H_
#define CAFFE2_OPERATORS_CONV_TRANSPOSE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/conv_transpose_unpool_op_base.h"

namespace caffe2 {

template <typename T, class Context>
class ConvTransposeOp final : public ConvTransposeUnpoolBase<Context> {
 public:
  USE_CONV_TRANSPOSE_UNPOOL_BASE_FUNCTIONS(Context);
  ConvTransposeOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvTransposeUnpoolBase<Context>(operator_def, ws) {}

  bool RunOnDeviceWithOrderNCHW() override;
  bool RunOnDeviceWithOrderNHWC() override;

 private:
  Tensor<Context> col_buffer_;
  Tensor<Context> bias_multiplier_;
  // Input: X, W, b
  // Output: Y
  INPUT_TAGS(INPUT, FILTER, BIAS);
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_CONV_TRANSPOSE_OP_H_
