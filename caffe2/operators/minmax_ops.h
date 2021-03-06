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

#ifndef CAFFE2_OPERATORS_MINMAX_OPS_H_
#define CAFFE2_OPERATORS_MINMAX_OPS_H_

#include "caffe2/core/common_omp.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/types.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class MaxMinOpBase : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(MaxMinOpBase)

  bool RunOnDevice() override {
    auto& input0 = Input(0);
    auto* output = Output(0);

    output->ResizeLike(input0);
    output->CopyFrom(input0, &context_);

    if (InputSize() == 1) {
      return true;
    }

    // Dimension checking
    for (int i = 1; i < InputSize(); ++i) {
      CAFFE_ENFORCE_EQ(
          output->dims(),
          Input(i).dims(),
          "Description: Input #",
          i,
          ", input dimension:",
          Input(i).dims(),
          " should match output dimension: ",
          output->dims());
    }

    return this->Compute();
  }

  virtual bool Compute() = 0;
};

template <typename T, class Context>
class MaxOp : public MaxMinOpBase<T, Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  MaxOp(const OperatorDef& operator_def, Workspace* ws)
      : MaxMinOpBase<T, Context>(operator_def, ws) {}
  virtual ~MaxOp() noexcept {}
  bool Compute() override;
};

template <typename T, class Context>
class MinOp : public MaxMinOpBase<T, Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  MinOp(const OperatorDef& operator_def, Workspace* ws)
      : MaxMinOpBase<T, Context>(operator_def, ws) {}
  virtual ~MinOp() noexcept {}
  bool Compute() override;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_MINMAX_OPS_H_
