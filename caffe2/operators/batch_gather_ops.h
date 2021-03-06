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

#ifndef CAFFE2_OPERATORS_BATCH_GATHER_OPS_H_
#define CAFFE2_OPERATORS_BATCH_GATHER_OPS_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
class BatchGatherOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(BatchGatherOp)

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, OperatorBase::Input<TensorCPU>(INDICES));
  }

  template <typename TInd>
  bool DoRunWithType() {
    auto& data = Input(DATA);
    auto& indices = Input(INDICES);
    auto* output = Output(0);

    CAFFE_ENFORCE_GE(data.ndim(), 2, "DATA should be at least 2-D");

    vector<TIndex> shape;
    shape.push_back(data.dim(0));
    shape.insert(shape.end(), indices.dims().begin(), indices.dims().end());
    shape.insert(shape.end(), data.dims().begin() + 2, data.dims().end());
    output->Resize(shape);

    auto block_size = data.size_from_dim(2);
    auto block_bytesize = block_size * data.meta().itemsize();
    auto N = indices.size();
    auto data_batch_bytesize = data.size_from_dim(1) * data.meta().itemsize();
    auto gathered_batch_bytesize =
        N * data.size_from_dim(2) * data.meta().itemsize();
    const TInd* idxs = indices.template data<TInd>();
    auto src_base = static_cast<const char*>(data.raw_data());
    auto out = static_cast<char*>(output->raw_mutable_data(data.meta()));

    for (auto batch = 0; batch < data.dim(0); ++batch) {
      for (auto i = 0; i < N; ++i) {
        auto idx = idxs[i];
        CAFFE_ENFORCE(
            0 <= idx && idx < data.dim(1),
            "INDICES element is out of DATA bounds, id=",
            idx,
            " data_dim=",
            data.dim(1));
        auto src =
            src_base + idx * block_bytesize + batch * data_batch_bytesize;
        auto dst = out + i * block_bytesize + batch * gathered_batch_bytesize;
        context_.template CopyItems<Context, Context>(
            data.meta(), block_size, src, dst);
      }
    }
    return true;
  }

  INPUT_TAGS(DATA, INDICES);
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_BATCH_GATHER_OPS_H_
