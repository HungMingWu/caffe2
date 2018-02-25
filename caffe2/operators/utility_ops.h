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

#ifndef CAFFE2_OPERATORS_UTILITY_OPS_H_
#define CAFFE2_OPERATORS_UTILITY_OPS_H_

#include <math.h>

#include "caffe2/core/common_omp.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/types.h"
#include "caffe2/utils/math.h"

#include <map>
#include <utility>

namespace caffe2 {

template <class Context>
class SumOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(SumOp);

  template <typename T, typename M>
  bool DoRunWithType() {
    auto& input0 = Input(0);
    auto* output = Output(0);
    if (InputSize() == 1) {
      output->CopyFrom(input0, &context_);
      return true;
    }
    output->ResizeLike(input0);
    T* output_data = output->template mutable_data<T>();
    // Dimension checking
    for (int i = 1; i < InputSize(); ++i) {
      if (output->dims() != Input(i).dims()) {
        CAFFE_THROW(
            "Check failed: output->dims() == Input(i).dims().",
            "Description: Input #",
            i,
            ", input dimension:",
            Input(i).dims(),
            " should match output dimension: ",
            output->dims());
      }
    }

    // Add the first two - works if in-place or not.
    math::Add(
        output->size(),
        input0.template data<T>(),
        Input(1).template data<T>(),
        output_data,
        &context_);
    // Add remaining.
    for (int i = 2; i < InputSize(); ++i) {
      math::Add(
          output->size(),
          output_data,
          Input(i).template data<T>(),
          output_data,
          &context_);
    }
    return true;
  }

  bool RunOnDevice() override {
    if (Input(0).template IsType<float>()) {
      return DoRunWithType<float, float>();
    } else if (Input(0).template IsType<int>()) {
      return DoRunWithType<int, int>();
    } else {
      CAFFE_THROW(
          "Sum operator only supports 32-bit float and ints, but",
          " input was of type ",
          Input(0).meta().name());
    }
  }
};

/**
 * @brief Update slices of the tensor in-place by overriding.
 *
 * Input:
 *   DATA - tensor to be updated
 *   INDICES - 1-D list of indices on the first dimension of X_0 that need to be
 *             updated
 *   SLICES - update slices, has to have shape of len(INDICES) + shape(X_0)[1:]
 *
 * Output:
 *   DATA - has to be exactly the same tensor as the input 0
 *
 * Note: The op pretty much ignores the exact shapes of the input arguments and
 * cares only about sizes. It's done for performance consideration to avoid
 * unnecessary reshapes. Only first dimension of X_0 is important, let's call it
 * N. If M is the total size of X_0 and K is the size of INDICES then X_i is
 * assumed to be of shape K x (M / N) regardless of the real shape.
 *
 * Note: Each update in INDICES is applied independently which means that if
 * duplicated elements are present in INDICES arbitrary one will win.
 *
 * For now really works only on CPU because of INDICES access
 */
template <class Context>
class ScatterAssignOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  virtual ~ScatterAssignOp() {}

  ScatterAssignOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        runners_({{{TensorProto_DataType_INT32, TensorProto_DataType_FLOAT},
                   &ScatterAssignOp::DoRun<int32_t, float>},
                  {{TensorProto_DataType_INT32, TensorProto_DataType_FLOAT16},
                   &ScatterAssignOp::DoRun<int32_t, float16>},
                  {{TensorProto_DataType_INT32, TensorProto_DataType_INT32},
                   &ScatterAssignOp::DoRun<int32_t, int32_t>},
                  {{TensorProto_DataType_INT32, TensorProto_DataType_INT64},
                   &ScatterAssignOp::DoRun<int32_t, int64_t>},
                  {{TensorProto_DataType_INT64, TensorProto_DataType_FLOAT},
                   &ScatterAssignOp::DoRun<int64_t, float>},
                  {{TensorProto_DataType_INT64, TensorProto_DataType_FLOAT16},
                   &ScatterAssignOp::DoRun<int64_t, float16>},
                  {{TensorProto_DataType_INT64, TensorProto_DataType_INT32},
                   &ScatterAssignOp::DoRun<int64_t, int32_t>},
                  {{TensorProto_DataType_INT64, TensorProto_DataType_INT64},
                   &ScatterAssignOp::DoRun<int64_t, int64_t>}}) {}

  bool RunOnDevice() override {
    const auto& data = Input(DATA);
    const auto& slices = Input(SLICES);
    auto& indices = Input(INDICES);

    const auto dataType = TypeMetaToDataType(data.meta());
    const auto slicesType = TypeMetaToDataType(slices.meta());
    const auto indicesType = TypeMetaToDataType(indices.meta());
    auto* output = Output(0);

    auto runner = GetRunner(dataType, slicesType, indicesType);
    (this->*runner)();
    return true;
  }

 private:
  typedef void (ScatterAssignOp::*RunnerType)();
  typedef std::
      map<std::pair<TensorProto_DataType, TensorProto_DataType>, RunnerType>
          RunnerMap;

  RunnerMap runners_;

  RunnerType GetRunner(
      const TensorProto_DataType dataType,
      const TensorProto_DataType slicesType,
      const TensorProto_DataType indicesType) {
    CAFFE_ENFORCE_EQ(dataType, slicesType, "Data and slice types must match");
    auto it = runners_.find({indicesType, dataType});
    CAFFE_ENFORCE(
        it != runners_.end(),
        "Could not find the runner corresponding to indicesType, dataType = ",
        indicesType,
        " ",
        dataType);
    return it->second;
  }

  template <typename Index, typename T>
  void DoRun() {
    auto& input = Input(DATA);
    auto& indices = Input(INDICES);
    auto& slices = Input(SLICES);
    auto* output = Output(0);
    CAFFE_ENFORCE_EQ(&input, output, "In place operation is required");

    CAFFE_ENFORCE_GT(input.ndim(), 0, "X0 has to be at least the vector");
    TIndex M = input.size();
    TIndex N = input.dim(0);
    TIndex K = indices.size();
    TIndex block_size = M / N;
    CAFFE_ENFORCE_EQ(slices.size(), block_size * K);
    // TODO(dzhulgakov): it can be made to work with arbitrary data type by
    // using raw_mutable_data
    T* data = output->template mutable_data<T>();
    const Index* idxs = indices.template data<Index>();
    const T* slicesData = slices.template data<T>();
    DoScatterAssign(data, idxs, slicesData, N, K, block_size);
  }

  template <typename Index, typename T>
  void DoScatterAssign(
      T* data,
      const Index* idxs,
      const T* slicesData,
      TIndex N,
      TIndex K,
      TIndex block_size) {
    for (int i = 0; i < K; ++i) {
      Index idx = idxs[i];
      // double-checking the indices, but it's fine as it's DCHECK only
      DCHECK(0 <= idx && idx < N) << "Index out of bounds: " << idx
                                  << ", range 0 to " << N;
      context_.template Copy<T, Context, Context>(
          block_size, slicesData + block_size * i, data + block_size * idx);
    }
  }

  INPUT_TAGS(DATA, INDICES, SLICES);
};

template <class Context>
class GatherOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(GatherOp);

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, OperatorBase::Input<TensorCPU>(INDICES));
  }

  template <typename Index>
  bool DoRunWithType() {
    // If we endup using it on GPU doing O(N) memcpy is probably not best :)
    // TODO: implement prefetching if it starts mattering (TF does it)
    auto& data = Input(DATA);
    auto& indices = Input(INDICES);
    auto* output = Output(0);

    CAFFE_ENFORCE_GE(data.ndim(), 1, "DATA should be at least 1-D");
    auto shape = indices.dims();
    shape.insert(shape.end(), data.dims().begin() + 1, data.dims().end());
    output->Resize(shape);

    int block_size = data.size_from_dim(1);
    auto block_bytesize = data.size_from_dim(1) * data.meta().itemsize();
    int N = indices.size();

    auto src_base = static_cast<const char*>(data.raw_data());
    const Index* idxs = indices.template data<Index>();
    auto out = static_cast<char*>(output->raw_mutable_data(data.meta()));

    for (int i = 0; i < N; ++i) {
      auto idx = idxs[i];
      CAFFE_ENFORCE(
          0 <= idx && idx < data.dim(0),
          "INDICES element is out of DATA bounds, id=",
          idx,
          " data_dim=",
          data.dim(0));
      auto src = src_base + idx * block_bytesize;
      context_.template CopyItems<Context, Context>(
          data.meta(), block_size, src, out + block_bytesize * i);
    }
    return true;
  }

  INPUT_TAGS(DATA, INDICES);
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_UTILITY_OPS_H_
