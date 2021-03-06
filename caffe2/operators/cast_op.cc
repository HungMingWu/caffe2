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

#include "caffe2/operators/cast_op.h"

namespace caffe2 {

template <>
template <typename DstType, typename SrcType>
bool CastOp<CPUContext>::DoRunWithType() {
  auto& input = Input(0);
  auto* output = Output(0);
  output->ResizeLike(input);
  const auto* data = input.template data<SrcType>();
  auto* out = output->template mutable_data<DstType>();
  auto N = input.size();
  for (TIndex i = 0; i < N; ++i) {
    out[i] = static_cast<DstType>(data[i]);
  }
  return true;
}

template <>
void CastOp<CPUContext>::SetBody(TensorProto_DataType to) {
  switch (to) {
    case TensorProto_DataType_FLOAT:
      // body_ = &CastOp::DoRunIncFp16WithDstType<float>;
      body_ = &CastOp<CPUContext>::DoRunWithDstType<float>;
      break;
    case TensorProto_DataType_INT32:
      body_ = &CastOp<CPUContext>::DoRunWithDstType<int>;
      break;
    case TensorProto_DataType_BYTE:
      LOG(FATAL) << "BYTE is deprecated";
      break;
    case TensorProto_DataType_STRING:
      CAFFE_THROW("Casting to and from strings is not supported yet");
      // break;
    case TensorProto_DataType_BOOL:
      body_ = &CastOp<CPUContext>::DoRunWithDstType<bool>;
      break;
    case TensorProto_DataType_UINT8:
      body_ = &CastOp<CPUContext>::DoRunWithDstType<uint8_t>;
      break;
    case TensorProto_DataType_INT8:
      body_ = &CastOp<CPUContext>::DoRunWithDstType<int8_t>;
      break;
    case TensorProto_DataType_UINT16:
      body_ = &CastOp<CPUContext>::DoRunWithDstType<uint16_t>;
      break;
    case TensorProto_DataType_INT16:
      body_ = &CastOp<CPUContext>::DoRunWithDstType<int16_t>;
      break;
    case TensorProto_DataType_INT64:
      body_ = &CastOp<CPUContext>::DoRunWithDstType<int64_t>;
      break;
    case TensorProto_DataType_FLOAT16:
      CAFFE_THROW("Casting to and from float16 on CPU is not supported yet");
      // break;
    case TensorProto_DataType_DOUBLE:
      //body_ = &CastOp::DoRunIncFp16WithDstType<double>;
      body_ = &CastOp<CPUContext>::DoRunWithDstType<double>;
      break;
    case TensorProto_DataType_UNDEFINED:
      CAFFE_THROW("Cast op must have 'to' argument of type DataType");
      // break;
    default:
      CAFFE_THROW("Unexpected 'to' argument value: ", to);
  }
}

template <>
template <typename DstType>
bool CastOp<CPUContext>::DoRunWithDstType() {
  return DispatchHelper<
      TensorTypes<
          float,
          int32_t,
          bool,
          uint8_t,
          int8_t,
          uint16_t,
          int16_t,
          int64_t,
          double>,
      DstType>::call(this, Input(0));
}

REGISTER_CPU_OPERATOR(Cast, CastOp<CPUContext>);

OPERATOR_SCHEMA(Cast)
    .NumInputs(1)
    .NumOutputs(1)
    .TensorInferenceFunction(
        [](const OperatorDef& def, const vector<TensorShape>& in) {
          ArgumentHelper helper(def);
          vector<TensorShape> out;
          out.push_back(in[0]);
          out[0].set_data_type(cast::GetCastDataType(helper, "to"));
          return out;
        })
    .SetDoc(R"DOC(
The operator casts the elements of a given input tensor to a data type
specified by the 'to' argument and returns an output tensor of the same size in
the converted type. The 'to' argument must be one of the data types specified
in the 'DataType' enum field in the TensorProto message. If the 'to' argument
is not provided or is not one of the enumerated types in DataType, Caffe2
throws an Enforce error.

NOTE: Casting to and from strings is not supported yet.
)DOC")
    .Arg(
        "to",
        "The data type to which the elements of the input tensor are cast."
        "Strictly must be one of the types from DataType enum in TensorProto")
    .Input(0, "input", "Input tensor to be cast.")
    .Output(
        0,
        "output",
        "Output tensor with the same shape as input with type "
        "specified by the 'to' argument");






}  // namespace caffe2
