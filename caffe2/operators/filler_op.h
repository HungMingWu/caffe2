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

#ifndef CAFFE2_OPERATORS_FILLER_OP_H_
#define CAFFE2_OPERATORS_FILLER_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

// FillerOp takes in either zero or one input.
//
// If the number of input is 1, the shape will be identical to that of the input
// at run time with optional additional dimensions appended at the end as
// specified by "extra_shape" argument. In that case the "shape" parameter
// should not be set.
//
// If the number of inputs is 0, the full shape must be provided via "shape"
// argument
template <class Context>
class FillerOp : public Operator<Context> {
 public:
  FillerOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        shape_(ToVectorTIndex(OperatorBase::GetRepeatedArgument<int>("shape"))),
        extra_shape_(ToVectorTIndex(
            OperatorBase::GetRepeatedArgument<int>("extra_shape"))),
        input_as_shape_(
            OperatorBase::GetSingleArgument<bool>("input_as_shape", false)) {
    if (InputSize()) {
      if (shape_.size() != 0) {
        CAFFE_THROW(
            "Cannot set the shape argument and pass in an input at "
            "the same time");
      }
    } else {
      if (!extra_shape_.empty()) {
        CAFFE_THROW("Cannot set extra_shape when there is no input");
      }
      if (input_as_shape_) {
        CAFFE_THROW("An input must be given if input_as_shape is true");
      }
      if (shape_.size() == 0 &&
          OperatorBase::HasSingleArgumentOfType<int>("shape")) {
        CAFFE_THROW("Fill 'shape' argument was a scalar, list expected");
      }
    }
  }

  virtual ~FillerOp() {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    auto* output = Operator<Context>::Output(0);
    if (InputSize()) {
      auto shape = vector<TIndex>{};
      if (input_as_shape_) {
        // Shape input must be in CPU context
        auto& input = OperatorBase::Input<Tensor<CPUContext>>(0);
        CAFFE_ENFORCE_EQ(
            input.ndim(),
            1,
            "When input_as_shape is true, the input must be a 1D tensor of "
            "data type TIndex");
        auto* shape_data = input.template data<TIndex>();
        shape.insert(shape.end(), shape_data, shape_data + input.dim32(0));
      } else {
        auto& input = Input(0);
        shape.insert(shape.end(), input.dims().begin(), input.dims().end());
      }
      shape.insert(shape.end(), extra_shape_.begin(), extra_shape_.end());
      output->Resize(shape);
    } else {
      output->Resize(shape_);
    }
    return Fill(output);
  }

  virtual bool Fill(Tensor<Context>* output) = 0;

 protected:
  vector<TIndex> shape_;
  vector<TIndex> extra_shape_;
  bool input_as_shape_;
};

template <class Context>
class ConstantFillOp final : public FillerOp<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  ConstantFillOp(const OperatorDef& operator_def, Workspace* ws)
      : FillerOp<Context>(operator_def, ws) {
    TensorProto_DataType dtype =
        static_cast<TensorProto_DataType>(OperatorBase::GetSingleArgument<int>(
            "dtype", TensorProto_DataType_FLOAT));

    if (!OperatorBase::HasArgument("dtype") &&
        OperatorBase::HasArgument("value")) {
      // If 'dtype' is not provided, infer type based on the type of 'value'
      // Currently, single argument contains either float, int64 or bytes
      if (OperatorBase::HasSingleArgumentOfType<float>("value")) {
        dtype = TensorProto_DataType_FLOAT;
      } else if (OperatorBase::HasSingleArgumentOfType<int64_t>("value")) {
        dtype = TensorProto_DataType_INT64;
      } else {
        CAFFE_THROW("Argument 'value' is of unexpected type");
      }
      VLOG(1) << "Argument 'dtype' is not provided. Assume the data type is "
              << "the same as that of argument 'value': " << dtype;
    }

    switch (dtype) {
      case TensorProto_DataType_FLOAT:
        body_ = &ConstantFillOp::FillWithType<float>;
        break;
      case TensorProto_DataType_DOUBLE:
        body_ = &ConstantFillOp::FillWithType<double>;
        break;
      case TensorProto_DataType_BOOL:
        body_ = &ConstantFillOp::FillWithType<bool>;
        break;
      case TensorProto_DataType_INT8:
        body_ = &ConstantFillOp::FillWithType<int8_t>;
        break;
      case TensorProto_DataType_INT16:
        body_ = &ConstantFillOp::FillWithType<int16_t>;
        break;
      case TensorProto_DataType_INT32:
        body_ = &ConstantFillOp::FillWithType<int>;
        break;
      case TensorProto_DataType_INT64:
        body_ = &ConstantFillOp::FillWithType<int64_t>;
        break;
      case TensorProto_DataType_UINT8:
        body_ = &ConstantFillOp::FillWithType<uint8_t>;
        break;
      case TensorProto_DataType_UINT16:
        body_ = &ConstantFillOp::FillWithType<uint16_t>;
        break;
      case TensorProto_DataType_STRING:
        body_ = &ConstantFillOp::FillWithString;
        break;
      case TensorProto_DataType_UNDEFINED:
        CAFFE_THROW("ConstantFill op cannot have undefined 'dtype' argument");
      // break;
      default:
        CAFFE_THROW("Unexpected 'dtype' argument value: ", dtype);
    }
  }

  bool Fill(Tensor<Context>* output) override {
    return (this->*body_)(output);
  }

  template <typename T>
  bool FillWithType(Tensor<Context>* output) {
    T value = OperatorBase::GetSingleArgument<T>("value", 0);
    auto* data = output->template mutable_data<T>();
    if (output->size()) {
      math::Set<T, Context>(output->size(), value, data, &context_);
    }
    return true;
  }

  bool FillWithString(Tensor<Context>* output) {
    auto value = OperatorBase::GetSingleArgument<std::string>("value", "");
    auto* data = output->template mutable_data<std::string>();
    for (int i = 0; i < output->size(); ++i) {
      data[i] = value;
    }
    return true;
  }

 private:
  bool (ConstantFillOp::*body_)(Tensor<Context>* output);
};

template <int VALUE_TYPE = TensorProto_DataType_FLOAT>
inline std::vector<TensorShape> FillerTensorInference(
    const OperatorDef& def,
    const vector<TensorShape>& in) {
  vector<TensorShape> out(1);
  ArgumentHelper helper(def);
  out[0].set_data_type(static_cast<TensorProto_DataType>(
      helper.GetSingleArgument<int>("dtype", VALUE_TYPE)));

  if (in.size()) {
    // TODO
    bool input_as_shape =
        helper.GetSingleArgument<bool>("input_as_shape", false);
    if (input_as_shape) {
      out[0].set_unknown_shape(true);
      return out;
    }
    for (int d : in[0].dims()) {
      out[0].add_dims(d);
    }
  } else {
    auto shape = helper.GetRepeatedArgument<int>("shape");
    for (int d : shape) {
      out[0].add_dims(d);
    }
  }
  return out;
}

} // namespace caffe2

#endif // CAFFE2_OPERATORS_FILLER_OP_H_
