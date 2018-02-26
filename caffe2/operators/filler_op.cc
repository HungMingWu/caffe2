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

#include "caffe2/operators/filler_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(ConstantFill, ConstantFillOp<CPUContext>);

OPERATOR_SCHEMA(ConstantFill)
    .NumInputs(0, 1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .TensorInferenceFunction(FillerTensorInference<>)
    .SetDoc(R"DOC(
The operator fills the elements of the output tensor with a constant value
specified by the 'value' argument.

The data type is specified by the 'dtype' argument. The 'dtype' argument must
be one of the data types specified in the 'DataType' enum field in the
TensorProto message. If the 'dtype' argument is not provided, the data type of
'value' is used.

The output tensor shape is specified by the 'shape' argument. If the number of
input is 1, the shape will be identical to that of the input at run time with
optional additional dimensions appended at the end as specified by 'extra_shape'
argument. In that case the 'shape' argument should not be set.

If input_as_shape is set to true, then the input should be a 1D tensor
containing the desired output shape (the dimensions specified in extra_shape
will also be appended)

NOTE: Currently, it supports data type of float, int32, int64, and bool.
)DOC")
    .Arg("value", "The value for the elements of the output tensor.")
    .Arg(
        "dtype",
        "The data type for the elements of the output tensor."
        "Strictly must be one of the types from DataType enum in TensorProto.")
    .Arg(
        "shape",
        "The shape of the output tensor."
        "Cannot set the shape argument and pass in an input at the same time.")
    .Arg(
        "extra_shape",
        "The additional dimensions appended at the end of the shape indicated"
        "by the input blob."
        "Cannot set the extra_shape argument when there is no input blob.")
    .Arg(
        "input_as_shape",
        "1D tensor containing the desired output shape.  First input must be in CPU context.")
    .Input(0, "input", "Input tensor (optional) to provide shape information.")
    .Output(
        0,
        "output",
        "Output tensor of constant values specified by 'value'"
        "argument and its type is specified by the 'dtype' argument");

}  // namespace caffe2
