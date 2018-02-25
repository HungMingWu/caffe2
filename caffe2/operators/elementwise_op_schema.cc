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

#include "caffe2/operators/elementwise_op.h"
#include "caffe2/utils/proto_utils.h"

namespace caffe2 {

const char* kBroadcastDoc = R"DOC(
If necessary the right-hand-side argument will be broadcasted to match the
shape of left-hand-side argument. When broadcasting is specified, the second
tensor can either be of size 1 (a scalar value), or having its shape as a
contiguous subset of the first tensor's shape. The starting of the mutually
equal shape is specified by the argument "axis", and if it is not set, suffix
matching is assumed. 1-dim expansion doesn't work yet.

For example, the following tensor shapes are supported (with broadcast=1):

  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0

Argument `broadcast=1` needs to be passed to enable broadcasting.
)DOC";

std::function<void(OpSchema&)> MathDocGenerator(const char* name) {
  return [=](OpSchema& schema) {
    string doc = R"DOC(
Performs element-wise binary {name} (with limited broadcast support).
{broadcast_doc})DOC";
    ReplaceAll(doc, "{name}", name);
    ReplaceAll(doc, "{broadcast_doc}", kBroadcastDoc);
    schema.SetDoc(doc);
    schema.Arg("broadcast", "Pass 1 to enable broadcasting");
    schema.Arg(
        "axis",
        "If set, defines the broadcast dimensions. See doc for details.");
    schema.Input(
        0,
        "A",
        "First operand, should share the type with the second operand.");
    schema.Input(
        1,
        "B",
        "Second operand. With broadcasting can be of smaller size than A. "
        "If broadcasting is disabled it should be of the same size.");
    schema.Output(0, "C", "Result, has same dimensions and type as A");
  };
}

OPERATOR_SCHEMA(Add)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{0, 0}, {1, 0}})
    .CostInferenceFunction(PointwiseCostInference<1>)
    .IdenticalTypeAndShapeOfInput(0)
    .FillUsing(MathDocGenerator("addition"));
OPERATOR_SCHEMA(Sub)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{0, 0}, {1, 0}})
    .CostInferenceFunction(PointwiseCostInference<1>)
    .IdenticalTypeAndShapeOfInput(0)
    .FillUsing(MathDocGenerator("subtraction"));
OPERATOR_SCHEMA(Mul)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{0, 0}, {1, 0}})
    .CostInferenceFunction(PointwiseCostInference<1>)
    .IdenticalTypeAndShapeOfInput(0)
    .FillUsing(MathDocGenerator("multiplication"));
OPERATOR_SCHEMA(Div)
    .NumInputs(2)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .CostInferenceFunction(PointwiseCostInference<1>)
    .IdenticalTypeAndShapeOfInput(0)
    .FillUsing(MathDocGenerator("division"));
OPERATOR_SCHEMA(DivGradient).NumInputs(3).NumOutputs(2).AllowInplace({{0, 0}});

std::function<void(OpSchema&)> ComparisonDocGenerator(
    const char* name,
    const char* desc) {
  return [=](OpSchema& schema) {
    string doc = R"DOC(
Performs element-wise {desc} comparison `{name}` (with limited broadcast support).
{broadcast_doc})DOC";
    ReplaceAll(doc, "{name}", name);
    ReplaceAll(doc, "{desc}", desc);
    ReplaceAll(doc, "{broadcast_doc}", kBroadcastDoc);
    schema.SetDoc(doc);
    schema.Arg("broadcast", "Pass 1 to enable broadcasting");
    schema.Arg(
        "axis",
        "If set, defines the broadcast dimensions. See doc for details.");
    schema.Input(
        0,
        "A",
        "First operand, should share the type with the second operand.");
    schema.Input(
        1,
        "B",
        "Second operand. With broadcasting can be of smaller size than A. "
        "If broadcasting is disabled it should be of the same size.");
    schema.Output(0, "C", "Result, has same dimensions and A and type `bool`");
  };
}

#define CAFFE2_SCHEMA_FOR_BINARY_COMPARISON_OP(name, symbol, desc) \
  OPERATOR_SCHEMA(name).NumInputs(2).NumOutputs(1).FillUsing(      \
      ComparisonDocGenerator(symbol, desc)); 

CAFFE2_SCHEMA_FOR_BINARY_COMPARISON_OP(LT, "<", "less than");
CAFFE2_SCHEMA_FOR_BINARY_COMPARISON_OP(LE, "<=", "less or equal than");
CAFFE2_SCHEMA_FOR_BINARY_COMPARISON_OP(GT, ">", "greater than");
CAFFE2_SCHEMA_FOR_BINARY_COMPARISON_OP(GE, ">=", "greater or equal than");
CAFFE2_SCHEMA_FOR_BINARY_COMPARISON_OP(EQ, "==", "equality");

std::function<void(OpSchema&)> LogicalDocGenerator(const char* name) {
  return [=](OpSchema& schema) {
    string doc = R"DOC(
Performs element-wise logical operation `{name}` (with limited broadcast support).
Both input operands should be of type `bool`.
{broadcast_doc})DOC";
    ReplaceAll(doc, "{name}", name);
    ReplaceAll(doc, "{broadcast_doc}", kBroadcastDoc);
    schema.SetDoc(doc);
    schema.Arg("broadcast", "Pass 1 to enable broadcasting");
    schema.Arg(
        "axis",
        "If set, defines the broadcast dimensions. See doc for details.");
    schema.Input(0, "A", "First operand.");
    schema.Input(
        1,
        "B",
        "Second operand. With broadcasting can be of smaller size than A. "
        "If broadcasting is disabled it should be of the same size.");
    schema.Output(0, "C", "Result, has same dimensions and A and type `bool`");
  };
}

#define CAFFE2_SCHEMA_FOR_BINARY_LOGICAL_OP(name, symbol) \
  OPERATOR_SCHEMA(name)                                   \
      .NumInputs(2)                                       \
      .NumOutputs(1)                                      \
      .AllowInplace({{0, 0}})                             \
      .FillUsing(LogicalDocGenerator(symbol)); 

CAFFE2_SCHEMA_FOR_BINARY_LOGICAL_OP(Or, "or");
CAFFE2_SCHEMA_FOR_BINARY_LOGICAL_OP(And, "and");
CAFFE2_SCHEMA_FOR_BINARY_LOGICAL_OP(Xor, "xor");

OPERATOR_SCHEMA(Not)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(Performs element-wise negation.)DOC")
    .Input(0, "X", "Input tensor of type `bool`.")
    .Output(0, "Y", "Output tensor of type `bool`.");

} // namespace caffe2
