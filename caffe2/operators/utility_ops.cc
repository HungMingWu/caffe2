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

#include "caffe2/operators/utility_ops.h"

#include <cmath>

namespace caffe2 {

REGISTER_CPU_OPERATOR(ScatterAssign, ScatterAssignOp<CPUContext>);
REGISTER_CPU_OPERATOR(Gather, GatherOp<CPUContext>);

OPERATOR_SCHEMA(ScatterAssign)
    .NumInputs(3)
    .NumOutputs(1)
    .EnforceInplace({{0, 0}})
    .SetDoc(R"DOC(
Update slices of the tensor in-place by overriding current value.

Note: The op pretty much ignores the exact shapes of the input arguments and
cares only about sizes. It's done for performance consideration to avoid
unnecessary reshapes. Only first dimension of X_0 is important, let's call it
N. If M is the total size of X_0 and K is the size of INDICES then X_i is
assumed to be of shape K x (M / N) regardless of the real shape.

Note: Each update in INDICES is applied independently which means that if
duplicated elements are present in INDICES arbitrary one will win.

Currently only works on CPU because of access to INDICES.
)DOC")
    .Input(0, "DATA", "Tensor to be updated.")
    .Input(
        1,
        "INDICES",
        "1-D list of indices on the first dimension"
        "of X_0 that need to be updated")
    .Input(
        2,
        "SLICES",
        "Update slices, with shape len(INDICES) + shape(X_0)[1:]")
    .Output(0, "DATA", "Has to be exactly the same tensor as the input 0");

OPERATOR_SCHEMA(Gather)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Given DATA tensor of rank r >= 1, and INDICES tensor of rank q, gather
entries of the outer-most dimension of DATA indexed by INDICES, and concatenate
them in an output tensor of rank q + (r - 1).

Example:
  DATA  = [
      [1.0, 1.2],
      [2.3, 3.4],
      [4.5, 5.7],
  ]
  INDICES = [
      [0, 1],
      [1, 2],
  ]
  OUTPUT = [
      [
          [1.0, 1.2],
          [2.3, 3.4],
      ],
      [
          [2.3, 3.4],
          [4.5, 5.7],
      ],
  ]
)DOC")
    .Input(0, "DATA", "Tensor of rank r >= 1.")
    .Input(1, "INDICES", "Tensor of int32/int64 indices, of any rank q.")
    .Output(0, "OUTPUT", "Tensor of rank q + (r - 1).")
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out(1);
      for (auto d : in[1].dims()) {
        out[0].add_dims(d);
      }
      for (int i = 1; i < in[0].dims_size(); ++i) {
        out[0].add_dims(in[0].dims(i));
      }
      out[0].set_data_type(in[0].data_type());
      return out;
    });

} // namespace caffe2
