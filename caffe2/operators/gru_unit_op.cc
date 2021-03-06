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

#include "gru_unit_op.h"

namespace caffe2 {
REGISTER_CPU_OPERATOR(GRUUnit, GRUUnitOp<float, CPUContext>);
OPERATOR_SCHEMA(GRUUnit)
    .NumInputs(3, 4)
    .NumOutputs(1)
    .SetDoc(R"DOC(
GRUUnit computes the activations of a standard GRU,
in a sequence-length aware fashion.

Concretely, given the (fused) inputs X (TxNxD), the previous hidden
state (NxD), and the sequence lengths (N), computes the GRU
activations, avoiding computation if the input is invalid (as in, the
value at X[t][n] >= seqLengths[n].

)DOC")
    .Arg(
        "drop_states",
        "Bool to determine if hidden state is zeroes or passed "
        "along for timesteps past the given sequence_length.")
    .Arg(
        "sequence_lengths",
        "When false, the sequence lengths input is left out, "
        "and all following inputs are shifted left by one.")
    .Output(0, "hidden", "The new GRU hidden state calculated by this op.");

} // namespace caffe2
