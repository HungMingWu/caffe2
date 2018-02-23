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

#include "lstm_unit_op.h"

namespace caffe2 {
REGISTER_CPU_OPERATOR(LSTMUnit, LSTMUnitOp<CPUContext>);
OPERATOR_SCHEMA(LSTMUnit)
    .NumInputs(4, 5)
    .NumOutputs(2)
    .SetDoc(R"DOC(
LSTMUnit computes the activations of a standard LSTM (without peephole
connections), in a sequence-length aware fashion.

Concretely, given the (fused) inputs X (TxNxD), the previous cell
state (NxD), and the sequence lengths (N), computes the LSTM
activations, avoiding computation if the input is invalid (as in, the
value at X{t][n] >= seqLengths[n].

)DOC")
    .Arg("forget_bias", "Bias term to add in while calculating forget gate")
    .Arg(
        "sequence_lengths",
        "When false, the sequence lengths input is left out, "
        "and all following inputs are shifted left by one.");
REGISTER_CPU_OPERATOR(LSTMUnitGradient, LSTMUnitGradientOp<CPUContext>);
OPERATOR_SCHEMA(LSTMUnitGradient)
    .NumInputs(8, 9)
    .NumOutputs(3)
    .Arg(
        "sequence_lengths",
        "When false, the sequence lengths input is left out, "
        "and all following inputs are shifted left by one.");

}
