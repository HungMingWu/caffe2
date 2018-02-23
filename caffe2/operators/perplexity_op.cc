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

#include "caffe2/operators/perplexity_op.h"

namespace caffe2 {

template <>
bool PerplexityOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);

  DCHECK_EQ(X.ndim(), 1);
  int N = X.dim32(0);

  Y->Resize(vector<TIndex>());
  const auto* Xdata = X.data<float>();

  float perplexity = 1.0;
  for (int i = 0; i < N; ++i) {
    perplexity *= pow(Xdata[i], -1.0/N);
  }
  *(Y->mutable_data<float>()) = perplexity;
  return true;
}

REGISTER_CPU_OPERATOR(Perplexity, PerplexityOp<float, CPUContext>);

OPERATOR_SCHEMA(Perplexity).NumInputs(1).NumOutputs(1)
.SetDoc(R"DOC(
Perplexity calculates how well a probability distribution predicts a sample.
Perplexity takes a 1-D tensor containing a batch of probabilities. Each value
in the tensor belongs to a different sample and represents the probability of
the model predicting the true label for that sample. The operator returns a
single (float) perplexity value for the batch.
)DOC")
.Input(0, "probabilities", "The input data as Tensor. It contains a batch of"
       "true label or target probabilities")
.Output(0, "output", "The output- a single (float) perplexity value for the "
        "batch");

}  // namespace caffe2
