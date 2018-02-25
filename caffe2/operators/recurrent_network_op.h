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

#ifndef CAFFE2_OPERATORS_RECURRENT_NETWORK_OP_H_
#define CAFFE2_OPERATORS_RECURRENT_NETWORK_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"
#include "caffe2/operators/recurrent_network_executor.h"
#include "caffe2/utils/conversions.h"
#include "caffe2/utils/math.h"

CAFFE2_DECLARE_bool(caffe2_rnn_executor);

namespace caffe2 {
namespace detail {

struct Param {
  std::string param;
  std::string grad;
  std::string cellGradient;
};

struct RecurrentInput {
  std::string state;
  std::string input;
};

struct RecurrentGradient {
  std::string param;
  std::string grad;
  std::string externalGrad;
  std::string lastExternalGrad;
  int32_t offset;
};

struct OffsetAlias {
  std::string src;
  std::string dst;
  int32_t offset{0};
};

struct Link {
  std::string internal;
  std::string external;
  int32_t offset{0};
  int32_t window{1};
};

struct ScratchWorkspaces {
  std::vector<std::shared_ptr<Workspace>> stepWorkspaces;
  std::shared_ptr<Workspace> sharedBlobsWs = nullptr;
};

inline void UpdateTimestepBlob(Workspace* ws, std::string blob_name, int t) {
  ws->CreateBlob(blob_name)->GetMutable<TensorCPU>()->Resize(1);
  auto timestepBlob = ws->GetBlob(blob_name);
  CAFFE_ENFORCE(timestepBlob);
  timestepBlob->GetMutable<TensorCPU>()->mutable_data<int32_t>()[0] = t;
}

std::map<string, string> GetRecurrentMapping(
  const std::vector<detail::Link>& links, bool backward);

template <typename T, typename Context>
void applyOffsetAlias(
    const OffsetAlias& oc,
    Workspace* ws,
    Context* /*context*/) {
  VLOG(1) << "Aliasing: " << oc.src << " to: " << oc.dst
          << " at offset: " << oc.offset;
  auto srcBlob = ws->GetBlob(oc.src);
  CAFFE_ENFORCE(srcBlob);
  auto* src = srcBlob->template GetMutable<Tensor<Context>>();
  auto* dst = ws->GetBlob(oc.dst)->template GetMutable<Tensor<Context>>();
  auto timestep = src->size() / src->dim(0);
  auto dims = src->dims();
  const int32_t startDstTimestep =
      oc.offset >= 0 ? oc.offset : src->dim(0) + oc.offset;
  const int32_t numDstTimesteps = src->dim(0) - startDstTimestep;
  CAFFE_ENFORCE(
      numDstTimesteps >= 1, "Invalid number of timesteps: ", numDstTimesteps);
  dims[0] = numDstTimesteps;
  dst->Resize(dims);
  CAFFE_ENFORCE(timestep == dst->size() / numDstTimesteps, "Invalid offset");
  dst->ShareExternalPointer(
      src->template mutable_data<T>() + startDstTimestep * timestep,
      dst->size());
}

template <typename T, class Context>
void repeatCopy(
    size_t repeat_n,
    size_t n,
    const T* src,
    T* dst,
    Context* context) {
  for (int i = 0; i < repeat_n; ++i) {
    context->template Copy<T, Context, Context>(n, src, dst + i * n);
  }
}

/**
 * Copy external input to the step net into the first item of
 * (T + 1) X batch_size X input_size tensor
 */
template <typename T, typename Context>
void initializeRecurrentInput(
    const RecurrentInput& rc,
    int32_t seqLen,
    int32_t batchSize,
    Workspace* ws,
    Context* context) {
  auto stateBlob = ws->GetBlob(rc.state);
  CAFFE_ENFORCE(stateBlob);
  auto* state = stateBlob->template GetMutable<Tensor<Context>>();

  auto inputBlob = ws->GetBlob(rc.input);
  CAFFE_ENFORCE(inputBlob);
  const auto& input = inputBlob->template Get<Tensor<Context>>();
  CAFFE_ENFORCE_GE(input.ndim(), 1, rc.input);
  CAFFE_ENFORCE_LE(input.ndim(), 3, rc.input);

  const auto stateSize = input.dim(input.ndim() - 1);
  // Sometimes we want to provide more than one initial step.
  // For example, if we do a convolution op in step net
  // and need a sufficient left padding around the input.
  // This could be used together with links where window != 1.
  auto initialStateLength = 1;
  if (input.ndim() == 3) {
    initialStateLength = input.dim(0);
  }
  // States at [0, ..., (T + initialStateLength - 1)] (inclusive)
  state->Resize(seqLen + initialStateLength, batchSize, stateSize);

  if (input.ndim() >= 2) {
    CAFFE_ENFORCE_EQ(input.dim(input.ndim() - 2), batchSize, rc.input);
    context->template Copy<T, Context, Context>(
        batchSize * stateSize * initialStateLength,
        input.template data<T>(),
        state->template mutable_data<T>());
  } else {
    // Usually, the initial state is the same for all inputs in the batch.
    // So the op conveniently accepts 1-D input and copies it batchSize times.
    repeatCopy<T, Context>(
          batchSize,
          stateSize,
          input.template data<T>(),
          state->template mutable_data<T>(),
          context);
  }
}

void PrependOps(std::vector<OperatorDef> ops, NetDef* netdef);

void AddApplyLinkOps(
    const vector<Link>& links,
    std::string timestep,
    const DeviceOption& device_option,
    NetDef* netdef);

void extractLinks(
    OperatorBase* op,
    const std::string& internalArg,
    const std::string& externalArg,
    const std::string& offsetArg,
    const std::string& windowArg,
    std::vector<detail::Link>* links);

NetDef extractNetDef(const OperatorDef& op, const std::string& argName);
} // namespace detail

template <class Context>
class RecurrentNetworkOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  RecurrentNetworkOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        sharedWs_(ws),
        enable_rnn_executor_(OperatorBase::template GetSingleArgument<bool>(
            "enable_rnn_executor",
            false)),
        timestep_(OperatorBase::template GetSingleArgument<std::string>(
            "timestep",
            "timestep")) {
    CAFFE_ENFORCE(ws);

    stepNetDef_ = detail::extractNetDef(operator_def, "step_net");

    recurrentInputs_ = constructRecurrentInputs(operator_def, sharedWs_);
    links_ = constructLinks();
    aliases_ = constructAliases();

    stepNetDef_.add_external_input(timestep_);
    detail::AddApplyLinkOps(
        links_, timestep_, operator_def.device_option(), &stepNetDef_);

    if (FLAGS_caffe2_rnn_executor && enable_rnn_executor_) {
      VLOG(1) << "Use RecurrentNetworkExecutor";
      auto recurrent_map = detail::GetRecurrentMapping(links_, false /* backward */);
      rnnExecutor_ =
          createRNNExecutor<Context>(
              stepNetDef_,
              recurrent_map,
              timestep_,
              ArgumentHelper(operator_def));
    } else {
      // Fix for legacy models that pass "rnn" type net
      if (stepNetDef_.type() == "rnn") {
        stepNetDef_.set_type("async_simple");
      }
      CAFFE_ENFORCE(stepNetDef_.type() != "async_dag");
    }
  }

  size_t NumObservers() override {
    size_t num = this->observers_list_.size();
    if (rnnExecutor_) {
      num += rnnExecutor_->NumObserversStepNet();
    }
    return num;
  }

  std::vector<detail::RecurrentInput> constructRecurrentInputs(
      const OperatorDef& operator_def,
      Workspace* sharedWs) {
    const auto states =
        OperatorBase::GetRepeatedArgument<std::string>("recurrent_states");
    const auto inputs =
        OperatorBase::GetRepeatedArgument<int>("initial_recurrent_state_ids");
    CAFFE_ENFORCE_EQ(states.size(), inputs.size(), "states/inputs mismatch");
    std::vector<detail::RecurrentInput> ris;
    for (auto i = 0; i < states.size(); ++i) {
      // States need to be "global" (since they are shared between
      // forward and backward).
      sharedWs->CreateBlob(states[i]);

      detail::RecurrentInput ri;
      ri.state = states[i];
      ri.input = operator_def.input(inputs[i]);
      ris.push_back(ri);
    }
    return ris;
  }

  std::vector<detail::OffsetAlias> constructAliases() {
    const auto& src =
        OperatorBase::GetRepeatedArgument<std::string>("alias_src");
    const auto& dst =
        OperatorBase::GetRepeatedArgument<std::string>("alias_dst");
    const auto& offset =
        OperatorBase::GetRepeatedArgument<int32_t>("alias_offset");
    CAFFE_ENFORCE(
        src.size() == offset.size(), "alias_src/alias_offset mismatch");
    CAFFE_ENFORCE(
        dst.size() == offset.size(), "alias_dst/alias_offset mismatch");
    std::vector<detail::OffsetAlias> aliases;
    for (auto i = 0; i < src.size(); ++i) {
      detail::OffsetAlias oc;
      oc.src = src[i];
      oc.dst = dst[i];
      oc.offset = offset[i];
      aliases.push_back(oc);
    }
    return aliases;
  }

  /**
    * Some blobs can be marked as to be recomputed on backward pass.
    * For those blobs, we do not want to allocate on each step workspace,
    * but we instead store that blob in the shared workspace so all
    * steps can use the same buffer on forward pass.
    */
  void initializeBlobsToRecomputeOnBackward(Workspace* sharedBlobsWs) {
    std::vector<std::string> v;
    const auto& blobs = OperatorBase::GetRepeatedArgument<std::string>(
        "recompute_blobs_on_backward", v);
    for (const auto& b : blobs) {
      // Note: if the blob already was created, this is a no-op.
      sharedBlobsWs->CreateBlob(b);
    }
  }

  std::vector<detail::Link> constructLinks() {
    std::vector<detail::Link> links;
    detail::extractLinks(
        this,
        "link_internal",
        "link_external",
        "link_offset",
        "link_window",
        &links);
    return links;
  }

  template<typename T>
  bool DoRunWithType() {
    const auto seqLen = Input(0).dim32(0);
    const auto batchSize = Input(0).dim32(1);
    for (const auto& ri : recurrentInputs_) {
      detail::initializeRecurrentInput<T, Context>(
          ri, seqLen, batchSize, sharedWs_, &context_);
    }

    // If we don't have a backward step net, this operator is forward_only
    // and we can avoid creating multiple workspaces.
    bool has_backward_pass =
        OperatorBase::HasSingleArgumentOfType<NetDef>("backward_step_net") ||
        (OperatorBase::HasSingleArgumentOfType<string>("backward_step_net") &&
         OperatorBase::GetSingleArgument<string>("backward_step_net", "") !=
             "");

    // With backward pass: we need to create workspace for each timestep
    detail::ScratchWorkspaces* scratch =
        OperatorBase::Output<detail::ScratchWorkspaces>(OutputSize() - 1);
    std::vector<std::shared_ptr<Workspace>>& stepWorkspaces =
        scratch->stepWorkspaces;
    std::shared_ptr<Workspace>& sharedBlobsWs = scratch->sharedBlobsWs;
    if (!sharedBlobsWs) {
      sharedBlobsWs = std::make_shared<Workspace>(sharedWs_);
    }

    // Caller can decide that some of the forward activations
    // are recomputed on backward pass. Then those activations do not
    // have to be stored in step workspaces but can be shared.
    initializeBlobsToRecomputeOnBackward(sharedBlobsWs.get());

    if (has_backward_pass && seqLen > stepWorkspaces.size()) {
      stepWorkspaces.resize(seqLen);
    }

    // In forward-only mode, we cycle over workspaces. This limits the amount
    // of parallelism over timesteps that the RNNExecutor provides. So with
    // RNN executor we use more workspaces to get better perf.
    int num_workspaces_on_fwd_only = rnnExecutor_ ? 4 : 2;

    if (!has_backward_pass && stepWorkspaces.size() < num_workspaces_on_fwd_only) {
      // Use alternating stepWorkspaces when forward_only=True.
      // Note that the step workspaces can be shared by other ops, thus
      // we cannot shrink it to 2 if there are more than 2 step workspaces.
      stepWorkspaces.resize(num_workspaces_on_fwd_only);
    }

    for (auto t = 0; t < seqLen; ++t) {
      auto& currentStepWorkspace =
          (has_backward_pass ? stepWorkspaces[t] :
              stepWorkspaces[t % num_workspaces_on_fwd_only]);
      if (!currentStepWorkspace) {
        currentStepWorkspace = std::make_shared<Workspace>(sharedBlobsWs.get());
      }

      if (rnnExecutor_) {
        if (!has_backward_pass) {
          // Need to limit timestep parallelism because we cycle over workspaces
          rnnExecutor_->SetMaxParallelTimesteps(num_workspaces_on_fwd_only);
        }
        rnnExecutor_->EnsureTimestepInitialized(
            t, currentStepWorkspace.get(), this->observers_list_);
      } else {
        // Use plain Caffe2 nets
        detail::UpdateTimestepBlob(currentStepWorkspace.get(), timestep_, t);
        auto* stepNet = currentStepWorkspace->GetNet(stepNetDef_.name());
        if (stepNet == nullptr) {
          stepNet = currentStepWorkspace->CreateNet(stepNetDef_);
        }
        CAFFE_ENFORCE(stepNet, "Step Net construction failure");
        // Since we have a SimpleNet, there are no races here.
        stepNet->RunAsync();
      }
    }

    if (rnnExecutor_) {
      rnnExecutor_->Run(seqLen);
    }

    for (const auto& alias : aliases_) {
      detail::applyOffsetAlias<T, Context>(alias, sharedWs_, &context_);
    }

    return true;
  }

  bool RunOnDevice() override {
    return DoRunWithType<float>();
  }

 protected:
  NetDef stepNetDef_;
  Workspace* sharedWs_;
  bool enable_rnn_executor_;
  std::unique_ptr<RecurrentNetworkExecutorBase> rnnExecutor_;

  std::vector<detail::Link> links_;
  std::vector<detail::OffsetAlias> aliases_;
  std::vector<detail::RecurrentInput> recurrentInputs_;
  std::string timestep_;
};

template <class Context>
class AccumulateInputGradientOp : public Operator<Context> {
 public:
  AccumulateInputGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        offset_(OperatorBase::GetSingleArgument<int>("offset", -1)) {
    CAFFE_ENFORCE(offset_ >= 0, "Offset not set");
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template<typename T>
  bool DoRunWithType() {
    const auto& t0 = OperatorBase::Input<Tensor<CPUContext>>(0);
    const auto t = t0.template data<int32_t>()[0];
    auto& og = Input(1);
    auto* g = Output(0);

    T* g_data = g->template mutable_data<T>();
    const auto timestep_size = g->size() / g->dim(0);

    CAFFE_ENFORCE(
        (t + offset_) * timestep_size + timestep_size <= g->size(),
        "Accumulation destination address over bounds");
    CAFFE_ENFORCE(
        t * timestep_size + timestep_size <= og.size(),
        "Accumulation source address out of bounds");

    math::Add<T, Context>(
        timestep_size,
        og.template data<T>() + t * timestep_size,
        g_data + (t + offset_) * timestep_size,
        g_data + (t + offset_) * timestep_size,
        &context_);
    return true;
  }

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float>>::call(this, Input(1));
  }

 private:
  int offset_;
};

template <class Context>
class RNNApplyLinkOp : public Operator<Context> {
 public:
  RNNApplyLinkOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        offset_(OperatorBase::GetSingleArgument<int>("offset", -1)),
        window_(OperatorBase::GetSingleArgument<int>("window", -1)) {
    CAFFE_ENFORCE(offset_ >= 0, "offset not set");
    CAFFE_ENFORCE(window_ >= 0, "window not set");
  }

  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <typename T>
  bool DoRunWithType() {
    // Both internal and external appear as both input and output to enforce
    // correct dependency computation.
    const auto& t0 = OperatorBase::Input<Tensor<CPUContext>>(0);
    const auto t = t0.template data<int32_t>()[0];
    auto& external = Input(1);

    auto* internal_out = Output(0);
    auto* external_out = Output(1);

    CAFFE_ENFORCE_GT(external.size(), 0);
    const TIndex externalTimestepSize = external.size() / external.dim(0);
    auto* externalData = external_out->template mutable_data<T>() +
        (t + offset_) * externalTimestepSize;
    auto internalDims = external_out->dims();
    internalDims[0] = window_;

    internal_out->Resize(internalDims);
    internal_out->ShareExternalPointer(
        externalData, externalTimestepSize * window_);
    return true;
  }

  bool RunOnDevice() override {
    return DoRunWithType<float>();
  }

 private:
  int offset_;
  int window_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_RECURRENT_NETWORK_OP_H_
