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

#include <google/protobuf/text_format.h>
#include <gtest/gtest.h>
#include "caffe2/core/net.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/scope_guard.h"

CAFFE2_DECLARE_bool(caffe2_disable_chaining);

namespace caffe2 {

namespace {

static std::atomic<int> counter;

// A net test dummy op that does nothing but scaffolding. Here, we
// inherit from OperatorBase because we instantiate on both CPU and
// GPU. In general, you want to only inherit from Operator<Context>.
class NetTestDummyOp final : public OperatorBase {
 public:
  using OperatorBase::OperatorBase;

  NetTestDummyOp(const OperatorDef& operator_def, Workspace* ws)
      : OperatorBase(operator_def, ws),
        fail_(OperatorBase::GetSingleArgument<bool>("fail", false)) {}

  bool Run(int /* unused */ /*stream_id*/) override {
    if (fail_) {
      return false;
    }
    counter.fetch_add(1);
    return true;
  }

  // Simulate CUDA operator behavior
  bool HasAsyncPart() const override {
    return debug_def().device_option().device_type() == CUDA;
  }

  bool SupportsAsyncScheduling() const override {
    return debug_def().device_option().device_type() == CUDA;
  }

 protected:
  const bool fail_;
};

REGISTER_CPU_OPERATOR(NetTestDummy, NetTestDummyOp);
REGISTER_CUDA_OPERATOR(NetTestDummy, NetTestDummyOp);
REGISTER_CPU_OPERATOR(NetTestDummy2, NetTestDummyOp);
REGISTER_CUDA_OPERATOR(NetTestDummy2, NetTestDummyOp);

OPERATOR_SCHEMA(NetTestDummy)
    .NumInputs(0, INT_MAX)
    .NumOutputs(0, INT_MAX)
    .AllowInplace({{0, 0}, {1, 1}});
OPERATOR_SCHEMA(NetTestDummy2)
    .NumInputs(0, INT_MAX)
    .NumOutputs(0, INT_MAX)
    .AllowInplace({{1, 0}});

unique_ptr<NetBase> CreateNetTestHelper(
    Workspace* ws,
    const vector<string>& input,
    const vector<string>& output) {
  NetDef net_def;
  {
    auto& op = *(net_def.add_op());
    op.set_type("NetTestDummy");
    op.add_input("in");
    op.add_output("hidden");
  }
  {
    auto& op = *(net_def.add_op());
    op.set_type("NetTestDummy");
    op.add_input("hidden");
    op.add_output("out");
  }

  for (const auto& name : input) {
    net_def.add_external_input(name);
  }
  for (const auto& name : output) {
    net_def.add_external_output(name);
  }
  return CreateNet(net_def, ws);
}

}  // namespace

TEST(NetTest, ConstructionNoDeclaredInputOutput) {
  Workspace ws;
  ws.CreateBlob("in");
  unique_ptr<NetBase> net(
      CreateNetTestHelper(&ws, vector<string>(), vector<string>()));
  EXPECT_TRUE(net.get() != nullptr);
}

TEST(NetTest, ConstructionDeclaredInput) {
  Workspace ws;
  ws.CreateBlob("in");
  unique_ptr<NetBase> net(
      CreateNetTestHelper(&ws, vector<string>{"in"}, vector<string>()));
  EXPECT_TRUE(net.get() != nullptr);
}

TEST(NetTest, ConstructionDeclaredOutput) {
  Workspace ws;
  ws.CreateBlob("in");
  unique_ptr<NetBase> net(
      CreateNetTestHelper(&ws, vector<string>(), vector<string>{"out"}));
  EXPECT_TRUE(net.get() != nullptr);
}

TEST(NetTest, DeclaredInputInsufficient) {
  Workspace ws;
  ws.CreateBlob("in");
  ASSERT_THROW(
      CreateNetTestHelper(&ws, vector<string>{"unuseful_in"},
                          vector<string>()),
      EnforceNotMet);
}

TEST(NetDeathTest, DeclaredOutputNotMet) {
  Workspace ws;
  ws.CreateBlob("in");
  ASSERT_THROW(
      CreateNetTestHelper(&ws, vector<string>(),
                          vector<string>{"unproduced_out"}),
      EnforceNotMet);
}

void testExecution(std::unique_ptr<NetBase>& net, int num_ops) {
  // Run 100 times
  for (int i = 0; i < 100; i++) {
    counter.exchange(0);
    net.get()->Run();
    ASSERT_EQ(num_ops, counter.load());
  }
}

} // namespace caffe2
