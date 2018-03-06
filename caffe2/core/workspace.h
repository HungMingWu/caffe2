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

#ifndef CAFFE2_CORE_WORKSPACE_H_
#define CAFFE2_CORE_WORKSPACE_H_

#include "caffe2/core/common.h"
#include "caffe2/core/observer.h"

#include <climits>
#include <cstddef>
#include <mutex>
#include <typeinfo>
#include <unordered_set>
#include <vector>

#include "caffe2/core/blob.h"
#include "caffe2/core/registry.h"
#include "caffe2/core/net.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/utils/signal_handler.h"

CAFFE2_DECLARE_bool(caffe2_print_blob_sizes_at_exit);

namespace caffe2 {

class NetBase;

struct StopOnSignal {
  StopOnSignal()
      : handler_(std::make_shared<SignalHandler>(
            SignalHandler::Action::STOP,
            SignalHandler::Action::STOP)) {}

  StopOnSignal(const StopOnSignal& other) : handler_(other.handler_) {}

  bool operator()(int /*iter*/) {
    return handler_->CheckForSignals() != SignalHandler::Action::STOP;
  }

  std::shared_ptr<SignalHandler> handler_;
};

/**
 * Workspace is a class that holds all the related objects created during
 * runtime: (1) all blobs, and (2) all instantiated networks. It is the owner of
 * all these objects and deals with the scaffolding logistics.
 */
class Workspace {
 public:
  typedef std::function<bool(int)> ShouldContinue;
  typedef CaffeMap<string, unique_ptr<Blob> > BlobMap;
  typedef CaffeMap<string, unique_ptr<NetBase> > NetMap;
  /**
   * Initializes an empty workspace.
   */
  Workspace() : shared_(nullptr) {}

  /**
   * Initializes a workspace with a shared workspace.
   *
   * When we access a Blob, we will first try to access the blob that exists
   * in the local workspace, and if not, access the blob that exists in the
   * shared workspace. The caller keeps the ownership of the shared workspace
   * and is responsible for making sure that its lifetime is longer than the
   * created workspace.
   */
  explicit Workspace(const Workspace* shared)
      : shared_(shared) {}

  /**
   * Initializes a workspace with a root folder and a shared workspace.
   */
  Workspace(Workspace* shared)
      : shared_(shared) {}

  ~Workspace() = default;

  /**
   * Return a list of blob names. This may be a bit slow since it will involve
   * creation of multiple temp variables. For best performance, simply use
   * HasBlob() and GetBlob().
   */
  vector<string> Blobs() const;

  /**
   * Checks if a blob with the given name is present in the current workspace.
   */
  inline bool HasBlob(const string& name) const {
    // First, check the local workspace,
    // Then, check the forwarding map, then the parent workspace
    if (blob_map_.count(name)) {
      return true;
    } else if (forwarded_blobs_.count(name)) {
      const auto parent_ws = forwarded_blobs_.at(name).first;
      const auto& parent_name = forwarded_blobs_.at(name).second;
      return parent_ws->HasBlob(parent_name);
    } else if (shared_) {
      return shared_->HasBlob(name);
    }
    return false;
  }

  /**
   * Creates a blob of the given name. The pointer to the blob is returned, but
   * the workspace keeps ownership of the pointer. If a blob of the given name
   * already exists, the creation is skipped and the existing blob is returned.
   */
  Blob* CreateBlob(const string& name);
  /**
   * Gets the blob with the given name as a const pointer. If the blob does not
   * exist, a nullptr is returned.
   */
  const Blob* GetBlob(const string& name) const;
  /**
   * Gets the blob with the given name as a mutable pointer. If the blob does
   * not exist, a nullptr is returned.
   */
  Blob* GetBlob(const string& name);

  /**
   * Creates a network with the given NetDef, and returns the pointer to the
   * network. If there is anything wrong during the creation of the network, a
   * nullptr is returned. The Workspace keeps ownership of the pointer.
   *
   * If there is already a net created in the workspace with the given name,
   * CreateNet will overwrite it if overwrite=true is specified. Otherwise, an
   * exception is thrown.
   */
  NetBase* CreateNet(const NetDef& net_def, bool overwrite = false);
  NetBase* CreateNet(
      const std::shared_ptr<const NetDef>& net_def,
      bool overwrite = false);
  /**
   * Gets the pointer to a created net. The workspace keeps ownership of the
   * network.
   */
  NetBase* GetNet(const string& net_name);
  /**
   * Deletes the instantiated network with the given name.
   */
  void DeleteNet(const string& net_name);
  /**
   * Finds and runs the instantiated network with the given name. If the network
   * does not exist or there are errors running the network, the function
   * returns false.
   */
  bool RunNet(const string& net_name);

  /**
   * Returns a list of names of the currently instantiated networks.
   */
  vector<string> Nets() const {
    vector<string> names;
    for (auto& entry : net_map_) {
      names.push_back(entry.first);
    }
    return names;
  }

  // RunOperatorOnce and RunNetOnce runs an operator or net once. The difference
  // between RunNet and RunNetOnce lies in the fact that RunNet allows you to
  // have a persistent net object, while RunNetOnce creates a net and discards
  // it on the fly - this may make things like database read and random number
  // generators repeat the same thing over multiple calls.
  bool RunOperatorOnce(const OperatorDef& op_def);
  bool RunNetOnce(const NetDef& net_def);

 public:
  std::atomic<int> last_failed_op_net_position;

 private:
  BlobMap blob_map_;
  NetMap net_map_;
  const Workspace* shared_;
  std::unordered_map<string, std::pair<const Workspace*, string>>
      forwarded_blobs_;

  DISABLE_COPY_AND_ASSIGN(Workspace);
};

}  // namespace caffe2

#endif  // CAFFE2_CORE_WORKSPACE_H_
