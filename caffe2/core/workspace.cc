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

#include "caffe2/core/workspace.h"

#include <algorithm>
#include <ctime>
#include <mutex>

#include "caffe2/core/logging.h"
#include "caffe2/core/net.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/plan_executor.h"
#include "caffe2/core/tensor.h"
#include "caffe2/proto/caffe2.pb.h"

namespace caffe2 {

vector<string> Workspace::Blobs() const {
  vector<string> names;
  names.reserve(blob_map_.size());
  for (auto& entry : blob_map_) {
    names.push_back(entry.first);
  }
  for (const auto& forwarded : forwarded_blobs_) {
    const auto parent_ws = forwarded.second.first;
    const auto& parent_name = forwarded.second.second;
    if (parent_ws->HasBlob(parent_name)) {
      names.push_back(forwarded.first);
    }
  }
  if (shared_) {
    const auto& shared_blobs = shared_->Blobs();
    names.insert(names.end(), shared_blobs.begin(), shared_blobs.end());
  }
  return names;
}

Blob* Workspace::CreateBlob(const string& name) {
  if (HasBlob(name)) {
    VLOG(1) << "Blob " << name << " already exists. Skipping.";
  } else if (forwarded_blobs_.count(name)) {
    // possible if parent workspace deletes forwarded blob
    VLOG(1) << "Blob " << name << " is already forwarded from parent workspace "
            << "(blob " << forwarded_blobs_[name].second << "). Skipping.";
  } else {
    VLOG(1) << "Creating blob " << name;
    blob_map_[name] = unique_ptr<Blob>(new Blob());
  }
  return GetBlob(name);
}

const Blob* Workspace::GetBlob(const string& name) const {
  if (blob_map_.count(name)) {
    return blob_map_.at(name).get();
  } else if (forwarded_blobs_.count(name)) {
    const auto parent_ws = forwarded_blobs_.at(name).first;
    const auto& parent_name = forwarded_blobs_.at(name).second;
    return parent_ws->GetBlob(parent_name);
  } else if (shared_ && shared_->HasBlob(name)) {
    return shared_->GetBlob(name);
  }
  LOG(WARNING) << "Blob " << name << " not in the workspace.";
  // TODO(Yangqing): do we want to always print out the list of blobs here?
  // LOG(WARNING) << "Current blobs:";
  // for (const auto& entry : blob_map_) {
  //   LOG(WARNING) << entry.first;
  // }
  return nullptr;
}

Blob* Workspace::GetBlob(const string& name) {
  return const_cast<Blob*>(static_cast<const Workspace*>(this)->GetBlob(name));
}

NetBase* Workspace::CreateNet(const NetDef& net_def, bool overwrite) {
  std::shared_ptr<NetDef> tmp_net_def(new NetDef(net_def));
  return CreateNet(tmp_net_def, overwrite);
}

NetBase* Workspace::CreateNet(
    const std::shared_ptr<const NetDef>& net_def,
    bool overwrite) {
  CAFFE_ENFORCE(net_def->has_name(), "Net definition should have a name.");
  if (net_map_.count(net_def->name()) > 0) {
    if (!overwrite) {
      CAFFE_THROW(
          "I respectfully refuse to overwrite an existing net of the same "
          "name \"",
          net_def->name(),
          "\", unless you explicitly specify overwrite=true.");
    }
    VLOG(1) << "Deleting existing network of the same name.";
    // Note(Yangqing): Why do we explicitly erase it here? Some components of
    // the old network, such as an opened LevelDB, may prevent us from creating
    // a new network before the old one is deleted. Thus we will need to first
    // erase the old one before the new one can be constructed.
    net_map_.erase(net_def->name());
  }
  // Create a new net with its name.
  VLOG(1) << "Initializing network " << net_def->name();
  net_map_[net_def->name()] =
      unique_ptr<NetBase>(caffe2::CreateNet(net_def, this));
  if (net_map_[net_def->name()].get() == nullptr) {
    LOG(ERROR) << "Error when creating the network."
               << "Maybe net type: [" << net_def->type() << "] does not exist";
    net_map_.erase(net_def->name());
    return nullptr;
  }
  return net_map_[net_def->name()].get();
}

NetBase* Workspace::GetNet(const string& name) {
  if (!net_map_.count(name)) {
    return nullptr;
  } else {
    return net_map_[name].get();
  }
}

void Workspace::DeleteNet(const string& name) {
  if (net_map_.count(name)) {
    net_map_.erase(name);
  }
}

bool Workspace::RunNet(const string& name) {
  if (!net_map_.count(name)) {
    LOG(ERROR) << "Network " << name << " does not exist yet.";
    return false;
  }
  return net_map_[name]->Run();
}

bool Workspace::RunOperatorOnce(const OperatorDef& op_def) {
  std::unique_ptr<OperatorBase> op(CreateOperator(op_def, this));
  if (op.get() == nullptr) {
    LOG(ERROR) << "Cannot create operator of type " << op_def.type();
    return false;
  }
  if (!op->Run()) {
    LOG(ERROR) << "Error when running operator " << op_def.type();
    return false;
  }
  return true;
}
bool Workspace::RunNetOnce(const NetDef& net_def) {
  std::unique_ptr<NetBase> net(caffe2::CreateNet(net_def, this));
  if (net == nullptr) {
    CAFFE_THROW(
        "Could not create net: " + net_def.name() + " of type " +
        net_def.type());
  }
  if (!net->Run()) {
    LOG(ERROR) << "Error when running network " << net_def.name();
    return false;
  }
  return true;
}

} // namespace caffe2
