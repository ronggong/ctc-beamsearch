/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <limits>
#include <string>
#include <algorithm>

#include "flashlight/lib/text/decoder/Trie.h"

namespace fl {
namespace lib {
namespace text {

const double kMinusLogThreshold = -39.14;

const TrieNode* Trie::getRoot() const {
  return root_.get();
}

TrieNodePtr
Trie::insert(const std::vector<int>& indices, int label, float score) {
  TrieNodePtr node = root_;
  //std::cout << "idx ";
  for (int i = 0; i < indices.size(); i++) {
    int idx = indices[i];
    if (idx < 0 || idx >= maxChildren_) {
      throw std::out_of_range(
          "[Trie] Invalid letter index: " + std::to_string(idx));
    }
    if (node->children.find(idx) == node->children.end()) {
      node->children[idx] = std::make_shared<TrieNode>(idx);
    }
    TrieNodePtr parent = node;
    node = node->children[idx];
    node->parent = parent;
    //std::cout << idx << " ";
  }
  //std::cout << std::endl;
  if (node->labels.size() < kTrieMaxLabel) {
    node->labels.push_back(label);
    node->scores.push_back(score);
    labelNodeMap_[label] = node;
    //std::cout << "label added " << label << std::endl;
  } else {
    std::cerr << "[Trie] Trie label number reached limit: " << kTrieMaxLabel
              << "\n";
  }
  return node;
}

TrieNodePtr
Trie::insert(const std::vector<int>& indices, const std::unordered_map<int, int>& label, float score) {
  TrieNodePtr node = root_;
  for (int i = 0; i < indices.size(); i++) {
    int idx = indices[i];
    if (idx < 0 || idx >= maxChildren_) {
      throw std::out_of_range(
          "[Trie] Invalid letter index: " + std::to_string(idx));
    }
    if (node->children.find(idx) == node->children.end()) {
      node->children[idx] = std::make_shared<TrieNode>(idx);
    }
    TrieNodePtr parent = node;
    node = node->children[idx];
    node->parent = parent;

    if (node->labels.size() < kTrieMaxLabel) {
      if (!label.count(i)) {
        continue;
      }
      if (std::find(node->labels.begin(), node->labels.end(), label.at(i)) == node->labels.end()) {
        node->labels.push_back(label.at(i));
        node->scores.push_back(score);
        labelNodeMap_[label.at(i)] = node;
      }
    } else {
      std::cerr << "[Trie] Trie label number reached limit: " << kTrieMaxLabel
                << "\n";
    }
  }
  return node;
}

TrieNodePtr Trie::search(const std::vector<int>& indices) {
  TrieNodePtr node = root_;
  for (auto idx : indices) {
    if (idx < 0 || idx >= maxChildren_) {
      throw std::out_of_range(
          "[Trie] Invalid letter index: " + std::to_string(idx));
    }
    if (node->children.find(idx) == node->children.end()) {
      return nullptr;
    }
    node = node->children[idx];
  }
  return node;
}

void Trie::del(const std::unordered_set<int> labels) {
  for (auto label : labels) {
    if (!labelNodeMap_.count(label)) {
      continue;
    }
    TrieNodePtr node = labelNodeMap_[label];
    auto it = std::find(node->labels.begin(), node->labels.end(), label);
    if (it != node->labels.end()) {
      node->labels.erase(it);
      node->scores.erase(node->scores.begin() + (it - node->labels.begin()));
    }
    // traceback the node, and remove all parents that have no children and no labels
    if (node->labels.empty() && node->children.size().empty()) {
      while (node->parent != nullptr) {
        TrieNodePtr parent = node->parent;
        parent->children.erase(node->idx);
        if (parent->children.size() > 0 || !parent->labels.empty() || parent == root_) {
          break;
        }
        node = parent;
      }
    }
  }
}

/* logadd */
double TrieLogAdd(double log_a, double log_b) {
  double minusdif;
  if (log_a < log_b) {
    std::swap(log_a, log_b);
  }
  minusdif = log_b - log_a;
  if (minusdif < kMinusLogThreshold) {
    return log_a;
  } else {
    return log_a + log1p(exp(minusdif));
  }
}

void smearNode(TrieNodePtr node, SmearingMode smearMode) {
  node->maxScore = -std::numeric_limits<float>::infinity();
  for (auto score : node->scores) {
    node->maxScore = TrieLogAdd(node->maxScore, score);
  }
  for (auto child : node->children) {
    auto childNode = child.second;
    smearNode(childNode, smearMode);
    if (smearMode == SmearingMode::LOGADD) {
      node->maxScore = TrieLogAdd(node->maxScore, childNode->maxScore);
    } else if (
        smearMode == SmearingMode::MAX &&
        childNode->maxScore > node->maxScore) {
      node->maxScore = childNode->maxScore;
    }
  }
}

void Trie::smear(SmearingMode smearMode) {
  if (smearMode != SmearingMode::NONE) {
    smearNode(root_, smearMode);
  }
}

void Trie::cleanMaxScoreNode(TrieNodePtr node) {
  node->maxScore = 0.0;
  for (auto child : node->children) {
    auto childNode = child.second;
    cleanMaxScoreNode(childNode);
  }
}

void Trie::cleanMaxScore() {
  cleanMaxScoreNode(root_);
}

int Trie::getMaxChildren() const
{
  return maxChildren_;
}

} // namespace text
} // namespace lib
} // namespace fl
