/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/lib/text/decoder/lm/KenLM.h"

#include <stdexcept>

#ifdef USE_KENLM_FROM_LANGTECH
#include "language_technology/jedi/lm/model.hh"
#else
#if __has_include(<kenlm/lm/model.hh>)
#include <kenlm/lm/model.hh>
#elif __has_include(<lm/model.hh>)
#include <lm/model.hh>
#else
#error "KenLM header not found (kenlm/lm/model.hh)."
#endif // __has_include
#endif // USE_KENLM_FROM_LANGTECH

namespace fl {
namespace lib {
namespace text {

KenLMState::KenLMState() : ken_(std::make_unique<lm::ngram::State>()) {}

KenLM::KenLM(const std::string& path, const Dictionary& usrTknDict) {
  // Load LM
  model_.reset(lm::ngram::LoadVirtual(path.c_str()));
  if (!model_) {
    throw std::runtime_error("[KenLM] LM loading failed.");
  }
  vocab_ = &model_->BaseVocabulary();
  if (!vocab_) {
    throw std::runtime_error("[KenLM] LM vocabulary loading failed.");
  }

  // Create index map
  usrToLmIdxMap_.resize(usrTknDict.indexSize());
  for (int i = 0; i < usrTknDict.indexSize(); i++) {
    auto token = usrTknDict.getEntry(i);
    int lmIdx = vocab_->Index(token.c_str());
    usrToLmIdxMap_[i] = lmIdx;
  }
}

void KenLM::updateIdxMap(const Dictionary& usrTknDict) {
  size_t idxMapSize = usrToLmIdxMap_.size();
  usrToLmIdxMap_.resize(idxMapSize + usrTknDict.indexSize());
  for (int i = 0; i < usrTknDict.indexSize(); i++) {
    auto token = usrTknDict.getEntry(i);
    int lmIdx = vocab_->Index(token.c_str());
    usrToLmIdxMap_[idxMapSize + i] = lmIdx;
  }
}

LMStatePtr KenLM::start(bool startWithNothing) {
  auto outState = std::make_shared<KenLMState>();
  if (startWithNothing) {
    model_->NullContextWrite(outState->ken());
  } else {
    model_->BeginSentenceWrite(outState->ken());
  }

  return outState;
}

std::pair<LMStatePtr, float> KenLM::score(
    const LMStatePtr& state,
    const int usrTokenIdx) {
  if (usrTokenIdx < 0 || usrTokenIdx >= usrToLmIdxMap_.size()) {
    throw std::runtime_error(
        "[KenLM] Invalid user token index: " + std::to_string(usrTokenIdx));
  }
  auto inState = std::static_pointer_cast<KenLMState>(state);
  auto outState = inState->child<KenLMState>(usrTokenIdx);
  float score = model_->BaseScore(
      inState->ken(), usrToLmIdxMap_[usrTokenIdx], outState->ken());
  return std::make_pair(std::move(outState), score);
}

std::pair<LMStatePtr, float> KenLM::finish(const LMStatePtr& state) {
  auto inState = std::static_pointer_cast<KenLMState>(state);
  auto outState = inState->child<KenLMState>(-1);
  float score =
      model_->BaseScore(inState->ken(), vocab_->EndSentence(), outState->ken());
  return std::make_pair(std::move(outState), score);
}
} // namespace text
} // namespace lib
} // namespace fl
