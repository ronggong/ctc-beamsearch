/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <unordered_map>
#include <set>

#include "flashlight/lib/text/decoder/Decoder.h"
#include "flashlight/lib/text/decoder/Trie.h"
#include "flashlight/lib/text/decoder/lm/LM.h"

namespace fl {
namespace lib {
namespace text {

struct LexiconDecoderOptions {
  int beamSize; // Maximum number of hypothesis we hold after each step
  int beamSizeToken; // Maximum number of tokens we consider at each step
  double beamThreshold; // Threshold to prune hypothesis
  double lmWeight; // Weight of lm
  double lm0Weight; // Weight of lm0
  double lm1Weight; // Weight of lm1
  double lm2Weight; // weight of lm2
  double wordScore; // Word insertion score
  double unkScore; // Unknown word insertion score
  double silScore; // Silence insertion score
  bool logAdd; // If or not use logadd when merging hypothesis
  CriterionType criterionType; // CTC or ASG
};

struct BoostOptions {
  bool wordBoost; // If or not word-level boost. If false do word piece level boost
  bool matchBegin; // If the boosting phrase should be matched from the first word or word piece
  bool matchEnd; // If the boosting phrase should be matched without extra tokens from the end
  bool matchIncr; // Incremental match. If true, not fallback the score once we have a phrase match
  double boostWeight; // Weight for the boosting
  double fixedScore; // fixed score for the boosting, added to each matched word or word piece
  std::vector<int> boostIgnore; // Word ids to be ignored during boosting
};

/**
 * LexiconDecoderState stores information for each hypothesis in the beam.
 */
struct LexiconDecoderState {
  double score; // Accumulated total score so far
  LMStatePtr lmState0; // Language model 0 state
  LMStatePtr lmState1; // Language model 1 state
  LMStatePtr lmState2; // Language model 2 state
  const TrieNode* lex; // Trie node in the lexicon
  const TrieNode* cmd; // Trie node in the command
  const TrieNode* uaw; // Trie node in the user added word Trie
  const LexiconDecoderState* parent; // Parent hypothesis
  int token; // Label of token
  int word; // Label of word (-1 if incomplete)
  bool prevBlank; // If previous hypothesis is blank (for CTC only)
  bool cmdBoostEnable; // If the command boost is enable for the hyp
  bool uawBoostEnable; // If the user added words boost is enable

  double emittingModelScore; // Accumulated AM score so far
  double lmScore; // Accumulated LM score so far
  double cmdScore; // Accumulated command score so far
  double uawScore; // Accumulated user added word score so far

  LexiconDecoderState(
      const double score,
      const LMStatePtr& lmState0,
      const LMStatePtr& lmState1,
      const LMStatePtr& lmState2,
      const TrieNode* lex,
      const TrieNode* cmd,
      const TrieNode* uaw,
      const LexiconDecoderState* parent,
      const int token,
      const int word,
      const bool prevBlank = false,
      const bool cmdBoostEnable = true,
      const bool uawBoostEnable = true,
      const double emittingModelScore = 0,
      const double lmScore = 0,
      const double cmdScore = 0,
      const double uawScore = 0)
      : score(score),
        lmState0(lmState0),
        lmState1(lmState1),
        lmState2(lmState2),
        lex(lex),
        cmd(cmd),
        uaw(uaw),
        parent(parent),
        token(token),
        word(word),
        prevBlank(prevBlank),
        cmdBoostEnable(cmdBoostEnable),
        uawBoostEnable(uawBoostEnable),
        emittingModelScore(emittingModelScore),
        lmScore(lmScore),
        cmdScore(cmdScore),
        uawScore(uawScore) {}

  LexiconDecoderState()
      : score(0.),
        lmState0(nullptr),
        lmState1(nullptr),
        lmState2(nullptr),
        lex(nullptr),
        cmd(nullptr),
        uaw(nullptr),
        parent(nullptr),
        token(-1),
        word(-1),
        prevBlank(false),
        cmdBoostEnable(true),
        uawBoostEnable(true),
        emittingModelScore(0.),
        lmScore(0.),
        cmdScore(0.),
        uawScore(0.) {}

  int compareNoScoreStates(const LexiconDecoderState* node) const {
    int lm0Cmp = lmState0->compare(node->lmState0);
    int lm1Cmp = lmState1->compare(node->lmState1);
    int lm2Cmp = lmState2->compare(node->lmState2);
    if (lm0Cmp != 0) {
      return lm0Cmp > 0 ? 1 : -1;
    } else if (lm1Cmp != 0) {
      return lm1Cmp > 0 ? 1 : -1;
    } else if (lm2Cmp != 0) {
      return lm2Cmp > 0 ? 1 : -1;
    } else if (lex != node->lex) {
      return lex > node->lex ? 1 : -1;
    } else if (token != node->token) {
      return token > node->token ? 1 : -1;
    } else if (prevBlank != node->prevBlank) {
      return prevBlank > node->prevBlank ? 1 : -1;
    }
    return 0;
  }

  int getWord() const {
    return word;
  }

  bool isComplete() const {
    return !parent || parent->word >= 0;
  }
};

/**
 * Decoder implements a beam seach decoder that finds the word transcription
 * W maximizing:
 *
 * AM(W) + lmWeight_ * log(P_{lm}(W)) + wordScore_ * |W_known| + unkScore_ *
 * |W_unknown| + silScore_ * |{i| pi_i = <sil>}|
 *
 * where P_{lm}(W) is the language model score, pi_i is the value for the i-th
 * frame in the path leading to W and AM(W) is the (unnormalized) emitting model
 * score of the transcription W. Note that the lexicon is used to limit the
 * search space and all candidate words are generated from it if unkScore is
 * -inf, otherwise <UNK> will be generated for OOVs.
 */
class LexiconDecoder : public Decoder {
 public:
  LexiconDecoder(
      LexiconDecoderOptions opt,
      BoostOptions cmdBoostOpt,
      BoostOptions uawBoostOpt,
      const TriePtr& lexicon,
      const LMPtr& lm0,
      const LMPtr& lm1,
      const LMPtr& lm2,
      const int sil,
      const int blank,
      const int unk,
      const std::vector<float>& transitions,
      const bool isLmToken)
      : opt_(std::move(opt)),
        cmdBoostOpt_(std::move(cmdBoostOpt)),
        uawBoostOpt_(std::move(uawBoostOpt)),
        lexicon_(lexicon),
        command_(std::make_shared<Trie>(Trie(0, sil))),
        uaw_(std::make_shared<Trie>(Trie(0, sil))),
        lm0_(lm0),
        lm1_(lm1),
        lm2_(lm2),
        sil_(sil),
        blank_(blank),
        unk_(unk),
        transitions_(transitions),
        isLmToken_(isLmToken) {}

  void decodeBegin() override;

  void decodeStep(const float* emissions, int T, int N) override;

  void decodeEnd() override;

  int nHypothesis() const;

  void prune(int lookBack = 0) override;

  int nDecodedFramesInBuffer() const override;

  DecodeResult getBestHypothesis(int lookBack = 0) const override;

  std::vector<DecodeResult> getAllFinalHypothesis() const override;

  void setCommandTrie(const TriePtr& command);

  void setUawTrie(const TriePtr& uaw);

  void setAlienWordIds(const std::set<int>& wordIds, const float& alienScore);

 private:
  void boostToken(double &score, double &accScore, bool &boostEnable,
                  const int &token, const double& prevHypScore,
                  const TrieNode* &node, const TrieNode* &prevNode,
                  const BoostOptions &opt, TriePtr trie);
  void boostWord(double &score, double &accScore, bool &boostEnable,
                 const int &token, const double& prevHypScore,
                 const TrieNode* &node, const TrieNode* &prevNode,
                 const BoostOptions &opt, TriePtr trie);

 protected:
  LexiconDecoderOptions opt_;
  // Boost options for commands
  BoostOptions cmdBoostOpt_;
  // Boost options for user added words
  BoostOptions uawBoostOpt_;
  // Lexicon trie to restrict beam-search decoder
  TriePtr lexicon_;
  // Command trie for boosting command words
  TriePtr command_;
  // Command trie for user added words boosting
  TriePtr uaw_;
  LMPtr lm0_;
  // second LM
  LMPtr lm1_;
  // third LM
  LMPtr lm2_;
  // Index of silence label
  int sil_;
  // Index of blank label (for CTC)
  int blank_;
  // Index of unknown word
  int unk_;
  // matrix of transitions (for ASG criterion)
  std::vector<float> transitions_;
  // if LM is token-level (operates on the same level as the emitting model)
  // or it is word-level (in case of false)
  bool isLmToken_;
  // Alien word ids
  std::set<int> alienWordIds_;
  // Score of the alien word
  float alienScore_;

  // All the hypothesis new candidates (can be larger than beamsize) proposed
  // based on the ones from previous frame
  std::vector<LexiconDecoderState> candidates_;

  // This vector is designed for efficient sorting and merging the candidates_,
  // so instead of moving around objects, we only need to sort pointers
  std::vector<LexiconDecoderState*> candidatePtrs_;

  // Best candidate score of current frame
  double candidatesBestScore_;

  // Vector of hypothesis for all the frames so far
  std::unordered_map<int, std::vector<LexiconDecoderState>> hyp_;

  // These 2 variables are used for online decoding, for hypothesis pruning
  int nDecodedFrames_; // Total number of decoded frames.
  int nPrunedFrames_; // Total number of pruned frames from hyp_.
};
} // namespace text
} // namespace lib
} // namespace fl
