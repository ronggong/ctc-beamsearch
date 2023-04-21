/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <unordered_map>

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
  double wordScore; // Word insertion score
  double unkScore; // Unknown word insertion score
  double silScore; // Silence insertion score
  bool logAdd; // If or not use logadd when merging hypothesis
  CriterionType criterionType; // CTC or ASG
};

struct BoostOptions {
  bool wordBoost; // If or not word-level boost. If false do word piece level boost
  bool cmdMatchBegin; // If the command should be matched from the first word or word piece
  bool cmdMatchEnd; // If the command should be matched without extra tokens from the end
  bool cmdMatchIncr; // Incremental match. If true, not fallback the score once we have a command match
  double cmdBoostWeight; // Weight for command boosting
  double cmdFixedScore; // fixed score for command boosting, added to each matched word or word piece
  std::vector<int> cmdBoostIgnore; // Word ids to be ignored during cmd boosting
};

/**
 * LexiconDecoderState stores information for each hypothesis in the beam.
 */
struct LexiconDecoderState {
  double score; // Accumulated total score so far
  LMStatePtr lmState; // Language model state
  const TrieNode* lex; // Trie node in the lexicon
  const TrieNode* cmd; // Trie node in the command
  const LexiconDecoderState* parent; // Parent hypothesis
  int token; // Label of token
  int word; // Label of word (-1 if incomplete)
  bool prevBlank; // If previous hypothesis is blank (for CTC only)
  bool cmdBoostEnable; // If the boost is enable for the hyp

  double emittingModelScore; // Accumulated AM score so far
  double lmScore; // Accumulated LM score so far
  double cmdScore; // Accumulated command score so far

  LexiconDecoderState(
      const double score,
      const LMStatePtr& lmState,
      const TrieNode* lex,
      const TrieNode* cmd,
      const LexiconDecoderState* parent,
      const int token,
      const int word,
      const bool prevBlank = false,
      const bool cmdBoostEnable = true,
      const double emittingModelScore = 0,
      const double lmScore = 0,
      const double cmdScore = 0)
      : score(score),
        lmState(lmState),
        lex(lex),
        cmd(cmd),
        parent(parent),
        token(token),
        word(word),
        prevBlank(prevBlank),
        cmdBoostEnable(cmdBoostEnable),
        emittingModelScore(emittingModelScore),
        lmScore(lmScore),
        cmdScore(cmdScore) {}

  LexiconDecoderState()
      : score(0.),
        lmState(nullptr),
        lex(nullptr),
        cmd(nullptr),
        parent(nullptr),
        token(-1),
        word(-1),
        prevBlank(false),
        cmdBoostEnable(true),
        emittingModelScore(0.),
        lmScore(0.),
        cmdScore(0.) {}

  int compareNoScoreStates(const LexiconDecoderState* node) const {
    int lmCmp = lmState->compare(node->lmState);
    if (lmCmp != 0) {
      return lmCmp > 0 ? 1 : -1;
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
      BoostOptions boostOpt,
      const TriePtr& lexicon,
      const LMPtr& lm,
      const int sil,
      const int blank,
      const int unk,
      const std::vector<float>& transitions,
      const bool isLmToken)
      : opt_(std::move(opt)),
        boostOpt_(std::move(boostOpt)),
        lexicon_(lexicon),
        command_(std::make_shared<Trie>(Trie(0, sil))),
        lm_(lm),
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

 private:
  void cmdBoostToken(double &cmdScore, double &accCmdScore, bool &cmdBoostEnable,
                     const int &token, const double& prevHypCmdScore,
                     const TrieNode* &cmd, const TrieNode* &prevCmd);
  void cmdBoostWord(double &cmdScore, double &accCmdScore, bool &cmdBoostEnable,
                    const int &token, const double& prevHypCmdScore,
                    const TrieNode* &cmd, const TrieNode* &prevCmd);

 protected:
  LexiconDecoderOptions opt_;
  BoostOptions boostOpt_;
  // Lexicon trie to restrict beam-search decoder
  TriePtr lexicon_;
  // Command trie for boosting command words
  TriePtr command_;
  LMPtr lm_;
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
