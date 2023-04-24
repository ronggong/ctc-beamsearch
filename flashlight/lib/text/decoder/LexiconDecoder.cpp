/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <unordered_map>
#include <iostream>

#include "flashlight/lib/text/decoder/LexiconDecoder.h"

namespace fl {
namespace lib {
namespace text {

void LexiconDecoder::decodeBegin() {
  hyp_.clear();
  hyp_.emplace(0, std::vector<LexiconDecoderState>());

  /* note: the lm reset itself with :start() */
  hyp_[0].emplace_back(
      0.0, lm_->start(0), lexicon_->getRoot(), command_->getRoot(), nullptr, sil_, -1);
  nDecodedFrames_ = 0;
  nPrunedFrames_ = 0;
}

void LexiconDecoder::decodeStep(const float* emissions, int T, int N) {
  int startFrame = nDecodedFrames_ - nPrunedFrames_;

  // Extend hyp_ buffer
  if (hyp_.size() < startFrame + T + 2) {
    for (int i = hyp_.size(); i < startFrame + T + 2; i++) {
      hyp_.emplace(i, std::vector<LexiconDecoderState>());
    }
  }

  std::vector<size_t> idx(N);
  for (int t = 0; t < T; t++) {
    std::iota(idx.begin(), idx.end(), 0);
    if (N > opt_.beamSizeToken) {
      std::partial_sort(
          idx.begin(),
          idx.begin() + opt_.beamSizeToken,
          idx.end(),
          [&t, &N, &emissions](const size_t& l, const size_t& r) {
            return emissions[t * N + l] > emissions[t * N + r];
          });
    }

    candidatesReset(candidatesBestScore_, candidates_, candidatePtrs_);
    for (const LexiconDecoderState& prevHyp : hyp_[startFrame + t]) {
      const TrieNode* prevLex = prevHyp.lex;
      const int prevIdx = prevHyp.token;
      const float lexMaxScore =
          prevLex == lexicon_->getRoot() ? 0 : prevLex->maxScore;
      const TrieNode* prevCmd = prevHyp.cmd;

      /* (1) Try children */
      for (int r = 0; r < std::min(opt_.beamSizeToken, N); ++r) {
        int n = idx[r];
        auto iter = prevLex->children.find(n);
        if (iter == prevLex->children.end()) {
          continue;
        }
        const TrieNodePtr& lex = iter->second;
        double emittingModelScore = emissions[t * N + n];
        if (nDecodedFrames_ + t > 0 &&
            opt_.criterionType == CriterionType::ASG) {
          emittingModelScore += transitions_[n * N + prevIdx];
        }
        double score = prevHyp.score + emittingModelScore;
        if (n == sil_) {
          score += opt_.silScore;
        }

        LMStatePtr lmState;
        double lmScore = 0.;

        if (isLmToken_) {
          auto lmStateScorePair = lm_->score(prevHyp.lmState, n);
          lmState = lmStateScorePair.first;
          lmScore = lmStateScorePair.second;
        }

        // We eat-up a new token
        if (opt_.criterionType != CriterionType::CTC || prevHyp.prevBlank ||
            n != prevIdx) {
          if (!lex->children.empty()) {
            if (!isLmToken_) {
              lmState = prevHyp.lmState;
              lmScore = lex->maxScore - lexMaxScore;
            }

            double cmdScore = 0.;
            double accCmdScore = 0.; // Accumulated command score
            const TrieNode* cmd; // command trie node
            bool cmdBoostEnable = prevHyp.cmdBoostEnable;

            if (command_->getMaxChildren() > 0 && boostOpt_.cmdBoostWeight > 0) {
              cmdBoostToken(cmdScore, accCmdScore, cmdBoostEnable, n, prevHyp.cmdScore, cmd, prevCmd);
            } else {
              cmd = command_->getRoot();
            }
            
            candidatesAdd(
                candidates_,
                candidatesBestScore_,
                opt_.beamThreshold,
                score + opt_.lmWeight * lmScore + boostOpt_.cmdBoostWeight * cmdScore,
                lmState,
                lex.get(),
                cmd,
                &prevHyp,
                n,
                -1,
                false, // prevBlank
                cmdBoostEnable,
                prevHyp.emittingModelScore + emittingModelScore,
                prevHyp.lmScore + lmScore,
                accCmdScore);
          }
        }

        // If we got a true word
        for (auto label : lex->labels) {
          if (prevLex == lexicon_->getRoot() && prevHyp.token == n) {
            // This is to avoid an situation that, when there is word with
            // single token spelling (e.g. X -> x) in the lexicon and token `x`
            // is predicted in several consecutive frames, multiple word `X`
            // will be emitted. This violates the property of CTC, where
            // there must be an blank token in between to predict 2 identical
            // tokens consecutively.
            continue;
          }

          if (!isLmToken_) {
            auto lmStateScorePair = lm_->score(prevHyp.lmState, label);
            lmState = lmStateScorePair.first;
            lmScore = lmStateScorePair.second - lexMaxScore;
          }
     
          double cmdScore = 0.;
          double accCmdScore = 0.; // Accumulated command score
          const TrieNode* cmd; // command trie node
          bool cmdBoostEnable = prevHyp.cmdBoostEnable;

          if (command_->getMaxChildren() > 0 && boostOpt_.cmdBoostWeight > 0) {
            int wpOrW = boostOpt_.wordBoost ? label : n; // depending on boosting type, search word or word piece
            cmdBoostWord(cmdScore, accCmdScore, cmdBoostEnable, wpOrW, prevHyp.cmdScore, cmd, prevCmd);
          } else {
            cmd = command_->getRoot();
          }

          candidatesAdd(
              candidates_,
              candidatesBestScore_,
              opt_.beamThreshold,
              score + opt_.lmWeight * lmScore + opt_.wordScore + boostOpt_.cmdBoostWeight * cmdScore,
              lmState,
              lexicon_->getRoot(),
              cmd,
              &prevHyp,
              n,
              label,
              false, // prevBlank
              cmdBoostEnable,
              prevHyp.emittingModelScore + emittingModelScore,
              prevHyp.lmScore + lmScore,
              accCmdScore);
        }

        // If we got an unknown word
        if (lex->labels.empty() && (opt_.unkScore > kNegativeInfinity)) {
          if (!isLmToken_) {
            auto lmStateScorePair = lm_->score(prevHyp.lmState, unk_);
            lmState = lmStateScorePair.first;
            lmScore = lmStateScorePair.second - lexMaxScore;
          }
          bool cmdBoostEnable = prevHyp.cmdBoostEnable;
          if ((boostOpt_.cmdMatchBegin || boostOpt_.cmdMatchEnd) && cmdBoostEnable) {
            // disable the boost if we have an unk
            cmdBoostEnable = false;
          }
          candidatesAdd(
              candidates_,
              candidatesBestScore_,
              opt_.beamThreshold,
              score + opt_.lmWeight * lmScore + opt_.unkScore - boostOpt_.cmdBoostWeight * prevHyp.cmdScore,
              lmState,
              lexicon_->getRoot(),
              command_->getRoot(),
              &prevHyp,
              n,
              unk_,
              false, // prevBlank
              cmdBoostEnable,
              prevHyp.emittingModelScore + emittingModelScore,
              prevHyp.lmScore + lmScore,
              0.);
        }
      }

      /* (2) Try same lexicon node */
      if (opt_.criterionType != CriterionType::CTC || !prevHyp.prevBlank ||
          prevLex == lexicon_->getRoot()) {
        int n = prevLex == lexicon_->getRoot() ? sil_ : prevIdx;
        double emittingModelScore = emissions[t * N + n];
        if (nDecodedFrames_ + t > 0 &&
            opt_.criterionType == CriterionType::ASG) {
          emittingModelScore += transitions_[n * N + prevIdx];
        }
        double score = prevHyp.score + emittingModelScore;
        if (n == sil_) {
          score += opt_.silScore;
        }

        candidatesAdd(
            candidates_,
            candidatesBestScore_,
            opt_.beamThreshold,
            score,
            prevHyp.lmState,
            prevLex,
            prevCmd,
            &prevHyp,
            n,
            -1,
            false, // prevBlank
            prevHyp.cmdBoostEnable,
            prevHyp.emittingModelScore + emittingModelScore,
            prevHyp.lmScore,
            prevHyp.cmdScore);
      }

      /* (3) CTC only, try blank */
      if (opt_.criterionType == CriterionType::CTC) {
        int n = blank_;
        double emittingModelScore = emissions[t * N + n];
        candidatesAdd(
            candidates_,
            candidatesBestScore_,
            opt_.beamThreshold,
            prevHyp.score + emittingModelScore,
            prevHyp.lmState,
            prevLex,
            prevCmd,
            &prevHyp,
            n,
            -1,
            true, // prevBlank
            prevHyp.cmdBoostEnable,
            prevHyp.emittingModelScore + emittingModelScore,
            prevHyp.lmScore,
            prevHyp.cmdScore);
      }
      // finish proposing
    }

    candidatesStore(
        candidates_,
        candidatePtrs_,
        hyp_[startFrame + t + 1],
        opt_.beamSize,
        candidatesBestScore_ - opt_.beamThreshold,
        opt_.logAdd,
        false);
    updateLMCache(lm_, hyp_[startFrame + t + 1]);
  }

  nDecodedFrames_ += T;
}

void LexiconDecoder::decodeEnd() {
  candidatesReset(candidatesBestScore_, candidates_, candidatePtrs_);
  bool hasNiceEnding = false;
  for (const LexiconDecoderState& prevHyp :
       hyp_[nDecodedFrames_ - nPrunedFrames_]) {
    if (prevHyp.lex == lexicon_->getRoot()) {
      hasNiceEnding = true;
      break;
    }
  }
  for (const LexiconDecoderState& prevHyp :
       hyp_[nDecodedFrames_ - nPrunedFrames_]) {
    const TrieNode* prevLex = prevHyp.lex;
    const LMStatePtr& prevLmState = prevHyp.lmState;

    if (!hasNiceEnding || prevHyp.lex == lexicon_->getRoot()) {
      auto lmStateScorePair = lm_->finish(prevLmState);
      auto lmScore = lmStateScorePair.second;
      double cmdScore = prevHyp.cmdScore;
      if (!boostOpt_.cmdMatchIncr && prevHyp.cmd == command_->getRoot() && cmdScore > 0) {
        // In incremental match case, accumulated score might not non-zero
        // If command trie node is at the root, meaning full command matched
        // we don't need to subtract the accumulated score
        cmdScore = 0.;
      }
      //std::cout << "finish decoding score subtracted " << cmdScore << std::endl;
      candidatesAdd(
          candidates_,
          candidatesBestScore_,
          opt_.beamThreshold,
          prevHyp.score + opt_.lmWeight * lmScore - boostOpt_.cmdBoostWeight * cmdScore, // clean unfinished boost
          lmStateScorePair.first,
          prevLex,
          command_->getRoot(),
          &prevHyp,
          sil_,
          -1,
          false, // prevBlank
          prevHyp.cmdBoostEnable,
          prevHyp.emittingModelScore,
          prevHyp.lmScore + lmScore,
          0.);
    }
  }

  candidatesStore(
      candidates_,
      candidatePtrs_,
      hyp_[nDecodedFrames_ - nPrunedFrames_ + 1],
      opt_.beamSize,
      candidatesBestScore_ - opt_.beamThreshold,
      opt_.logAdd,
      true);
  ++nDecodedFrames_;
}

std::vector<DecodeResult> LexiconDecoder::getAllFinalHypothesis() const {
  int finalFrame = nDecodedFrames_ - nPrunedFrames_;
  if (finalFrame < 1) {
    return std::vector<DecodeResult>{};
  }

  return getAllHypothesis(hyp_.find(finalFrame)->second, finalFrame);
}

DecodeResult LexiconDecoder::getBestHypothesis(int lookBack) const {
  if (nDecodedFrames_ - nPrunedFrames_ - lookBack < 1) {
    return DecodeResult();
  }

  const LexiconDecoderState* bestNode = findBestAncestor(
      hyp_.find(nDecodedFrames_ - nPrunedFrames_)->second, lookBack);
  return getHypothesis(bestNode, nDecodedFrames_ - nPrunedFrames_ - lookBack);
}

int LexiconDecoder::nHypothesis() const {
  int finalFrame = nDecodedFrames_ - nPrunedFrames_;
  return hyp_.find(finalFrame)->second.size();
}

int LexiconDecoder::nDecodedFramesInBuffer() const {
  return nDecodedFrames_ - nPrunedFrames_ + 1;
}

void LexiconDecoder::prune(int lookBack) {
  if (nDecodedFrames_ - nPrunedFrames_ - lookBack < 1) {
    return; // Not enough decoded frames to prune
  }

  /* (1) Find the last emitted word in the best path */
  const LexiconDecoderState* bestNode = findBestAncestor(
      hyp_.find(nDecodedFrames_ - nPrunedFrames_)->second, lookBack);
  if (!bestNode) {
    return; // Not enough decoded frames to prune
  }

  int startFrame = nDecodedFrames_ - nPrunedFrames_ - lookBack;
  if (startFrame < 1) {
    return; // Not enough decoded frames to prune
  }

  /* (2) Move things from back of hyp_ to front and normalize scores */
  pruneAndNormalize(hyp_, startFrame, lookBack);

  nPrunedFrames_ = nDecodedFrames_ - lookBack;
}

void LexiconDecoder::setCommandTrie(const TriePtr& command) {
  command_ = command;
}

void LexiconDecoder::cmdBoostToken(double &cmdScore, double &accCmdScore,
                                   bool &cmdBoostEnable, const int &token,
                                   const double& prevHypCmdScore,
                                   const TrieNode* &cmd, const TrieNode* &prevCmd) {
  if (boostOpt_.cmdMatchEnd && !cmdBoostEnable && prevHypCmdScore > 0 && \
      std::find(boostOpt_.cmdBoostIgnore.begin(), \
      boostOpt_.cmdBoostIgnore.end(), token) != boostOpt_.cmdBoostIgnore.end()) {
    // if we have an extra token matched after the command tokens
    // and this token is not in the ignored list
    // then this hypothesis does not only contain command
    // we do score fallback
    cmd = prevCmd;
    cmdScore = -prevHypCmdScore;
    accCmdScore = 0.;
  } else if (boostOpt_.wordBoost || !cmdBoostEnable) {
    // bypass the word piece level boosting if we do word level boosting
    // or the boost is disabled for this hyp
    cmd = prevCmd;
    accCmdScore = prevHypCmdScore;
  } else {
    // Command boosting logic:
    // First try to find the token in the current command trienode's
    //  children's nodes. 
    // If the token is in the children, add to the overall score a boost.
    // If the token is NOT in the children
    //  It is possible the token is in the ignored token list, e.g. separator
    //   In this case, we do nothing for the current hypothesis
    //  If the token is not in the ignored token list
    //   We subtract the overall score by the accumulated boost score
    //   for boosting fallback.
    //   We also goes back to the trie root
    //   We immediately restart the search from the trie root
    auto iter = prevCmd->children.find(token);
    if (iter != prevCmd->children.end()) {
      //std::cout << startFrame + t << " new token: index found in the trie " << token << std::endl;
      cmdScore = boostOpt_.cmdFixedScore;
      cmd = (iter->second).get();
      accCmdScore = prevHypCmdScore + cmdScore;
    } else if (std::find(boostOpt_.cmdBoostIgnore.begin(), \
        boostOpt_.cmdBoostIgnore.end(), token) != boostOpt_.cmdBoostIgnore.end()) {
      // word piece in the ignored list
      //std::cout << startFrame + t << " new token: index ignored " << token << std::endl;
      cmd = prevCmd;
      accCmdScore = prevHypCmdScore;
    } else {
      // word piece not in the trie
      cmdScore = -prevHypCmdScore;
      cmd = command_->getRoot();
      accCmdScore = 0;
      if (boostOpt_.cmdMatchBegin) {
        // disable the boost because we have a no match
        cmdBoostEnable = false;
      } else {
        // restart the search immediately
        iter = cmd->children.find(token);
        if (iter != cmd->children.end()) {
          //std::cout << startFrame + t << " new token: restart from root, index found in the trie " << token << std::endl;
          cmdScore += boostOpt_.cmdFixedScore;
          cmd = (iter->second).get();
          accCmdScore += boostOpt_.cmdFixedScore;
        }
      }
    }
  }
}

void LexiconDecoder::cmdBoostWord(double &cmdScore, double &accCmdScore,
                                  bool &cmdBoostEnable, const int &token,
                                  const double &prevHypCmdScore,
                                  const TrieNode* &cmd, const TrieNode* &prevCmd) {
  // Command boosting logic:
  // First try to find the word or WP (word piece) in the command current trienode's
  //  children nodes. 
  // If the word is in the children, add to the overall score a boost.
  //  Check if the updated node has complete commands by checking the node labels.
  //  If it has commands (labels is not empty), which means a command is recognized,
  //   then we don't fallback the accumulated boost score.
  //   We again check if it is a leaf node by its children
  //   If it is a leaf node, we go back to the root
  //  If it doesn't have command, which means a command is not recognized yet
  //   We accumulate the boost score for potential fallback
  // If the word is NOT in the children
  //  Check if the node is in the ignored list
  //  If it's not in the ignored list
  //   We subtract the score by the accumulated boost score, and goes back 
  //   to the trie root for fallback
  //   We also restart the search immediately from the root
  if (boostOpt_.cmdMatchEnd && !cmdBoostEnable && prevHypCmdScore > 0 && \
      std::find(boostOpt_.cmdBoostIgnore.begin(), \
      boostOpt_.cmdBoostIgnore.end(), token) != boostOpt_.cmdBoostIgnore.end()) {
    // if we have an extra token matched after the command tokens
    // and this token is not in the ignored list
    // then this hypothesis does not only contain command
    // we do score fallback
    cmd = prevCmd;
    cmdScore = -prevHypCmdScore;
    accCmdScore = 0.;
  } else if (!cmdBoostEnable) {
    cmd = prevCmd;
    accCmdScore = prevHypCmdScore;
  } else {
    auto iter = prevCmd->children.find(token);
    if (iter != prevCmd->children.end()) {
      cmdScore = boostOpt_.cmdFixedScore;
      cmd = (iter->second).get();
      if ((cmd->labels).empty()) {
        //std::cout << startFrame + t << " word end: index found, command not finished " << token << " word label " << label << std::endl;
        accCmdScore = prevHypCmdScore + cmdScore;
      } else {
        // full command match
        //std::cout << startFrame + t << " word end: index found, command finished " << token << " word label " << label << " boost score " << prevHypCmdScore + cmdScore << " cmd id " << cmd->labels[0] << std::endl;
        if (boostOpt_.cmdMatchIncr) {
          accCmdScore = 0; // no score fallback
        } else {
          accCmdScore = prevHypCmdScore + cmdScore;
        }
        if (cmd->children.empty()) {
          // leaf node
          cmd = command_->getRoot();
          if (boostOpt_.cmdMatchEnd) {
            // disable boosting as we reached to the end of the command
            cmdBoostEnable = false;
          }
        }
      }
    } else if (std::find(boostOpt_.cmdBoostIgnore.begin(), \
      boostOpt_.cmdBoostIgnore.end(), token) != boostOpt_.cmdBoostIgnore.end()) {
      // word in the ignore list
      //std::cout << startFrame + t << " word end: index ignored " << token << std::endl;
      cmd = prevCmd;
      accCmdScore = prevHypCmdScore;
    } else {
      // word not in the command trieNode children
      cmdScore = -prevHypCmdScore;
      cmd = command_->getRoot();
      accCmdScore = 0;
      if (boostOpt_.cmdMatchBegin) {
        // disable the boost if we have a no match
        cmdBoostEnable = false;
      } else {
        // restart the search immediately
        iter = cmd->children.find(token);
        if (iter != cmd->children.end()) {
          cmdScore += boostOpt_.cmdFixedScore;
          cmd = (iter->second).get();
          if ((cmd->labels).empty()) {
            // not a leaf node
            //std::cout << startFrame + t << " word end: restart from root, index found, command finished " << token << " word label " << label << " boost score " << cmdScore << std::endl;
            accCmdScore += boostOpt_.cmdFixedScore;
          } else if (cmd->children.empty()) {
            // a leaf node
            cmd = command_->getRoot();
          }
        }
      }
    }
  }

}

} // namespace text
} // namespace lib
} // namespace fl
