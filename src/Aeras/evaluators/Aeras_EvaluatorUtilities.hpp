//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AERAS_EVALUATORUTILITIES_HPP
#define AERAS_EVALUATORUTILITIES_HPP

namespace Aeras {

/* Aeras uses just 1 workset quite often. This memoizer (which is not really a
 * true memoizer at present, and may never need to be) detects whether that fact
 * holds. If it does, an evaluator doesn't have to do calculations more than
 * once.
 *
 * WARNING: However, it is a bug to use this class if the evaluator's output
 * fields change over time.
 */
template<typename Traits>
class MDFieldMemoizer {
  int prev_workset_index_;

public:
  MDFieldMemoizer () : prev_workset_index_(-1) {}

  bool haveStoredData (typename Traits::EvalData workset) {
    const bool stored = workset.wsIndex == prev_workset_index_;
    prev_workset_index_ = workset.wsIndex;
    return stored;
  }
};

} // namespace Aeras

#endif
