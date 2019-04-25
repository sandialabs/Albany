//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ATO_OPTIMIZER_DOTK_HPP
#define ATO_OPTIMIZER_DOTK_HPP

#include "Teuchos_ParameterList.hpp"

#include "ATO_Optimizer.hpp"

namespace ATO {

// ATO forward declarations
class ATO_DOTk_ContinuousOperators;

class Optimizer_DOTk : public Optimizer {
public:
  Optimizer_DOTk(const Teuchos::ParameterList& optimizerParams);
  ~Optimizer_DOTk();
  void Initialize();
  void Optimize();
private:

  ATO_DOTk_ContinuousOperators* myCoOperators;
};

} // namespace ATO

#endif // ATO_OPTIMIZER_DOTK_HPP
