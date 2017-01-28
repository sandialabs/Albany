//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ATOT_Optimizer_DOTk_HPP
#define ATOT_Optimizer_DOTk_HPP


#include "Albany_StateManager.hpp"

#include <string>
#include <vector>
#include <tr1/memory>

#include "Teuchos_ParameterList.hpp"

#include "DOTk/DOTk_Types.hpp"
#include "DOTk/vector.hpp"

#include "ATOT_Optimizer.hpp"
#include "ATO_DOTk_ContinuousOperators.hpp"

namespace ATOT {

class Optimizer_DOTk : public Optimizer {
 public:
  Optimizer_DOTk(const Teuchos::ParameterList& optimizerParams);
  ~Optimizer_DOTk();
  void Initialize();
  void Optimize();
 private:

  ATO_DOTk_ContinuousOperators* myCoOperators;
};

}
#endif
