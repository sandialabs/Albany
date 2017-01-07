//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ATOT_Optimizer_Pareto_HPP
#define ATOT_Optimizer_Pareto_HPP


#include "Albany_StateManager.hpp"

#include <string>
#include <vector>
#include <tr1/memory>

#include "Teuchos_ParameterList.hpp"
#include "ATOT_Optimizer.hpp"

namespace ATOT {

class Optimizer_Pareto : public Optimizer {
 public:
  Optimizer_Pareto(const Teuchos::ParameterList& optimizerParams);
  ~Optimizer_Pareto();
  void Initialize();
  void Optimize();
 private:

  void computeUpdatedTopology(double volfrac);

  double f;
  double f_last;
  double* p;
  double* p_last;
  double* dfdp;

  int numOptDofs;

  double _optVolume;
  double _minDensity;

  double _volFrac;

  double _volFracHigh;
  double _volFracLow;
  int    _nVolFracSteps;

  double _volConvTol;
  double _volMaxIter;
};

}
#endif
