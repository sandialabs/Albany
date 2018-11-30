//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_TIMEDEPBC_HPP
#define PHAL_TIMEDEPBC_HPP

#include <vector>
#include "PHAL_Dirichlet.hpp"

namespace PHAL {
/** \brief Time Dependent BC Dirichlet evaluator.
 */

template <typename EvalT, typename Traits>
class TimeDepDBC_Base : public PHAL::Dirichlet<EvalT, Traits>
{
 private:
  typedef typename EvalT::ScalarT ScalarT;

 public:
  TimeDepDBC_Base(Teuchos::ParameterList& p);
  ScalarT
  computeVal(RealType time);

 protected:
  const int             offset;
  std::vector<RealType> timeValues;
  std::vector<RealType> BCValues;
};

template <typename EvalT, typename Traits>
class TimeDepDBC : public TimeDepDBC_Base<EvalT, Traits>
{
 public:
  TimeDepDBC(Teuchos::ParameterList& p);
  void
  evaluateFields(typename Traits::EvalData d);

 private:
  typedef typename EvalT::ScalarT ScalarT;
};

}  // namespace PHAL

#endif
