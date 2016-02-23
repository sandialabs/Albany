//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef TIMEDEPBC_HPP
#define TIMEDEPBC_HPP

#include "PHAL_Dirichlet.hpp"
#include <vector>

namespace LCM {
/** \brief Time Dependent BC Dirichlet evaluator.
 */

template <typename EvalT, typename Traits>
class TimeDepBC_Base : public PHAL::Dirichlet<EvalT, Traits> {
private:
  typedef typename EvalT::ScalarT ScalarT;

public:
  TimeDepBC_Base(Teuchos::ParameterList& p);
  ScalarT computeVal(RealType time);

protected:
  const int offset;
  std::vector< RealType > timeValues;
  std::vector< RealType > BCValues;
};

template<typename EvalT, typename Traits>
class TimeDepBC : public TimeDepBC_Base<EvalT, Traits> {
public:
  TimeDepBC(Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename EvalT::ScalarT ScalarT;
};

}

#endif
