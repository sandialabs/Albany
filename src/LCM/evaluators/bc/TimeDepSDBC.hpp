//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_TimeDepSDBC_hpp)
#define LCM_TimeDepSDBC_hpp

#include <vector>

#include "PHAL_SDirichlet.hpp"

namespace LCM {

///
/// Time-dependendt, strongly-enforced Dirichlet BC evauator
///
template<typename EvalT, typename Traits>
class TimeDepSDBC_Base: public PHAL::SDirichlet<EvalT, Traits>
{
private:
  using ScalarT = typename EvalT::ScalarT;

public:
  TimeDepSDBC_Base(Teuchos::ParameterList & p);

  ScalarT
  computeVal(RealType time);

protected:
  int const
  offset_;

  std::vector<RealType>
  times_;

  std::vector<RealType>
  values_;
};

template<typename EvalT, typename Traits>
class TimeDepSDBC: public TimeDepSDBC_Base<EvalT, Traits>
{
public:
  TimeDepSDBC(Teuchos::ParameterList & p);

  void
  evaluateFields(typename Traits::EvalData d);

private:
  using ScalarT = typename EvalT::ScalarT;
};

}

#endif // LCM_TimeDepSDBC_hpp
