//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_Utils.hpp"
#include "PHAL_TimeDepSDBC.hpp"
#include "Phalanx_DataLayout.hpp"

#include "PHAL_SDirichlet_Def.hpp"

namespace PHAL {

//
//
//
template <typename EvalT, typename Traits>
TimeDepSDBC_Base<EvalT, Traits>::TimeDepSDBC_Base(Teuchos::ParameterList& p)
    : offset_(p.get<int>("Equation Offset")), PHAL::SDirichlet<EvalT, Traits>(p)
{
  times_  = p.get<Teuchos::Array<RealType>>("Time Values").toVector();
  values_  = p.get<Teuchos::Array<RealType>>("BC Values").toVector();

  ALBANY_ASSERT(
      times_.size() == values_.size(),
      "Number of times and number of values must match");
}

//
//
//
template <typename EvalT, typename Traits>
typename TimeDepSDBC_Base<EvalT, Traits>::ScalarT
TimeDepSDBC_Base<EvalT, Traits>::computeVal(RealType time)
{
  ScalarT value{0.0};

  size_t index{0};

  while (times_[index] < time) index++;

  if (index == 0) {
    value = values_[index];
  } else {
    RealType const slope = (values_[index] - values_[index - 1]) /
                           (times_[index] - times_[index - 1]);

    value = values_[index - 1] + slope * (time - times_[index - 1]);
  }

  return value;
}

//
//
//
template <typename EvalT, typename Traits>
TimeDepSDBC<EvalT, Traits>::TimeDepSDBC(Teuchos::ParameterList& p)
    : TimeDepSDBC_Base<EvalT, Traits>(p)
{
}

//
//
//
template <typename EvalT, typename Traits>
void
TimeDepSDBC<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  this->value = this->computeVal(workset.current_time);
  PHAL::SDirichlet<EvalT, Traits>::evaluateFields(workset);
}

}  // namespace PHAL
