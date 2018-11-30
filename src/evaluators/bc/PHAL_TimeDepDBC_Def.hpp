//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"

namespace PHAL {

template <typename EvalT, typename Traits>
TimeDepDBC_Base<EvalT, Traits>::TimeDepDBC_Base(Teuchos::ParameterList& p)
    : offset(p.get<int>("Equation Offset")), PHAL::Dirichlet<EvalT, Traits>(p)
{
  timeValues = p.get<Teuchos::Array<RealType>>("Time Values").toVector();
  BCValues   = p.get<Teuchos::Array<RealType>>("BC Values").toVector();

  TEUCHOS_TEST_FOR_EXCEPTION(
      !(timeValues.size() == BCValues.size()),
      Teuchos::Exceptions::InvalidParameter,
      "Dimension of \"Time Values\" and \"BC Values\" do not match");
}

template <typename EvalT, typename Traits>
typename TimeDepDBC_Base<EvalT, Traits>::ScalarT
TimeDepDBC_Base<EvalT, Traits>::computeVal(RealType time)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      time > timeValues.back(),
      Teuchos::Exceptions::InvalidParameter,
      "Time is growing unbounded!");

  ScalarT      val;
  RealType     slope;
  unsigned int index(0);

  while (timeValues[index] < time) index++;

  if (index == 0)
    val = BCValues[index];
  else {
    slope =
        ((BCValues[index] - BCValues[index - 1]) /
         (timeValues[index] - timeValues[index - 1]));
    val = BCValues[index - 1] + slope * (time - timeValues[index - 1]);
  }

  return val;
}

template <typename EvalT, typename Traits>
TimeDepDBC<EvalT, Traits>::TimeDepDBC(Teuchos::ParameterList& p)
    : TimeDepDBC_Base<EvalT, Traits>(p)
{
}

template <typename EvalT, typename Traits>
void
TimeDepDBC<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  this->value = this->computeVal(workset.current_time);
  PHAL::Dirichlet<EvalT, Traits>::evaluateFields(workset);
}

}  // namespace PHAL
