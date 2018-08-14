//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <string>
#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"

namespace LCM {

//**********************************************************************
template <typename EvalT, typename Traits>
RigidContactBC_Base<EvalT, Traits>::RigidContactBC_Base(
    Teuchos::ParameterList& p)
    : PHAL::Neumann<EvalT, Traits>(p)
{
  timeValues = p.get<Teuchos::Array<RealType>>("Time Values").toVector();
  BCValues   = p.get<Teuchos::TwoDArray<RealType>>("BC Values");

  if (this->bc_type == PHAL::NeumannBase<EvalT, Traits>::COORD)

    TEUCHOS_TEST_FOR_EXCEPTION(
        !(this->cellDims == BCValues.getNumCols()),
        Teuchos::Exceptions::InvalidParameter,
        "Dimension of the current problem and \"BC Values\" do not match");

  TEUCHOS_TEST_FOR_EXCEPTION(
      !(timeValues.size() == BCValues.getNumRows()),
      Teuchos::Exceptions::InvalidParameter,
      "Dimension of \"Time Values\" and \"BC Values\" do not match");
}

//**********************************************************************
template <typename EvalT, typename Traits>
void
RigidContactBC_Base<EvalT, Traits>::computeVal(RealType time)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      time > timeValues.back(),
      Teuchos::Exceptions::InvalidParameter,
      "Time is growing unbounded!");
  ScalarT      Val;
  RealType     slope;
  unsigned int Index(0);

  while (timeValues[Index] < time) Index++;

  if (Index == 0)
    this->const_val = BCValues(0, Index);
  else {
    slope = (BCValues(0, Index) - BCValues(0, Index - 1)) /
            (timeValues[Index] - timeValues[Index - 1]);
    this->const_val =
        BCValues(0, Index - 1) + slope * (time - timeValues[Index - 1]);
  }

  return;
}

template <typename EvalT, typename Traits>
void
RigidContactBC_Base<EvalT, Traits>::computeCoordVal(RealType time)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      time > timeValues.back(),
      Teuchos::Exceptions::InvalidParameter,
      "Time is growing unbounded!");
  ScalarT      Val;
  RealType     slope;
  unsigned int Index(0);

  while (timeValues[Index] < time) Index++;

  if (Index == 0)
    for (int dim = 0; dim < this->cellDims; dim++)
      this->dudx[dim] = BCValues(dim, Index);
  else {
    for (size_t dim = 0; dim < this->cellDims; dim++) {
      slope = (BCValues(dim, Index) - BCValues(dim, Index - 1)) /
              (timeValues[Index] - timeValues[Index - 1]);
      this->dudx[dim] =
          BCValues(dim, Index - 1) + slope * (time - timeValues[Index - 1]);
    }
  }

  return;
}

template <typename EvalT, typename Traits>
RigidContactBC<EvalT, Traits>::RigidContactBC(Teuchos::ParameterList& p)
    : RigidContactBC_Base<EvalT, Traits>(p)
{
}

template <typename EvalT, typename Traits>
void
RigidContactBC<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  RealType time = workset.current_time;

  switch (this->bc_type) {
    case PHAL::NeumannBase<EvalT, Traits>::INTJUMP:
    case PHAL::NeumannBase<EvalT, Traits>::PRESS:
    case PHAL::NeumannBase<EvalT, Traits>::NORMAL:
      // calculate scalar value of BC based on current time

      this->computeVal(time);
      break;

    case PHAL::NeumannBase<EvalT, Traits>::COORD:
      // calculate a value of BC for each coordinate based on current time

      this->computeCoordVal(time);
      break;

    default:

      TEUCHOS_TEST_FOR_EXCEPTION(
          true,
          std::logic_error,
          "Time dependent Neumann boundary condition of type - "
              << this->bc_type << " is not supported");
      break;
  }

  PHAL::Neumann<EvalT, Traits>::evaluateFields(workset);
}

}  // namespace LCM
