/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Glen Hansen, gahanse@sandia.gov                    *
\********************************************************************/


#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include <string>


namespace LCM {

//**********************************************************************
template<typename EvalT, typename Traits>
TimeTracBC_Base<EvalT, Traits>::
TimeTracBC_Base(Teuchos::ParameterList& p) :
  PHAL::Neumann<EvalT, Traits>(p) {

  timeValues = p.get<Teuchos::Array<RealType> >("Time Values").toVector();
  BCValues = p.get<Teuchos::TwoDArray<RealType> >("BC Values");

  TEUCHOS_TEST_FOR_EXCEPTION( !(this->cellDims == BCValues.getNumCols()),
			      Teuchos::Exceptions::InvalidParameter,
			      "Dimension of the current problem and \"BC Values\" do not match" );

  TEUCHOS_TEST_FOR_EXCEPTION( !(timeValues.size() == BCValues.getNumRows()),
			      Teuchos::Exceptions::InvalidParameter,
			      "Dimension of \"Time Values\" and \"BC Values\" do not match" );

}

//**********************************************************************
template<typename EvalT, typename Traits>
void
TimeTracBC_Base<EvalT, Traits>::
computeVal(RealType time)
{
  TEUCHOS_TEST_FOR_EXCEPTION( time > timeValues.back(),
			      Teuchos::Exceptions::InvalidParameter,
			      "Time is growing unbounded!" );
  ScalarT Val;
  RealType slope;
  unsigned int Index(0);

  while( timeValues[Index] < time )
    Index++;

  if (Index == 0)
    for(int dim = 0; dim < this->cellDims; dim++)
      this->dudx[dim] = BCValues(dim, Index);
  else
  {
    for(size_t dim = 0; dim < this->cellDims; dim++){
      slope = ( BCValues(dim, Index) - BCValues(dim, Index - 1) ) / ( timeValues[Index] - timeValues[Index - 1] );
      this->dudx[dim] = BCValues(dim, Index-1) + slope * ( time - timeValues[Index - 1] );
    }
  }

  return;

}

template<typename EvalT, typename Traits>
TimeTracBC<EvalT,Traits>::
TimeTracBC(Teuchos::ParameterList& p)
  : TimeTracBC_Base<EvalT,Traits>(p)
{
}

template<typename EvalT, typename Traits>
void TimeTracBC<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

  RealType time = workset.current_time;
  this->computeVal(time);

  this->template PHAL::Neumann<EvalT, Traits>::evaluateFields(workset);

}

}
