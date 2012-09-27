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


#ifndef TIMETRACBC_HPP
#define TIMETRACBC_HPP

#include "PHAL_Neumann.hpp"

namespace LCM {

/** \brief Time dependent Neumann boundary condition evaluator

*/

template<typename EvalT, typename Traits>
class TimeTracBC_Base : public PHAL::Neumann<EvalT, Traits> {

public:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  TimeTracBC_Base(Teuchos::ParameterList& p);

  void computeVal(RealType time);

protected:

  std::vector< RealType > timeValues;
  Teuchos::TwoDArray< RealType > BCValues;

};

template<typename EvalT, typename Traits>
class TimeTracBC : public TimeTracBC_Base<EvalT, Traits>  {

public:
  TimeTracBC(Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename EvalT::ScalarT ScalarT;
};


}

#endif
