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
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


#ifndef TIMEDEPBC_HPP
#define TIMEDEPBC_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Epetra_Vector.h"

#include "Sacado_ParameterAccessor.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Dirichlet.hpp"
#include <vector>

namespace LCM {
/** \brief Time Dependent BC Dirichlet evaluator

*/


// **************************************************************
// **************************************************************
// * Specialization of the DirichletBase class
// **************************************************************
// **************************************************************

template<typename EvalT, typename Traits> class TimeDepBC;

template <typename EvalT, typename Traits> 
class TimeDepBC_Base : public PHAL::DirichletBase<EvalT, Traits> {
public:
  typedef typename EvalT::ScalarT ScalarT;
  TimeDepBC_Base(Teuchos::ParameterList& p);
  ScalarT computeVal(RealType time);

protected:
  const int offset;
  std::vector< RealType > timeValues;
  std::vector< RealType > BCValues;
};

// **************************************************************
// Residual 
// **************************************************************
template<typename Traits>
class TimeDepBC<PHAL::AlbanyTraits::Residual,Traits>
  : public TimeDepBC_Base<PHAL::AlbanyTraits::Residual, Traits> {
public:
  TimeDepBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Jacobian
// **************************************************************
template<typename Traits>
class TimeDepBC<PHAL::AlbanyTraits::Jacobian,Traits>
   : public TimeDepBC_Base<PHAL::AlbanyTraits::Jacobian, Traits> {
public:
  TimeDepBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Tangent
// **************************************************************
template<typename Traits>
class TimeDepBC<PHAL::AlbanyTraits::Tangent,Traits>
   : public TimeDepBC_Base<PHAL::AlbanyTraits::Tangent, Traits> {
public:
  TimeDepBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::Tangent::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Stochastic Galerkin Residual 
// **************************************************************
template<typename Traits>
class TimeDepBC<PHAL::AlbanyTraits::SGResidual,Traits>
   : public TimeDepBC_Base<PHAL::AlbanyTraits::SGResidual, Traits> {
public:
  TimeDepBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::SGResidual::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Stochastic Galerkin Jacobian
// **************************************************************
template<typename Traits>
class TimeDepBC<PHAL::AlbanyTraits::SGJacobian,Traits>
   : public TimeDepBC_Base<PHAL::AlbanyTraits::SGJacobian, Traits> {
public:
  TimeDepBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::SGJacobian::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Stochastic Galerkin Tangent
// **************************************************************
template<typename Traits>
class TimeDepBC<PHAL::AlbanyTraits::SGTangent,Traits>
   : public TimeDepBC_Base<PHAL::AlbanyTraits::SGTangent, Traits> {
public:
  TimeDepBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::SGTangent::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Multi-point Residual 
// **************************************************************
template<typename Traits>
class TimeDepBC<PHAL::AlbanyTraits::MPResidual,Traits>
   : public TimeDepBC_Base<PHAL::AlbanyTraits::MPResidual, Traits> {
public:
  TimeDepBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::MPResidual::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Multi-point Jacobian
// **************************************************************
template<typename Traits>
class TimeDepBC<PHAL::AlbanyTraits::MPJacobian,Traits>
   : public TimeDepBC_Base<PHAL::AlbanyTraits::MPJacobian, Traits> {
public:
  TimeDepBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::MPJacobian::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Multi-point Tangent
// **************************************************************
template<typename Traits>
class TimeDepBC<PHAL::AlbanyTraits::MPTangent,Traits>
   : public TimeDepBC_Base<PHAL::AlbanyTraits::MPTangent, Traits> {
public:
  TimeDepBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::MPTangent::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

}

#endif
