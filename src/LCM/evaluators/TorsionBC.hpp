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


#ifndef TORSIONBC_HPP
#define TORSIONBC_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Epetra_Vector.h"

#include "Sacado_ParameterAccessor.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Dirichlet.hpp"

namespace LCM {
/** \brief Torsion BC Dirichlet evaluator

*/


// **************************************************************
// **************************************************************
// * Specialization of the DirichletBase class
// **************************************************************
// **************************************************************

template<typename EvalT, typename Traits> class TorsionBC;

template <typename EvalT, typename Traits> 
class TorsionBC_Base : public PHAL::DirichletBase<EvalT, Traits> {
public:
  typedef typename EvalT::ScalarT ScalarT;
  TorsionBC_Base(Teuchos::ParameterList& p);
  void computeBCs(double* coord, ScalarT& Xval, ScalarT& Yval, 
                  const RealType time);

  RealType thetaDot;
};

// **************************************************************
// Residual 
// **************************************************************
template<typename Traits>
class TorsionBC<PHAL::AlbanyTraits::Residual,Traits>
  : public TorsionBC_Base<PHAL::AlbanyTraits::Residual, Traits> {
public:
  TorsionBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Jacobian
// **************************************************************
template<typename Traits>
class TorsionBC<PHAL::AlbanyTraits::Jacobian,Traits>
   : public TorsionBC_Base<PHAL::AlbanyTraits::Jacobian, Traits> {
public:
  TorsionBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Tangent
// **************************************************************
template<typename Traits>
class TorsionBC<PHAL::AlbanyTraits::Tangent,Traits>
   : public TorsionBC_Base<PHAL::AlbanyTraits::Tangent, Traits> {
public:
  TorsionBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::Tangent::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Stochastic Galerkin Residual 
// **************************************************************
template<typename Traits>
class TorsionBC<PHAL::AlbanyTraits::SGResidual,Traits>
   : public TorsionBC_Base<PHAL::AlbanyTraits::SGResidual, Traits> {
public:
  TorsionBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::SGResidual::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Stochastic Galerkin Jacobian
// **************************************************************
template<typename Traits>
class TorsionBC<PHAL::AlbanyTraits::SGJacobian,Traits>
   : public TorsionBC_Base<PHAL::AlbanyTraits::SGJacobian, Traits> {
public:
  TorsionBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::SGJacobian::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Stochastic Galerkin Tangent
// **************************************************************
template<typename Traits>
class TorsionBC<PHAL::AlbanyTraits::SGTangent,Traits>
   : public TorsionBC_Base<PHAL::AlbanyTraits::SGTangent, Traits> {
public:
  TorsionBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::SGTangent::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Multi-point Residual 
// **************************************************************
template<typename Traits>
class TorsionBC<PHAL::AlbanyTraits::MPResidual,Traits>
   : public TorsionBC_Base<PHAL::AlbanyTraits::MPResidual, Traits> {
public:
  TorsionBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::MPResidual::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Multi-point Jacobian
// **************************************************************
template<typename Traits>
class TorsionBC<PHAL::AlbanyTraits::MPJacobian,Traits>
   : public TorsionBC_Base<PHAL::AlbanyTraits::MPJacobian, Traits> {
public:
  TorsionBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::MPJacobian::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Multi-point Tangent
// **************************************************************
template<typename Traits>
class TorsionBC<PHAL::AlbanyTraits::MPTangent,Traits>
   : public TorsionBC_Base<PHAL::AlbanyTraits::MPTangent, Traits> {
public:
  TorsionBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::MPTangent::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

}

#endif
