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


#ifndef KFIELDBC_HPP
#define KFIELDBC_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Epetra_Vector.h"

#include "Sacado_ParameterAccessor.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Dirichlet.hpp"

/** \brief KfieldBC Dirichlet evaluator

*/

namespace LCM {

// **************************************************************
// **************************************************************
// * Specialization of the DirichletBase class
// **************************************************************
// **************************************************************

template<typename EvalT, typename Traits> class KfieldBC;

// **************************************************************
// Residual 
// **************************************************************
template<typename Traits>
class KfieldBC<PHAL::AlbanyTraits::Residual,Traits>
  : public PHAL::DirichletBase<PHAL::AlbanyTraits::Residual, Traits> {
public:
  KfieldBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
  ScalarT& getValue(const std::string &n);
  RealType mu, nu, KIval, KIIval;
  ScalarT KI, KII;
};

// **************************************************************
// Jacobian
// **************************************************************
template<typename Traits>
class KfieldBC<PHAL::AlbanyTraits::Jacobian,Traits>
   : public PHAL::DirichletBase<PHAL::AlbanyTraits::Jacobian, Traits> {
public:
  KfieldBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
  ScalarT& getValue(const std::string &n);
  RealType mu, nu, KIval, KIIval;
  ScalarT KI, KII;
};

// **************************************************************
// Tangent
// **************************************************************
template<typename Traits>
class KfieldBC<PHAL::AlbanyTraits::Tangent,Traits>
   : public PHAL::DirichletBase<PHAL::AlbanyTraits::Tangent, Traits> {
public:
  KfieldBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::Tangent::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
  ScalarT& getValue(const std::string &n);
  RealType mu, nu, KIval, KIIval;
  ScalarT KI, KII;
};

// **************************************************************
// Stochastic Galerkin Residual 
// **************************************************************
template<typename Traits>
class KfieldBC<PHAL::AlbanyTraits::SGResidual,Traits>
   : public PHAL::DirichletBase<PHAL::AlbanyTraits::SGResidual, Traits> {
public:
  KfieldBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::SGResidual::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
  ScalarT& getValue(const std::string &n);
  RealType mu, nu, KIval, KIIval;
  ScalarT KI, KII;
};

// **************************************************************
// Stochastic Galerkin Jacobian
// **************************************************************
template<typename Traits>
class KfieldBC<PHAL::AlbanyTraits::SGJacobian,Traits>
   : public PHAL::DirichletBase<PHAL::AlbanyTraits::SGJacobian, Traits> {
public:
  KfieldBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::SGJacobian::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
  ScalarT& getValue(const std::string &n);
  RealType mu, nu, KIval, KIIval;
  ScalarT KI, KII;
};

// **************************************************************
// Multi-point Residual 
// **************************************************************
template<typename Traits>
class KfieldBC<PHAL::AlbanyTraits::MPResidual,Traits>
   : public PHAL::DirichletBase<PHAL::AlbanyTraits::MPResidual, Traits> {
public:
  KfieldBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::MPResidual::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
  ScalarT& getValue(const std::string &n);
  RealType mu, nu, KIval, KIIval;
  ScalarT KI, KII;
};

// **************************************************************
// Multi-point Jacobian
// **************************************************************
template<typename Traits>
class KfieldBC<PHAL::AlbanyTraits::MPJacobian,Traits>
   : public PHAL::DirichletBase<PHAL::AlbanyTraits::MPJacobian, Traits> {
public:
  KfieldBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::MPJacobian::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
  ScalarT& getValue(const std::string &n);
  RealType mu, nu, KIval, KIIval;
  ScalarT KI, KII;
};

}

#endif
