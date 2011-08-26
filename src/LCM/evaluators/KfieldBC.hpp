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

namespace LCM {
/** \brief KfieldBC Dirichlet evaluator

*/


// **************************************************************
// **************************************************************
// * Specialization of the DirichletBase class
// **************************************************************
// **************************************************************

template<typename EvalT, typename Traits> class KfieldBC;

template <typename EvalT, typename Traits> 
class KfieldBC_Base : public PHAL::DirichletBase<EvalT, Traits> {
public:
  typedef typename EvalT::ScalarT ScalarT;
  KfieldBC_Base(Teuchos::ParameterList& p);
  ScalarT& getValue(const std::string &n);
  void computeBCs(double* coord, ScalarT& Xval, ScalarT& Yval);

  RealType mu, nu, KIval, KIIval;
  ScalarT KI, KII;
  std::string KI_name, KII_name;
};

// **************************************************************
// Residual 
// **************************************************************
template<typename Traits>
class KfieldBC<PHAL::AlbanyTraits::Residual,Traits>
  : public KfieldBC_Base<PHAL::AlbanyTraits::Residual, Traits> {
public:
  KfieldBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Jacobian
// **************************************************************
template<typename Traits>
class KfieldBC<PHAL::AlbanyTraits::Jacobian,Traits>
   : public KfieldBC_Base<PHAL::AlbanyTraits::Jacobian, Traits> {
public:
  KfieldBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Tangent
// **************************************************************
template<typename Traits>
class KfieldBC<PHAL::AlbanyTraits::Tangent,Traits>
   : public KfieldBC_Base<PHAL::AlbanyTraits::Tangent, Traits> {
public:
  KfieldBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::Tangent::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Stochastic Galerkin Residual 
// **************************************************************
template<typename Traits>
class KfieldBC<PHAL::AlbanyTraits::SGResidual,Traits>
   : public KfieldBC_Base<PHAL::AlbanyTraits::SGResidual, Traits> {
public:
  KfieldBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::SGResidual::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Stochastic Galerkin Jacobian
// **************************************************************
template<typename Traits>
class KfieldBC<PHAL::AlbanyTraits::SGJacobian,Traits>
   : public KfieldBC_Base<PHAL::AlbanyTraits::SGJacobian, Traits> {
public:
  KfieldBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::SGJacobian::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Stochastic Galerkin Tangent
// **************************************************************
template<typename Traits>
class KfieldBC<PHAL::AlbanyTraits::SGTangent,Traits>
   : public KfieldBC_Base<PHAL::AlbanyTraits::SGTangent, Traits> {
public:
  KfieldBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::SGTangent::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Multi-point Residual 
// **************************************************************
template<typename Traits>
class KfieldBC<PHAL::AlbanyTraits::MPResidual,Traits>
   : public KfieldBC_Base<PHAL::AlbanyTraits::MPResidual, Traits> {
public:
  KfieldBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::MPResidual::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Multi-point Jacobian
// **************************************************************
template<typename Traits>
class KfieldBC<PHAL::AlbanyTraits::MPJacobian,Traits>
   : public KfieldBC_Base<PHAL::AlbanyTraits::MPJacobian, Traits> {
public:
  KfieldBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::MPJacobian::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Multi-point Tangent
// **************************************************************
template<typename Traits>
class KfieldBC<PHAL::AlbanyTraits::MPTangent,Traits>
   : public KfieldBC_Base<PHAL::AlbanyTraits::MPTangent, Traits> {
public:
  KfieldBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::MPTangent::ScalarT ScalarT;
  void evaluateFields(typename Traits::EvalData d);
};

}

#endif
