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


#ifndef PHAL_DIRICHLET_HPP
#define PHAL_DIRICHLET_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Epetra_Vector.h"

#include "Sacado_ParameterAccessor.hpp"
#include "PHAL_AlbanyTraits.hpp"

/** \brief Gathers solution values from the Newton solution vector into 
    the nodal fields of the field manager

    Currently makes an assumption that the stride is constant for dofs
    and that the nmber of dofs is equal to the size of the solution
    names vector.

*/
// **************************************************************
// Generic Template Impelementation for constructor and PostReg
// **************************************************************
namespace PHAL {

template<typename EvalT, typename Traits>
class DirichletBase
  : public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits>,
    public Sacado::ParameterAccessor<EvalT, SPL_Traits>  
   {
  
private:

  //typedef typename Traits::Residual::ScalarT ScalarT;
  typedef typename EvalT::ScalarT ScalarT;

public:
  
  DirichletBase(Teuchos::ParameterList& p);
  
  void postRegistrationSetup(typename Traits::SetupData d,
                             PHX::FieldManager<Traits>& vm);
  
  // This function will be overloaded with template specialized code
  void evaluateFields(typename Traits::EvalData d)=0;
  
  virtual ScalarT& getValue(const std::string &n) { return value; };

protected:
  const int offset;
  const int neq;
  ScalarT value;
  std::string nodeSetID;
};

// **************************************************************
// **************************************************************
// * Specializations
// **************************************************************
// **************************************************************

template<typename EvalT, typename Traits> class Dirichlet;

// **************************************************************
// Residual 
// **************************************************************
template<typename Traits>
class Dirichlet<PHAL::AlbanyTraits::Residual,Traits>
   : public DirichletBase<PHAL::AlbanyTraits::Residual, Traits> {
public:
  Dirichlet(Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Jacobian
// **************************************************************
template<typename Traits>
class Dirichlet<PHAL::AlbanyTraits::Jacobian,Traits>
   : public DirichletBase<PHAL::AlbanyTraits::Jacobian, Traits> {
public:
  Dirichlet(Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Tangent
// **************************************************************
template<typename Traits>
class Dirichlet<PHAL::AlbanyTraits::Tangent,Traits>
   : public DirichletBase<PHAL::AlbanyTraits::Tangent, Traits> {
public:
  Dirichlet(Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Stochastic Galerkin Residual 
// **************************************************************
template<typename Traits>
class Dirichlet<PHAL::AlbanyTraits::SGResidual,Traits>
   : public DirichletBase<PHAL::AlbanyTraits::SGResidual, Traits> {
public:
  Dirichlet(Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Stochastic Galerkin Jacobian
// **************************************************************
template<typename Traits>
class Dirichlet<PHAL::AlbanyTraits::SGJacobian,Traits>
   : public DirichletBase<PHAL::AlbanyTraits::SGJacobian, Traits> {
public:
  Dirichlet(Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Multi-point Residual 
// **************************************************************
template<typename Traits>
class Dirichlet<PHAL::AlbanyTraits::MPResidual,Traits>
   : public DirichletBase<PHAL::AlbanyTraits::MPResidual, Traits> {
public:
  Dirichlet(Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Multi-point Jacobian
// **************************************************************
template<typename Traits>
class Dirichlet<PHAL::AlbanyTraits::MPJacobian,Traits>
   : public DirichletBase<PHAL::AlbanyTraits::MPJacobian, Traits> {
public:
  Dirichlet(Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// **************************************************************
// Evaluator to aggregate all Dirichlet BCs into one "field"
// **************************************************************
template<typename EvalT, typename Traits>
class DirichletAggregator
  : public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits>
{
private:

  typedef typename EvalT::ScalarT ScalarT;

public:
  
  DirichletAggregator(Teuchos::ParameterList& p);
  
  void postRegistrationSetup(typename Traits::SetupData d,
                             PHX::FieldManager<Traits>& vm) {};
  
  // This function will be overloaded with template specialized code
  void evaluateFields(typename Traits::EvalData d) {};
};
}

#endif
