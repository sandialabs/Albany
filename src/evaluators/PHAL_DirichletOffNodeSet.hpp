//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/13/14: only Epetra is SG and MP

#ifndef PHAL_DIRICHLET_OFF_SIDE_SET_HPP
#define PHAL_DIRICHLET_OFF_SIDE_SET_HPP 1

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Teuchos_ParameterList.hpp"
#if defined(ALBANY_EPETRA)
#include "Epetra_Vector.h"
#endif

#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Dirichlet.hpp"

namespace PHAL {

/** \brief Dirichlet evaluator for nodes outside a given node set

    This evaluator is needed when the given problem has equations defined only on a side set.
    In that case, the Jacobian entries on the remaining nodes (not on the side set) MUST be
    handled (typically, J(dof,dof)=1, J(dof,:)=0 and res(dof)=x-datum), otherwise the linear
    solvers may complain (nan). To this end, we exploit the addition "computeNodeSetsFromSideSets"
    in STKDiscretization, which for each sideset creates a nodeset with the same name of the
    sideset. Then, this evaluator loops on all the nodes in the partition: if the nodes belongs
    to the sideset, it does nothing; otherwise, it sets the residual appropriately.
*/

template<typename EvalT, typename Traits>
class DirichletOffNodeSet;

// **************************************************************
// Residual
// **************************************************************
template<typename Traits>
class DirichletOffNodeSet<PHAL::AlbanyTraits::Residual,Traits>
            : public DirichletBase<PHAL::AlbanyTraits::Residual, Traits>
{
public:
  DirichletOffNodeSet(Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  std::vector<std::string>  nodeSets;
};

// **************************************************************
// Jacobian
// **************************************************************
template<typename Traits>
class DirichletOffNodeSet<PHAL::AlbanyTraits::Jacobian,Traits>
            : public DirichletBase<PHAL::AlbanyTraits::Jacobian, Traits>
{
public:
  DirichletOffNodeSet(Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  std::vector<std::string> nodeSets;
};

// **************************************************************
// Tangent
// **************************************************************
template<typename Traits>
class DirichletOffNodeSet<PHAL::AlbanyTraits::Tangent,Traits>
   : public DirichletBase<PHAL::AlbanyTraits::Tangent, Traits> {
public:
  DirichletOffNodeSet(Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  std::vector<std::string>  nodeSets;
};

// **************************************************************
// Distributed Parameter Derivative
// **************************************************************
template<typename Traits>
class DirichletOffNodeSet<PHAL::AlbanyTraits::DistParamDeriv,Traits>
   : public DirichletBase<PHAL::AlbanyTraits::DistParamDeriv, Traits> {
public:
  DirichletOffNodeSet(Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  std::vector<std::string>  nodeSets;
};

// **************************************************************
// Stochastic Galerkin Residual
// **************************************************************
#ifdef ALBANY_SG
template<typename Traits>
class DirichletOffNodeSet<PHAL::AlbanyTraits::SGResidual,Traits>
   : public DirichletBase<PHAL::AlbanyTraits::SGResidual, Traits> {
public:
  DirichletOffNodeSet(Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Stochastic Galerkin Jacobian
// **************************************************************
template<typename Traits>
class DirichletOffNodeSet<PHAL::AlbanyTraits::SGJacobian,Traits>
   : public DirichletBase<PHAL::AlbanyTraits::SGJacobian, Traits> {
public:
  DirichletOffNodeSet(Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Stochastic Galerkin Tangent
// **************************************************************
template<typename Traits>
class DirichletOffNodeSet<PHAL::AlbanyTraits::SGTangent,Traits>
   : public DirichletBase<PHAL::AlbanyTraits::SGTangent, Traits> {
public:
  DirichletOffNodeSet(Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
};
#endif

#ifdef ALBANY_ENSEMBLE
// **************************************************************
// Multi-point Residual
// **************************************************************
template<typename Traits>
class DirichletOffNodeSet<PHAL::AlbanyTraits::MPResidual,Traits>
   : public DirichletBase<PHAL::AlbanyTraits::MPResidual, Traits> {
public:
  DirichletOffNodeSet(Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Multi-point Jacobian
// **************************************************************
template<typename Traits>
class DirichletOffNodeSet<PHAL::AlbanyTraits::MPJacobian,Traits>
   : public DirichletBase<PHAL::AlbanyTraits::MPJacobian, Traits> {
public:
  DirichletOffNodeSet(Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Multi-point Tangent
// **************************************************************
template<typename Traits>
class DirichletOffNodeSet<PHAL::AlbanyTraits::MPTangent,Traits>
   : public DirichletBase<PHAL::AlbanyTraits::MPTangent, Traits> {
public:
  DirichletOffNodeSet(Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
};
#endif

} // Namespace PHAL

#endif // PHAL_DIRICHLET_OFF_SIDE_SET_HPP
