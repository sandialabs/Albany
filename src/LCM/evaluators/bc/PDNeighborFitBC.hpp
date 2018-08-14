//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_PDNEIGHBORFIT_HPP
#define PHAL_PDNEIGHBORFIT_HPP

#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"

#include "Teuchos_ParameterList.hpp"

#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Dirichlet.hpp"
#include "Sacado_ParameterAccessor.hpp"

namespace LCM {
/** \brief BC Dirichlet evaluator for general coordinate functions

*/

// **************************************************************
// **************************************************************
// * Specialization of the DirichletBase class
// **************************************************************
// **************************************************************

template <typename EvalT, typename Traits>
class PDNeighborFitBC;

template <typename EvalT, typename Traits>
class PDNeighborFitBC_Base : public PHAL::DirichletBase<EvalT, Traits>
{
 public:
  typedef typename EvalT::ScalarT ScalarT;
  PDNeighborFitBC_Base(Teuchos::ParameterList& p);

 protected:
  double perturbDirichlet;
  double timeStep;
};

// **************************************************************
// Residual
// **************************************************************
template <typename Traits>
class PDNeighborFitBC<PHAL::AlbanyTraits::Residual, Traits>
    : public PDNeighborFitBC_Base<PHAL::AlbanyTraits::Residual, Traits>
{
 public:
  PDNeighborFitBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
  void
  evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Jacobian
// **************************************************************
template <typename Traits>
class PDNeighborFitBC<PHAL::AlbanyTraits::Jacobian, Traits>
    : public PDNeighborFitBC_Base<PHAL::AlbanyTraits::Jacobian, Traits>
{
 public:
  PDNeighborFitBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
  void
  evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Tangent
// **************************************************************
template <typename Traits>
class PDNeighborFitBC<PHAL::AlbanyTraits::Tangent, Traits>
    : public PDNeighborFitBC_Base<PHAL::AlbanyTraits::Tangent, Traits>
{
 public:
  PDNeighborFitBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::Tangent::ScalarT ScalarT;
  void
  evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Distributed Parameter Derivative
//  -- Currently assuming no parameter derivative
// **************************************************************
template <typename Traits>
class PDNeighborFitBC<PHAL::AlbanyTraits::DistParamDeriv, Traits>
    : public PDNeighborFitBC_Base<PHAL::AlbanyTraits::DistParamDeriv, Traits>
{
 public:
  PDNeighborFitBC(Teuchos::ParameterList& p);
  typedef typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT ScalarT;
  void
  evaluateFields(typename Traits::EvalData d);
};

}  // namespace LCM

#endif
