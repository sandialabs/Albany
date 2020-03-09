//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(PHAL_SDirichlet_hpp)
#define PHAL_SDirichlet_hpp

#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Dirichlet.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"
#include "Sacado_ParameterAccessor.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_ThyraTypes.hpp"

namespace PHAL {

///
/// Strong Dirichlet boundary condition evaluator
///
template <typename EvalT, typename Traits>
class SDirichlet
{
};

//
// Specializations for different Albany Traits.
//

//
// Residual
//
template <typename Traits>
class SDirichlet<PHAL::AlbanyTraits::Residual, Traits>
    : public PHAL::DirichletBase<PHAL::AlbanyTraits::Residual, Traits>
{
 public:
  using ScalarT = typename PHAL::AlbanyTraits::Residual::ScalarT;

  SDirichlet(Teuchos::ParameterList& p);

  void
  preEvaluate(typename Traits::EvalData d);

  void
  evaluateFields(typename Traits::EvalData d);
};

//
// Jacobian
//
template <typename Traits>
class SDirichlet<PHAL::AlbanyTraits::Jacobian, Traits>
    : public PHAL::DirichletBase<PHAL::AlbanyTraits::Jacobian, Traits>
{
 public:
  using ScalarT = typename PHAL::AlbanyTraits::Jacobian::ScalarT;

  SDirichlet(Teuchos::ParameterList& p);

  void
  preEvaluate(typename Traits::EvalData d);

  void
  evaluateFields(typename Traits::EvalData d);

  void
  set_row_and_col_is_dbc(typename Traits::EvalData d);

 protected:
  Teuchos::RCP<Thyra_Vector> row_is_dbc_;
  Teuchos::RCP<Thyra_Vector> col_is_dbc_;
};

//
// Tangent
//
template <typename Traits>
class SDirichlet<PHAL::AlbanyTraits::Tangent, Traits>
    : public PHAL::DirichletBase<PHAL::AlbanyTraits::Tangent, Traits>
{
 public:
  using ScalarT = typename PHAL::AlbanyTraits::Tangent::ScalarT;

  SDirichlet(Teuchos::ParameterList& p);

  void
  preEvaluate(typename Traits::EvalData d);

  void
  evaluateFields(typename Traits::EvalData d);
};

//
// Distributed Parameter Derivative
//
template <typename Traits>
class SDirichlet<PHAL::AlbanyTraits::DistParamDeriv, Traits>
    : public PHAL::DirichletBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>
{
 public:
  using ScalarT = typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT;

  SDirichlet(Teuchos::ParameterList& p);

  void
  evaluateFields(typename Traits::EvalData d);
};

}  // namespace PHAL

#endif  // PHAL_SDirichlet_hpp
