//
// Albany 3.0: Copyright 2016 National Technology & Engineering Solutions of
// Sandia, LLC (NTESS). This Software is released under the BSD license detailed
// in the file license.txt in the top-level Albany directory.
//

#if !defined(PHAL_ExprEvalSDBC_hpp)
#define PHAL_ExprEvalSDBC_hpp

#include "Albany_ThyraTypes.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Dirichlet.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_config.hpp"
#include "Sacado_ParameterAccessor.hpp"
#include "Teuchos_ParameterList.hpp"

namespace PHAL {

///
/// Strong Dirichlet boundary condition evaluator
///
template <typename EvalT, typename Traits>
class ExprEvalSDBC
{
};

//
// Specializations for different Albany Traits.
//

//
// Residual
//
template <typename Traits>
class ExprEvalSDBC<PHAL::AlbanyTraits::Residual, Traits>
    : public PHAL::DirichletBase<PHAL::AlbanyTraits::Residual, Traits>
{
 public:
  using ScalarT = typename PHAL::AlbanyTraits::Residual::ScalarT;

  ExprEvalSDBC(Teuchos::ParameterList& p);

  void
  preEvaluate(typename Traits::EvalData d);

  void
  evaluateFields(typename Traits::EvalData d);

 protected:
  std::string expression{""};
};

//
// Jacobian
//
template <typename Traits>
class ExprEvalSDBC<PHAL::AlbanyTraits::Jacobian, Traits>
    : public PHAL::DirichletBase<PHAL::AlbanyTraits::Jacobian, Traits>
{
 public:
  using ScalarT = typename PHAL::AlbanyTraits::Jacobian::ScalarT;

  ExprEvalSDBC(Teuchos::ParameterList& p);

  void
  evaluateFields(typename Traits::EvalData d);

  void
  set_row_and_col_is_dbc(typename Traits::EvalData d);

 protected:
  Teuchos::RCP<Thyra_Vector> row_is_dbc_{Teuchos::null};
  Teuchos::RCP<Thyra_Vector> col_is_dbc_{Teuchos::null};
};

// **************************************************************
// Tangent
// **************************************************************
template<typename Traits>
class ExprEvalSDBC<PHAL::AlbanyTraits::Tangent,Traits>
   : public PHAL::DirichletBase<PHAL::AlbanyTraits::Tangent, Traits> {
public:
  ExprEvalSDBC(Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Distributed Parameter Derivative
// **************************************************************
template<typename Traits>
class ExprEvalSDBC<PHAL::AlbanyTraits::DistParamDeriv,Traits>
   : public PHAL::DirichletBase<PHAL::AlbanyTraits::DistParamDeriv, Traits> {
public:
  ExprEvalSDBC(Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
};

}  // namespace PHAL

#endif  // PHAL_ExprEvalSDBC_hpp
