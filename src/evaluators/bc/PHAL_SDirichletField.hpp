//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_SDIRICHLET_FIELD_HPP
#define PHAL_SDIRICHLET_FIELD_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Sacado_ParameterAccessor.hpp"

#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Dirichlet.hpp"
#include "PHAL_IdentityCoordinateFunctionTraits.hpp"

namespace PHAL {



// **************************************************************
// **************************************************************
// * Specialization of the DirichletBase class
// **************************************************************
// **************************************************************

/// Strong Dirichlet boundary condition evaluator prescribing a field
/// Note, the field must be available before the volume field manager evaluation

template<typename EvalT, typename Traits>
class SDirichletField;

template <typename EvalT, typename Traits>
class SDirichletField_Base : public PHAL::DirichletBase<EvalT, Traits> {
  public:
    typedef typename EvalT::ScalarT ScalarT;
    SDirichletField_Base(Teuchos::ParameterList& p);

  protected:
    /// name of the field used to prescribe boundary conditions
    /// Note, the field must be available before the volume field manager evaluation
    std::string field_name;
};

// **************************************************************
// Residual
// **************************************************************
template<typename Traits>
class SDirichletField<PHAL::AlbanyTraits::Residual, Traits>
    : public SDirichletField_Base<PHAL::AlbanyTraits::Residual, Traits> {
  public:
    using ScalarT = typename PHAL::AlbanyTraits::Residual::ScalarT;

    SDirichletField(Teuchos::ParameterList& p);

    void preEvaluate(typename Traits::EvalData d);

    void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Jacobian
// **************************************************************
template<typename Traits>
class SDirichletField<PHAL::AlbanyTraits::Jacobian, Traits>
    : public SDirichletField_Base<PHAL::AlbanyTraits::Jacobian, Traits> {
  public:
    using ScalarT = typename PHAL::AlbanyTraits::Jacobian::ScalarT;

    SDirichletField(Teuchos::ParameterList& p);

    void preEvaluate(typename Traits::EvalData d);

    void evaluateFields(typename Traits::EvalData d);

    void set_row_and_col_is_dbc(typename Traits::EvalData d);

   protected:
    Teuchos::RCP<Thyra_Vector> row_is_dbc_;
    Teuchos::RCP<Thyra_Vector> col_is_dbc_;
};

// **************************************************************
// Tangent
// **************************************************************
template<typename Traits>
class SDirichletField<PHAL::AlbanyTraits::Tangent, Traits>
    : public SDirichletField_Base<PHAL::AlbanyTraits::Tangent, Traits> {
  public:
    using ScalarT = typename PHAL::AlbanyTraits::Tangent::ScalarT;

    void preEvaluate(typename Traits::EvalData d);

    SDirichletField(Teuchos::ParameterList& p);

    void evaluateFields(typename Traits::EvalData d);
};

// **************************************************************
// Distributed Parameter Derivative
//  -- Currently assuming no parameter derivative
// **************************************************************
template<typename Traits>
class SDirichletField<PHAL::AlbanyTraits::DistParamDeriv, Traits>
    : public SDirichletField_Base<PHAL::AlbanyTraits::DistParamDeriv, Traits> {
  public:
    using ScalarT = typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT;

    SDirichletField(Teuchos::ParameterList& p);

    void evaluateFields(typename Traits::EvalData d);
};

}

#endif
