//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/13/14: only Epetra is SG and MP

#ifndef PHAL_DIRICHLET_STATE_HPP
#define PHAL_DIRICHLET_STATE_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Shards_CellTopologyData.h"

#include "Teuchos_ParameterList.hpp"
#if defined(ALBANY_EPETRA)
#include "Epetra_Vector.h"
#endif

#include "PHAL_AlbanyTraits.hpp"

namespace FELIX
{
template <typename EvalT, typename Traits>
class HydrologyDirichlet;

template <typename EvalT, typename Traits>
class HydrologyDirichletBase : public PHX::EvaluatorWithBaseImpl<Traits>,
                               public PHX::EvaluatorDerived<EvalT, Traits>
{
  public:
    HydrologyDirichletBase(Teuchos::ParameterList& p);

    void postRegistrationSetup(typename Traits::SetupData d,
                               PHX::FieldManager<Traits>& fm) {}

    typedef typename EvalT::ScalarT ScalarT;

  protected:

    std::string   nodeSetID;
    int           offset;

    std::string   H_name;
    std::string   s_name;

    double        rho_w;
    double        g;
};

// **************************************************************
// Residual
// **************************************************************
template<typename Traits>
class HydrologyDirichlet<PHAL::AlbanyTraits::Residual, Traits>
    : public HydrologyDirichletBase<PHAL::AlbanyTraits::Residual, Traits> {
  public:
    HydrologyDirichlet(Teuchos::ParameterList& p);
    void evaluateFields(typename Traits::EvalData d);

    typedef HydrologyDirichletBase<PHAL::AlbanyTraits::Residual, Traits>  super;
};

// **************************************************************
// Jacobian
// **************************************************************
template<typename Traits>
class HydrologyDirichlet<PHAL::AlbanyTraits::Jacobian, Traits>
    : public HydrologyDirichletBase<PHAL::AlbanyTraits::Jacobian, Traits> {
  public:
    HydrologyDirichlet(Teuchos::ParameterList& p);
    void evaluateFields(typename Traits::EvalData d);

    typedef HydrologyDirichletBase<PHAL::AlbanyTraits::Jacobian, Traits>  super;
};

// **************************************************************
// Tangent
// **************************************************************
template<typename Traits>
class HydrologyDirichlet<PHAL::AlbanyTraits::Tangent, Traits>
    : public HydrologyDirichletBase<PHAL::AlbanyTraits::Tangent, Traits> {
  public:
    HydrologyDirichlet(Teuchos::ParameterList& p);
    void evaluateFields(typename Traits::EvalData d);

    typedef HydrologyDirichletBase<PHAL::AlbanyTraits::Tangent, Traits>  super;
};

// **************************************************************
// Distributed Parameter Derivative
//  -- Currently assuming no parameter derivative
// **************************************************************
template<typename Traits>
class HydrologyDirichlet<PHAL::AlbanyTraits::DistParamDeriv, Traits>
    : public HydrologyDirichletBase<PHAL::AlbanyTraits::DistParamDeriv, Traits> {
  public:
    HydrologyDirichlet(Teuchos::ParameterList& p);
    void evaluateFields(typename Traits::EvalData d);

    typedef HydrologyDirichletBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>  super;
};

} // Namespace FELIX

#endif
