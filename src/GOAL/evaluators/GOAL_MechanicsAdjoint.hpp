//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef GOAL_MECHANICSADJOINT_HPP
#define GOAL_MECHANICSADJOINT_HPP

#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_MDField.hpp>
#include <Phalanx_DataLayout.hpp>
#include <Teuchos_ParameterList.hpp>
#include "Albany_ProblemUtils.hpp"

namespace GOAL {

template<typename EvalT, typename Traits>
class MechanicsAdjointBase : 
    public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits>
{
  public:
    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;

    MechanicsAdjointBase(Teuchos::ParameterList& p,
        const Teuchos::RCP<Albany::Layouts>& dl,
        const Albany::MeshSpecsStruct* mesh_specs);

    void postRegistrationSetup(typename Traits::SetupData d,
        PHX::FieldManager<Traits>& vm);

    void preEvaluate(typename Traits::PreEvalData d) = 0;
    void postEvaluate(typename Traits::PostEvalData d) = 0;
    void evaluateFields(typename Traits::EvalData d) = 0;

    Teuchos::RCP<const PHX::FieldTag> getEvaluatedFieldTag() const {
      return fieldTag_;
    }

    Teuchos::RCP<const PHX::FieldTag> getResponseFieldTag() const {
      return fieldTag_;
    }

  protected:

    Albany::StateManager* stateManager_;

    Teuchos::RCP< PHX::Tag<ScalarT> > fieldTag_;

};

template<typename EvalT, typename Traits>
class MechanicsAdjoint : public MechanicsAdjointBase<EvalT, Traits>
{
  public:
    MechanicsAdjoint(Teuchos::ParameterList& p,
        const Teuchos::RCP<Albany::Layouts>& dl,
        const Albany::MeshSpecsStruct* mesh_specs)
      : MechanicsAdjointBase<EvalT, Traits>(p, dl, mesh_specs) {}
    void preEvaluate(typename Traits::PreEvalData d) {}
    void postEvaluate(typename Traits::PostEvalData d) {}
    void evaluateFields(typename Traits::EvalData d) {}
};

// **************************************************************
// **************************************************************
// * Specializations
// **************************************************************
// **************************************************************

// **************************************************************
// Residual 
// **************************************************************
template<typename Traits>
class MechanicsAdjoint<PHAL::AlbanyTraits::Residual,Traits> :
public MechanicsAdjointBase<PHAL::AlbanyTraits::Residual, Traits>
{
  public:
    MechanicsAdjoint(Teuchos::ParameterList& p,
        const Teuchos::RCP<Albany::Layouts>& dl,
        const Albany::MeshSpecsStruct* mesh_specs);
    void preEvaluate(typename Traits::PreEvalData d);
    void postEvaluate(typename Traits::PostEvalData d);
    void evaluateFields(typename Traits::EvalData d);

  private:
};

}

#endif
