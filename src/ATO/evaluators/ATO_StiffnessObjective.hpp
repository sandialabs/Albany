//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ATO_STIFFNESSOBJECTIVE_HPP
#define ATO_STIFFNESSOBJECTIVE_HPP

#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Albany_ProblemUtils.hpp"
#include "ATO_TopoTools.hpp"

namespace ATO {
/**
 * \brief Description
 */
  template<typename EvalT, typename Traits>
  class StiffnessObjectiveBase :
    public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits>
  {
  public:
    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;
    StiffnessObjectiveBase(Teuchos::ParameterList& p,
		      const Teuchos::RCP<Albany::Layouts>& dl);

    void postRegistrationSetup(typename Traits::SetupData d,
			       PHX::FieldManager<Traits>& vm);

    // These functions are defined in the specializations
    void preEvaluate(typename Traits::PreEvalData d) = 0;
    void postEvaluate(typename Traits::PostEvalData d) = 0;
    void evaluateFields(typename Traits::EvalData d) = 0;

    Teuchos::RCP<const PHX::FieldTag> getEvaluatedFieldTag() const {
      return stiffness_objective_tag;
    }

    Teuchos::RCP<const PHX::FieldTag> getResponseFieldTag() const {
      return stiffness_objective_tag;
    }

  protected:

    std::string dFdpName;
    std::string FName;
    static const std::string className;

    PHX::MDField<ScalarT> gradX;
    PHX::MDField<ScalarT> workConj;
    PHX::MDField<MeshScalarT,Cell,QuadPoint> qp_weights;
    PHX::MDField<RealType,Cell,Node,QuadPoint> BF;


    Teuchos::RCP< PHX::Tag<ScalarT> > stiffness_objective_tag;
    Albany::StateManager* pStateMgr;

    Teuchos::RCP<Topology> topology;

  };

template<typename EvalT, typename Traits>
class StiffnessObjective
   : public StiffnessObjectiveBase<EvalT, Traits> {

   using StiffnessObjectiveBase<EvalT,Traits>::topology;
   using StiffnessObjectiveBase<EvalT,Traits>::dFdpName;
   using StiffnessObjectiveBase<EvalT,Traits>::FName;
   using StiffnessObjectiveBase<EvalT,Traits>::className;
   using StiffnessObjectiveBase<EvalT,Traits>::gradX;
   using StiffnessObjectiveBase<EvalT,Traits>::workConj;
   using StiffnessObjectiveBase<EvalT,Traits>::qp_weights;
   using StiffnessObjectiveBase<EvalT,Traits>::BF;
   using StiffnessObjectiveBase<EvalT,Traits>::stiffness_objective_tag;
   using StiffnessObjectiveBase<EvalT,Traits>::pStateMgr;

public:
  StiffnessObjective(Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl) :
  StiffnessObjectiveBase<EvalT, Traits>(p, dl){}
  void preEvaluate(typename Traits::PreEvalData d){}
  void postEvaluate(typename Traits::PostEvalData d){}
  void evaluateFields(typename Traits::EvalData d){}
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
class StiffnessObjective<PHAL::AlbanyTraits::Residual,Traits>
   : public StiffnessObjectiveBase<PHAL::AlbanyTraits::Residual, Traits> {

   using StiffnessObjectiveBase<PHAL::AlbanyTraits::Residual, Traits>::topology;
   using StiffnessObjectiveBase<PHAL::AlbanyTraits::Residual, Traits>::dFdpName;
   using StiffnessObjectiveBase<PHAL::AlbanyTraits::Residual, Traits>::FName;
   using StiffnessObjectiveBase<PHAL::AlbanyTraits::Residual, Traits>::className;
   using StiffnessObjectiveBase<PHAL::AlbanyTraits::Residual, Traits>::gradX;
   using StiffnessObjectiveBase<PHAL::AlbanyTraits::Residual, Traits>::workConj;
   using StiffnessObjectiveBase<PHAL::AlbanyTraits::Residual, Traits>::qp_weights;
   using StiffnessObjectiveBase<PHAL::AlbanyTraits::Residual, Traits>::BF;
   using StiffnessObjectiveBase<PHAL::AlbanyTraits::Residual, Traits>::stiffness_objective_tag;
   using StiffnessObjectiveBase<PHAL::AlbanyTraits::Residual, Traits>::pStateMgr;

public:
  StiffnessObjective(Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl);
  void preEvaluate(typename Traits::PreEvalData d);
  void postEvaluate(typename Traits::PostEvalData d);
  void evaluateFields(typename Traits::EvalData d);
};

}

#endif
