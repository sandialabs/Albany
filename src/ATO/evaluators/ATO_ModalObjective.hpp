//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ATO_MODALBJECTIVE_HPP
#define ATO_MODALOBJECTIVE_HPP

#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Albany_ProblemUtils.hpp"
#include "ATO_TopoTools.hpp"
#include "Albany_StateManager.hpp"

namespace ATO {
/**
 * \brief Description
 */
  template<typename EvalT, typename Traits>
  class ModalObjectiveBase :
    public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits>
  {
  public:
    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;
    ModalObjectiveBase(Teuchos::ParameterList& p,
	 	       const Teuchos::RCP<Albany::Layouts>& dl);

    void postRegistrationSetup(typename Traits::SetupData d,
			       PHX::FieldManager<Traits>& vm);

    // These functions are defined in the specializations
    void preEvaluate(typename Traits::PreEvalData d) = 0;
    void postEvaluate(typename Traits::PostEvalData d) = 0;
    void evaluateFields(typename Traits::EvalData d) = 0;

    Teuchos::RCP<const PHX::FieldTag> getEvaluatedFieldTag() const {
      return modal_objective_tag;
    }

    Teuchos::RCP<const PHX::FieldTag> getResponseFieldTag() const {
      return modal_objective_tag;
    }

  protected:

    std::string dFdpName;
    std::string FName;
    static const std::string className;

    PHX::MDField<MeshScalarT,Cell,QuadPoint> qp_weights;
    PHX::MDField<RealType,Cell,Node,QuadPoint> BF;
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim> val_qp;
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> gradX;
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> workConj;
    PHX::MDField<ScalarT> eigval;

    Teuchos::RCP< PHX::Tag<ScalarT> > modal_objective_tag;
    Albany::StateManager* pStateMgr;

    Teuchos::RCP<Topology> topology;
    int functionIndex;

  };

template<typename EvalT, typename Traits>
class ModalObjective
   : public ModalObjectiveBase<EvalT, Traits> {

   using ModalObjectiveBase<EvalT,Traits>::topology;
   using ModalObjectiveBase<EvalT,Traits>::functionIndex;
   using ModalObjectiveBase<EvalT,Traits>::dFdpName;
   using ModalObjectiveBase<EvalT,Traits>::FName;
   using ModalObjectiveBase<EvalT,Traits>::className;
   using ModalObjectiveBase<EvalT,Traits>::qp_weights;
   using ModalObjectiveBase<EvalT,Traits>::BF;
   using ModalObjectiveBase<EvalT,Traits>::val_qp;
   using ModalObjectiveBase<EvalT,Traits>::gradX;
   using ModalObjectiveBase<EvalT,Traits>::workConj;
   using ModalObjectiveBase<EvalT,Traits>::eigval;
   using ModalObjectiveBase<EvalT,Traits>::modal_objective_tag;
   using ModalObjectiveBase<EvalT,Traits>::pStateMgr;

public:
  ModalObjective(Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl) :
  ModalObjectiveBase<EvalT, Traits>(p, dl){}
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
class ModalObjective<PHAL::AlbanyTraits::Residual,Traits>
   : public ModalObjectiveBase<PHAL::AlbanyTraits::Residual, Traits> {

   using ModalObjectiveBase<PHAL::AlbanyTraits::Residual, Traits>::topology;
   using ModalObjectiveBase<PHAL::AlbanyTraits::Residual, Traits>::functionIndex;
   using ModalObjectiveBase<PHAL::AlbanyTraits::Residual, Traits>::dFdpName;
   using ModalObjectiveBase<PHAL::AlbanyTraits::Residual, Traits>::FName;
   using ModalObjectiveBase<PHAL::AlbanyTraits::Residual, Traits>::className;
   using ModalObjectiveBase<PHAL::AlbanyTraits::Residual, Traits>::qp_weights;
   using ModalObjectiveBase<PHAL::AlbanyTraits::Residual, Traits>::BF;
   using ModalObjectiveBase<PHAL::AlbanyTraits::Residual, Traits>::val_qp;
   using ModalObjectiveBase<PHAL::AlbanyTraits::Residual, Traits>::gradX;
   using ModalObjectiveBase<PHAL::AlbanyTraits::Residual, Traits>::workConj;
   using ModalObjectiveBase<PHAL::AlbanyTraits::Residual, Traits>::eigval;
   using ModalObjectiveBase<PHAL::AlbanyTraits::Residual, Traits>::modal_objective_tag;
   using ModalObjectiveBase<PHAL::AlbanyTraits::Residual, Traits>::pStateMgr;

public:
  ModalObjective(Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl);
  void preEvaluate(typename Traits::PreEvalData d);
  void postEvaluate(typename Traits::PostEvalData d);
  void evaluateFields(typename Traits::EvalData d);
};

}

#endif
