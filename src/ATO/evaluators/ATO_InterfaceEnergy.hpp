//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ATO_INTERFACE_ENERGY_HPP
#define ATO_INTERFACE_ENERGY_HPP

#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_StateManager.hpp"
#include "ATO_TopoTools.hpp"
#include "ATO_PenaltyModel.hpp"

namespace ATO {
/**
 * \brief Description
 */
  template<typename EvalT, typename Traits>
  class InterfaceEnergyBase :
    public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits>
  {
  public:
    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;
    InterfaceEnergyBase(Teuchos::ParameterList& p,
		      const Teuchos::RCP<Albany::Layouts>& dl,
                      const Albany::MeshSpecsStruct* meshSpecs);

    void postRegistrationSetup(typename Traits::SetupData d,
			       PHX::FieldManager<Traits>& vm);

    // These functions are defined in the specializations
    void preEvaluate(typename Traits::PreEvalData d) = 0;
    void postEvaluate(typename Traits::PostEvalData d) = 0;
    void evaluateFields(typename Traits::EvalData d) = 0;

    Teuchos::RCP<const PHX::FieldTag> getEvaluatedFieldTag() const {
      return interface_energy_tag;
    }

    Teuchos::RCP<const PHX::FieldTag> getResponseFieldTag() const {
      return interface_energy_tag;
    }

  protected:

    RealType m_scaling;
    std::string m_FName;
    std::string m_topoName;
    std::string m_dFdpName;
    std::string elementBlockName;
    static const std::string className;

    PHX::MDField<MeshScalarT,Cell,QuadPoint> qp_weights;
    PHX::MDField<RealType,Cell,Node,QuadPoint,Dim> GradBF;

    Teuchos::RCP< PHX::Tag<ScalarT> > interface_energy_tag;
    Albany::StateManager* pStateMgr;

    bool m_excludeBlock;

  };

template<typename EvalT, typename Traits>
class InterfaceEnergy
   : public InterfaceEnergyBase<EvalT, Traits> {

   using InterfaceEnergyBase<EvalT,Traits>::m_excludeBlock;
   using InterfaceEnergyBase<EvalT,Traits>::m_dFdpName;
   using InterfaceEnergyBase<EvalT,Traits>::m_FName;
   using InterfaceEnergyBase<EvalT,Traits>::m_topoName;
   using InterfaceEnergyBase<EvalT,Traits>::elementBlockName;
   using InterfaceEnergyBase<EvalT,Traits>::className;
   using InterfaceEnergyBase<EvalT,Traits>::qp_weights;
   using InterfaceEnergyBase<EvalT,Traits>::GradBF;
   using InterfaceEnergyBase<EvalT,Traits>::interface_energy_tag;
   using InterfaceEnergyBase<EvalT,Traits>::pStateMgr;

public:
  InterfaceEnergy(Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl,
              const Albany::MeshSpecsStruct* meshSpecs) :
  InterfaceEnergyBase<EvalT, Traits>(p, dl, meshSpecs){}
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
class InterfaceEnergy<PHAL::AlbanyTraits::Residual,Traits>
   : public InterfaceEnergyBase<PHAL::AlbanyTraits::Residual, Traits> {

   using InterfaceEnergyBase<PHAL::AlbanyTraits::Residual, Traits>::m_excludeBlock;
   using InterfaceEnergyBase<PHAL::AlbanyTraits::Residual, Traits>::m_dFdpName;
   using InterfaceEnergyBase<PHAL::AlbanyTraits::Residual, Traits>::m_FName;
   using InterfaceEnergyBase<PHAL::AlbanyTraits::Residual, Traits>::m_topoName;
   using InterfaceEnergyBase<PHAL::AlbanyTraits::Residual, Traits>::elementBlockName;
   using InterfaceEnergyBase<PHAL::AlbanyTraits::Residual, Traits>::className;
   using InterfaceEnergyBase<PHAL::AlbanyTraits::Residual, Traits>::qp_weights;
   using InterfaceEnergyBase<PHAL::AlbanyTraits::Residual, Traits>::GradBF;
   using InterfaceEnergyBase<PHAL::AlbanyTraits::Residual, Traits>::interface_energy_tag;
   using InterfaceEnergyBase<PHAL::AlbanyTraits::Residual, Traits>::pStateMgr;

public:
  InterfaceEnergy(Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl,
              const Albany::MeshSpecsStruct* meshSpecs);
  void preEvaluate(typename Traits::PreEvalData d);
  void postEvaluate(typename Traits::PostEvalData d);
  void evaluateFields(typename Traits::EvalData d);
};

}

#endif
