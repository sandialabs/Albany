//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef TDM_ENERGY_DOT_HPP
#define TDM_ENERGY_DOT_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace TDM {
  ///
  /// \brief  Rate of energy evaluator
  ///
  /// This evaluator computes the rate of energy
  ///
  template<typename EvalT, typename Traits>
  class Energy_Dot : 
    public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits>
  {

  public:

    Energy_Dot(const Teuchos::ParameterList& p,
	       const Teuchos::RCP<Albany::Layouts>& dl);

    void 
    postRegistrationSetup(typename Traits::SetupData d,
			  PHX::FieldManager<Traits>& vm);

    void 
    evaluateFields(typename Traits::EvalData d);

  private:

    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;

    PHX::MDField<ScalarT,Cell,QuadPoint> T_;
    PHX::MDField<ScalarT,Cell,QuadPoint> T_dot_;
    PHX::MDField<ScalarT,Cell,QuadPoint> phi1_;
    PHX::MDField<ScalarT,Cell,QuadPoint> phi1_dot_;
    PHX::MDField<ScalarT,Cell,QuadPoint> phi2_;
    PHX::MDField<ScalarT,Cell,QuadPoint> phi2_dot_;
    PHX::MDField<ScalarT,Cell,QuadPoint> psi1_;
    PHX::MDField<ScalarT,Cell,QuadPoint> psi2_;
    PHX::MDField<ScalarT,Cell,QuadPoint> psi1_dot_;
    PHX::MDField<ScalarT,Cell,QuadPoint> psi2_dot_;
    PHX::MDField<ScalarT,Cell,QuadPoint> rho_Cp_;
    PHX::MDField<ScalarT,Dummy> time_;
    PHX::MDField<ScalarT,Dummy> deltaTime_;

    PHX::MDField<ScalarT,Cell,QuadPoint> energyDot_;

    unsigned int num_qps_;
    unsigned int num_dims_;
    unsigned int num_nodes_;
    unsigned int workset_size_;
  
    // Volumetric heat capacity in liquid
    ScalarT Cl_;
    // Latent heat of fusion/melting
    ScalarT Lm_;
    // Melting temperature
    ScalarT Tm_;
    // Delta temperature to compute phi
    ScalarT Tc_;
    // Dense state volumetric heat capacity
    ScalarT Cd_;
    //Latent Heat of Vaporization
    ScalarT Lv_;
    //Volumetric heat capacity of Vapour
    ScalarT Cv_;	
    // Vaporization Temperature
    ScalarT Tv_;
    // Initial porosity epsilon not
    ScalarT initial_porosity;



    // old temperature name
    std::string Temperature_Name_;

    // old phi name
    std::string Phi1_old_name_;
    std::string Phi2_old_name_;

    //old psi name
    std::string Psi1_old_name_;
    std::string Psi2_old_name_;

    Teuchos::RCP<const Teuchos::ParameterList>
    getValidEnergy_DotParameters() const;
  };
}

#endif
