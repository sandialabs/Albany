//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef TDM_THCOND_HPP
#define TDM_THCOND_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Epetra_Vector.h"
#include "Sacado_ParameterAccessor.hpp"
#include "Teuchos_Array.hpp"
#include "Albany_Layouts.hpp"

namespace TDM {
  ///
  /// This evaluator computes the thermal conductivity
  ///
  template<typename EvalT, typename Traits>
  class ThCond : 
    public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits>
  {

  public:

    ThCond(Teuchos::ParameterList& p,
	   const Teuchos::RCP<Albany::Layouts>& dl);

    void 
    postRegistrationSetup(typename Traits::SetupData d,
			  PHX::FieldManager<Traits>& vm);

    void 
    evaluateFields(typename Traits::EvalData d);

  private:

    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;


    // Parameters to define the functional form of thermal conductivity as: Ks_(T) = a + bT + cT^2 + d/T + e/T^2 
    ScalarT aPre;
    ScalarT bPre;
    ScalarT cPre; 
    ScalarT dPre;
    ScalarT ePre;

    ScalarT aL;
    ScalarT bL;
    ScalarT cL; 
    ScalarT dL;
    ScalarT eL;

    ScalarT aPo;
    ScalarT bPo;
    ScalarT cPo; 
    ScalarT dPo;
    ScalarT ePo;

    ScalarT aV;
    ScalarT bV;
    ScalarT cV; 
    ScalarT dV;
    ScalarT eV;

    // Thermal conductivity in the pre-melted material
    ScalarT Kp_;
    //Thermal conductivity of the post-melted material	
    ScalarT Kd_;
	
    //Thermal conductivity of the solid material, accounting for porosity
    ScalarT Ks_;
	
    // Thermal conductivity in the liquid phase
    ScalarT Kl_;

    // Thermal conductivity  in the vapor phase
    ScalarT Kv_;
	
    //  Thermal conductivity in the powder phase
    ScalarT Kpowder_;

    // Powder initial porosity (detecting if initially powder or solid)
    ScalarT initial_porosity_;


    //	std::string sim_type;

    PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim> coord_;
    PHX::MDField<ScalarT,Cell,QuadPoint> T_;
    PHX::MDField<ScalarT,Cell,QuadPoint> k_;
    PHX::MDField<ScalarT,Cell,QuadPoint> phi1_;
    PHX::MDField<ScalarT,Cell,QuadPoint> phi2_;    
    PHX::MDField<ScalarT,Cell,QuadPoint> psi1_;
    PHX::MDField<ScalarT,Cell,QuadPoint> psi2_;
    PHX::MDField<ScalarT,Cell,QuadPoint> depth_;

    unsigned int num_qps_;
    unsigned int num_dims_;
    unsigned int num_nodes_;
    unsigned int workset_size_;
 
    Teuchos::RCP<const Teuchos::ParameterList>
    getValidThCondParameters() const;

  };
}

#endif
