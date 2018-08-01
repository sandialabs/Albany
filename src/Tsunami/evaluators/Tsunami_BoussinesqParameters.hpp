//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef TSUNAMI_BOUSSINESQPARAMETERS_HPP
#define TSUNAMI_BOUSSINESQPARAMETERS_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"
#include "PHAL_Utilities.hpp"

namespace Tsunami {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits>
class BoussinesqParameters : public PHX::EvaluatorWithBaseImpl<Traits>,
		    public PHX::EvaluatorDerived<EvalT, Traits> {

public:

  typedef typename EvalT::ScalarT ScalarT;

  BoussinesqParameters(const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);


private:
 
  typedef typename EvalT::MeshScalarT MeshScalarT;
  typedef typename EvalT::ParamScalarT  ParamScalarT; 

  // Input:  
  PHX::MDField<const ParamScalarT,Cell,QuadPoint>    waterdepthQPin;
  PHX::MDField<const ParamScalarT,Cell,QuadPoint>    zalphaQPin;
  
  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint>          waterdepthQP;
  PHX::MDField<ScalarT,Cell,QuadPoint>          zalphaQP;
  PHX::MDField<ScalarT,Cell,QuadPoint>          betaQP;

  unsigned int numQPs, numDims;

  double h, zAlpha;
 
  bool use_params_on_mesh;

  bool enable_memoizer;  

  PHAL::MDFieldMemoizer<Traits> memoizer;
};
}

#endif
