//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_STOKESFOBODYFORCE_HPP
#define FELIX_STOKESFOBODYFORCE_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace FELIX {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits>
class StokesFOBodyForce : public PHX::EvaluatorWithBaseImpl<Traits>,
		    public PHX::EvaluatorDerived<EvalT, Traits> {

public:

  typedef typename EvalT::ScalarT ScalarT;

  StokesFOBodyForce(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);


private:
 
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:  
  PHX::MDField<ScalarT,Cell,QuadPoint> muFELIX;
  PHX::MDField<MeshScalarT,Cell,QuadPoint, Dim> coordVec;
  PHX::MDField<ScalarT,Cell,QuadPoint, Dim> surfaceGrad;
  Teuchos::Array<double> gravity;
  
  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim> force;

   //Radom field types
  enum BFTYPE {NONE, FO_INTERP_SURF_GRAD, POISSON, FO_SINCOS2D, FO_COSEXP2D, FO_COSEXP2DFLIP, FO_COSEXP2DALL,
	  FO_SINCOSZ, FO_SINEXP2D, FO_DOME};
  BFTYPE bf_type;

  std::size_t numQPs;
  std::size_t numDims;
  std::size_t vecDim;

  //Glen's law parameters
  double n; 
  double A;
  //ISMIP-HOM parameter
  double alpha; 
};
}

#endif
