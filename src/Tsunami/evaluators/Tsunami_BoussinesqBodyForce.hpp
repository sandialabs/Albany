//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef TSUNAMI_BOUSSINESQBODYFORCE_HPP
#define TSUNAMI_BOUSSINESQBODYFORCE_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace Tsunami {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits>
class BoussinesqBodyForce : public PHX::EvaluatorWithBaseImpl<Traits>,
		    public PHX::EvaluatorDerived<EvalT, Traits> {

public:

  typedef typename EvalT::ScalarT ScalarT;

  BoussinesqBodyForce(const Teuchos::ParameterList& p,
                      const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:
 
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<const MeshScalarT,Cell,QuadPoint, Dim> coordVec;
  PHX::MDField<const ScalarT,Cell,QuadPoint> waterDepthQP;
  PHX::MDField<const ScalarT,Cell,QuadPoint> betaQP;
  PHX::MDField<const ScalarT,Cell,QuadPoint> zalphaQP;

  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim> force;

  unsigned int numQPs, numDims, vecDim;

  double muSqr, epsilon; 

  double C1, C2, C3; 
 
  Teuchos::RCP<Teuchos::FancyOStream> out; 
 
   //Body force types
  enum BFTYPE {NONE, ONED_SOL_WAVE};
  BFTYPE bf_type;
};
}

#endif
