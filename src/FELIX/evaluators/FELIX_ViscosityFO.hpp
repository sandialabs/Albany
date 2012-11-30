//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_VISCOSITYFO_HPP
#define FELIX_VISCOSITYFO_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Sacado_ParameterAccessor.hpp" 

namespace FELIX {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits>
class ViscosityFO : public PHX::EvaluatorWithBaseImpl<Traits>,
		    public PHX::EvaluatorDerived<EvalT, Traits>,
		    public Sacado::ParameterAccessor<EvalT, SPL_Traits> {

public:

  typedef typename EvalT::ScalarT ScalarT;

  ViscosityFO(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

  ScalarT& getValue(const std::string &n); 

private:
 
  typedef typename EvalT::MeshScalarT MeshScalarT;

  ScalarT homotopyParam;

  //coefficients for Glen's law
  double A; 
  double n; 

  // Input:
  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim,Dim> Ugrad;
  PHX::MDField<MeshScalarT,Cell,QuadPoint, Dim> coordVec;

  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint> mu;

  unsigned int numQPs, numDims, numNodes;
  
  enum VISCTYPE {CONSTANT, EXPTRIG, GLENSLAW};
  VISCTYPE visc_type;
 
};
}

#endif
