//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_COMPRNSVISCOSITY_HPP
#define PHAL_COMPRNSVISCOSITY_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

namespace PHAL {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits>
class ComprNSViscosity : public PHX::EvaluatorWithBaseImpl<Traits>,
		    public PHX::EvaluatorDerived<EvalT, Traits> {

public:

  typedef typename EvalT::ScalarT ScalarT;

  ComprNSViscosity(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);


private:
 
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:  
  PHX::MDField<MeshScalarT,Cell,QuadPoint, Dim> coordVec;
  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim> qFluct; //vector q' containing fluid fluctuations in primitive variables
  //reference values for viscosities
  double muref; 
  double kapparef; 
  double Tref; //reference temperature -- needed for Sutherland's viscosity law   
  double Pr; //Prandtl number
  double Cp; //specific heat at constant pressure 
 
  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint> mu;
  PHX::MDField<ScalarT,Cell,QuadPoint> kappa;
  PHX::MDField<ScalarT,Cell,QuadPoint> lambda;

   //Force types
  enum VISCTYPE {CONSTANT, SUTHERLAND};
  VISCTYPE visc_type;
  
  std::size_t numQPs;
  std::size_t numDims;
  std::size_t vecDim;

};
}

#endif
