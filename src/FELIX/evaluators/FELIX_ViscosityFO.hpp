//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_VISCOSITYFO_HPP
#define FELIX_VISCOSITYFO_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Sacado_ParameterAccessor.hpp" 
#include "Albany_Layouts.hpp"

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

  ViscosityFO(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

  ScalarT& getValue(const std::string &n); 

  typedef typename PHX::Device execution_space;  
  KOKKOS_INLINE_FUNCTION
  void operator () (const int i) const;

private:
 
  typedef typename EvalT::MeshScalarT MeshScalarT;

  ScalarT homotopyParam;
  ScalarT dummyParam;

  //coefficients for Glen's law
  double A; 
  double n; 

  // Input:
  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim,Dim> Ugrad;
  PHX::MDField<MeshScalarT,Cell,QuadPoint, Dim> coordVec;
  PHX::MDField<ScalarT,Cell> temperature;
  PHX::MDField<ScalarT,Cell> flowFactorA;  //this is the coefficient A.  To distinguish it from the scalar flowFactor defined in the body of the function, it is called flowFactorA.  Probably this should be changed at some point...

  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint> mu;

  unsigned int numQPs, numDims, numNodes, numCells;
  
  enum VISCTYPE {CONSTANT, EXPTRIG, GLENSLAW, GLENSLAW_XZ};
  enum FLOWRATETYPE {UNIFORM, TEMPERATUREBASED, FROMFILE, FROMCISM};
  VISCTYPE visc_type;
  FLOWRATETYPE flowRate_type;
 
};

}

#endif
