//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_STOKESBODYFORCE_HPP
#define FELIX_STOKESBODYFORCE_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace FELIX {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits>
class StokesBodyForce : public PHX::EvaluatorWithBaseImpl<Traits>,
		    public PHX::EvaluatorDerived<EvalT, Traits> {

public:

  typedef typename EvalT::ScalarT ScalarT;

  StokesBodyForce(const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);


private:
 
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:  
  PHX::MDField<const ScalarT,Cell,QuadPoint> muFELIX;
  PHX::MDField<const MeshScalarT,Cell,QuadPoint, Dim> coordVec;
  Teuchos::Array<double> gravity;
  
  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> force;

   //Radom field types
  enum BFTYPE {NONE, GRAVITY, TESTAMMF, POLY, POLYSACADO, SINSIN, SINSINGLEN, FULLSTOKESBASAL, SINCOSZ};
  BFTYPE bf_type;

  unsigned int numQPs, numDims;

  //Glen's law parameters
  double n; 
  double A;
};
}

#endif
