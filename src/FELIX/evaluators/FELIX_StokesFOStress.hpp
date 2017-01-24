//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_STOKESFOSTRESS_HPP
#define FELIX_STOKESFOSTRESS_HPP

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
class StokesFOStress : public PHX::EvaluatorWithBaseImpl<Traits>,
            public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  StokesFOStress(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
           PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<ScalarT,Cell,QuadPoint>       surfaceHeight;

  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim,Dim>   Ugrad;

  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim>   U;

  PHX::MDField<ScalarT,Cell,QuadPoint>              muFELIX;

  PHX::MDField<ScalarT,Cell,QuadPoint,Dim, Dim>     Stress;

  PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>      coordVec;

  // Output:
  PHX::MDField<ScalarT,Cell,Node,VecDim> Residual;

  std::size_t numNodes;
  std::size_t numQPs;
  std::size_t numDims;
  std::size_t vecDimFO;
  Teuchos::ParameterList* stereographicMapList;
  bool useStereographicMap;
  double rho_g;
};
}

#endif
