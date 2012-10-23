//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_STOKESCONTRAVARIENTMETRICTENSOR_HPP
#define FELIX_STOKESCONTRAVARIENTMETRICTENSOR_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Intrepid_CellTools.hpp"
#include "Intrepid_Cubature.hpp"

namespace FELIX {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits>
class StokesContravarientMetricTensor : 
    public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  StokesContravarientMetricTensor(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::MeshScalarT MeshScalarT;
  int  numDims, numQPs;

  // Input:
  //! Coordinate vector at vertices
  PHX::MDField<MeshScalarT,Cell,Vertex,Dim> coordVec;
  Teuchos::RCP<Intrepid::Cubature<RealType> > cubature;
  Teuchos::RCP<shards::CellTopology> cellType;

  // Temporary FieldContainers
  Intrepid::FieldContainer<RealType> refPoints;
  Intrepid::FieldContainer<RealType> refWeights;
  Intrepid::FieldContainer<MeshScalarT> jacobian;
  Intrepid::FieldContainer<MeshScalarT> jacobian_inv;

  // Output:
  PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim,Dim> Gc;
};
}

#endif
