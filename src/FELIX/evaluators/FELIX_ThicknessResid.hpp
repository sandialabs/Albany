//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_THICKNESSRESID_HPP
#define FELIX_THICKNESSRESID_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

#include "Intrepid_CellTools.hpp"
#include "Intrepid_Cubature.hpp"

namespace FELIX {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits>
class ThicknessResid : public PHX::EvaluatorWithBaseImpl<Traits>,
		        public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  ThicknessResid(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:

  PHX::MDField<ScalarT,Cell,Node> H;
  PHX::MDField<ScalarT,Cell,Node> H0;
  PHX::MDField<ScalarT,Cell,Node,Dim> V;
  
  // Output:
  PHX::MDField<ScalarT,Cell,Node> Residual;


  int  cellDims, sideDims, numQPsSide, numNodes, cubatureDegree;
  double dt;

  std::size_t numQPs;
  std::size_t numDims;
  std::size_t numVecDims;

  PHX::MDField<ScalarT,Cell,Node,VecDim> velocity_field;
  PHX::MDField<ScalarT,Cell,Node,VecDim> surfaceVelocity_field;
  PHX::MDField<ScalarT,Cell,Node,VecDim> velocityRMS_field;
  PHX::MDField<ScalarT,Cell,Node> basal_friction_field;
  PHX::MDField<MeshScalarT,Cell,Vertex,Dim> coordVec;

  Teuchos::RCP<shards::CellTopology> cellType;
  Teuchos::RCP<shards::CellTopology> sideType;
  Teuchos::RCP<Intrepid::Cubature<RealType> > cubatureCell;
  Teuchos::RCP<Intrepid::Cubature<RealType> > cubatureSide;

  // The basis
  Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > intrepidBasis;

  // Temporary FieldContainers
  Intrepid::FieldContainer<RealType> cubPointsSide;
  //const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs;
  Intrepid::FieldContainer<RealType> refPointsSide;
  Intrepid::FieldContainer<RealType> cubWeightsSide;
  Intrepid::FieldContainer<MeshScalarT> physPointsSide;
  Intrepid::FieldContainer<MeshScalarT> jacobianSide;
  Intrepid::FieldContainer<MeshScalarT> invJacobianSide;
  Intrepid::FieldContainer<MeshScalarT> jacobianSide_det;

  Intrepid::FieldContainer<MeshScalarT> physPointsCell;

  Intrepid::FieldContainer<MeshScalarT> weighted_measure;
  Intrepid::FieldContainer<RealType> basis_refPointsSide;
  Intrepid::FieldContainer<RealType> basisGrad_refPointsSide;
  Intrepid::FieldContainer<MeshScalarT> trans_basis_refPointsSide;
  Intrepid::FieldContainer<MeshScalarT> trans_gradBasis_refPointsSide;
  Intrepid::FieldContainer<MeshScalarT> weighted_trans_basis_refPointsSide;

  Intrepid::FieldContainer<ScalarT> dofCell;
  Intrepid::FieldContainer<ScalarT> dofSide;

  Intrepid::FieldContainer<ScalarT> dofCellVec;
  Intrepid::FieldContainer<ScalarT> dofSideVec;

  std::string sideSetID;

};
}

#endif
