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

#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_Cubature.hpp"

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

  PHX::MDField<ScalarT,Cell,Node> dH;
  PHX::MDField<ScalarT,Cell,Node> H0;
  PHX::MDField<ScalarT,Cell,Node,Dim> V;
  Teuchos::RCP<PHX::MDField<ScalarT,Cell,Node> > SMB_ptr;
  
  // Output:
  PHX::MDField<ScalarT,Cell,Node> Residual;


  int  cellDims, sideDims, numQPsSide, numNodes, cubatureDegree;
  Teuchos::RCP<double> dt;
  bool have_SMB;
  std::string meshPart;

  std::size_t numQPs;
  std::size_t numVecFODims;

  PHX::MDField<MeshScalarT,Cell,Vertex,Dim> coordVec;


  Teuchos::RCP<shards::CellTopology> cellType;
  Teuchos::RCP<shards::CellTopology> sideType;
  Teuchos::RCP<Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,PHX::Device> > > cubatureSide;

  // The basis
  Teuchos::RCP<Intrepid2::Basis<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > > intrepidBasis;

  // Temporary FieldContainers
  Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> cubPointsSide;
  //const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs;
  Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> refPointsSide;
  Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> cubWeightsSide;
  Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device> physPointsSide;
  Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device> jacobianSide;
  Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device> invJacobianSide;
  Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device> jacobianSide_det;

  Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device> physPointsCell;

  Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device> weighted_measure;
  Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> basis_refPointsSide;
  Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> basisGrad_refPointsSide;
  Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device> trans_basis_refPointsSide;
  Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device> trans_gradBasis_refPointsSide;
  Intrepid2::FieldContainer_Kokkos<MeshScalarT, PHX::Layout, PHX::Device> weighted_trans_basis_refPointsSide;

  Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> dofCell;
  Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> dofSide;

  Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> dofCellVec;
  Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> dofSideVec;

  std::string sideSetID;

};
}

#endif
