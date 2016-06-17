//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
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
  typedef typename EvalT::ParamScalarT ParamScalarT;

  // Input:

  PHX::MDField<ScalarT,Cell,Node> dH;
  PHX::MDField<ParamScalarT,Cell,Node> H0;
  PHX::MDField<ScalarT,Cell,Node,Dim> V;
  PHX::MDField<ParamScalarT,Cell,Node> SMB;
  
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
  Teuchos::RCP<Intrepid2::Cubature<PHX::Device> > cubatureSide;

  // The basis
  Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> > intrepidBasis;

  // Temporary Views
  Kokkos::DynRankView<MeshScalarT, PHX::Device> physPointsCell;
  Kokkos::DynRankView<ScalarT, PHX::Device> dofCell;
  Kokkos::DynRankView<ScalarT, PHX::Device> dofCellVec;

  Kokkos::DynRankView<RealType, PHX::Device> cubPointsSide;
  Kokkos::DynRankView<RealType, PHX::Device> refPointsSide;
  Kokkos::DynRankView<RealType, PHX::Device> cubWeightsSide;
  Kokkos::DynRankView<RealType, PHX::Device> basis_refPointsSide;
  Kokkos::DynRankView<RealType, PHX::Device> basisGrad_refPointsSide;

  Kokkos::DynRankView<MeshScalarT, PHX::Device> physPointsSide;
  Kokkos::DynRankView<MeshScalarT, PHX::Device> jacobianSide;
  Kokkos::DynRankView<MeshScalarT, PHX::Device> invJacobianSide;
  Kokkos::DynRankView<MeshScalarT, PHX::Device> jacobianSide_det;
  Kokkos::DynRankView<MeshScalarT, PHX::Device> weighted_measure;
  Kokkos::DynRankView<MeshScalarT, PHX::Device> trans_basis_refPointsSide;
  Kokkos::DynRankView<MeshScalarT, PHX::Device> trans_gradBasis_refPointsSide;
  Kokkos::DynRankView<MeshScalarT, PHX::Device> weighted_trans_basis_refPointsSide;

  Kokkos::DynRankView<ScalarT, PHX::Device> dofSide;
  Kokkos::DynRankView<ScalarT, PHX::Device> dofSideVec;

  std::string sideSetID;

};
}

#endif
