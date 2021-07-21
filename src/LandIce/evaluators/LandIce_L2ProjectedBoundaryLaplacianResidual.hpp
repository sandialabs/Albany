//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_L2_PROJECTED_BOUNDARY_LAPLACIAN_RESIDUAL_HPP
#define LANDICE_L2_PROJECTED_BOUNDARY_LAPLACIAN_RESIDUAL_HPP

#include "Shards_CellTopology.hpp"

#include "PHAL_SeparableScatterScalarResponse.hpp"

namespace LandIce {

template<typename EvalT, typename Traits, typename FieldScalarT>
class L2ProjectedBoundaryLaplacianResidualBase : public PHX::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalT, Traits>
{
public:
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;
  typedef typename EvalT::ParamScalarT ParamScalarT;

  L2ProjectedBoundaryLaplacianResidualBase(Teuchos::ParameterList& p,
     const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
           PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);


private:

  std::string sideName, bdEdgesName;
  Kokkos::View<int**, PHX::Device> sideNodes;
  Teuchos::RCP<shards::CellTopology> cellType;

  unsigned int numNodes;

  unsigned int numSideNodes;
  unsigned int numBasalQPs;
  unsigned int sideDim;

  Albany::LocalSideSetInfo sideSet;

  PHX::MDField<const ScalarT,Cell,Node>                   solution;
  PHX::MDField<const FieldScalarT,Side,Node>              field;
  PHX::MDField<const FieldScalarT,Side,QuadPoint,Dim>     gradField;
  PHX::MDField<const MeshScalarT,Side,Node,QuadPoint,Dim> gradBF;
  PHX::MDField<const MeshScalarT,Side,QuadPoint>          w_side_measure;
  PHX::MDField<const MeshScalarT,Side,QuadPoint,Dim,Dim>  side_tangents;
  PHX::MDField<const MeshScalarT>                         coordVec;

  PHX::MDField<ScalarT,Cell,Node> bdLaplacian_L2Projection_res;


  ScalarT p_reg, reg;
  double laplacian_coeff, mass_coeff, robin_coeff;

};

// Some shortcut names
template<typename EvalT, typename Traits>
using L2ProjectedBoundaryLaplacianResidual = L2ProjectedBoundaryLaplacianResidualBase<EvalT,Traits,typename EvalT::ScalarT>;

// Some shortcut names
template<typename EvalT, typename Traits>
using L2ProjectedBoundaryLaplacianResidualMesh = L2ProjectedBoundaryLaplacianResidualBase<EvalT,Traits,typename EvalT::MeshScalarT>;

// Some shortcut names
template<typename EvalT, typename Traits>
using L2ProjectedBoundaryLaplacianResidualParam = L2ProjectedBoundaryLaplacianResidualBase<EvalT,Traits,typename EvalT::ParamScalarT>;

} // Namespace LandIce

#endif // LANDICE_RESPONSE_SURFACE_VELOCITY_MISMATCH_HPP
