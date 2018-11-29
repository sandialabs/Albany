//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_THICKNESS_RESID_HPP
#define LANDICE_THICKNESS_RESID_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_Cubature.hpp"

#include "PHAL_Dimension.hpp"
#include "Albany_Layouts.hpp"
#include "Albany_ScalarOrdinalTypes.hpp"

namespace LandIce {
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

  PHX::MDField<const ScalarT,Cell,Node> dH;
  PHX::MDField<const ParamScalarT,Cell,Node> H0;
  PHX::MDField<const ScalarT,Side,Cell,Node,Dim> V;
  PHX::MDField<const ParamScalarT,Cell,Node> SMB;
  PHX::MDField<const MeshScalarT,Cell,Vertex,Dim> coordVec;
  
  // Output:
  PHX::MDField<ScalarT,Cell,Node> Residual;


  int  cellDims, numNodes, cubatureDegree;
  Teuchos::RCP<double> dt;
  bool have_SMB;
  std::string meshPart;

  std::size_t numVecFODims;


  Teuchos::RCP<shards::CellTopology> cellType;
  Teuchos::RCP<shards::CellTopology> sideType;
  Teuchos::RCP<Intrepid2::Cubature<PHX::Device> > cubatureSide;

  // The basis
  Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> > intrepidBasis;

  // Temporary Views
  Kokkos::DynRankView<MeshScalarT, PHX::Device> physPointsCell;

  std::string sideSetID;

};

} // namespace LandIce

#endif // LANDICE_THICKNESS_RESID_HPP
