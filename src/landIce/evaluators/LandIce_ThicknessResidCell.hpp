//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_THICKNESS_RESID_CELL_HPP
#define LANDICE_THICKNESS_RESID_CELL_HPP

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

    This evaluator computes the thickness evolution with Galerkin discretization and different stabilizations
    We assume that the donmain has no inflow boundary (defined as the part of the boundary where the outward normal velocity is negative), or that the thickness is zero on the inflow boundary
    Supported stabilizations: SUPG, Graph Viscosity, Edge Stabilization 
*/

template<typename EvalT, typename Traits>
class ThicknessResidCell : public PHX::EvaluatorWithBaseImpl<Traits>,
		        public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  ThicknessResidCell(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;
  typedef typename EvalT::ParamScalarT ParamScalarT;

  // Input:

  PHX::MDField<const ScalarT,Cell,Node>       Hdiff;  //[km]
  PHX::MDField<const ScalarT,Cell,Node>       dHdt;   //[m/yr]
  PHX::MDField<const ParamScalarT,Cell,Node>  H0;     //[km]
  PHX::MDField<const RealType>                V;      //[m/yr]                
  PHX::MDField<const RealType,Cell,Node>      forcing;    //[m/yr]
  PHX::MDField<const MeshScalarT,Cell,Vertex,Dim> coordVec;  //[km]
  
  // Output:
  PHX::MDField<ScalarT,Cell,Node> Residual;


  unsigned int  cellDim, numNodes, cubatureDegree;
  Teuchos::RCP<double> dt;
  std::string sideSetName, lateralSideSetName;

  std::size_t numVecFODims;

  Teuchos::RCP<shards::CellTopology> cellType;
  Teuchos::RCP<shards::CellTopology> sideType;
  Teuchos::RCP<Intrepid2::Cubature<PHX::Device> > cubatureSide;

  // The basis
  Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> > intrepidBasis;

  // Temporary Views
  Kokkos::DynRankView<MeshScalarT, PHX::Device> physPointsCell;

  std::string sideSetID;
  bool unsteady;
  bool supg; 
  bool graph_viscosity;
  bool edge_stabilization;
  bool lump_mass; 

};

} // namespace LandIce

#endif // LANDICE_THICKNESS_RESID_HPP
