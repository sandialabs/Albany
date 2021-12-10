//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"
#include "Shards_CellTopology.hpp"

#include "Albany_DiscretizationUtils.hpp"
#include "LandIce_StokesFOLateralResid.hpp"
#include "Albany_KokkosUtils.hpp"

namespace LandIce {

//**********************************************************************
template<typename EvalT, typename Traits, typename ThicknessScalarT>
StokesFOLateralResid<EvalT, Traits, ThicknessScalarT>::
StokesFOLateralResid (const Teuchos::ParameterList& p,
                      const Teuchos::RCP<Albany::Layouts>& dl)
{
  // Get side layouts
  lateralSideName = p.get<std::string>("Side Set Name");
  TEUCHOS_TEST_FOR_EXCEPTION (dl->side_layouts.find(lateralSideName)==dl->side_layouts.end(), std::runtime_error,
                              "Error! Lateral side data layout not found.\n");
  Teuchos::RCP<Albany::Layouts> dl_lateral = dl->side_layouts.at(lateralSideName);

  // Create dependent fields
  thickness  = decltype(thickness)(p.get<std::string> ("Ice Thickness Variable Name"), dl_lateral->qp_scalar);
  BF         = decltype(BF)(p.get<std::string> ("BF Side Name"), dl_lateral->node_qp_scalar);
  normals    = decltype(normals)(p.get<std::string> ("Side Normal Name"), dl_lateral->qp_vector_spacedim);
  w_measure  = decltype(w_measure)(p.get<std::string> ("Weighted Measure Name"), dl_lateral->qp_scalar);

  this->addDependentField(thickness);
  this->addDependentField(BF);
  this->addDependentField(normals);
  this->addDependentField(w_measure);

  Teuchos::ParameterList& bc_pl = *p.get<Teuchos::ParameterList*>("Lateral BC Parameters");
  immerse_ratio_provided = bc_pl.isParameter("Immersed Ratio");
  if (immerse_ratio_provided) {
    given_immersed_ratio = bc_pl.get<double>("Immersed Ratio");
  } else {
    elevation = decltype(elevation)(p.get<std::string> ("Ice Surface Elevation Variable Name"), dl_lateral->qp_scalar);
    this->addDependentField(elevation);
  }

  add_melange_force=false;
  melange_force_value = melange_thickness_threshold =0;

  if (bc_pl.isParameter("Melange Force")) {
    melange_force_value = bc_pl.get<double>("Melange Force");
    if (melange_force_value != 0) {
      add_melange_force = true;
      melange_thickness_threshold = bc_pl.get<double>("Melange Submerged Thickness Threshold");
    }
  }

  // Create evaluated field
  residual = decltype(residual)(p.get<std::string> ("Residual Variable Name"), dl->node_vector);
  this->addContributedField(residual);

  // Get stereographic map info
  Teuchos::ParameterList* stereographicMapList = p.get<Teuchos::ParameterList*>("Stereographic Map");
  use_stereographic_map = stereographicMapList->get("Use Stereographic Map", false);
  if (use_stereographic_map) {
    double R = stereographicMapList->get<double>("Earth Radius", 6371);
    X_0 = stereographicMapList->get<double>("X_0", 0);//-136);
    Y_0 = stereographicMapList->get<double>("Y_0", 0);//-2040);
    R2 = std::pow(R,2);

    coords_qp = decltype(coords_qp)(p.get<std::string>("Coordinate Vector Variable Name"), dl_lateral->qp_coords);
    this->addDependentField(coords_qp);
  }

  // Get physical parameters
  const Teuchos::ParameterList& physical_params = *p.get<Teuchos::ParameterList*>("Physical Parameters");
  rho_w = physical_params.get<double>("Water Density"); // [Kg m^{-3}]
  rho_i = physical_params.get<double>("Ice Density"); // [Kg m^{-3}]
  g     = physical_params.get<double>("Gravity Acceleration"); // [m s^{-2}]

  // Get dimensions
  std::vector<PHX::DataLayout::size_type> dims;
  dl_lateral->node_qp_gradient->dimensions(dims);
  numSideNodes = dims[1];
  numSideQPs   = dims[2];
  dl->node_vector->dimensions(dims);
  vecDimFO     = std::min((int)dims[2],2);

  // Index of the nodes on the sides in the numeration of the cell
  Teuchos::RCP<shards::CellTopology> cellType;
  cellType = p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type");
  unsigned int numSides = cellType->getSideCount();
  int sideDim = cellType->getDimension()-1;
  int nodeMax = 0;
  for (unsigned int side=0; side<numSides; ++side) {
    int thisSideNodes = cellType->getNodeCount(sideDim,side);
    nodeMax = std::max(nodeMax, thisSideNodes);
  }
  sideNodes = Kokkos::View<int**, PHX::Device>("sideNodes", numSides, nodeMax);
  for (unsigned int side=0; side<numSides; ++side) {
    // Need to get the subcell exact count, since different sides may have different number of nodes (e.g., Wedge)
    int thisSideNodes = cellType->getNodeCount(sideDim,side);
    for (int node=0; node<thisSideNodes; ++node) {
      sideNodes(side,node) = cellType->getNodeMap(sideDim,side,node);
    }
  }

  this->setName("StokesFOLateralResid"+PHX::print<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits, typename ThicknessScalarT>
void StokesFOLateralResid<EvalT, Traits, ThicknessScalarT>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  // Inputs
  this->utils.setFieldData(thickness,fm);
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(normals,fm);
  this->utils.setFieldData(w_measure,fm);
  if (!immerse_ratio_provided) {
    this->utils.setFieldData(elevation,fm);
  }

  // Output
  this->utils.setFieldData(residual,fm);
  d.fill_field_dependencies(this->dependentFields(),this->contributedFields());
}

// *********************************************************************
// Kokkos functor
template<typename EvalT, typename Traits, typename ThicknessScalarT>
KOKKOS_INLINE_FUNCTION
void StokesFOLateralResid<EvalT, Traits, ThicknessScalarT>::
operator() (const GivenImmersedRatio_Tag&, const int& sideSet_idx) const {
  
  const double scale = 1e-6; //[k^2]

  // Get the local data of side and cell
  const int cell = sideSet.elem_LID(sideSet_idx);
  const int side = sideSet.side_local_id(sideSet_idx);

  for (unsigned int qp=0; qp<numSideQPs; ++qp) {
    const ThicknessScalarT H = thickness(sideSet_idx,qp); //[km]
    OutputScalarT w_normal_stress = -0.5 * g * H * (rho_i - rho_w*given_immersed_ratio*given_immersed_ratio); //[kPa]
    if(add_melange_force)
      w_normal_stress += scale * melange_force_value * KU::min(given_immersed_ratio*H/melange_thickness_threshold, 1.0) / H;

    w_normal_stress *= w_measure(sideSet_idx,qp);

    if (use_stereographic_map) {
      const MeshScalarT x = coords_qp(sideSet_idx,qp,0) - X_0;
      const MeshScalarT y = coords_qp(sideSet_idx,qp,1) - Y_0;
      const MeshScalarT h = 4.0*R2/(4.0*R2 + x*x + y*y);
      w_normal_stress *= h;
    }
    for (unsigned int node=0; node<numSideNodes; ++node) {
      int sideNode = sideNodes(side,node);
      // NOTE: we are RELYING on the fact that the lateral side is vertical, so that u*n = ux*nx+uy*ny.
      const OutputScalarT w_normal_stress_bf = w_normal_stress * BF(sideSet_idx,node,qp);
      for (unsigned int dim=0; dim<vecDimFO; ++dim) {
        KU::atomic_add<ExecutionSpace>(&residual(cell,sideNode,dim), w_normal_stress_bf * normals(sideSet_idx,qp,dim));
      }
    }
  }

}

template<typename EvalT, typename Traits, typename ThicknessScalarT>
KOKKOS_INLINE_FUNCTION
void StokesFOLateralResid<EvalT, Traits, ThicknessScalarT>::
operator() (const ComputedImmersedRatio_Tag&, const int& sideSet_idx) const {
  
  const double scale = 1e-6; //[k^2]

  // Get the local data of side and cell
  const int cell = sideSet.elem_LID(sideSet_idx);
  const int side = sideSet.side_local_id(sideSet_idx);

  const OutputScalarT zero (0.0);
  const ThicknessScalarT threshold (1e-8);
  const OutputScalarT one (1.0);

  for (unsigned int qp=0; qp<numSideQPs; ++qp) {
    const ThicknessScalarT H = thickness(sideSet_idx,qp); //[km]
    const ThicknessScalarT s = elevation(sideSet_idx,qp); //[km]
    const OutputScalarT immersed_ratio = H>threshold ? KU::max(zero,KU::min(one,1-s/H)) : zero;
    OutputScalarT w_normal_stress = -0.5 * g * H * (rho_i - rho_w*immersed_ratio*immersed_ratio); //[kPa]
    if(add_melange_force)
      w_normal_stress += scale * melange_force_value * KU::min(immersed_ratio*H/melange_thickness_threshold, 1.0) / H;
    
    w_normal_stress *= w_measure(sideSet_idx,qp);
    
    if (use_stereographic_map) {
      const MeshScalarT x = coords_qp(sideSet_idx,qp,0) - X_0;
      const MeshScalarT y = coords_qp(sideSet_idx,qp,1) - Y_0;
      const MeshScalarT h = 4.0*R2/(4.0*R2 + x*x + y*y);
      w_normal_stress *= h;
    }
    for (unsigned int node=0; node<numSideNodes; ++node) {
      int sideNode = sideNodes(side,node);
      // The immersed ratio should be between 0 and 1. If s>=H, it is 0, since the ice bottom is at s-H, which is >=0.
      // If s<=0, it is 1, since the top is already under water. If 0<s<H it is somewhere in (0,1), since the top is above the sea level,
      // but the bottom is s-H<0, which is below the sea level.
      // NOTE: we are RELYING on the fact that the lateral side is vertical, so that u*n = ux*nx+uy*ny.
      const OutputScalarT w_normal_stress_bf = w_normal_stress * BF(sideSet_idx,node,qp);
      for (unsigned int dim=0; dim<vecDimFO; ++dim) {
        KU::atomic_add<ExecutionSpace>(&residual(cell,sideNode,dim), w_normal_stress_bf * normals(sideSet_idx,qp,dim));
      }
    }
  }
  
}

//**********************************************************************
template<typename EvalT, typename Traits, typename ThicknessScalarT>
void StokesFOLateralResid<EvalT, Traits, ThicknessScalarT>::evaluateFields (typename Traits::EvalData workset)
{
  if (workset.sideSetViews->find(lateralSideName)==workset.sideSetViews->end()) return;

  sideSet = workset.sideSetViews->at(lateralSideName);

  if (immerse_ratio_provided) {
    evaluate_with_given_immersed_ratio(workset);
  } else {
    evaluate_with_computed_immersed_ratio(workset);
  }
}

template<typename EvalT, typename Traits, typename ThicknessScalarT>
void StokesFOLateralResid<EvalT, Traits, ThicknessScalarT>::
evaluate_with_computed_immersed_ratio (typename Traits::EvalData /* workset */)
{
  Kokkos::parallel_for(ComputedImmersedRatio_Policy(0, sideSet.size), *this);
}

template<typename EvalT, typename Traits, typename ThicknessScalarT>
void StokesFOLateralResid<EvalT, Traits, ThicknessScalarT>::
evaluate_with_given_immersed_ratio (typename Traits::EvalData /* workset */)
{
  Kokkos::parallel_for(GivenImmersedRatio_Policy(0, sideSet.size), *this);  
}

} // Namespace LandIce
