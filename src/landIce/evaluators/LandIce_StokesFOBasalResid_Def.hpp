//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"
#include "Shards_CellTopology.hpp"

#include "Albany_DiscretizationUtils.hpp"
#include "LandIce_StokesFOBasalResid.hpp"
#include "Albany_KokkosUtils.hpp"

//uncomment the following line if you want debug output to be printed to screen
// #define OUTPUT_TO_SCREEN

#ifdef OUTPUT_TO_SCREEN
#include "Teuchos_VerboseObject.hpp"
#endif

namespace LandIce {

//**********************************************************************
template<typename EvalT, typename Traits, typename BetaScalarT>
StokesFOBasalResid<EvalT, Traits, BetaScalarT>::StokesFOBasalResid (const Teuchos::ParameterList& p,
                                           const Teuchos::RCP<Albany::Layouts>& dl) :
  residual (p.get<std::string> ("Residual Variable Name"),dl->node_vector)
{
  basalSideName = p.get<std::string>("Side Set Name");

  TEUCHOS_TEST_FOR_EXCEPTION (dl->side_layouts.find(basalSideName)==dl->side_layouts.end(), std::runtime_error,
                              "Error! Basal side data layout not found.\n");

  Teuchos::RCP<Albany::Layouts> dl_basal = dl->side_layouts.at(basalSideName);

  u         = decltype(u)(p.get<std::string> ("Velocity Side QP Variable Name"), dl_basal->qp_vector);
  beta      = decltype(beta)(p.get<std::string> ("Basal Friction Coefficient Side QP Variable Name"), dl_basal->qp_scalar);
  BF        = decltype(BF)(p.get<std::string> ("BF Side Name"), dl_basal->node_qp_scalar);
  w_measure = decltype(w_measure)(p.get<std::string> ("Weighted Measure Name"), dl_basal->qp_scalar);
  normals   = decltype(normals)(p.get<std::string> ("Side Normal Name"), dl_basal->qp_vector_spacedim);

  //If true, the tangential velocity is the same as the horizontal velocity vector
  flat_approx = p.get<bool>("Flat Bed Approximation"); 

  this->addDependentField(u);
  this->addDependentField(beta);
  this->addDependentField(BF);
  this->addDependentField(w_measure);
  if(!flat_approx)
    this->addDependentField(normals);

  this->addContributedField(residual);

  std::vector<PHX::DataLayout::size_type> dims;
  dl_basal->node_qp_gradient->dimensions(dims);
  numSideNodes = dims[1];
  numSideQPs   = dims[2];

  dl->node_vector->dimensions(dims);
  vecDimFO     = std::min((int)dims[2],2);
  vecDim       = dims[2];

  // Index of the nodes on the sides in the numeration of the cell
  Teuchos::RCP<shards::CellTopology> cellType;
  cellType = p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type");
  int sideDim = cellType->getDimension()-1;
  int numSides = cellType->getSideCount();
  int nodeMax = 0;
  for (int side=0; side<numSides; ++side) {
    int thisSideNodes = cellType->getNodeCount(sideDim,side);
    nodeMax = std::max(nodeMax, thisSideNodes);
  }
  sideNodes = Kokkos::DualView<int**, PHX::Device>("sideNodes", numSides, nodeMax);
  for (int side=0; side<numSides; ++side) {
    // Need to get the subcell exact count, since different sides may have different number of nodes (e.g., Wedge)
    int thisSideNodes = cellType->getNodeCount(sideDim,side);
    for (int node=0; node<thisSideNodes; ++node) {
      sideNodes.h_view(side,node) = cellType->getNodeMap(sideDim,side,node);
    }
  }
  sideNodes.modify_host();
  sideNodes.sync_device();

  this->setName("StokesFOBasalResid"+PHX::print<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits, typename BetaScalarT>
void StokesFOBasalResid<EvalT, Traits, BetaScalarT>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(u,fm);
  this->utils.setFieldData(beta,fm);
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(w_measure,fm);
  this->utils.setFieldData(residual,fm);
  d.fill_field_dependencies(this->dependentFields(),this->contributedFields());
}

// *********************************************************************
// Kokkos functor
template<typename EvalT, typename Traits, typename BetaScalarT>
KOKKOS_INLINE_FUNCTION
void StokesFOBasalResid<EvalT, Traits, BetaScalarT>::
operator() (const StokesFOBasalResid_Tag&, const int& sideSet_idx) const {
  
  // Get the local data of side and cell
  const int cell = sideSet.ws_elem_idx.d_view(sideSet_idx);
  const int side = sideSet.side_pos.d_view(sideSet_idx);

  ScalarT local_res[2];

  MeshScalarT bx(0.0), by(0.0); 
  for (unsigned int node=0; node<numSideNodes; ++node) {
    local_res[0] = 0.0;
    local_res[1] = 0.0;

    for (unsigned int qp=0; qp<numSideQPs; ++qp) {
      if(!flat_approx) {
        bx = -normals(sideSet_idx,qp,0)/normals(sideSet_idx,qp,2);
        by = -normals(sideSet_idx,qp,1)/normals(sideSet_idx,qp,2);
      }

      ScalarT u0 = u(sideSet_idx,qp,0);
      ScalarT u1 = u(sideSet_idx,qp,1);
      local_res[0] += (beta(sideSet_idx,qp)*(u0*(1.0+bx*bx) + u1*bx*by))*BF(sideSet_idx,node,qp)*w_measure(sideSet_idx,qp);
      local_res[1] += (beta(sideSet_idx,qp)*(u0*bx*by + u1*(1.0+by*by)))*BF(sideSet_idx,node,qp)*w_measure(sideSet_idx,qp);
    }
    for (unsigned int dim=0; dim<vecDimFO; ++dim)
      KU::atomic_add<ExecutionSpace>(&residual(cell,sideNodes.d_view(side,node),dim), local_res[dim]);
  }

}

//**********************************************************************
template<typename EvalT, typename Traits, typename BetaScalarT>
void StokesFOBasalResid<EvalT, Traits, BetaScalarT>::evaluateFields (typename Traits::EvalData workset)
{
  if (workset.sideSetViews->find(basalSideName)==workset.sideSetViews->end()) return;

  sideSet = workset.sideSetViews->at(basalSideName);

  Kokkos::parallel_for(StokesFOBasalResid_Policy(0, sideSet.size), *this);
}

} // Namespace LandIce
