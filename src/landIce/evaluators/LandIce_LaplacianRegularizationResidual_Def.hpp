//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Teuchos_CommHelpers.hpp"

#include "LandIce_LaplacianRegularizationResidual.hpp"
#include "PHAL_Utilities.hpp"

template<typename EvalT, typename Traits>
LandIce::LaplacianRegularizationResidual<EvalT, Traits>::
LaplacianRegularizationResidual(Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl)
{

  laplacian_coeff = p.get<double>("Laplacian Coefficient");
  mass_coeff = p.get<double>("Mass Coefficient");
  robin_coeff = p.get<double>("Robin Coefficient");
  lumpedMassMatrix = p.get<bool>("Lump Mass Matrix");

  // Setting up the fields required by the regularizations
  sideName = p.get<std::string> ("Side Set Name");

  TEUCHOS_TEST_FOR_EXCEPTION (dl->side_layouts.find(sideName)==dl->side_layouts.end(), std::runtime_error,
                              "Error! side data layout not found.\n");
  Teuchos::RCP<Albany::Layouts> dl_side = dl->side_layouts.at(sideName);

  const std::string& field_name     = p.get<std::string>("Field Variable Name");
  const std::string& forcing_name   = p.get<std::string>("Forcing Field Name");
  const std::string& gradField_name = p.get<std::string>("Field Gradient Variable Name");
  const std::string& gradBFname     = p.get<std::string>("Gradient BF Name");
  const std::string& w_measure_name = p.get<std::string>("Weighted Measure Name");
  const std::string& residual_name = p.get<std::string>("Laplacian Residual Name");
  const std::string& w_side_measure_name = p.get<std::string>("Weighted Measure Side Name");

  if(lumpedMassMatrix) {
    forcing        = decltype(forcing)(forcing_name, dl->node_scalar);
    field          = decltype(field)(field_name, dl->node_scalar);
  } else {
    forcing        = decltype(forcing)(forcing_name, dl->qp_scalar);
    field          = decltype(field)(field_name, dl->qp_scalar);  
    side_field     = decltype(side_field)(p.get<std::string>("Side Field Variable Name"), dl_side->qp_scalar); 
    BF             = decltype(BF)(p.get<std::string>("BF Name"), dl->node_qp_scalar); 
    side_BF        = decltype(side_BF)(p.get<std::string>("Side BF Name"), dl_side->node_qp_scalar);     
  }
  gradField      = decltype(gradField)(gradField_name, dl->qp_gradient);
  gradBF         = decltype(gradBF)(gradBFname,dl->node_qp_gradient),
  w_measure      = decltype(w_measure)(w_measure_name, dl->qp_scalar);
  residual       = decltype(residual)(residual_name, dl->node_scalar);
  w_side_measure = decltype(w_side_measure)(w_side_measure_name, dl_side->qp_scalar);

  Teuchos::RCP<shards::CellTopology> cellType;
  cellType = p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type");

  // Get Dimensions
  numCells  = dl->node_scalar->extent(0);
  numNodes  = dl->node_scalar->extent(1);

  numQPs = dl->qp_scalar->extent(1);
  cellDim  = cellType->getDimension();

  numSideNodes  = dl_side->node_scalar->extent(1);
  numSideQPs = dl_side->qp_scalar->extent(1);


  unsigned int numSides = cellType->getSideCount();
  sideDim  = cellType->getDimension()-1;

  this->addDependentField(forcing);
  this->addDependentField(field);
  this->addDependentField(gradField);
  this->addDependentField(gradBF);
  this->addDependentField(gradBF);
  this->addDependentField(w_measure);
  this->addDependentField(w_side_measure);
  if(!lumpedMassMatrix) {
    this->addDependentField(side_field);
    this->addDependentField(BF);
    this->addDependentField(side_BF);
  }


  this->addEvaluatedField(residual);

  this->setName("Laplacian Regularization Residual" + PHX::print<EvalT>());

  using PHX::MDALayout;

  unsigned int nodeMax = 0;
  for (unsigned int side=0; side<numSides; ++side) {
    unsigned int thisSideNodes = cellType->getNodeCount(sideDim,side);
    nodeMax = std::max(nodeMax, thisSideNodes);
  }
  sideNodes = Kokkos::DualView<int**, PHX::Device>("sideNodes", numSides, nodeMax);
  for (unsigned int side=0; side<numSides; ++side) {
    unsigned int thisSideNodes = cellType->getNodeCount(sideDim,side);
    for (unsigned int node=0; node<thisSideNodes; ++node) {
      sideNodes.view_host()(side,node) = cellType->getNodeMap(sideDim,side,node);
    }
  }
  sideNodes.modify_host();
  sideNodes.sync_device();
}

// **********************************************************************
template<typename EvalT, typename Traits>
void LandIce::LaplacianRegularizationResidual<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData /* d */,
                      PHX::FieldManager<Traits>& /* fm */)
{}

//**********************************************************************
//Kokkos functor
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void LandIce::LaplacianRegularizationResidual<EvalT, Traits>::
operator() (const LaplacianRegularization_Cell_Tag&, const int& cell) const {
  
  MeshScalarT trapezoid_weights = 0;
  if(lumpedMassMatrix) {
    for (unsigned int qp=0; qp<numQPs; ++qp)
      trapezoid_weights += w_measure(cell, qp);
    trapezoid_weights /= numNodes;
  }
  for (unsigned int inode=0; inode<numNodes; ++inode) {
      ScalarT t = 0;
      for (unsigned int qp=0; qp<numQPs; ++qp)
        for (unsigned int idim=0; idim<cellDim; ++idim)
          t += laplacian_coeff*gradField(cell,qp,idim)*gradBF(cell,inode, qp,idim)*w_measure(cell, qp);

      if(lumpedMassMatrix) {
        //using trapezoidal rule to get diagonal mass matrix
        t += (mass_coeff*field(cell,inode)-forcing(cell,inode))* trapezoid_weights;
      } else {
        for (unsigned int qp=0; qp<numQPs; ++qp)
          t += (mass_coeff*field(cell,qp)-forcing(cell,qp))*BF(cell,inode, qp)*w_measure(cell, qp);
      }
      residual(cell,inode) = t;
  }

}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void LandIce::LaplacianRegularizationResidual<EvalT, Traits>::
operator() (const LaplacianRegularization_Side_Tag&, const int& sideSet_idx) const {

  // Get the local data of side and cell
  const int cell = sideSet.ws_elem_idx.view_device()(sideSet_idx);
  const int side = sideSet.side_pos.view_device()(sideSet_idx);
  if(lumpedMassMatrix) {
    MeshScalarT side_trapezoid_weights= 0;
    for (unsigned int qp=0; qp<numSideQPs; ++qp)
      side_trapezoid_weights += w_side_measure(sideSet_idx, qp);
    side_trapezoid_weights /= numSideNodes;

    for (unsigned int inode=0; inode<numSideNodes; ++inode) {
      auto cell_node = sideNodes.view_device()(side,inode);
      residual(cell,cell_node) += robin_coeff*field(cell,cell_node)* side_trapezoid_weights;
    }
  } else {
    for (unsigned int inode=0; inode<numSideNodes; ++inode) {
      auto cell_node = sideNodes.view_device()(side,inode);
      for (unsigned int qp=0; qp<numSideQPs; ++qp) {
        residual(cell,cell_node) += robin_coeff*side_field(sideSet_idx,qp)*side_BF(sideSet_idx, inode, qp)*w_side_measure(sideSet_idx, qp);
      }
    }
  }
}

// **********************************************************************
template<typename EvalT, typename Traits>
void LandIce::LaplacianRegularizationResidual<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{

  Kokkos::parallel_for(LaplacianRegularization_Cell_Policy(0, numCells), *this);

  //compute robin term using lumped boundary mass matrix
  if (workset.sideSets->find(sideName) != workset.sideSets->end())
  {
    sideSet = workset.sideSetViews->at(sideName);
    Kokkos::parallel_for(LaplacianRegularization_Side_Policy(0, sideSet.size), *this);
  }

}
