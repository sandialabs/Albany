//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Teuchos_CommHelpers.hpp"

#include "LandIce_EllipticMockModelResidual.hpp"
#include "PHAL_Utilities.hpp"

template<typename EvalT, typename Traits>
LandIce::EllipticMockModelResidual<EvalT, Traits>::
EllipticMockModelResidual(Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl)
{

  laplacian_coeff = p.get<RealType>("Laplacian Coefficient");
  robin_coeff = p.get<RealType>("Robin Coefficient");
  linearized_model = p.get<bool>("Linearize Model");
  eps = p.get<RealType>("Viscosity Regularization");
  n = p.get<RealType>("Glen's Law Exponent");
  

  // Setting up the fields required
  sideName = p.get<std::string> ("Side Set Name");

  TEUCHOS_TEST_FOR_EXCEPTION (dl->side_layouts.find(sideName)==dl->side_layouts.end(), std::runtime_error,
                              "Error! side data layout not found.\n");
  Teuchos::RCP<Albany::Layouts> dl_side = dl->side_layouts.at(sideName);

  const std::string& field_name      = p.get<std::string>("Field Variable Name");
  const std::string& field0_name     = p.get<std::string>("Nominal Field Name");
  const std::string& param0_name     = p.get<std::string>("Nominal Parameter Name");
  const std::string& param_name      = p.get<std::string>("Parameter Name");
  const std::string& forcing_name    = p.get<std::string>("Forcing Name");
  const std::string& gradField_name  = p.get<std::string>("Field Gradient Variable Name");
  const std::string& gradField0_name = p.get<std::string>("Nominal Field Gradient Name");
  const std::string& gradBFname      = p.get<std::string>("Gradient BF Name");
  const std::string& w_measure_name  = p.get<std::string>("Weighted Measure Name");
  const std::string& residual_name   = p.get<std::string>("Elliptic Mock Model Residual Name");
  const std::string& w_side_measure_name = p.get<std::string>("Weighted Measure Side Name");


  forcing        = decltype(forcing)(forcing_name, dl->qp_scalar);
  field          = decltype(field)(field_name, dl->qp_scalar);  
  param          = decltype(param)(param_name, dl->qp_scalar);
  side_field     = decltype(side_field)(p.get<std::string>("Side Field Variable Name"), dl_side->qp_scalar); 
  BF             = decltype(BF)(p.get<std::string>("BF Name"), dl->node_qp_scalar); 
  side_BF        = decltype(side_BF)(p.get<std::string>("Side BF Name"), dl_side->node_qp_scalar);       
  gradField      = decltype(gradField)(gradField_name, dl->qp_gradient);
  gradBF         = decltype(gradBF)(gradBFname,dl->node_qp_gradient),
  w_measure      = decltype(w_measure)(w_measure_name, dl->qp_scalar);
  residual       = decltype(residual)(residual_name, dl->node_scalar);
  w_side_measure = decltype(w_side_measure)(w_side_measure_name, dl_side->qp_scalar);
  if(linearized_model) {
    field0       = decltype(field0)(field0_name, dl->qp_scalar);
    gradField0   = decltype(gradField0)(gradField0_name, dl->qp_gradient);      
    param0       = decltype(param0)(param0_name, dl->qp_scalar);  
  }

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
  this->addDependentField(param);
  this->addDependentField(gradField);
  this->addDependentField(gradBF);
  this->addDependentField(gradBF);
  this->addDependentField(w_measure);
  this->addDependentField(w_side_measure);
  this->addDependentField(side_field);
  this->addDependentField(BF);
  this->addDependentField(side_BF);
  if(linearized_model){
    this->addDependentField(field0);
    this->addDependentField(param0);
    this->addDependentField(gradField0);
  }


  this->addEvaluatedField(residual);

  this->setName("Mock Elliptic Equation Residual" + PHX::print<EvalT>());

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
      sideNodes.h_view(side,node) = cellType->getNodeMap(sideDim,side,node);
    }
  }
  sideNodes.modify_host();
  sideNodes.sync_device();
}

// **********************************************************************
template<typename EvalT, typename Traits>
void LandIce::EllipticMockModelResidual<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData /* d */,
                      PHX::FieldManager<Traits>& /* fm */)
{}

//**********************************************************************
//Kokkos functor
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void LandIce::EllipticMockModelResidual<EvalT, Traits>::
operator() (const EllipticMockModel_Cell_Tag&, const int& cell) const {
  RealType q=1.0/n-1.0;
  if(linearized_model) {
    for (unsigned int inode=0; inode<numNodes; ++inode) {
        ScalarT t = 0;
        for (unsigned int qp=0; qp<numQPs; ++qp) {
          ScalarT dot0 = 0;
          ScalarT dot1 = 0;
          ScalarT dot2 = 0;
          ScalarT dot3 = 0;
          for (unsigned int idim=0; idim<cellDim; ++idim) {
            dot0 += std::pow(gradField0(cell,qp,idim),2);
            dot1 += gradField0(cell,qp,idim)*(gradField(cell,qp,idim)-gradField0(cell,qp,idim));
            dot2 += gradField0(cell,qp,idim)*gradBF(cell,inode, qp,idim);
            dot3 += gradField(cell,qp,idim)*gradBF(cell,inode, qp,idim);
          }
          t += laplacian_coeff*(q*pow(dot0+eps*eps, (q-2.0)/2.0)*dot1*dot2+pow(dot0+eps*eps, q/2.0)*dot3)*w_measure(cell, qp);

          ParamScalarT beta0 = exp(param0(cell,qp));
          t += (beta0*field(cell,qp)+(param(cell,qp)-param0(cell,qp))*beta0*field0(cell,qp)-forcing(cell,qp))*BF(cell,inode, qp)*w_measure(cell, qp);
        }
        residual(cell,inode) = t;
    }
  } else {
    for (unsigned int inode=0; inode<numNodes; ++inode) {
      ScalarT t = 0;
      for (unsigned int qp=0; qp<numQPs; ++qp) {
        ScalarT dot0 = 0;
        ScalarT dot1 = 0;
        for (unsigned int idim=0; idim<cellDim; ++idim) {
          dot0 += std::pow(gradField(cell,qp,idim),2);
          dot1 += gradField(cell,qp,idim)*gradBF(cell,inode, qp,idim);
        }
        t += laplacian_coeff*(pow(dot0+eps*eps, q/2.0)*dot1)*w_measure(cell, qp);

        ParamScalarT beta = exp(param(cell,qp));
        t += (beta*field(cell,qp)-forcing(cell,qp))*BF(cell,inode, qp)*w_measure(cell, qp);
      }
      residual(cell,inode) = t;
    }
  }
}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void LandIce::EllipticMockModelResidual<EvalT, Traits>::
operator() (const EllipticMockModel_Side_Tag&, const int& sideSet_idx) const {

  // Get the local data of side and cell
  const int cell = sideSet.ws_elem_idx.d_view(sideSet_idx);
  const int side = sideSet.side_pos.d_view(sideSet_idx);
  for (unsigned int inode=0; inode<numSideNodes; ++inode) {
    auto cell_node = sideNodes.d_view(side,inode);
    for (unsigned int qp=0; qp<numSideQPs; ++qp) {
      residual(cell,cell_node) += robin_coeff*side_field(sideSet_idx,qp)*side_BF(sideSet_idx, inode, qp)*w_side_measure(sideSet_idx, qp);
    }
  }
}

// **********************************************************************
template<typename EvalT, typename Traits>
void LandIce::EllipticMockModelResidual<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{

  Kokkos::parallel_for(EllipticMockModel_Cell_Policy(0, numCells), *this);

  //compute robin term using lumped boundary mass matrix
  if (workset.sideSets->find(sideName) != workset.sideSets->end())
  {
    sideSet = workset.sideSetViews->at(sideName);
    Kokkos::parallel_for(EllipticMockModel_Side_Policy(0, sideSet.size), *this);
  }

}
