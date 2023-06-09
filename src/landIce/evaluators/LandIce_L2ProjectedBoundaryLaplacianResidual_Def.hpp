//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>

#include "Teuchos_TestForException.hpp"
#include "Teuchos_CommHelpers.hpp"
#include "Phalanx_DataLayout.hpp"

#include "PHAL_Utilities.hpp"

#include "Albany_AbstractDiscretization.hpp"

#include "LandIce_L2ProjectedBoundaryLaplacianResidual.hpp"

template<typename EvalT, typename Traits, typename FieldScalarT>
LandIce::L2ProjectedBoundaryLaplacianResidualBase<EvalT, Traits, FieldScalarT>::
L2ProjectedBoundaryLaplacianResidualBase(Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl)
{

  laplacian_coeff = p.get<double>("Laplacian Coefficient");
  mass_coeff = p.get<double>("Mass Coefficient");
  robin_coeff = p.get<double>("Robin Coefficient");


  // Setting up the fields required by the regularizations
  sideName = p.get<std::string> ("Side Set Name");
  bdEdgesName = p.get<std::string> ("Boundary Edges Set Name"); //name of the set containing boundary edges of the side mesh

  TEUCHOS_TEST_FOR_EXCEPTION (dl->side_layouts.find(sideName)==dl->side_layouts.end(), std::runtime_error,
                              "Error! Basal side data layout not found.\n");
  Teuchos::RCP<Albany::Layouts> dl_side = dl->side_layouts.at(sideName);

  const std::string& solution_name       = p.get<std::string>("Solution Variable Name");
  const std::string& field_name          = p.get<std::string>("Field Name");
  const std::string& gradField_name      = p.get<std::string>("Field Gradient Name");
  const std::string& gradBFname          = p.get<std::string>("Gradient BF Side Name");
  const std::string& w_side_measure_name = p.get<std::string>("Weighted Measure Side Name");
  const std::string& side_tangents_name  = p.get<std::string>("Tangents Side Name");
  const std::string& residual_name       = p.get<std::string>("L2 Projected Boundary Laplacian Residual Name");
  const std::string& coords_name         = p.get<std::string>("Coordinate Vector Variable Name");

  solution           = decltype(solution)(solution_name, dl->node_scalar);
  field              = decltype(field)(field_name, dl_side->node_scalar);
  gradField          = decltype(gradField)(gradField_name, dl_side->qp_gradient);
  gradBF             = decltype(gradBF)(gradBFname, dl_side->node_qp_gradient),
  w_side_measure     = decltype(w_side_measure)(w_side_measure_name, dl_side->qp_scalar);
  side_tangents      = decltype(side_tangents)(side_tangents_name, dl_side->qp_tensor_cd_sd);
  bdLaplacian_L2Projection_res = decltype(bdLaplacian_L2Projection_res)(residual_name, dl->node_scalar);
  coordVec = decltype(coordVec)(coords_name, dl->vertices_vector);

  cellType = p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type");

  // Get Dimensions
  numNodes  = dl->node_scalar->extent(1);
  TEUCHOS_TEST_FOR_EXCEPTION (cellType->getBaseKey() != shards::Wedge<>::key, std::runtime_error,
      "Error! This evaluator works only with Wedge nodal finite elements.\n");

  numSideNodes  = dl_side->node_scalar->extent(1);
  numBasalQPs = dl_side->qp_scalar->extent(1);
  unsigned int numSides = cellType->getSideCount();
  sideDim  = cellType->getDimension()-1;

  this->addDependentField(solution);
  this->addDependentField(field);
  this->addDependentField(gradField);
  this->addDependentField(gradBF);
  this->addDependentField(w_side_measure);
  this->addDependentField(side_tangents);
  this->addDependentField(coordVec);

  this->addEvaluatedField(bdLaplacian_L2Projection_res);

  this->setName("Boundary Laplacian L2 Projection Residual" + PHX::print<EvalT>());

  using PHX::MDALayout;

  int nodeMax = 0;
  for (unsigned int side=0; side<numSides; ++side) {
    int thisSideNodes = cellType->getNodeCount(sideDim,side);
    nodeMax = std::max(nodeMax, thisSideNodes);
  }
  sideNodes = Kokkos::DualView<int**, PHX::Device>("sideNodes", numSides, nodeMax);
  for (unsigned int side=0; side<numSides; ++side) {
    // Need to get the subcell exact count, since different sides may have different number of nodes (e.g., Wedge)
    int thisSideNodes = cellType->getNodeCount(sideDim,side);
    for (int node=0; node<thisSideNodes; ++node) {
      sideNodes.h_view(side,node) = cellType->getNodeMap(sideDim,side,node);
    }
  }
  sideNodes.sync<PHX::Device>();
}

// **********************************************************************
template<typename EvalT, typename Traits, typename FieldScalarT>
void LandIce::L2ProjectedBoundaryLaplacianResidualBase<EvalT, Traits, FieldScalarT>::
postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
{

  this->utils.setFieldData(solution, fm);
  this->utils.setFieldData(field, fm);
  this->utils.setFieldData(gradField, fm);
  this->utils.setFieldData(gradBF, fm);
  this->utils.setFieldData(w_side_measure, fm);
  this->utils.setFieldData(side_tangents, fm);

  this->utils.setFieldData(bdLaplacian_L2Projection_res, fm);
  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
}

// **********************************************************************
template<typename EvalT, typename Traits, typename FieldScalarT>
void LandIce::L2ProjectedBoundaryLaplacianResidualBase<EvalT, Traits, FieldScalarT>::evaluateFields(typename Traits::EvalData workset)
{
  TEUCHOS_TEST_FOR_EXCEPTION (workset.sideSets==Teuchos::null, std::logic_error,
                              "Side sets defined in input file but not properly specified on the mesh" << std::endl);

  // First initialize the residual on all the nodes (to guarantee a non-singular Jacobian),
  // then overwrite the residual at the sidesets nodes with the correct value.
  // Note, this initialization works for Wedges but not for Tetrahedrons,
  // because a Tetrahedron might share a node with a basal side without sharing the side,
  // which would cause the residual to not being overwritten at those nodes.
  for (unsigned int cell=0; cell<workset.numCells; ++cell)
    for (unsigned int inode=0; inode<numNodes; ++inode)
      bdLaplacian_L2Projection_res(cell,inode) = solution(cell,inode);

  if (workset.sideSets->find(sideName) != workset.sideSets->end())
  {
    sideSet = workset.sideSetViews->at(sideName);
    
    for (int sideSet_idx = 0; sideSet_idx < sideSet.size; ++sideSet_idx)
    {
      // Get the local data of side and cell
      const int cell = sideSet.ws_elem_idx.h_view(sideSet_idx);
      const int side = sideSet.side_pos.h_view(sideSet_idx);

      MeshScalarT trapezoid_weights= 0;
      for (unsigned int qp=0; qp<numBasalQPs; ++qp)
        trapezoid_weights += w_side_measure(sideSet_idx, qp);
      trapezoid_weights /= numSideNodes;

      for (unsigned int inode=0; inode<numSideNodes; ++inode) {
        ScalarT t = 0;
        for (unsigned int qp=0; qp<numBasalQPs; ++qp) {
          for (unsigned int icoor=0; icoor<sideDim; ++icoor) {
            ScalarT gradField_i(0.0), gradBF_i(0.0);
            for (unsigned int itan=0; itan<sideDim; ++itan) {
              gradField_i += side_tangents(sideSet_idx,qp,icoor,itan)*gradField(sideSet_idx,qp,itan);
              gradBF_i    += side_tangents(sideSet_idx,qp,icoor,itan)*gradBF(sideSet_idx,inode,qp,itan);
            }

            t -= laplacian_coeff * gradField_i*gradBF_i * w_side_measure(sideSet_idx,qp);
          }
        }

        //using trapezoidal rule to get diagonal mass matrix
        t += (solution(cell,sideNodes.h_view(side,inode))-mass_coeff*field(sideSet_idx,inode))* trapezoid_weights;

        bdLaplacian_L2Projection_res(cell,sideNodes.h_view(side,inode)) = t;
      }
    }

    for (int sideSet_idx = 0; sideSet_idx < sideSet.size; ++sideSet_idx)
    {
      // Get the local data of side and cell
      const int cell = sideSet.ws_elem_idx.h_view(sideSet_idx);
      const int side = sideSet.side_pos.h_view(sideSet_idx);
      shards::CellTopology cell2dType(cellType->getCellTopologyData(sideDim,side));
      auto side_disc = workset.disc->getSideSetDiscretizations().at(sideName);
      auto side_gid = sideSet.side_GID.h_view(sideSet_idx);

      // The following line associates to each 3d-side GID the corresponding 2d-cell.
      // It's needed when the 3d mesh is not built online
      side_gid = workset.disc->getSideToSideSetCellMap().at(sideName).at(side_gid);

      //This map associates the nodes of a 2d cell to the nodes of the corresponding side of the 3d mesh
      const auto& node_map = workset.disc->getSideNodeNumerationMap().at(sideName).at(side_gid);
      auto side_workset = side_disc->getElemGIDws()[side_gid].ws;
      auto side_side_set = side_disc->getSideSets(side_workset);
      if(side_side_set.find(bdEdgesName) != side_side_set.end()) {
        auto bdEdges_set = side_side_set[bdEdgesName];
        for (auto const& it_bdEdge : bdEdges_set){
          if(it_bdEdge.elem_GID == side_gid) {
            unsigned int cell2d_nodes[2] = {cell2dType.getNodeMap(sideDim-1,it_bdEdge.side_pos,0),cell2dType.getNodeMap(sideDim-1,it_bdEdge.side_pos,1)};

            unsigned int side_nodes[2];
            for(unsigned int side_node=0; side_node < numSideNodes; ++side_node) {
              unsigned int cell2d_node = static_cast<unsigned int>(node_map[side_node]);
              if(cell2d_node == cell2d_nodes[0])
                side_nodes[0] = side_node;
              if(cell2d_node == cell2d_nodes[1])
                side_nodes[1] = side_node;
            }

            int cell_nodes[2] = {sideNodes.h_view(side,side_nodes[0]),sideNodes.h_view(side,side_nodes[1])};
            MeshScalarT bdEdge_measure_sqr = (coordVec(cell,cell_nodes[1],0)-coordVec(cell,cell_nodes[0],0))*(coordVec(cell,cell_nodes[1],0)-coordVec(cell,cell_nodes[0],0))+(coordVec(cell,cell_nodes[1],1)-coordVec(cell,cell_nodes[0],1))*(coordVec(cell,cell_nodes[1],1)-coordVec(cell,cell_nodes[0],1));
            MeshScalarT bdEdge_measure = 0.0; 
            if (bdEdge_measure_sqr > 0.0) 
              bdEdge_measure = std::sqrt(bdEdge_measure_sqr); 
            for(int i=0; i<2; ++i) {
              bdLaplacian_L2Projection_res(cell,cell_nodes[i]) -= robin_coeff*field(sideSet_idx,side_nodes[i])*bdEdge_measure/2.0;
            }
          }
        }
      }
    }

  }
}
