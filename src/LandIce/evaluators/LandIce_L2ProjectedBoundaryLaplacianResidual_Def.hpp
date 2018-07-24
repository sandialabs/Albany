//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Teuchos_CommHelpers.hpp"
#include "PHAL_Utilities.hpp"

template<typename EvalT, typename Traits>
LandIce::L2ProjectedBoundaryLaplacianResidual<EvalT, Traits>::
L2ProjectedBoundaryLaplacianResidual(Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl)
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
  gradBF             = decltype(gradBF)(gradBFname,dl_side->node_qp_gradient),
  w_side_measure     = decltype(w_side_measure)(w_side_measure_name, dl_side->qp_scalar);
  side_tangents      = decltype(side_tangents)(side_tangents_name, dl_side->qp_tensor_cd_sd);
  bdLaplacian_L2Projection_res = decltype(bdLaplacian_L2Projection_res)(residual_name, dl->node_scalar);
  coordVec = decltype(coordVec)(coords_name, dl->vertices_vector);

  cellType = p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type");

  // Get Dimensions
  numCells  = dl->node_scalar->dimension(0);
  numNodes  = dl->node_scalar->dimension(1);

  numSideNodes  = dl_side->node_scalar->dimension(2);
  numBasalQPs = dl_side->qp_scalar->dimension(2);
  int numSides = dl_side->node_scalar->dimension(1);
  sideDim  = cellType->getDimension()-1;

  this->addDependentField(solution);
  this->addDependentField(field);
  this->addDependentField(gradField);
  this->addDependentField(gradBF);
  this->addDependentField(w_side_measure);
  this->addDependentField(side_tangents);
  this->addDependentField(coordVec);

  this->addEvaluatedField(bdLaplacian_L2Projection_res);

  this->setName("Boundary Laplacian L2 Projection Residual" + PHX::typeAsString<EvalT>());

  using PHX::MDALayout;


  sideNodes.resize(numSides);
  for (int side=0; side<numSides; ++side)
  {
    //Need to get the subcell exact count, since different sides may have different number of nodes (e.g., Wedge)
    int thisSideNodes = cellType->getNodeCount(sideDim,side);
    sideNodes[side].resize(thisSideNodes);
    for (int node=0; node<thisSideNodes; ++node)
      sideNodes[side][node] = cellType->getNodeMap(sideDim,side,node);
  }
}

// **********************************************************************
template<typename EvalT, typename Traits>
void LandIce::L2ProjectedBoundaryLaplacianResidual<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& fm)
{

  this->utils.setFieldData(solution, fm);
  this->utils.setFieldData(field, fm);
  this->utils.setFieldData(gradField, fm);
  this->utils.setFieldData(gradBF, fm);
  this->utils.setFieldData(w_side_measure, fm);
  this->utils.setFieldData(side_tangents, fm);

  this->utils.setFieldData(bdLaplacian_L2Projection_res, fm);
}


// **********************************************************************
template<typename EvalT, typename Traits>
void LandIce::L2ProjectedBoundaryLaplacianResidual<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  TEUCHOS_TEST_FOR_EXCEPTION (workset.sideSets==Teuchos::null, std::logic_error,
                              "Side sets defined in input file but not properly specified on the mesh" << std::endl);


  for (int cell=0; cell<numCells; ++cell)
    for (int inode=0; inode<numNodes; ++inode)
      bdLaplacian_L2Projection_res(cell,inode) = solution(cell,inode);

  if (workset.sideSets->find(sideName) != workset.sideSets->end())
  {
    const std::vector<Albany::SideStruct>& sideSet = workset.sideSets->at(sideName);
    for (auto const& it_side : sideSet)
    {
      // Get the local data of side and cell
      const int cell = it_side.elem_LID;
      const int side = it_side.side_local_id;

      MeshScalarT trapezoid_weights= 0;
      for (int qp=0; qp<numBasalQPs; ++qp)
        trapezoid_weights += w_side_measure(cell,side, qp);
      trapezoid_weights /= numSideNodes;

      for (int inode=0; inode<numSideNodes; ++inode) {
        ScalarT t = 0;
        for (int qp=0; qp<numBasalQPs; ++qp) {
          for (int icoor=0; icoor<sideDim; ++icoor) {
            ScalarT gradField_i(0.0), gradBF_i(0.0);
            for (int itan=0; itan<sideDim; ++itan) {
              gradField_i += side_tangents(cell,side,qp,icoor,itan)*gradField(cell,side,qp,itan);
              gradBF_i    += side_tangents(cell,side,qp,icoor,itan)*gradBF(cell,side,inode,qp,itan);
            }

            t -= laplacian_coeff * gradField_i*gradBF_i * w_side_measure(cell,side,qp);
          }
        }
        //for (int qp=0; qp<numBasalQPs; ++qp)
        //  for (int idim=0; idim<sideDim; ++idim)
        //    for (int jdim=0; jdim<sideDim; ++jdim)
        //      for (int icoor=0; icoor<sideDim; ++icoor) // Note: if icoor<cellDim, then tangents(...)*tangents(...)=metric
        //        t -= laplacian_coeff*side_tangents(cell,side,qp,icoor,idim)*gradField(cell,side,qp,idim)
        //                            *side_tangents(cell,side,qp,icoor,jdim)*gradBF(cell,side,inode, qp,jdim)
        //                            *w_side_measure(cell,side, qp);

        //using trapezoidal rule to get diagonal mass matrix
        t += (solution(cell,sideNodes[side][inode])-mass_coeff*field(cell,side,inode))* trapezoid_weights;

        bdLaplacian_L2Projection_res(cell,sideNodes[side][inode]) = t;
      }
    }

    for (auto const& it_side : sideSet)
    {
      //compute robin term using lumped boundary mass matrix
      const int cell = it_side.elem_LID;
      const int side = it_side.side_local_id;
      shards::CellTopology sideType(cellType->getCellTopologyData(sideDim,side));
      auto side_disc = workset.disc->getSideSetDiscretizations().at(sideName);
      auto side_gid =it_side.side_GID ;
      side_gid = workset.disc->getSideToSideSetCellMap().at(sideName).at(side_gid);
      auto side_workset = side_disc->getElemGIDws()[side_gid].ws;
      auto side_side_set = side_disc->getSideSets(side_workset);
      if(side_side_set.find(bdEdgesName) != side_side_set.end()) {
        auto bdEdges_set = side_side_set[bdEdgesName];
        for (auto const& it_bdEdge : bdEdges_set){
          if(it_bdEdge.elem_GID == side_gid) {
            unsigned int side_nodes[2] = {sideType.getNodeMap(sideDim-1,it_bdEdge.side_local_id,0),sideType.getNodeMap(sideDim-1,it_bdEdge.side_local_id,1)};
            int cell_nodes[2] = {sideNodes[side][side_nodes[0]],sideNodes[side][side_nodes[1]]};
            MeshScalarT bdEdge_measure = std::sqrt(std::pow(coordVec(cell,cell_nodes[1],0)-coordVec(cell,cell_nodes[0],0),2)+std::pow(coordVec(cell,cell_nodes[1],1)-coordVec(cell,cell_nodes[0],1),2));
            for(int i=0; i<2; ++i) {
              bdLaplacian_L2Projection_res(cell,cell_nodes[i]) -= robin_coeff*field(cell,side,side_nodes[i])*bdEdge_measure/2.0;
            }
          }
        }
      }
    }
  }
}
