/*
 * LandIce_VelocityZ_Def.hpp
 *
 *  Created on: Jun 7, 2016
 *      Author: mperego, abarone
 */

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"
#include "Shards_CellTopology.hpp"
#include "Shards_BasicTopologies.hpp"

#include "Albany_SacadoTypes.hpp"
#include "Albany_DiscretizationUtils.hpp"

#include "LandIce_w_Resid.hpp"

namespace LandIce
{

  template<typename EvalT, typename Traits, typename VelocityType>
  w_Resid<EvalT,Traits,VelocityType>::
  w_Resid(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl):
  BF           (p.get<std::string> ("BF Variable Name"), dl->node_qp_scalar),
  wBF          (p.get<std::string> ("Weighted BF Variable Name"), dl->node_qp_scalar),
  wGradBF      (p.get<std::string> ("Weighted Gradient BF Variable Name"),dl->node_qp_gradient),
  coordVec     (p.get<std::string> ("Coordinate Vector Name"),dl->vertices_vector),
  GradVelocity   (p.get<std::string> ("Velocity Gradient QP Variable Name"), dl->qp_vecgradient),
  w_z        (p.get<std::string> ("w Gradient QP Variable Name"), dl->qp_gradient),
  Residual     (p.get<std::string> ("Residual Variable Name"), dl->node_scalar)
  {
    Teuchos::RCP<shards::CellTopology> cellType;
    cellType = p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type");

    sideName = p.get<std::string> ("Side Set Name");

    TEUCHOS_TEST_FOR_EXCEPTION (dl->side_layouts.find(sideName)==dl->side_layouts.end(), std::runtime_error,
                                "Error! Basal side data layout not found.\n");
    Teuchos::RCP<Albany::Layouts> dl_side = dl->side_layouts.at(sideName);

    sideBF = decltype(sideBF)(p.get<std::string> ("BF Side Name"), dl_side->node_qp_scalar);
    side_w_measure = decltype(side_w_measure)(p.get<std::string> ("Weighted Measure Side Name"), dl_side->qp_scalar);
    side_w_qp  = decltype(side_w_qp)(p.get<std::string> ("w Side QP Variable Name"), dl_side->qp_scalar);
    side_velocity_qp = decltype(side_velocity_qp)(p.get<std::string> ("Velocity Side QP Variable Name"), dl_side->qp_vector);
    basalVerticalVelocitySideQP = decltype(basalVerticalVelocitySideQP)(p.get<std::string>("Basal Vertical Velocity Side QP Variable Name"), dl_side->qp_scalar);
    normals    = decltype(normals)(p.get<std::string> ("Side Normal Name"), dl_side->qp_vector_spacedim);

    std::vector<PHX::Device::size_type> dims;
    dl->node_qp_vector->dimensions(dims);
    numNodes = dims[1];
    numQPs   = dims[2];
    numSideQPs   = dl_side->qp_scalar->extent(1);
    numSideNodes  = dl_side->node_scalar->extent(1);

    // The stabilization needs the local column height, computed from the vertical edges of the cell.
    // Extruded meshes provide Wedge/Hexahedron cells with the bottom-face nodes first.
    const bool isExtrudedCell = cellType->getKey() == shards::Wedge<6>::key ||
                                cellType->getKey() == shards::Hexahedron<8>::key;
    TEUCHOS_TEST_FOR_EXCEPTION (!isExtrudedCell || numNodes != dl->vertices_vector->extent(1), std::runtime_error,
                                "Error! w_Resid requires Wedge<6> or Hexahedron<8> cells of an extruded mesh, with one node per vertex.\n");
    numVertEdges = numNodes/2;

    unsigned int numSides = cellType->getSideCount();
    unsigned int sideDim  = cellType->getDimension()-1;

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

    this->addDependentField(GradVelocity);
    this->addDependentField(side_velocity_qp);
    this->addDependentField(basalVerticalVelocitySideQP);
    this->addDependentField(BF);
    this->addDependentField(wBF);
    this->addDependentField(wGradBF);
    this->addDependentField(coordVec);
    this->addDependentField(sideBF);
    this->addDependentField(side_w_qp);
    this->addDependentField(side_w_measure);
    this->addDependentField(w_z);
    this->addDependentField(normals);

    this->addEvaluatedField(Residual);
    this->setName("W Residual");
  }

  //**********************************************************************
  //Kokkos functor
  template<typename EvalT, typename Traits, typename VelocityType>
  KOKKOS_INLINE_FUNCTION
  void w_Resid<EvalT,Traits,VelocityType>::
  operator() (const wResid_Cell_Tag&, const int& cell) const {

    // Incompressibility, w_z + u_x + v_y = 0, with a consistent streamline-upwind term in the vertical:
    // the test function is v + 0.5*dz*v_z, which makes the scheme integrate upward from the bed.
    // With linear basis functions, the w-w block becomes block lower triangular (layer by layer)
    // with positive definite diagonal blocks, whereas the Galerkin form has zero diagonal entries.
    for (std::size_t qp = 0; qp < numQPs; ++qp) {
      // Local column height at the quadrature point: dz = sum_k lambda_k (z_top_k - z_bot_k), where
      // lambda_k = BF(bot_k) + BF(top_k) is the horizontal weight of vertical edge k. For extruded cells
      // this equals 2 dz/dzeta, so 0.5*dz is exactly the stabilization parameter that gives the upwind structure.
      MeshScalarT dz(0);
      for (std::size_t k = 0; k < numVertEdges; ++k)
        dz += (BF(cell,k,qp) + BF(cell,k+numVertEdges,qp)) * (coordVec(cell,k+numVertEdges,2) - coordVec(cell,k,2));

      ScalarT divU = w_z(cell,qp,2) + GradVelocity(cell,qp,0,0) + GradVelocity(cell,qp,1,1);
      for (std::size_t node = 0; node < numNodes; ++node)
        Residual(cell,node) += divU * wBF(cell,node,qp) + 0.5*dz * divU * wGradBF(cell,node,qp,2);
    }

  }

  template<typename EvalT, typename Traits, typename VelocityType>
  KOKKOS_INLINE_FUNCTION
  void w_Resid<EvalT,Traits,VelocityType>::
  operator() (const wResid_Side_Tag&, const int& side_idx) const {

    // Get the local data of side and cell
    const int cell = sideSet.ws_elem_idx.view_device()(side_idx);
    const int side = sideSet.side_pos.view_device()(side_idx);

    for (unsigned int snode=0; snode<numSideNodes; ++snode){
      int cnode = sideNodes.view_device()(side,snode);
      Residual(cell,cnode) =0;
      }

    for (unsigned int snode=0; snode<numSideNodes; ++snode) {
      int cnode = sideNodes.view_device()(side,snode);
      for (std::size_t qp = 0; qp < numSideQPs; ++qp) {
      // No penetration condition at the bed
      Residual(cell,cnode) += (side_w_qp(side_idx,qp) * normals(side_idx,qp,2) +
                                  side_velocity_qp(side_idx,qp,0)  * normals(side_idx,qp,0) +
                                  side_velocity_qp(side_idx,qp,1)  * normals(side_idx,qp,1) +
                                  basalVerticalVelocitySideQP(side_idx, qp)) *
                              sideBF(side_idx,snode,qp) * side_w_measure(side_idx,qp);
      }
    }

  }

  //**********************************************************************
  template<typename EvalT, typename Traits, typename VelocityType>
  void w_Resid<EvalT,Traits,VelocityType>::
  postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>&)
  {  
    d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
  }

  template<typename EvalT, typename Traits, typename VelocityType>
  void w_Resid<EvalT,Traits,VelocityType>::
  evaluateFields(typename Traits::EvalData d)
  {
    Residual.deep_copy(0.0);

    Kokkos::parallel_for(wResid_Cell_Policy(0, d.numCells), *this);

    if (d.sideSetViews->find(sideName)==d.sideSetViews->end()) return;

    sideSet = d.sideSetViews->at(sideName);
    Kokkos::parallel_for(wResid_Side_Policy(0, sideSet.size), *this);
  }
}
