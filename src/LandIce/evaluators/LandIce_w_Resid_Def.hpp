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

#include "Albany_SacadoTypes.hpp"
#include "Albany_DiscretizationUtils.hpp"
#include "Albany_KokkosUtils.hpp"

#include "LandIce_w_Resid.hpp"

namespace LandIce
{

  template<typename Type>
  Type distance (const Type& x0, const Type& x1, const Type& x2,
                 const Type& y0, const Type& y1, const Type& y2)
  {
    return std::sqrt(std::pow(x0-y0,2) +
                     std::pow(x1-y1,2) +
                     std::pow(x2-y2,2));
  }

  template<typename EvalT, typename Traits, typename VelocityType>
  w_Resid<EvalT,Traits,VelocityType>::
  w_Resid(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl):
  wBF          (p.get<std::string> ("Weighted BF Variable Name"), dl->node_qp_scalar),
  wGradBF      (p.get<std::string> ("Weighted Gradient BF Variable Name"),dl->node_qp_gradient),
  GradVelocity   (p.get<std::string> ("Velocity Gradient QP Variable Name"), dl->qp_vecgradient),
  velocity (p.get<std::string> ("Velocity QP Variable Name"), dl->qp_vector),
  w_z        (p.get<std::string> ("w Gradient QP Variable Name"), dl->qp_gradient),
  coordVec     (p.get<std::string> ("Coordinate Vector Name"),dl->vertices_vector),
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
    basalVerticalVelocitySideQP = decltype(basalVerticalVelocitySideQP)(p.get<std::string>("Basal Vertical Velocity Side QP Variable Name"), dl_side->qp_scalar);
    normals    = decltype(normals)(p.get<std::string> ("Side Normal Name"), dl_side->qp_vector_spacedim);

    std::vector<PHX::Device::size_type> dims;
    dl->node_qp_vector->dimensions(dims);
    numNodes = dims[1];
    numQPs   = dims[2];
    numSideQPs   = dl_side->qp_scalar->extent(1);
    numSideNodes  = dl_side->node_scalar->extent(1);

    unsigned int numSides = cellType->getSideCount();
    unsigned int sideDim  = cellType->getDimension()-1;

    unsigned int nodeMax = 0;
    for (unsigned int side=0; side<numSides; ++side) {
      unsigned int thisSideNodes = cellType->getNodeCount(sideDim,side);
      nodeMax = std::max(nodeMax, thisSideNodes);
    }
    sideNodes = Kokkos::View<int**, PHX::Device>("sideNodes", numSides, nodeMax);
    for (unsigned int side=0; side<numSides; ++side) {
      unsigned int thisSideNodes = cellType->getNodeCount(sideDim,side);
      for (unsigned int node=0; node<thisSideNodes; ++node) {
        sideNodes(side,node) = cellType->getNodeMap(sideDim,side,node);
      }
    }

    this->addDependentField(GradVelocity);
    this->addDependentField(velocity);
    this->addDependentField(basalVerticalVelocitySideQP);
    this->addDependentField(wBF);
    this->addDependentField(wGradBF);
    this->addDependentField(sideBF);
    this->addDependentField(side_w_qp);
    this->addDependentField(side_w_measure);
    this->addDependentField(w_z);
    this->addDependentField(coordVec);
    this->addDependentField(normals);

    this->addEvaluatedField(Residual);
    this->setName("W Residual");
  }

  //**********************************************************************
  //Kokkos functor
  template<typename EvalT, typename Traits, typename VelocityType>
  KOKKOS_INLINE_FUNCTION
  void w_Resid<EvalT,Traits,VelocityType>::
  operator() (const wResid_Cell_Tag& tag, const int& cell) const {

    MeshScalarT diam_z(0);//, diam_xy(0), diam_z(0);
    for (std::size_t i = 0; i < numNodes; ++i) {
      //  diam = std::max(diam,distance<MeshScalarT>(coordVec(cell,i,0),coordVec(cell,i,1),coordVec(cell,i,2),
      //                                              coordVec(cell,0,0),coordVec(cell,0,1),coordVec(cell,j,2)));
      //  diam_xy = std::max(diam_xy,distance<MeshScalarT>(coordVec(cell,i,0),coordVec(cell,i,1),MeshScalarT(0.0),coordVec(cell,0,0),coordVec(cell,0,1),MeshScalarT(0.0)));
      diam_z = KU::max(diam_z,std::abs(coordVec(cell,i,2) - coordVec(cell,0,2)));
    }
    for (std::size_t node = 0; node < numNodes; ++node)
      for (std::size_t qp = 0; qp < numQPs; ++qp)
        Residual(cell,node) += ( w_z(cell,qp,2) + GradVelocity(cell,qp,0,0) +  GradVelocity(cell,qp,1,1) ) * wBF(cell,node,qp)
                            + 0.0*  diam_z * w_z(cell,qp,2) * wGradBF(cell,node,qp,2);// + diam_xy * GradVelocity(cell,qp,0,0) * wGradBF(cell,node,qp,0);// +  diam_xy * GradVelocity(cell,qp,1,1) * wGradBF(cell,node,qp,1);

  }

  template<typename EvalT, typename Traits, typename VelocityType>
  KOKKOS_INLINE_FUNCTION
  void w_Resid<EvalT,Traits,VelocityType>::
  operator() (const wResid_Side_Tag& tag, const int& side_idx) const {

    // Get the local data of side and cell
    const int cell = sideSet.elem_LID(side_idx);
    const int side = sideSet.side_local_id(side_idx);

    for (unsigned int snode=0; snode<numSideNodes; ++snode){
      int cnode = sideNodes(side,snode);
      Residual(cell,cnode) =0;
      }

    for (unsigned int snode=0; snode<numSideNodes; ++snode) {
      int cnode = sideNodes(side,snode);
      for (std::size_t qp = 0; qp < numSideQPs; ++qp) {
      Residual(cell,cnode) += (side_w_qp(side_idx,qp) * normals(side_idx,qp,2) +
                                  velocity(cell,qp,0)  * normals(side_idx,qp,0) +
                                  velocity(cell,qp,1)  * normals(side_idx,qp,1) +
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
