//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"
#include "Sacado.hpp"

#include "Albany_AbstractDiscretization.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_GlobalLocalIndexer.hpp"

#include "LandIce_Gather2DField.hpp"

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN

namespace LandIce {

//**********************************************************************

template<typename EvalT, typename Traits>
Gather2DFieldBase<EvalT, Traits>::
Gather2DFieldBase(const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Albany::Layouts>& dl)
 : field2D(p.get<std::string>("2D Field Name"), dl->node_scalar)
{
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

  this->addEvaluatedField(field2D);
  cell_topo = p.get<Teuchos::RCP<const CellTopologyData> >("Cell Topology");

  std::vector<PHX::DataLayout::size_type> dims;

  dl->node_gradient->dimensions(dims);
  numNodes = dims[1];

  this->setName("Gather2DField"+PHX::print<EvalT>());

  if (p.isType<int>("Offset of First DOF")) {
    offset = p.get<int>("Offset of First DOF");
  } else {
    offset = 2;
  }

  fieldLevel = p.get<int>("Field Level");
}

//**********************************************************************
template<typename EvalT, typename Traits>
void Gather2DFieldBase<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData /* d */,
                      PHX::FieldManager<Traits>& fm)
{
    this->utils.setFieldData(field2D,fm);
}

//**********************************************************************

template<typename Traits>
Gather2DField<PHAL::AlbanyTraits::Residual, Traits>::
Gather2DField(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl)
 : Gather2DFieldBase<PHAL::AlbanyTraits::Residual, Traits>(p,dl)
{
  if (p.isType<const std::string>("Mesh Part")) {
    this->meshPart = p.get<const std::string>("Mesh Part");
  } else {
    this->meshPart = "upperside";
  }
}

template<typename Traits>
void Gather2DField<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  auto nodeID = workset.wsElNodeEqID;
  Teuchos::ArrayRCP<const ST> x_constView = Albany::getLocalData(workset.x);

  const Albany::SideSetList& ssList = *(workset.sideSets);
  Albany::SideSetList::const_iterator it = ssList.find(this->meshPart);

  if (it != ssList.end()) {
    const std::vector<Albany::SideStruct>& sideSet = it->second;

    for (std::size_t iSide = 0; iSide < sideSet.size(); ++iSide) { // loop over the sides on this ws and name
      // Get the data that corresponds to the side
      const int elem_LID = sideSet[iSide].elem_LID;
      const int elem_side = sideSet[iSide].side_local_id;
      const CellTopologyData_Subcell& side =  this->cell_topo->side[elem_side];
      unsigned int numSideNodes = side.topology->node_count;
      for (unsigned int i = 0; i < numSideNodes; ++i){
        std::size_t node = side.node[i];
        this->field2D(elem_LID,node) = x_constView[nodeID(elem_LID,node,this->offset)];
      }
    }
  }
}

template<typename Traits>
Gather2DField<PHAL::AlbanyTraits::Jacobian, Traits>::
Gather2DField(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl)
 : Gather2DFieldBase<PHAL::AlbanyTraits::Jacobian, Traits>(p,dl)
{
  if (p.isType<const std::string>("Mesh Part")) {
    this->meshPart = p.get<const std::string>("Mesh Part");
  } else {
    this->meshPart = "upperside";
  }
}

template<typename Traits>
void Gather2DField<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  auto nodeID = workset.wsElNodeEqID;
  Teuchos::ArrayRCP<const ST> x_constView = Albany::getLocalData(workset.x);

  TEUCHOS_TEST_FOR_EXCEPTION(workset.sideSets.is_null(), std::logic_error,
                             "Side sets defined in input file but not properly specified on the mesh.\n");

  const Albany::SideSetList& ssList = *(workset.sideSets);
  Albany::SideSetList::const_iterator it = ssList.find(this->meshPart);
  const int neq = nodeID.extent(2);

  if (it != ssList.end()) {
    const std::vector<Albany::SideStruct>& sideSet = it->second;

    // Loop over the sides that form the boundary condition
    for (std::size_t iSide = 0; iSide < sideSet.size(); ++iSide) { // loop over the sides on this ws and name

      // Get the data that corresponds to the side
      const int elem_LID = sideSet[iSide].elem_LID;
      const int elem_side = sideSet[iSide].side_local_id;
      const CellTopologyData_Subcell& side =  this->cell_topo->side[elem_side];
      unsigned int numSideNodes = side.topology->node_count;

      for (unsigned int i = 0; i < numSideNodes; ++i){
        std::size_t node = side.node[i];
        typename PHAL::Ref<ScalarT>::type val = (this->field2D)(elem_LID,node);
        val = FadType(val.size(), x_constView[nodeID(elem_LID,node,this->offset)]);
        val.fastAccessDx(node*neq+this->offset) = workset.j_coeff;
      }
    }
  }
}

template<typename Traits>
Gather2DField<PHAL::AlbanyTraits::Tangent, Traits>::
Gather2DField(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl)
 : Gather2DFieldBase<PHAL::AlbanyTraits::Tangent, Traits>(p,dl)
{
  // Nothing to do here
}

template<typename Traits>
Gather2DField<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
Gather2DField(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl)
 : Gather2DFieldBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p,dl)
{
  // Nothing to do here
}

template<typename Traits>
Gather2DField<PHAL::AlbanyTraits::HessianVec, Traits>::
Gather2DField(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl)
 : Gather2DFieldBase<PHAL::AlbanyTraits::HessianVec, Traits>(p,dl)
{
  // Nothing to do here
}

template<typename Traits>
void Gather2DField<PHAL::AlbanyTraits::HessianVec, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  auto nodeID = workset.wsElNodeEqID;
  Teuchos::ArrayRCP<const ST> x_constView = Albany::getLocalData(workset.x);
  Teuchos::RCP<const Thyra_MultiVector> direction_x = workset.hessianWorkset.direction_x;
  Teuchos::ArrayRCP<const ST> direction_x_constView;

  bool g_xx_is_active = !workset.hessianWorkset.hess_vec_prod_g_xx.is_null();
  bool g_xp_is_active = !workset.hessianWorkset.hess_vec_prod_g_xp.is_null();
  bool g_px_is_active = !workset.hessianWorkset.hess_vec_prod_g_px.is_null();
  bool f_xx_is_active = !workset.hessianWorkset.hess_vec_prod_f_xx.is_null();
  bool f_xp_is_active = !workset.hessianWorkset.hess_vec_prod_f_xp.is_null();
  bool f_px_is_active = !workset.hessianWorkset.hess_vec_prod_f_px.is_null();

  // is_x_active is true if we compute the Hessian-vector product contributions of either:
  // Hv_g_xx, Hv_g_xp, Hv_f_xx, or Hv_f_xp, i.e. if the first derivative is w.r.t. the solution.
  // If one of those is active, we have to initialize the first level of AD derivatives:
  // .fastAccessDx().val().
  const bool is_x_active = g_xx_is_active || g_xp_is_active || f_xx_is_active || f_xp_is_active;

  // is_x_direction_active is true if we compute the Hessian-vector product contributions of either:
  // Hv_g_xx, Hv_g_px, Hv_f_xx, or Hv_f_px, i.e. if the second derivative is w.r.t. the solution direction.
  // If one of those is active, we have to initialize the second level of AD derivatives:
  // .val().fastAccessDx().
  const bool is_x_direction_active = g_xx_is_active || g_px_is_active || f_xx_is_active || f_px_is_active;

  if(is_x_direction_active) {
    TEUCHOS_TEST_FOR_EXCEPTION(
        direction_x.is_null(),
        Teuchos::Exceptions::InvalidParameter,
        "\nError in Gather2DField<HessianVec, Traits>: "
        "direction_x is not set and hess_vec_prod_g_xx or"
        "hess_vec_prod_g_px is set.\n");
    direction_x_constView = Albany::getLocalData(direction_x->col(0));
  }

  TEUCHOS_TEST_FOR_EXCEPTION(workset.sideSets.is_null(), std::logic_error,
                             "Side sets defined in input file but not properly specified on the mesh.\n");

  int numLayers = workset.disc->getLayeredMeshNumbering()->numLayers;
  this->fieldLevel = (this->fieldLevel < 0) ? numLayers : this->fieldLevel;

  const Albany::SideSetList& ssList = *(workset.sideSets);
  Albany::SideSetList::const_iterator it = ssList.find(this->meshPart);

  if (it != ssList.end()) {
    const std::vector<Albany::SideStruct>& sideSet = it->second;
    const int neq = nodeID.extent(2);

    // Loop over the sides that form the boundary condition
    for (std::size_t iSide = 0; iSide < sideSet.size(); ++iSide) { // loop over the sides on this ws and name

      // Get the data that corresponds to the side
      const int elem_LID = sideSet[iSide].elem_LID;
      const int elem_side = sideSet[iSide].side_local_id;
      const CellTopologyData_Subcell& side =  this->cell_topo->side[elem_side];
      unsigned int numSideNodes = side.topology->node_count;

      for (unsigned int i = 0; i < numSideNodes; ++i){
        std::size_t node = side.node[i];
        typename PHAL::Ref<ScalarT>::type val = (this->field2D)(elem_LID,node);
        val = HessianVecFad(val.size(), x_constView[nodeID(elem_LID,node,this->offset)]);
        // If we differentiate w.r.t. the solution, we have to set the first
        // derivative to workset.j_coeff
        if (is_x_active)
          val.fastAccessDx(neq*node+this->offset).val() = workset.j_coeff;
        // If we differentiate w.r.t. the solution direction, we have to set
        // the second derivative to the related direction value
        if (is_x_direction_active)
          val.val().fastAccessDx(0) = direction_x_constView[nodeID(elem_LID,node,this->offset)];
      }
    }
  }
}

//********************************

template<typename Traits>
GatherExtruded2DField<PHAL::AlbanyTraits::Residual, Traits>::
GatherExtruded2DField(const Teuchos::ParameterList& p,
                      const Teuchos::RCP<Albany::Layouts>& dl)
 : Gather2DFieldBase<PHAL::AlbanyTraits::Residual, Traits>(p,dl)
{
  this->setName("GatherExtruded2DField Residual");
}

template<typename Traits>
void GatherExtruded2DField<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::ArrayRCP<const ST> x_constView = Albany::getLocalData(workset.x);

  TEUCHOS_TEST_FOR_EXCEPTION (workset.disc->getLayeredMeshNumbering().is_null(),
    std::runtime_error, "Error! No layered numbering in the mesh.\n");

  const Albany::LayeredMeshNumbering<GO>& layeredMeshNumbering = *workset.disc->getLayeredMeshNumbering();
  const Albany::NodalDOFManager& solDOFManager = workset.disc->getOverlapDOFManager(workset.disc->solution_dof_name());
  const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> >& wsElNodeID  = workset.disc->getWsElNodeID()[workset.wsIndex];
  const auto& indexer = *workset.disc->getOverlapNodeGlobalLocalIndexer();

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<GO>& elNodeID = wsElNodeID[cell];

    for (std::size_t node = 0; node < this->numNodes; ++node) {
      // Retrieve corresponding 2D node
      const GO base_id = layeredMeshNumbering.getColumnId(elNodeID[node]);
      GO gnode = layeredMeshNumbering.getId(base_id, this->fieldLevel);
      LO lnode = indexer.getLocalElement(gnode);
      (this->field2D)(cell,node) = x_constView[solDOFManager.getLocalDOF(lnode, this->offset)];
    }
  }
}

template<typename Traits>
GatherExtruded2DField<PHAL::AlbanyTraits::Jacobian, Traits>::
GatherExtruded2DField(const Teuchos::ParameterList& p,
                      const Teuchos::RCP<Albany::Layouts>& dl)
 : Gather2DFieldBase<PHAL::AlbanyTraits::Jacobian, Traits>(p,dl)
{
  this->setName("GatherExtruded2DField Jacobian");
}

template<typename Traits>
void GatherExtruded2DField<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  auto nodeID = workset.wsElNodeEqID;
  Teuchos::ArrayRCP<const ST> x_constView = Albany::getLocalData(workset.x);

  TEUCHOS_TEST_FOR_EXCEPTION (workset.disc->getLayeredMeshNumbering().is_null(),
    std::runtime_error, "Error! No layered numbering in the mesh.\n");

  const Albany::LayeredMeshNumbering<GO>& layeredMeshNumbering = *workset.disc->getLayeredMeshNumbering();
  const Albany::NodalDOFManager& solDOFManager = workset.disc->getOverlapDOFManager("ordinary_solution");
  const auto& indexer = *workset.disc->getOverlapGlobalLocalIndexer();

  int numLayers = layeredMeshNumbering.numLayers;
  this->fieldLevel = (this->fieldLevel < 0) ? numLayers : this->fieldLevel;
  const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> >& wsElNodeID  = workset.disc->getWsElNodeID()[workset.wsIndex];

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<GO>& elNodeID = wsElNodeID[cell];
    const int neq = nodeID.extent(2);

    for (std::size_t node = 0; node < this->numNodes; ++node) {
      int firstunk = neq * node + this->offset;
      const GO base_id = layeredMeshNumbering.getColumnId(elNodeID[node]);
      GO gnode = layeredMeshNumbering.getId(base_id, this->fieldLevel);
      GO gdof = solDOFManager.getGlobalDOF(gnode, this->offset);
      typename PHAL::Ref<ScalarT>::type val = (this->field2D)(cell,node);

      LO ldof = indexer.getLocalElement(gdof);
      val = FadType(val.size(), x_constView[ldof]);
      val.setUpdateValue(!workset.ignore_residual);
      val.fastAccessDx(firstunk) = workset.j_coeff;
    }
  }
}

template<typename Traits>
GatherExtruded2DField<PHAL::AlbanyTraits::Tangent, Traits>::
GatherExtruded2DField(const Teuchos::ParameterList& p,
                      const Teuchos::RCP<Albany::Layouts>& dl)
 : Gather2DFieldBase<PHAL::AlbanyTraits::Tangent, Traits>(p,dl)
{
  this->setName("GatherExtruded2DField Tangent");
}

template<typename Traits>
GatherExtruded2DField<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
GatherExtruded2DField(const Teuchos::ParameterList& p,
                      const Teuchos::RCP<Albany::Layouts>& dl)
 : Gather2DFieldBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p,dl)
{
  this->setName("GatherExtruded2DField DistParamDeriv");
}

template<typename Traits>
GatherExtruded2DField<PHAL::AlbanyTraits::HessianVec, Traits>::
GatherExtruded2DField(const Teuchos::ParameterList& p,
                      const Teuchos::RCP<Albany::Layouts>& dl)
 : Gather2DFieldBase<PHAL::AlbanyTraits::HessianVec, Traits>(p,dl)
{
  this->setName("GatherExtruded2DField HessianVec");
}

template<typename Traits>
void GatherExtruded2DField<PHAL::AlbanyTraits::HessianVec, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  auto nodeID = workset.wsElNodeEqID;
  Teuchos::ArrayRCP<const ST> x_constView = Albany::getLocalData(workset.x);
  Teuchos::RCP<const Thyra_MultiVector> direction_x = workset.hessianWorkset.direction_x;
  Teuchos::ArrayRCP<const ST> direction_x_constView;

  bool g_xx_is_active = !workset.hessianWorkset.hess_vec_prod_g_xx.is_null();
  bool g_xp_is_active = !workset.hessianWorkset.hess_vec_prod_g_xp.is_null();
  bool g_px_is_active = !workset.hessianWorkset.hess_vec_prod_g_px.is_null();
  bool f_xx_is_active = !workset.hessianWorkset.hess_vec_prod_f_xx.is_null();
  bool f_xp_is_active = !workset.hessianWorkset.hess_vec_prod_f_xp.is_null();
  bool f_px_is_active = !workset.hessianWorkset.hess_vec_prod_f_px.is_null();

  // is_x_active is true if we compute the Hessian-vector product contributions of either:
  // Hv_g_xx, Hv_g_xp, Hv_f_xx, or Hv_f_xp, i.e. if the first derivative is w.r.t. the solution.
  // If one of those is active, we have to initialize the first level of AD derivatives:
  // .fastAccessDx().val().
  const bool is_x_active = g_xx_is_active || g_xp_is_active || f_xx_is_active || f_xp_is_active;

  // is_x_direction_active is true if we compute the Hessian-vector product contributions of either:
  // Hv_g_xx, Hv_g_px, Hv_f_xx, or Hv_f_px, i.e. if the second derivative is w.r.t. the solution direction.
  // If one of those is active, we have to initialize the second level of AD derivatives:
  // .val().fastAccessDx().
  const bool is_x_direction_active = g_xx_is_active || g_px_is_active || f_xx_is_active || f_px_is_active;

  if(is_x_direction_active) {
    TEUCHOS_TEST_FOR_EXCEPTION(
        direction_x.is_null(),
        Teuchos::Exceptions::InvalidParameter,
        "\nError in GatherExtruded2DField<HessianVec, Traits>: "
        "direction_x is not set and hess_vec_prod_g_xx or"
        "hess_vec_prod_g_px is set.\n");
    direction_x_constView = Albany::getLocalData(direction_x->col(0));
  }

  const Albany::LayeredMeshNumbering<GO>& layeredMeshNumbering = *workset.disc->getLayeredMeshNumbering();
  const Albany::NodalDOFManager& solDOFManager = workset.disc->getOverlapDOFManager("ordinary_solution");

  int numLayers = layeredMeshNumbering.numLayers;
  this->fieldLevel = (this->fieldLevel < 0) ? numLayers : this->fieldLevel;
  const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> >& wsElNodeID  = workset.disc->getWsElNodeID()[workset.wsIndex];

  const auto& indexer = *workset.disc->getOverlapGlobalLocalIndexer();
  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<GO>& elNodeID = wsElNodeID[cell];
    const int neq = nodeID.extent(2);

    for (std::size_t node = 0; node < this->numNodes; ++node) {
      int firstunk = neq * node + this->offset;
      const GO base_id = layeredMeshNumbering.getColumnId(elNodeID[node]);
      GO gnode = layeredMeshNumbering.getId(base_id, this->fieldLevel);
      GO gdof = solDOFManager.getGlobalDOF(gnode, this->offset);
      LO ldof = indexer.getLocalElement(gdof);
      typename PHAL::Ref<ScalarT>::type val = (this->field2D)(cell,node);
      val = HessianVecFad(val.size(), x_constView[ldof]);
      // If we differentiate w.r.t. the solution, we have to set the first
      // derivative to workset.j_coeff
      if (is_x_active)
        val.fastAccessDx(firstunk).val() = workset.j_coeff;
      // If we differentiate w.r.t. the solution direction, we have to set
      // the second derivative to the related direction value
      if (is_x_direction_active)
        val.val().fastAccessDx(0) = direction_x_constView[ldof];
    }
  }
}

} // namespace LandIce
