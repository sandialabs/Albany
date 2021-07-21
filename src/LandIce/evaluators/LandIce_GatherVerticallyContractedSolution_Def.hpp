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

#include "LandIce_GatherVerticallyContractedSolution.hpp"

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN

namespace LandIce {

//**********************************************************************

template<typename EvalT, typename Traits>
GatherVerticallyContractedSolutionBase<EvalT, Traits>::
GatherVerticallyContractedSolutionBase(const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Albany::Layouts>& dl)
{
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());
  cell_topo = p.get<Teuchos::RCP<const CellTopologyData> >("Cell Topology");
  const auto& opType = p.get<std::string>("Contraction Operator");
  if(opType == "Vertical Sum")
    op = VerticalSum;
  else if (opType == "Vertical Average")
    op = VerticalAverage;
  else {
    TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error,
                                  "Error! \"" << opType << "\" is not a valid Contraction Operator. Valid Operators are: \"Vertical Sum\" and \"Vertical Average\"");
  }

  isVector =  p.get<bool>("Is Vector");

  offset = p.get<int>("Solution Offset");

  std::vector<PHX::DataLayout::size_type> dims;

  dl->node_vector->dimensions(dims);
  numNodes = dims[1];
  vecDim = isVector ? dims[2] : 1;

  meshPart = p.get<std::string>("Mesh Part");

  std::string sideSetName  = p.get<std::string> ("Side Set Name");
  TEUCHOS_TEST_FOR_EXCEPTION (dl->side_layouts.find(sideSetName)==dl->side_layouts.end(), std::runtime_error,
                              "Error! Layout for side set " << sideSetName << " not found.\n");
  Teuchos::RCP<Albany::Layouts> dl_side = dl->side_layouts.at(sideSetName);

  if(isVector)
    contractedSol = decltype(contractedSol)(p.get<std::string>("Contracted Solution Name"), dl_side->node_vector);
  else
    contractedSol = decltype(contractedSol)(p.get<std::string>("Contracted Solution Name"), dl_side->node_scalar);

  this->addEvaluatedField(contractedSol);

  this->setName("GatherVerticallyContractedSolution"+PHX::print<EvalT>());

  int numSides = cell_topo->side_count;  
  this->numSideNodes = Kokkos::View<int*, PHX::Device>("numSideNodes", numSides);
  for (int side=0; side<numSides; ++side) {
    this->numSideNodes(side) = cell_topo->side[side].topology->node_count;
  }

}

template<typename EvalT, typename Traits>
void GatherVerticallyContractedSolutionBase<EvalT, Traits>::
computeQuadWeights(const Albany::LayeredMeshNumbering<GO>& layeredMeshNumbering)
{
  if(this->op == this->VerticalSum){
    for (int i=0; i<this->numLayers+1; ++i)
      this->quadWeights(i) = 1.0;
  } else  { //Average
    const Teuchos::ArrayRCP<double>& layers_ratio = layeredMeshNumbering.layers_ratio;
    this->quadWeights(0) = 0.5*layers_ratio[0]; 
    this->quadWeights(this->numLayers) = 0.5*layers_ratio[this->numLayers-1];
    for(int i=1; i<this->numLayers; ++i)
      this->quadWeights(i) = 0.5*(layers_ratio[i-1] + layers_ratio[i]);
  }
}

//**********************************************************************

template<typename EvalT, typename Traits>
void GatherVerticallyContractedSolutionBase<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{

  this->numLayers = d.get_num_layers();
  this->quadWeights = Kokkos::View<double*, PHX::Device>("quadWeights", this->numLayers+1);

  this->utils.setFieldData(contractedSol,fm);
  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields(),false);

}

//**********************************************************************

template<typename Traits>
GatherVerticallyContractedSolution<PHAL::AlbanyTraits::Residual, Traits>::
GatherVerticallyContractedSolution(const Teuchos::ParameterList& p,
                                 const Teuchos::RCP<Albany::Layouts>& dl)
 : GatherVerticallyContractedSolutionBase<PHAL::AlbanyTraits::Residual, Traits>(p,dl)
{
  // Nothing to do here
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherVerticallyContractedSolution<PHAL::AlbanyTraits::Residual, Traits>::
operator() (const ResidualScalar_Tag& tag, const int& sideSet_idx) const {

  const int side = this->sideSet.side_local_id(sideSet_idx);
  const int numSideNodes = this->numSideNodes(side);

  //we only consider elements on the top.
  for (int node = 0; node < numSideNodes; ++node) {
    double contrSol[3] = {0.0, 0.0, 0.0};
    for(int il=0; il<this->numLayers+1; ++il) {
      for(unsigned int comp=0; comp<this->vecDim; ++comp)
        contrSol[comp] += this->x_constView_device(this->localDOFView(sideSet_idx, node, il, comp+this->offset))*this->quadWeights(il);
    }
    this->contractedSol(sideSet_idx,node) = contrSol[0];
  }

}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherVerticallyContractedSolution<PHAL::AlbanyTraits::Residual, Traits>::
operator() (const ResidualVector_Tag& tag, const int& sideSet_idx) const {

  const int side = this->sideSet.side_local_id(sideSet_idx);
  const int numSideNodes = this->numSideNodes(side);

  //we only consider elements on the top.
  for (int node = 0; node < numSideNodes; ++node) {
    double contrSol[3] = {0.0, 0.0, 0.0};
    for(int il=0; il<this->numLayers+1; ++il) {
      for(unsigned int comp=0; comp<this->vecDim; ++comp)
        contrSol[comp] += this->x_constView_device(this->localDOFView(sideSet_idx, node, il, comp+this->offset))*this->quadWeights(il);
    }
    for(unsigned int comp=0; comp<this->vecDim; ++comp) {
      this->contractedSol(sideSet_idx,node,comp) = contrSol[comp];
    }
  }

}

template<typename Traits>
void GatherVerticallyContractedSolution<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  this->x_constView = Albany::getLocalData(workset.x);
  this->x_constView_device = Albany::getDeviceData(workset.x);

  Kokkos::deep_copy(this->contractedSol.get_view(), ScalarT(0.0));

  TEUCHOS_TEST_FOR_EXCEPTION(workset.sideSetViews.is_null(), std::logic_error,
                             "Side sets defined in input file but not properly specified on the mesh.\n");

  // Get layered mesh numbering object
  const Albany::LayeredMeshNumbering<GO>& layeredMeshNumbering = *workset.disc->getLayeredMeshNumbering();

  // Compute quadWeights
  this->computeQuadWeights(layeredMeshNumbering);

  // Get sideSetViews and iterator
  const Albany::LocalSideSetInfoList& ssList = *(workset.sideSetViews);
  const std::map<std::string, Kokkos::View<LO****, PHX::Device>>& localDOFList = *(workset.localDOFViews);
  Albany::LocalSideSetInfoList::const_iterator it = ssList.find(this->meshPart);

  if (it != ssList.end()) {

    this->sideSet = it->second;

    // Each sideset has a localDOFView that can be accessed on device in a coalesced fashion
    this->localDOFView = localDOFList.at(it->first);

    if (this->isVector) {
      Kokkos::parallel_for(ResidualVector_Policy(0, this->sideSet.size), *this);
    } else {
      Kokkos::parallel_for(ResidualScalar_Policy(0, this->sideSet.size), *this);
    }
  }
}

template<typename Traits>
GatherVerticallyContractedSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
GatherVerticallyContractedSolution(const Teuchos::ParameterList& p,
                                 const Teuchos::RCP<Albany::Layouts>& dl)
 : GatherVerticallyContractedSolutionBase<PHAL::AlbanyTraits::Jacobian, Traits>(p,dl)
{
  // Nothing to do here
}

template<typename Traits>
void GatherVerticallyContractedSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::ArrayRCP<const ST> x_constView = Albany::getLocalData(workset.x);
  
  TEUCHOS_TEST_FOR_EXCEPTION(workset.sideSetViews.is_null(), std::logic_error,
                             "Side sets defined in input file but not properly specified on the mesh.\n");

  Kokkos::deep_copy(this->contractedSol.get_view(), ScalarT(0.0));

  // Get layered mesh numbering object
  const Albany::LayeredMeshNumbering<GO>& layeredMeshNumbering = *workset.disc->getLayeredMeshNumbering();

  // Compute quadWeights
  this->computeQuadWeights(layeredMeshNumbering);

  const Albany::LocalSideSetInfoList& ssList = *(workset.sideSetViews);
  Albany::LocalSideSetInfoList::const_iterator it = ssList.find(this->meshPart);

  if (it != ssList.end()) {
    const unsigned int  neq = workset.wsElNodeEqID.extent(2);

    const Albany::LocalSideSetInfo& sideSet = it->second;

    // Loop over the sides that form the boundary condition
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> >& wsElNodeID  = workset.disc->getWsElNodeID()[workset.wsIndex];
    const Albany::NodalDOFManager& solDOFManager = workset.disc->getOverlapDOFManager("ordinary_solution");
    const auto& ov_node_indexer = *workset.disc->getOverlapNodeGlobalLocalIndexer();

    for (int sideSet_idx = 0; sideSet_idx < sideSet.size; ++sideSet_idx) { // loop over the sides on this ws and name
      // Get the data that corresponds to the side
      const unsigned int elem_LID = sideSet.elem_LID(sideSet_idx);
      const unsigned int elem_side = sideSet.side_local_id(sideSet_idx);
      const CellTopologyData_Subcell& side =  this->cell_topo->side[elem_side];
      unsigned int numSideNodes = side.topology->node_count;

      const Teuchos::ArrayRCP<GO>& elNodeID = wsElNodeID[elem_LID];
      std::vector<double> velx(this->numNodes,0), vely(this->numNodes,0);

      GO baseId;
      for (unsigned int i = 0; i < numSideNodes; ++i) {
        const std::size_t node = side.node[i];
        baseId = layeredMeshNumbering.getColumnId(elNodeID[node]);
        std::vector<double> contrSol(this->vecDim,0);
        for(int il=0; il<this->numLayers+1; ++il) {
          const GO gnode = layeredMeshNumbering.getId(baseId, il);
          const LO inode = ov_node_indexer.getLocalElement(gnode);
          for(unsigned int comp=0; comp<this->vecDim; ++comp)
            contrSol[comp] += x_constView[solDOFManager.getLocalDOF(inode, comp+this->offset)]*this->quadWeights(il);
        }

        if(this->isVector) {
          for(unsigned int comp=0; comp<this->vecDim; ++comp) {
            this->contractedSol(sideSet_idx,i,comp) = FadType(this->contractedSol(sideSet_idx,i,comp).size(), contrSol[comp]);
            for(int il=0; il<this->numLayers+1; ++il)
              this->contractedSol(sideSet_idx,i,comp).fastAccessDx(neq*(this->numNodes+numSideNodes*il+i)+comp+this->offset) = this->quadWeights(il)*workset.j_coeff;
          }
        } else {
          this->contractedSol(sideSet_idx,i) = FadType(this->contractedSol(sideSet_idx,i).size(), contrSol[0]);
          for(int il=0; il<this->numLayers+1; ++il)
            this->contractedSol(sideSet_idx,i).fastAccessDx(neq*(this->numNodes+numSideNodes*il+i)+this->offset) = this->quadWeights(il)*workset.j_coeff;
        }
      }
    }
  }
}

template<typename Traits>
GatherVerticallyContractedSolution<PHAL::AlbanyTraits::Tangent, Traits>::
GatherVerticallyContractedSolution(const Teuchos::ParameterList& p,
                                 const Teuchos::RCP<Albany::Layouts>& dl)
 : GatherVerticallyContractedSolutionBase<PHAL::AlbanyTraits::Tangent, Traits>(p,dl)
{
  // Nothing to do here
}

template<typename Traits>
void GatherVerticallyContractedSolution<PHAL::AlbanyTraits::Tangent, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::ArrayRCP<const ST> x_constView = Albany::getLocalData(workset.x);

  Kokkos::deep_copy(this->contractedSol.get_view(), ScalarT(0.0));

  TEUCHOS_TEST_FOR_EXCEPTION(workset.sideSetViews.is_null(), std::logic_error,
                             "Side sets defined in input file but not properly specified on the mesh.\n");

  // Get layered mesh numbering object
  const Albany::LayeredMeshNumbering<GO>& layeredMeshNumbering = *workset.disc->getLayeredMeshNumbering();

  // Compute quadWeights
  this->computeQuadWeights(layeredMeshNumbering);

  const Albany::LocalSideSetInfoList& ssList = *(workset.sideSetViews);
  Albany::LocalSideSetInfoList::const_iterator it = ssList.find(this->meshPart);

  if (it != ssList.end()) {
    const Albany::LocalSideSetInfo& sideSet = it->second;

    // Loop over the sides that form the boundary condition
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> >& wsElNodeID  = workset.disc->getWsElNodeID()[workset.wsIndex];
    const Albany::NodalDOFManager& solDOFManager = workset.disc->getOverlapDOFManager("ordinary_solution");
    const auto& ov_node_indexer = *workset.disc->getOverlapNodeGlobalLocalIndexer();

    for (int sideSet_idx = 0; sideSet_idx < sideSet.size; ++sideSet_idx) { // loop over the sides on this ws and name
      // Get the data that corresponds to the side
      const unsigned int elem_LID = sideSet.elem_LID(sideSet_idx);
      const unsigned int elem_side = sideSet.side_local_id(sideSet_idx);
      const CellTopologyData_Subcell& side =  this->cell_topo->side[elem_side];
      unsigned int numSideNodes = side.topology->node_count;

      const Teuchos::ArrayRCP<GO>& elNodeID = wsElNodeID[elem_LID];

      //we only consider elements on the top.
      GO baseId;
      for (unsigned int i = 0; i < numSideNodes; ++i) {
        const std::size_t node = side.node[i];
        baseId = layeredMeshNumbering.getColumnId(elNodeID[node]);
        std::vector<double> contrSol(this->vecDim,0);
        for(int il=0; il<this->numLayers+1; ++il) {
          const GO gnode = layeredMeshNumbering.getId(baseId, il);
          const LO inode = ov_node_indexer.getLocalElement(gnode);
          for(unsigned int comp=0; comp<this->vecDim; ++comp)
            contrSol[comp] += x_constView[solDOFManager.getLocalDOF(inode, comp+this->offset)]*this->quadWeights(il);
        }
        if(this->isVector) {
          for(unsigned int comp=0; comp<this->vecDim; ++comp)
            this->contractedSol(sideSet_idx,i,comp) = contrSol[comp];
        } else {
          this->contractedSol(sideSet_idx,i) = contrSol[0];
        }
        if (workset.Vx != Teuchos::null && workset.j_coeff != 0.0) {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Not Implemented yet" << std::endl);
        }
      }
    }
  }
}

template<typename Traits>
GatherVerticallyContractedSolution<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
GatherVerticallyContractedSolution(const Teuchos::ParameterList& p,
                                 const Teuchos::RCP<Albany::Layouts>& dl)
 : GatherVerticallyContractedSolutionBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p,dl)
{
  // Nothing to do here
}

template<typename Traits>
void GatherVerticallyContractedSolution<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::ArrayRCP<const ST> x_constView = Albany::getLocalData(workset.x);

  Kokkos::deep_copy(this->contractedSol.get_view(), ScalarT(0.0));

  TEUCHOS_TEST_FOR_EXCEPTION(workset.sideSetViews.is_null(), std::logic_error,
                             "Side sets defined in input file but not properly specified on the mesh.\n");

  // Get layered mesh numbering object
  const Albany::LayeredMeshNumbering<GO>& layeredMeshNumbering = *workset.disc->getLayeredMeshNumbering();

  // Compute quadWeights
  this->computeQuadWeights(layeredMeshNumbering);

  const Albany::LocalSideSetInfoList& ssList = *(workset.sideSetViews);
  Albany::LocalSideSetInfoList::const_iterator it = ssList.find(this->meshPart);

  if (it != ssList.end()) {
    const Albany::LocalSideSetInfo& sideSet = it->second;

    // Loop over the sides that form the boundary condition
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> >& wsElNodeID  = workset.disc->getWsElNodeID()[workset.wsIndex];
    const Albany::NodalDOFManager& solDOFManager = workset.disc->getOverlapDOFManager("ordinary_solution");
    const auto& ov_node_indexer = *workset.disc->getOverlapNodeGlobalLocalIndexer();

    for (int sideSet_idx = 0; sideSet_idx < sideSet.size; ++sideSet_idx) { // loop over the sides on this ws and name
      // Get the data that corresponds to the side
      const unsigned int elem_LID = sideSet.elem_LID(sideSet_idx);
      const unsigned int elem_side = sideSet.side_local_id(sideSet_idx);
      const CellTopologyData_Subcell& side =  this->cell_topo->side[elem_side];
      unsigned int numSideNodes = side.topology->node_count;

      const Teuchos::ArrayRCP<GO>& elNodeID = wsElNodeID[elem_LID];

      //we only consider elements on the top.
      GO baseId;
      for (unsigned int i = 0; i < numSideNodes; ++i) {
        const std::size_t node = side.node[i];
        baseId = layeredMeshNumbering.getColumnId(elNodeID[node]);
        std::vector<double> contrSol(this->vecDim,0);
        for (int il=0; il<this->numLayers+1; ++il) {
          const GO gnode = layeredMeshNumbering.getId(baseId, il);
          const LO inode = ov_node_indexer.getLocalElement(gnode);
          for(unsigned int comp=0; comp<this->vecDim; ++comp)
            contrSol[comp] += x_constView[solDOFManager.getLocalDOF(inode, comp+this->offset)]*this->quadWeights(il);
        }

        if(this->isVector) {
          for(unsigned int comp=0; comp<this->vecDim; ++comp)
            this->contractedSol(sideSet_idx,i,comp) = contrSol[comp];
        } else {
          this->contractedSol(sideSet_idx,i) = contrSol[0];
        }
      }
    }
  }
}

template<typename Traits>
GatherVerticallyContractedSolution<PHAL::AlbanyTraits::HessianVec, Traits>::
GatherVerticallyContractedSolution(const Teuchos::ParameterList& p,
                                 const Teuchos::RCP<Albany::Layouts>& dl)
 : GatherVerticallyContractedSolutionBase<PHAL::AlbanyTraits::HessianVec, Traits>(p,dl)
{
  // Nothing to do here
}

template<typename Traits>
void GatherVerticallyContractedSolution<PHAL::AlbanyTraits::HessianVec, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::ArrayRCP<const ST> x_constView = Albany::getLocalData(workset.x);
  Teuchos::RCP<const Thyra_MultiVector> direction_x = workset.hessianWorkset.direction_x;
  Teuchos::ArrayRCP<const ST> direction_x_constView;

  int neq = workset.wsElNodeEqID.extent(2);

  bool g_xx_is_active = !workset.hessianWorkset.hess_vec_prod_g_xx.is_null();
  bool g_xp_is_active = !workset.hessianWorkset.hess_vec_prod_g_xp.is_null();
  bool g_px_is_active = !workset.hessianWorkset.hess_vec_prod_g_px.is_null();
  bool f_xx_is_active = !workset.hessianWorkset.hess_vec_prod_f_xx.is_null();
  //bool f_xp_is_active = !workset.hessianWorkset.hess_vec_prod_f_xp.is_null();
  bool f_px_is_active = !workset.hessianWorkset.hess_vec_prod_f_px.is_null();

  // is_x_active is true if we compute the Hessian-vector product contributions of either:
  // Hv_g_xx, Hv_g_xp, Hv_f_xx, or Hv_f_xp, i.e. if the first derivative is w.r.t. the solution.
  // If one of those is active, we have to initialize the first level of AD derivatives:
  // .fastAccessDx().val().
  //const bool is_x_active = g_xx_is_active || g_xp_is_active || f_xx_is_active || f_xp_is_active;

  // is_x_direction_active is true if we compute the Hessian-vector product contributions of either:
  // Hv_g_xx, Hv_g_px, Hv_f_xx, or Hv_f_px, i.e. if the second derivative is w.r.t. the solution direction.
  // If one of those is active, we have to initialize the second level of AD derivatives:
  // .val().fastAccessDx().
  const bool is_x_direction_active = g_xx_is_active || g_px_is_active || f_xx_is_active || f_px_is_active;

  if(is_x_direction_active) {
    TEUCHOS_TEST_FOR_EXCEPTION(
        direction_x.is_null(),
        Teuchos::Exceptions::InvalidParameter,
        "\nError in GatherSolution<HessianVec, Traits>: "
        "direction_x is not set and hess_vec_prod_g_xx or"
        "hess_vec_prod_g_px is set.\n");
    direction_x_constView = Albany::getLocalData(direction_x->col(0));
  }

  TEUCHOS_TEST_FOR_EXCEPTION(workset.sideSetViews.is_null(), std::logic_error,
                             "Side sets defined in input file but not properly specified on the mesh.\n");

  // Get layered mesh numbering object
  const Albany::LayeredMeshNumbering<GO>& layeredMeshNumbering = *workset.disc->getLayeredMeshNumbering();

  // Compute quadWeights
  this->computeQuadWeights(layeredMeshNumbering);

  Kokkos::deep_copy(this->contractedSol.get_view(), ScalarT(0.0));

  const Albany::LocalSideSetInfoList& ssList = *(workset.sideSetViews);
  Albany::LocalSideSetInfoList::const_iterator it = ssList.find(this->meshPart);

  if (it != ssList.end()) {
    const Albany::LocalSideSetInfo& sideSet = it->second;

    // Loop over the sides that form the boundary condition
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> >& wsElNodeID  = workset.disc->getWsElNodeID()[workset.wsIndex];
    const Albany::NodalDOFManager& solDOFManager = workset.disc->getOverlapDOFManager("ordinary_solution");
    const auto& ov_node_indexer = *workset.disc->getOverlapNodeGlobalLocalIndexer();

    for (int sideSet_idx = 0; sideSet_idx < sideSet.size; ++sideSet_idx) { // loop over the sides on this ws and name
      // Get the data that corresponds to the side
      const unsigned int elem_LID = sideSet.elem_LID(sideSet_idx);
      const unsigned int elem_side = sideSet.side_local_id(sideSet_idx);
      const CellTopologyData_Subcell& side =  this->cell_topo->side[elem_side];
      unsigned int numSideNodes = side.topology->node_count;

      const Teuchos::ArrayRCP<GO>& elNodeID = wsElNodeID[elem_LID];

      //we only consider elements on the top.
      GO baseId;
      for (unsigned int i = 0; i < numSideNodes; ++i) {
        std::size_t node = side.node[i];
        baseId = layeredMeshNumbering.getColumnId(elNodeID[node]);
        std::vector<double> contrSol(this->vecDim,0);
        for (int il=0; il<this->numLayers+1; ++il) {
          const GO gnode = layeredMeshNumbering.getId(baseId, il);
          const LO inode = ov_node_indexer.getLocalElement(gnode);
          for(unsigned int comp=0; comp<this->vecDim; ++comp)
            contrSol[comp] += x_constView[solDOFManager.getLocalDOF(inode, comp+this->offset)]*this->quadWeights(il);
        }
        std::vector<double> contrDirection(this->vecDim,0);

        if (g_xx_is_active||g_px_is_active)
          for (int il=0; il<this->numLayers+1; ++il) {
            const GO gnode = layeredMeshNumbering.getId(baseId, il);
            const LO inode = ov_node_indexer.getLocalElement(gnode);
            for(unsigned int comp=0; comp<this->vecDim; ++comp)
              contrDirection[comp] += direction_x_constView[solDOFManager.getLocalDOF(inode, comp+this->offset)]*this->quadWeights(il);
          }

        if(this->isVector) {
          for(unsigned int comp=0; comp<this->vecDim; ++comp) {
            this->contractedSol(sideSet_idx,i,comp) = HessianVecFad(this->contractedSol(sideSet_idx,i,comp).size(), contrSol[comp]);
            if (g_xx_is_active||g_px_is_active)
              this->contractedSol(sideSet_idx,i,comp).val().fastAccessDx(0) = contrDirection[comp];
            if (g_xx_is_active||g_xp_is_active)
              for(int il=0; il<this->numLayers+1; ++il)
                this->contractedSol(sideSet_idx,i,comp).fastAccessDx(neq*(this->numNodes+numSideNodes*il+i)+comp+this->offset).val() = this->quadWeights(il) * workset.j_coeff;
          }
        } else {
          this->contractedSol(sideSet_idx,i) = HessianVecFad(this->contractedSol(sideSet_idx,i).size(), contrSol[0]);
          if (g_xx_is_active||g_px_is_active)
            this->contractedSol(sideSet_idx,i).val().fastAccessDx(0) = contrDirection[0];
          if (g_xx_is_active||g_xp_is_active)
            for(int il=0; il<this->numLayers+1; ++il)
              this->contractedSol(sideSet_idx,i).fastAccessDx(neq*(this->numNodes+numSideNodes*il+i)+this->offset).val() = this->quadWeights(il) * workset.j_coeff;
        }
      }
    }
  }
}

} // namespace LandIce
