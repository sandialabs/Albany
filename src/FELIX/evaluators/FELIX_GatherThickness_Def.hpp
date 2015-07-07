//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"
#include "Sacado.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

//uncomment the following line if you want debug output to be printed to screen
#define OUTPUT_TO_SCREEN

namespace FELIX {

//**********************************************************************

template<typename EvalT, typename Traits>
GatherThicknessBase<EvalT, Traits>::
GatherThicknessBase(const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Albany::Layouts>& dl) :
  thickness(p.get<std::string>("Thickness Name"), dl->node_scalar)
{
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

  this->addEvaluatedField(thickness);
  cell_topo = p.get<Teuchos::RCP<const CellTopologyData> >("Cell Topology");

  std::vector<PHX::DataLayout::size_type> dims;

  dl->node_gradient->dimensions(dims);
  numNodes = dims[1];
  vecDim = dims[2];
  vecDimFO = std::min(PHX::DataLayout::size_type(2), dims[2]); //this->vecDim (dims[2]) can be greater than 2 for coupled problems and = 1 for the problem in the xz plane

  this->setName("GatherThickness"+PHX::typeAsString<EvalT>());

  if (p.isType<int>("Offset of First DOF"))
    offset = p.get<int>("Offset of First DOF");
  else offset = 2;

  if (p.isType<int>("H level"))
    HLevel = p.get<int>("H level");
  else HLevel = -1;
}


//**********************************************************************
template<typename EvalT, typename Traits>
void GatherThicknessBase<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
    this->utils.setFieldData(thickness,fm);
}

//**********************************************************************



template<typename Traits>
GatherThickness<PHAL::AlbanyTraits::Residual, Traits>::
GatherThickness(const Teuchos::ParameterList& p,
          const Teuchos::RCP<Albany::Layouts>& dl)
          : GatherThicknessBase<PHAL::AlbanyTraits::Residual, Traits>(p,dl)
{
  if (p.isType<const std::string>("Mesh Part"))
    this->meshPart = p.get<const std::string>("Mesh Part");
  else
    this->meshPart = "upperside";
}

template<typename Traits>
void GatherThickness<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::RCP<const Tpetra_Vector> xT = workset.xT;
  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();


  Kokkos::deep_copy(this->thickness.get_kokkos_view(), ScalarT(0.0));

  const Albany::SideSetList& ssList = *(workset.sideSets);
  Albany::SideSetList::const_iterator it = ssList.find(this->meshPart);

  if (it != ssList.end()) {
    const std::vector<Albany::SideStruct>& sideSet = it->second;

    const Albany::NodalDOFManager& solDOFManager = workset.disc->getOverlapDOFManager("ordinary_solution");

    for (std::size_t iSide = 0; iSide < sideSet.size(); ++iSide) { // loop over the sides on this ws and name

      // Get the data that corresponds to the side
      const int elem_GID = sideSet[iSide].elem_GID;
      const int elem_LID = sideSet[iSide].elem_LID;
      const int elem_side = sideSet[iSide].side_local_id;
      const CellTopologyData_Subcell& side =  this->cell_topo->side[elem_side];
      int numSideNodes = side.topology->node_count;
      const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[elem_LID];
      for (int i = 0; i < numSideNodes; ++i){
        std::size_t node = side.node[i];
        const Teuchos::ArrayRCP<int>& eqID  = nodeID[node];
        this->thickness(elem_LID,node) = xT_constView[eqID[this->offset]];
      }
    }
  }
}


template<typename Traits>
GatherThickness<PHAL::AlbanyTraits::Jacobian, Traits>::
GatherThickness(const Teuchos::ParameterList& p,
          const Teuchos::RCP<Albany::Layouts>& dl)
          : GatherThicknessBase<PHAL::AlbanyTraits::Jacobian, Traits>(p,dl)
{
  if (p.isType<const std::string>("Mesh Part"))
    this->meshPart = p.get<const std::string>("Mesh Part");
  else
    this->meshPart = "upperside";
}

template<typename Traits>
void GatherThickness<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::RCP<const Tpetra_Vector> xT = workset.xT;
  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();

  if (workset.sideSets == Teuchos::null)
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Side sets defined in input file but not properly specified on the mesh" << std::endl);


  const Albany::LayeredMeshNumbering<LO>& layeredMeshNumbering = *workset.disc->getLayeredMeshNumbering();
  int numLayers = workset.disc->getLayeredMeshNumbering()->numLayers;
  this->HLevel = (this->HLevel < 0) ? numLayers : this->HLevel;

  Kokkos::deep_copy(this->thickness.get_kokkos_view(), ScalarT(0.0));

  const Albany::SideSetList& ssList = *(workset.sideSets);
  Albany::SideSetList::const_iterator it = ssList.find(this->meshPart);


  if (it != ssList.end()) {
    const std::vector<Albany::SideStruct>& sideSet = it->second;

    // Loop over the sides that form the boundary condition
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> >& wsElNodeID  = workset.disc->getWsElNodeID()[workset.wsIndex];
    const Albany::NodalDOFManager& solDOFManager = workset.disc->getOverlapDOFManager("ordinary_solution");

     for (std::size_t iSide = 0; iSide < sideSet.size(); ++iSide) { // loop over the sides on this ws and name

      // Get the data that corresponds to the side
      const int elem_GID = sideSet[iSide].elem_GID;
      const int elem_LID = sideSet[iSide].elem_LID;
      const int elem_side = sideSet[iSide].side_local_id;
      const CellTopologyData_Subcell& side =  this->cell_topo->side[elem_side];
      int numSideNodes = side.topology->node_count;

      const Teuchos::ArrayRCP<GO>& elNodeID = wsElNodeID[elem_LID];
      const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[elem_LID];

      for (int i = 0; i < numSideNodes; ++i){
        std::size_t node = side.node[i];
        const Teuchos::ArrayRCP<int>& eqID  = nodeID[node];
        this->thickness(elem_LID,node) = FadType(numSideNodes*this->vecDim*(numLayers+1), xT_constView[eqID[this->offset]]);
        this->thickness(elem_LID,node).fastAccessDx(numSideNodes*this->vecDim*this->HLevel+this->vecDim*i+this->offset) = workset.j_coeff;
      }
    }
  }
}

template<typename Traits>
GatherThickness<PHAL::AlbanyTraits::Tangent, Traits>::
GatherThickness(const Teuchos::ParameterList& p,
          const Teuchos::RCP<Albany::Layouts>& dl)
          : GatherThicknessBase<PHAL::AlbanyTraits::Tangent, Traits>(p,dl)
            {}

template<typename Traits>
void GatherThickness<PHAL::AlbanyTraits::Tangent, Traits>::
evaluateFields(typename Traits::EvalData workset)
{}

template<typename Traits>
GatherThickness<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
GatherThickness(const Teuchos::ParameterList& p,
          const Teuchos::RCP<Albany::Layouts>& dl)
          : GatherThicknessBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p,dl)
            {}

template<typename Traits>
void GatherThickness<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData workset)
{}


//********************************


template<typename Traits>
GatherThickness3D<PHAL::AlbanyTraits::Residual, Traits>::
GatherThickness3D(const Teuchos::ParameterList& p,
          const Teuchos::RCP<Albany::Layouts>& dl)
          : GatherThicknessBase<PHAL::AlbanyTraits::Residual, Traits>(p,dl)
            {}

template<typename Traits>
void GatherThickness3D<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::RCP<const Tpetra_Vector> xT = workset.xT;
  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();

  Kokkos::deep_copy(this->thickness.get_kokkos_view(), ScalarT(0.0));

  const Albany::LayeredMeshNumbering<LO>& layeredMeshNumbering = *workset.disc->getLayeredMeshNumbering();
  const Albany::NodalDOFManager& solDOFManager = workset.disc->getOverlapDOFManager("ordinary_solution");

  int numLayers = layeredMeshNumbering.numLayers;
  this->HLevel = (this->HLevel < 0) ? numLayers : this->HLevel;
  const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> >& wsElNodeID  = workset.disc->getWsElNodeID()[workset.wsIndex];

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<GO>& elNodeID = wsElNodeID[cell];

    for (std::size_t node = 0; node < this->numNodes; ++node) {
      LO lnodeId = workset.disc->getOverlapNodeMapT()->getLocalElement(elNodeID[node]);
      LO base_id, ilayer;
      layeredMeshNumbering.getIndices(lnodeId, base_id, ilayer);
      LO inode = layeredMeshNumbering.getId(base_id, this->HLevel);

      (this->thickness)(cell,node) = xT_constView[solDOFManager.getLocalDOF(inode, this->offset)];
    }
  }
}


template<typename Traits>
GatherThickness3D<PHAL::AlbanyTraits::Jacobian, Traits>::
GatherThickness3D(const Teuchos::ParameterList& p,
          const Teuchos::RCP<Albany::Layouts>& dl)
          : GatherThicknessBase<PHAL::AlbanyTraits::Jacobian, Traits>(p,dl)
            {}

template<typename Traits>
void GatherThickness3D<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::RCP<const Tpetra_Vector> xT = workset.xT;
  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();

  Kokkos::deep_copy(this->thickness.get_kokkos_view(), ScalarT(0.0));

  const Albany::LayeredMeshNumbering<LO>& layeredMeshNumbering = *workset.disc->getLayeredMeshNumbering();
  const Albany::NodalDOFManager& solDOFManager = workset.disc->getOverlapDOFManager("ordinary_solution");

  int numLayers = layeredMeshNumbering.numLayers;
  this->HLevel = (this->HLevel < 0) ? numLayers : this->HLevel;
  const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> >& wsElNodeID  = workset.disc->getWsElNodeID()[workset.wsIndex];


  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<GO>& elNodeID = wsElNodeID[cell];
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];
    const int neq = nodeID[0].size();
    const std::size_t num_dof = neq * this->numNodes;

    for (std::size_t node = 0; node < this->numNodes; ++node) {
      int firstunk = node;
      LO lnodeId = workset.disc->getOverlapNodeMapT()->getLocalElement(elNodeID[node]);
      LO base_id, ilayer;
      layeredMeshNumbering.getIndices(lnodeId, base_id, ilayer);
      LO inode = layeredMeshNumbering.getId(base_id, this->HLevel);
      typename PHAL::Ref<ScalarT>::type val = (this->thickness)(cell,node);

      val = FadType(this->numNodes, xT_constView[solDOFManager.getLocalDOF(inode, this->offset)]);
      val.setUpdateValue(!workset.ignore_residual);
      val.fastAccessDx(firstunk) = workset.j_coeff;
    }
  }
}


template<typename Traits>
GatherThickness3D<PHAL::AlbanyTraits::Tangent, Traits>::
GatherThickness3D(const Teuchos::ParameterList& p,
          const Teuchos::RCP<Albany::Layouts>& dl)
          : GatherThicknessBase<PHAL::AlbanyTraits::Tangent, Traits>(p,dl)
            {}

template<typename Traits>
void GatherThickness3D<PHAL::AlbanyTraits::Tangent, Traits>::
evaluateFields(typename Traits::EvalData workset)
{}

template<typename Traits>
GatherThickness3D<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
GatherThickness3D(const Teuchos::ParameterList& p,
          const Teuchos::RCP<Albany::Layouts>& dl)
          : GatherThicknessBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p,dl)
            {}

template<typename Traits>
void GatherThickness3D<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData workset)
{}


}

