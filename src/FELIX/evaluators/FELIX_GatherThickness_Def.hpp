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

  std::vector<PHX::DataLayout::size_type> dims;

  dl->node_gradient->dimensions(dims);
  numNodes = dims[1];
  vecDim = dims[2];
  vecDimFO = std::min(std::size_t(2), dims[2]); //this->vecDim (dims[2]) can be greater than 2 for coupled problems and = 1 for the problem in the xz plane

  this->setName("GatherThickness"+PHX::typeAsString<EvalT>());

  if (p.isType<int>("Offset of First DOF"))
    offset = p.get<int>("Offset of First DOF");
  else offset = 2;
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
            {}

template<typename Traits>
void GatherThickness<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::RCP<const Tpetra_Vector> xT = workset.xT;
  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();


  Kokkos::deep_copy(this->thickness.get_kokkos_view(), ScalarT(0.0));

  /*
  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    for (std::size_t node = 0; node < this->numNodes; ++node) {
      (this->thickness)(cell,node) = 0;
    }
  }
*/
  const Albany::SideSetList& ssList = *(workset.sideSets);
  Albany::SideSetList::const_iterator it = ssList.find("upperside");

  if (it != ssList.end()) {
    const std::vector<Albany::SideStruct>& sideSet = it->second;

    // Loop over the sides that form the boundary condition
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> >& wsElNodeID  = workset.disc->getWsElNodeID()[workset.wsIndex];
    const Albany::LayeredMeshNumbering<LO>& layeredMeshNumbering = *workset.disc->getLayeredMeshNumbering();
    const Albany::NodalDOFManager& solDOFManager = workset.disc->getOverlapDOFManager("ordinary_solution");

    const Teuchos::ArrayRCP<double>& layers_ratio = layeredMeshNumbering.layers_ratio;
    int numLayers = layeredMeshNumbering.numLayers;
    std::map<LO, std::size_t> baseIds;

    for (std::size_t side = 0; side < sideSet.size(); ++side) { // loop over the sides on this ws and name

      baseIds.clear();
      // Get the data that corresponds to the side
      const int elem_GID = sideSet[side].elem_GID;
      const int elem_LID = sideSet[side].elem_LID;
      const int elem_side = sideSet[side].side_local_id;

      const Teuchos::ArrayRCP<GO>& elNodeID = wsElNodeID[elem_LID];
      const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[elem_LID];
      for (std::size_t node = 0; node < this->numNodes; ++node) {
        LO lnodeId = workset.disc->getOverlapNodeMapT()->getLocalElement(elNodeID[node]);
        LO base_id, ilayer;
        layeredMeshNumbering.getIndices(lnodeId, base_id, ilayer);
        if(ilayer == numLayers)
          baseIds[base_id] = node;
      }
      std::map<LO, std::size_t>::const_iterator it = baseIds.begin();
      if(baseIds.size()==3) {
        for (int i=0; it != baseIds.end(); ++it, ++i){
          const Teuchos::ArrayRCP<int>& eqID  = nodeID[it->second];
          this->thickness(elem_LID,it->second) = xT_constView[eqID[this->offset]];
        }
      }
    }
  }
}


template<typename Traits>
GatherThickness<PHAL::AlbanyTraits::Jacobian, Traits>::
GatherThickness(const Teuchos::ParameterList& p,
          const Teuchos::RCP<Albany::Layouts>& dl)
          : GatherThicknessBase<PHAL::AlbanyTraits::Jacobian, Traits>(p,dl)
            {}

template<typename Traits>
void GatherThickness<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::RCP<const Tpetra_Vector> xT = workset.xT;
  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();

  if (workset.sideSets == Teuchos::null)
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Side sets defined in input file but not properly specified on the mesh" << std::endl);


  const Albany::LayeredMeshNumbering<LO>& layeredMeshNumbering = *workset.disc->getLayeredMeshNumbering();
  int numLayers = layeredMeshNumbering.numLayers;

  Kokkos::deep_copy(this->thickness.get_kokkos_view(), ScalarT(0.0));

  /*
  for (std::size_t cell=0; cell < workset.numCells; ++cell )
    for (std::size_t node = 0; node < this->numNodes; ++node)
      this->thickness(cell,node) = FadType(3*this->vecDim*(numLayers+1), 0.0);
*/

  const Albany::SideSetList& ssList = *(workset.sideSets);
  Albany::SideSetList::const_iterator it = ssList.find("upperside");


  if (it != ssList.end()) {
    const std::vector<Albany::SideStruct>& sideSet = it->second;

    // Loop over the sides that form the boundary condition
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> >& wsElNodeID  = workset.disc->getWsElNodeID()[workset.wsIndex];
    const Albany::NodalDOFManager& solDOFManager = workset.disc->getOverlapDOFManager("ordinary_solution");

    const Teuchos::ArrayRCP<double>& layers_ratio = layeredMeshNumbering.layers_ratio;

    std::map<LO, std::size_t> baseIds;

    for (std::size_t side = 0; side < sideSet.size(); ++side) { // loop over the sides on this ws and name

      baseIds.clear();
      // Get the data that corresponds to the side
      const int elem_GID = sideSet[side].elem_GID;
      const int elem_LID = sideSet[side].elem_LID;
      const int elem_side = sideSet[side].side_local_id;

      const Teuchos::ArrayRCP<GO>& elNodeID = wsElNodeID[elem_LID];
      const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[elem_LID];
      std::vector<double> H(this->numNodes,0);
      for (std::size_t node = 0; node < this->numNodes; ++node) {
        LO lnodeId = workset.disc->getOverlapNodeMapT()->getLocalElement(elNodeID[node]);
        LO base_id, ilayer;
        layeredMeshNumbering.getIndices(lnodeId, base_id, ilayer);
        if(ilayer == numLayers)
          baseIds[base_id] = node;
      }
      std::map<LO, std::size_t>::const_iterator it = baseIds.begin();
      if(baseIds.size()==3) {
        for (int i=0; it != baseIds.end(); ++it, ++i){
          const Teuchos::ArrayRCP<int>& eqID  = nodeID[it->second];
          this->thickness(elem_LID,it->second) = FadType(baseIds.size()*this->vecDim*(numLayers+1), xT_constView[eqID[this->offset]]);
     //     std::cout << "("<<(xT_constView[eqID[0]]) << ","<< (xT_constView[eqID[1]]) << "," <<   (xT_constView[eqID[2]]) <<") ";
          this->thickness(elem_LID,it->second).fastAccessDx(baseIds.size()*this->vecDim*numLayers+this->vecDim*i+this->offset) = workset.j_coeff;
        }
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

}

