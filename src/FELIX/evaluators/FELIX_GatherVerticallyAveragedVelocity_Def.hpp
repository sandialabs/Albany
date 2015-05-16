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
GatherVerticallyAveragedVelocityBase<EvalT, Traits>::
GatherVerticallyAveragedVelocityBase(const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Albany::Layouts>& dl) :
  averagedVel(p.get<std::string>("Averaged Velocity Name"), dl->node_vector)
{
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

  this->addEvaluatedField(averagedVel);

  std::vector<PHX::DataLayout::size_type> dims;

  dl->node_gradient->dimensions(dims);
  numNodes = dims[1];
  vecDim = dims[2];
  vecDimFO = std::min(std::size_t(2), dims[2]); //vecDim (dims[2]) can be greater than 2 for coupled problems and = 1 for the problem in the xz plane

  this->setName("GatherVerticallyAveragedVelocity"+PHX::typeAsString<EvalT>());
}


//**********************************************************************
template<typename EvalT, typename Traits>
void GatherVerticallyAveragedVelocityBase<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
    this->utils.setFieldData(averagedVel,fm);
}

//**********************************************************************



template<typename Traits>
GatherVerticallyAveragedVelocity<PHAL::AlbanyTraits::Residual, Traits>::
GatherVerticallyAveragedVelocity(const Teuchos::ParameterList& p,
          const Teuchos::RCP<Albany::Layouts>& dl)
          : GatherVerticallyAveragedVelocityBase<PHAL::AlbanyTraits::Residual, Traits>(p,dl)
            {}

template<typename Traits>
void GatherVerticallyAveragedVelocity<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::RCP<const Tpetra_Vector> xT = workset.xT;
  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();

  Kokkos::deep_copy(this->averagedVel.get_kokkos_view(), ScalarT(0.0));

  if (workset.sideSets == Teuchos::null)
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Side sets defined in input file but not properly specified on the mesh" << std::endl);

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

    Teuchos::ArrayRCP<double> quadWeights(numLayers+1); //doing trapezoidal rule
    quadWeights[0] = 0.5*layers_ratio[0]; quadWeights[numLayers] = 0.5*layers_ratio[numLayers-1];
    for(int i=1; i<numLayers; ++i)
      quadWeights[i] = 0.5*(layers_ratio[i-1] + layers_ratio[i]);

    std::map<LO, std::size_t> baseIds;
    for (std::size_t side = 0; side < sideSet.size(); ++side) { // loop over the sides on this ws and name
      baseIds.clear();
      // Get the data that corresponds to the side
      const int elem_GID = sideSet[side].elem_GID;
      const int elem_LID = sideSet[side].elem_LID;
      const int elem_side = sideSet[side].side_local_id;
      const Teuchos::ArrayRCP<GO>& elNodeID = wsElNodeID[elem_LID];
      for (std::size_t node = 0; node < this->numNodes; ++node) {
        LO lnodeId = workset.disc->getOverlapNodeMapT()->getLocalElement(elNodeID[node]);
        LO base_id, ilayer;
        layeredMeshNumbering.getIndices(lnodeId, base_id, ilayer);
        if(ilayer == numLayers)
          baseIds[base_id] = node;
      }
      //we only consider elements on the top.
      std::map<LO, std::size_t>::const_iterator it = baseIds.begin();
      for (int i=0; it != baseIds.end(); ++it, ++i){
        std::vector<double> avVel(this->vecDimFO,0);
        for(int il=0; il<numLayers+1; ++il)
        {
          LO inode = layeredMeshNumbering.getId(it->first, il);
          for(int comp=0; comp<this->vecDimFO; ++comp)
            avVel[comp] += xT_constView[solDOFManager.getLocalDOF(inode, comp)]*quadWeights[il];
        }
        for(int comp=0; comp<this->vecDimFO; ++comp)
          this->averagedVel(elem_LID,it->second,comp) = avVel[comp];
      }
    }
  }
}


template<typename Traits>
GatherVerticallyAveragedVelocity<PHAL::AlbanyTraits::Jacobian, Traits>::
GatherVerticallyAveragedVelocity(const Teuchos::ParameterList& p,
          const Teuchos::RCP<Albany::Layouts>& dl)
          : GatherVerticallyAveragedVelocityBase<PHAL::AlbanyTraits::Jacobian, Traits>(p,dl)
            {}

template<typename Traits>
void GatherVerticallyAveragedVelocity<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::RCP<const Tpetra_Vector> xT = workset.xT;
  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();

  if (workset.sideSets == Teuchos::null)
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Side sets defined in input file but not properly specified on the mesh" << std::endl);


  const Albany::LayeredMeshNumbering<LO>& layeredMeshNumbering = *workset.disc->getLayeredMeshNumbering();
  int numLayers = layeredMeshNumbering.numLayers;

  Kokkos::deep_copy(this->averagedVel.get_kokkos_view(), ScalarT(0.0));
/*
  for (std::size_t cell=0; cell < workset.numCells; ++cell )
    for (std::size_t node = 0; node < this->numNodes; ++node)
      for(int comp=0; comp<this->vecDimFO; ++comp)
        this->averagedVel(cell,node,comp) = FadType(3*this->vecDim*(numLayers+1), 0.0);
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

    Teuchos::ArrayRCP<double> quadWeights(numLayers+1); //doing trapezoidal rule

    quadWeights[0] = 0.5*layers_ratio[0]; quadWeights[numLayers] = 0.5*layers_ratio[numLayers-1];
    for(int i=1; i<numLayers; ++i)
      quadWeights[i] = 0.5*(layers_ratio[i-1] + layers_ratio[i]);

    for (std::size_t side = 0; side < sideSet.size(); ++side) { // loop over the sides on this ws and name

      baseIds.clear();
      // Get the data that corresponds to the side
      const int elem_GID = sideSet[side].elem_GID;
      const int elem_LID = sideSet[side].elem_LID;
      const int elem_side = sideSet[side].side_local_id;
      const Teuchos::ArrayRCP<GO>& elNodeID = wsElNodeID[elem_LID];
      std::vector<double> velx(this->numNodes,0), vely(this->numNodes,0);
      for (std::size_t node = 0; node < this->numNodes; ++node) {
        LO lnodeId = workset.disc->getOverlapNodeMapT()->getLocalElement(elNodeID[node]);
        LO base_id, ilayer;
        layeredMeshNumbering.getIndices(lnodeId, base_id, ilayer);
        if(ilayer == numLayers)
          baseIds[base_id] = node;
      }
      std::map<LO, std::size_t>::const_iterator it = baseIds.begin();
      for (int i=0; it != baseIds.end(); ++it, ++i){
        std::vector<double> avVel(this->vecDimFO,0);
        for(int il=0; il<numLayers+1; ++il)
        {
          LO inode = layeredMeshNumbering.getId(it->first, il);
          for(int comp=0; comp<this->vecDimFO; ++comp)
            avVel[comp] += xT_constView[solDOFManager.getLocalDOF(inode, comp)]*quadWeights[il];
        }

        for(int comp=0; comp<this->vecDimFO; ++comp) {
          this->averagedVel(elem_LID,it->second,comp) = FadType(baseIds.size()*this->vecDim*(numLayers+1), avVel[comp]);
          for(int il=0; il<numLayers+1; ++il)
            this->averagedVel(elem_LID,it->second,comp).fastAccessDx(baseIds.size()*this->vecDim*il+this->vecDim*i+comp) = quadWeights[il]*workset.j_coeff;
        }
      }
    }
  }
}


template<typename Traits>
GatherVerticallyAveragedVelocity<PHAL::AlbanyTraits::Tangent, Traits>::
GatherVerticallyAveragedVelocity(const Teuchos::ParameterList& p,
          const Teuchos::RCP<Albany::Layouts>& dl)
          : GatherVerticallyAveragedVelocityBase<PHAL::AlbanyTraits::Tangent, Traits>(p,dl)
            {}

template<typename Traits>
void GatherVerticallyAveragedVelocity<PHAL::AlbanyTraits::Tangent, Traits>::
evaluateFields(typename Traits::EvalData workset)
{}


template<typename Traits>
GatherVerticallyAveragedVelocity<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
GatherVerticallyAveragedVelocity(const Teuchos::ParameterList& p,
          const Teuchos::RCP<Albany::Layouts>& dl)
          : GatherVerticallyAveragedVelocityBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p,dl)
            {}

template<typename Traits>
void GatherVerticallyAveragedVelocity<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData workset)
{}

}

