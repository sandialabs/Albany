//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Albany_Layouts.hpp"

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN

namespace FELIX {


 
//**********************************************************************
template<typename EvalT, typename Traits>
UpdateZCoordinate<EvalT, Traits>::
UpdateZCoordinate(const Teuchos::ParameterList& p,
            const Teuchos::RCP<Albany::Layouts>& dl) :
  coordVecIn (p.get<std::string> ("Old Coords Name"), dl->vertices_vector),
  coordVecOut(p.get<std::string> ("New Coords Name"), dl->vertices_vector),
  H0(p.get<std::string> ("Past Thickness Name"), dl->node_scalar),
  dH(p.get<std::string> ("Thickness Increment Name"), dl->node_scalar),
  elevation(p.get<std::string> ("Elevation Name"), dl->node_scalar)
{
  this->addEvaluatedField(coordVecOut);

  this->addDependentField(coordVecIn);
  this->addDependentField(H0);
  this->addDependentField(dH);
  this->addDependentField(elevation);

  std::vector<PHX::DataLayout::size_type> dims;
  dl->vertices_vector->dimensions(dims);
  numNodes = dims[1];
  numDims = dims[2];
  this->setName("Update Z Coordinate");
}

//**********************************************************************
template<typename EvalT, typename Traits>
void UpdateZCoordinate<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(coordVecIn,fm);
  this->utils.setFieldData(coordVecOut,fm);
  this->utils.setFieldData(H0, fm);
  this->utils.setFieldData(dH,fm);
  this->utils.setFieldData(elevation,fm);
}

//**********************************************************************
//Kokkos functors

//**********************************************************************
template<typename EvalT, typename Traits>
void UpdateZCoordinate<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::RCP<const Tpetra_Vector> xT = workset.xT;
  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();

  const Albany::LayeredMeshNumbering<LO>& layeredMeshNumbering = *workset.disc->getLayeredMeshNumbering();
  const Albany::NodalDOFManager& solDOFManager = workset.disc->getOverlapDOFManager("ordinary_solution");

  int numLayers = layeredMeshNumbering.numLayers;
  const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> >& wsElNodeID  = workset.disc->getWsElNodeID()[workset.wsIndex];
  const Teuchos::ArrayRCP<double>& layers_ratio = layeredMeshNumbering.layers_ratio;
  Teuchos::ArrayRCP<double> sigmaLevel(numLayers+1);
  sigmaLevel[0] = 0.; sigmaLevel[numLayers] = 1.;
  for(int i=1; i<numLayers; ++i)
    sigmaLevel[i] = sigmaLevel[i-1] + layers_ratio[i-1];

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<GO>& elNodeID = wsElNodeID[cell];
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];
    const int neq = nodeID[0].size();
    const std::size_t num_dof = neq * this->numNodes;

    for (std::size_t node = 0; node < this->numNodes; ++node) {
      LO lnodeId = workset.disc->getOverlapNodeMapT()->getLocalElement(elNodeID[node]);
      LO base_id, ilevel;
      layeredMeshNumbering.getIndices(lnodeId, base_id,  ilevel);
      for(std::size_t icomp=0; icomp< numDims; icomp++) {
        typename PHAL::Ref<MeshScalarT>::type val = coordVecOut(cell,node,icomp);
        val = (icomp==2) ?
            (H0(cell,node)+dH(cell,node)>1e-4) ? MeshScalarT(elevation(cell,node)- H0(cell,node) + sigmaLevel[ ilevel]*(H0(cell,node)+dH(cell,node)))
                                               : MeshScalarT(elevation(cell,node)- H0(cell,node) + sigmaLevel[ ilevel]*1e-4)
           : coordVecIn(cell,node,icomp);
      }
    }
  }

}

}
