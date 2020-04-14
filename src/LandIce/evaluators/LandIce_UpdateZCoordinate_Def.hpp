//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Albany_Layouts.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_GlobalLocalIndexer.hpp"
#include "Albany_AbstractDiscretization.hpp"
#include "Albany_NodalDOFManager.hpp"
#include "PHAL_AlbanyTraits.hpp"

#include "LandIce_UpdateZCoordinate.hpp"

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN

namespace LandIce {

//**********************************************************************
template<typename EvalT, typename Traits>
UpdateZCoordinateMovingTop<EvalT, Traits>::
UpdateZCoordinateMovingTop (const Teuchos::ParameterList& p,
                            const Teuchos::RCP<Albany::Layouts>& dl) :
  coordVecIn (p.get<std::string> ("Old Coords Name"), dl->vertices_vector),
  coordVecOut(p.get<std::string> ("New Coords Name"), dl->vertices_vector),
  topSurface(p.get<std::string>("Top Surface Name"), dl->node_scalar),
  bedTopo(p.get<std::string> ("Bed Topography Name"), dl->node_scalar)
{
  this->addDependentField(coordVecIn);
  this->addDependentField(bedTopo);
  if(p.isParameter("Thickness Name")) {
    H = decltype(H)(p.get<std::string> ("Thickness Name"), dl->node_scalar);
    this->addDependentField(H);
    haveThickness = true;
  }
  else {
    H0 = decltype(H0)(p.get<std::string> ("Past Thickness Name"), dl->node_scalar);
    dH = decltype(dH)(p.get<std::string> ("Thickness Increment Name"), dl->node_scalar);
    this->addDependentField(H0);
    this->addDependentField(dH);
    haveThickness = false;
  }

  this->addEvaluatedField(coordVecOut);
  this->addEvaluatedField(topSurface);

  minH = p.isParameter("Minimum Thickness") ? p.get<double>("Minimum Thickness") : 1e-5;
  std::vector<PHX::DataLayout::size_type> dims;
  dl->vertices_vector->dimensions(dims);
  numNodes = dims[1];
  numDims = dims[2];
  this->setName("Update Z Coordinate Moving Top");

  Teuchos::ParameterList* p_list = p.get<Teuchos::ParameterList*>("Physical Parameter List");
  rho_i = p_list->get<double>("Ice Density");
  rho_w = p_list->get<double>("Water Density");
}

//**********************************************************************
template<typename EvalT, typename Traits>
void UpdateZCoordinateMovingTop<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  auto nodeID = workset.wsElNodeEqID;

  const Albany::LayeredMeshNumbering<LO>& layeredMeshNumbering = *workset.disc->getLayeredMeshNumbering();
  // const Albany::NodalDOFManager& solDOFManager = workset.disc->getOverlapDOFManager("ordinary_solution");

  int numLayers = layeredMeshNumbering.numLayers;
  const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> >& wsElNodeID  = workset.disc->getWsElNodeID()[workset.wsIndex];
  const Teuchos::ArrayRCP<double>& layers_ratio = layeredMeshNumbering.layers_ratio;
  Teuchos::ArrayRCP<double> sigmaLevel(numLayers+1);
  sigmaLevel[0] = 0.; sigmaLevel[numLayers] = 1.;
  for(int i=1; i<numLayers; ++i) {
    sigmaLevel[i] = sigmaLevel[i-1] + layers_ratio[i-1];
  }

  auto ov_node_indexer = Albany::createGlobalLocalIndexer(workset.disc->getOverlapNodeVectorSpace());
  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<GO>& elNodeID = wsElNodeID[cell];
    // const int neq = nodeID.extent(2);
    // const std::size_t num_dof = neq * this->numNodes;

    for (std::size_t node = 0; node < this->numNodes; ++node) {
      const LO lnodeId = ov_node_indexer->getLocalElement(elNodeID[node]);
      LO base_id, ilevel;
      layeredMeshNumbering.getIndices(lnodeId, base_id,  ilevel);
      MeshScalarT h;
      if(haveThickness) {
        h = std::max(H(cell,node), MeshScalarT(minH));
      } else {
        h = std::max(H0(cell,node) + dH(cell,node), MeshScalarT(minH));
      }
      MeshScalarT bed = bedTopo(cell,node);
      auto floating = (rho_i*h + rho_w*bed) < 0.0;// && (h+bed > 0.0);

      MeshScalarT lowSurf = floating ? -h*rho_i/rho_w : bed;
      typename PHAL::Ref<MeshScalarT>::type vals = topSurface(cell,node);
      vals = lowSurf+h;

      for(std::size_t icomp=0; icomp< numDims; icomp++) {
        typename PHAL::Ref<MeshScalarT>::type val = coordVecOut(cell,node,icomp);
        val = (icomp==2) ? MeshScalarT(lowSurf + sigmaLevel[ ilevel]*h)
                         : coordVecIn(cell,node,icomp);
      }
    }
  }
}

//**********************************************************************
template<typename EvalT, typename Traits>
UpdateZCoordinateMovingBed<EvalT, Traits>::
UpdateZCoordinateMovingBed (const Teuchos::ParameterList& p,
                            const Teuchos::RCP<Albany::Layouts>& dl) :
  coordVecIn (p.get<std::string> ("Old Coords Name"), dl->vertices_vector),
  coordVecOut(p.get<std::string> ("New Coords Name"), dl->vertices_vector),
  H(p.get<std::string> ("Thickness Name"), dl->node_scalar),
  bedTopo(p.get<std::string> ("Bed Topography Name"), dl->node_scalar),
  bedTopoOut(p.get<std::string> ("Updated Bed Topography Name"), dl->node_scalar),
  topSurface(p.get<std::string> ("Top Surface Name"), dl->node_scalar),
  topSurfaceOut(p.get<std::string> ("Updated Top Surface Name"), dl->node_scalar)
{
  this->addEvaluatedField(coordVecOut);

  this->addDependentField(coordVecIn);

  this->addDependentField(H);
  this->addDependentField(bedTopo);
  this->addDependentField(topSurface);
  this->addEvaluatedField(topSurfaceOut);
  this->addEvaluatedField(bedTopoOut);

  minH = p.isParameter("Minimum Thickness") ? p.get<double>("Minimum Thickness") : 1e-4;
  std::vector<PHX::DataLayout::size_type> dims;
  dl->vertices_vector->dimensions(dims);
  numNodes = dims[1];
  numDims = dims[2];

  Teuchos::ParameterList* p_list = p.get<Teuchos::ParameterList*>("Physical Parameter List");
  rho_i = p_list->get<double>("Ice Density");
  rho_w = p_list->get<double>("Water Density");

  this->setName("Update Z Coordinate Moving Bed");
}

//**********************************************************************
//Kokkos functors

//**********************************************************************
template<typename EvalT, typename Traits>
void UpdateZCoordinateMovingBed<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  auto nodeID = workset.wsElNodeEqID;

  const Albany::LayeredMeshNumbering<LO>& layeredMeshNumbering = *workset.disc->getLayeredMeshNumbering();
  // const Albany::NodalDOFManager& solDOFManager = workset.disc->getOverlapDOFManager("ordinary_solution");

  int numLayers = layeredMeshNumbering.numLayers;
  const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> >& wsElNodeID  = workset.disc->getWsElNodeID()[workset.wsIndex];
  const Teuchos::ArrayRCP<double>& layers_ratio = layeredMeshNumbering.layers_ratio;
  Teuchos::ArrayRCP<double> sigmaLevel(numLayers+1);
  sigmaLevel[0] = 0.; sigmaLevel[numLayers] = 1.;
  for(int i=1; i<numLayers; ++i)
    sigmaLevel[i] = sigmaLevel[i-1] + layers_ratio[i-1];

  auto ov_node_indexer = Albany::createGlobalLocalIndexer(workset.disc->getOverlapNodeVectorSpace());
  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<GO>& elNodeID = wsElNodeID[cell];
    // const int neq = nodeID.extent(2); 
    // const std::size_t num_dof = neq * this->numNodes;

    for (std::size_t node = 0; node < this->numNodes; ++node) {
      const LO lnodeId = ov_node_indexer->getLocalElement(elNodeID[node]);
      LO base_id, ilevel;
      layeredMeshNumbering.getIndices(lnodeId, base_id,  ilevel);
     MeshScalarT h = H(cell,node);
      MeshScalarT top = topSurface(cell,node);
      typename PHAL::Ref<MeshScalarT>::type vals = topSurfaceOut(cell,node);
      typename PHAL::Ref<MeshScalarT>::type valb = bedTopoOut(cell,node);
      MeshScalarT bed = bedTopo(cell,node);

      //floating when the floating condition is met with the old bed
      // or with the new (top-h) bed.
      auto floating = (rho_i*h + rho_w*std::min(bed, MeshScalarT(top-h))) < 0.0;

      top = floating ? h*(1.0 - rho_i/rho_w) : top; //adjust surface when floating
      vals = top;
      bed = floating ? std::min(bed, MeshScalarT(-rho_i/rho_w*h)) : top - h;
      valb = bed;

      for(std::size_t icomp=0; icomp< numDims; icomp++) {
        typename PHAL::Ref<MeshScalarT>::type val = coordVecOut(cell,node,icomp);
        val = (icomp==2) ?
            MeshScalarT(top - (1.0- sigmaLevel[ ilevel])*h)
           : coordVecIn(cell,node,icomp);
      }
    }
  }
}

}  // namespace LandIce
