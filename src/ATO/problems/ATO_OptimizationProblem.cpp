//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "ATO_OptimizationProblem.hpp"
#include "Albany_AbstractDiscretization.hpp"
#include "Epetra_Export.h"
#include "Adapt_NodalDataBlock.hpp"
#include "ATO_TopoTools.hpp"

/******************************************************************************/
ATO::OptimizationProblem::
OptimizationProblem( const Teuchos::RCP<Teuchos::ParameterList>& _params,
                     const Teuchos::RCP<ParamLib>& _paramLib,
                     const int _numDim) :
Albany::AbstractProblem(_params, _paramLib, _numDim) {}
/******************************************************************************/


/******************************************************************************/
void
ATO::OptimizationProblem::
ComputeVolume(double& v)
/******************************************************************************/
{
  double localv = 0.0;

  const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > > >::type&
    wsElNodeEqID = disc->getWsElNodeEqID();
  const Albany::WorksetArray<int>::type& wsPhysIndex = disc->getWsPhysIndex();


  int numWorksets = wsElNodeEqID.size();

  for(int ws=0; ws<numWorksets; ws++){

    int physIndex = wsPhysIndex[ws];

    int numCells = wsElNodeEqID[ws].size();
    int numQPs = cubatures[physIndex]->getNumPoints();
    
    for(int cell=0; cell<numCells; cell++)
      for(int qp=0; qp<numQPs; qp++)
        localv += weighted_measure[ws](cell,qp);
  }

  comm->SumAll(&localv, &v, 1);
}
/******************************************************************************/
void
ATO::OptimizationProblem::
ComputeVolume(const double* p, double& v, double* dvdp)
/******************************************************************************/
{
  double localv = 0.0;

  const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > >::type&
    wsElNodeID = disc->getWsElNodeID();
  const Albany::WorksetArray<int>::type& wsPhysIndex = disc->getWsPhysIndex();

  int numWorksets = wsElNodeID.size();


  if( topology->getCentering() == "Element" ){
    int wsOffset = 0;
    for(int ws=0; ws<numWorksets; ws++){
  
      int physIndex = wsPhysIndex[ws];
  
      int numCells = weighted_measure[ws].dimension(0);
      int numQPs   = weighted_measure[ws].dimension(1);
      
      for(int cell=0; cell<numCells; cell++){
        double elVol = 0.0;
        for(int qp=0; qp<numQPs; qp++)
          elVol += weighted_measure[ws](cell,qp);
        localv += elVol*p[wsOffset+cell];
      }
      wsOffset += numCells;
    }
    comm->SumAll(&localv, &v, 1);

    if( dvdp != NULL ){
      int wsOffset = 0;
      for(int ws=0; ws<numWorksets; ws++){
    
        int physIndex = wsPhysIndex[ws];
    
        int numCells = weighted_measure[ws].dimension(0);
        int numQPs   = weighted_measure[ws].dimension(1);
        
        for(int cell=0; cell<numCells; cell++){
          double elVol = 0.0;
          for(int qp=0; qp<numQPs; qp++)
            elVol += weighted_measure[ws](cell,qp);
          dvdp[wsOffset+cell] = elVol;
        }
        wsOffset += numCells;
      }
    }
  } else 
  if( topology->getCentering() == "Node" ){
    Teuchos::RCP<const Epetra_BlockMap>
      overlapNodeMap = stateMgr->getNodalDataBlock()->getOverlapMap();
    for(int ws=0; ws<numWorksets; ws++){
  
      int physIndex = wsPhysIndex[ws];
      int numNodes  = basisAtQPs[physIndex].dimension(0);
      int numCells  = weighted_measure[ws].dimension(0);
      int numQPs    = weighted_measure[ws].dimension(1);
      
      for(int cell=0; cell<numCells; cell++){
        double elVol = 0.0;
        for(int node=0; node<numNodes; node++)
          for(int qp=0; qp<numQPs; qp++){
            int gid = wsElNodeID[ws][cell][node];
            int lid = overlapNodeMap->LID(gid);
            elVol += p[lid]*basisAtQPs[physIndex](node,qp)*weighted_measure[ws](cell,qp);
          }
        localv += elVol;
      }
    }
    comm->SumAll(&localv, &v, 1);

    if( dvdp != NULL ){
      localVec->PutScalar(0.0);
      overlapVec->PutScalar(0.0);
      double* odvdp; overlapVec->ExtractView(&odvdp);

      for(int ws=0; ws<numWorksets; ws++){
  
        int physIndex = wsPhysIndex[ws];
        int numNodes  = basisAtQPs[physIndex].dimension(0);
        int numCells  = weighted_measure[ws].dimension(0);
        int numQPs    = weighted_measure[ws].dimension(1);
      
        for(int cell=0; cell<numCells; cell++){
          for(int node=0; node<numNodes; node++){
            double elVol = 0.0;
            int gid = wsElNodeID[ws][cell][node];
            int lid = overlapNodeMap->LID(gid);
            for(int qp=0; qp<numQPs; qp++){
              elVol += basisAtQPs[physIndex](node,qp)*weighted_measure[ws](cell,qp);
            }
            odvdp[lid] += elVol;
          }
        }
      }
      localVec->Export(*overlapVec, *exporter, Add);
      int numLocalNodes = localVec->MyLength();
      double* lvec; localVec->ExtractView(&lvec);
      std::memcpy((void*)dvdp, (void*)lvec, numLocalNodes*sizeof(double));
    }
  }

}

/******************************************************************************/
void
ATO::OptimizationProblem::
setupTopOpt( Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  _meshSpecs,
             Albany::StateManager& _stateMgr)
/******************************************************************************/
{
  meshSpecs=_meshSpecs; 
  stateMgr=&_stateMgr;

  topology = params->get<Teuchos::RCP<Topology> >("Topology");
  double initValue = topology->getInitialValue();

  Teuchos::ParameterList& aggParams = params->get<Teuchos::ParameterList>("Objective Aggregator");
  std::string derName = aggParams.get<std::string>("dFdTopology Name");
  std::string objName = aggParams.get<std::string>("Objective Name");



  int numPhysSets = meshSpecs.size();

  cellTypes.resize(numPhysSets);
  cubatures.resize(numPhysSets);
  intrepidBasis.resize(numPhysSets);

  refPoints.resize(numPhysSets);
  refWeights.resize(numPhysSets);
  basisAtQPs.resize(numPhysSets);
  for(int i=0; i<numPhysSets; i++){
    cellTypes[i] = Teuchos::rcp(new shards::CellTopology (&meshSpecs[i]->ctd));
    Intrepid::DefaultCubatureFactory<double> cubFactory;
    cubatures[i] = cubFactory.create(*(cellTypes[i]), meshSpecs[i]->cubatureDegree);
    intrepidBasis[i] = Albany::getIntrepidBasis(meshSpecs[i]->ctd);

    int wsSize   = meshSpecs[i]->worksetSize;
    int numVerts = cellTypes[i]->getNodeCount();
    int numNodes = intrepidBasis[i]->getCardinality();
    int numQPs   = cubatures[i]->getNumPoints();
    int numDims  = cubatures[i]->getDimension();

    refPoints[i].resize(numQPs, numDims);
    refWeights[i].resize(numQPs);
    basisAtQPs[i].resize(numNodes, numQPs);
    cubatures[i]->getCubature(refPoints[i],refWeights[i]);

    intrepidBasis[i]->getValues(basisAtQPs[i], refPoints[i], Intrepid::OPERATOR_VALUE);

    Teuchos::RCP<Albany::Layouts> dl = 
      Teuchos::rcp( new Albany::Layouts(wsSize, numVerts, numNodes, numQPs, numDims));

    //tpetra-conversion If registerOldState is ever made true, the code will
    // likely break.
    stateMgr->registerStateVariable(objName, dl->workset_scalar, meshSpecs[i]->ebName, 
                                   "scalar", 0.0, /*registerOldState=*/ false, true);

    if( topology->getCentering() == "Element" ){
      stateMgr->registerStateVariable(topology->getName(), dl->cell_scalar, meshSpecs[i]->ebName, 
                                     "scalar", initValue, /*registerOldState=*/ false, true);
      stateMgr->registerStateVariable(derName, dl->cell_scalar, meshSpecs[i]->ebName, 
                                     "scalar", initValue, /*registerOldState=*/ false, true);
    } else
    if( topology->getCentering() == "Node" ){
      stateMgr->registerStateVariable(derName, dl->node_scalar, meshSpecs[i]->ebName, 
                                     "scalar", initValue, /*registerOldState=*/ false, false);
      stateMgr->registerStateVariable(topology->getName(), dl->node_scalar, meshSpecs[i]->ebName, 
                                     "scalar", initValue, /*registerOldState=*/ false, false);
      stateMgr->registerStateVariable(topology->getName()+"_node", dl->node_node_scalar, "all",
                                     "scalar", initValue, /*registerOldState=*/ false, true);
      if( topology->OutputFilteredTopology() )
        stateMgr->registerStateVariable(topology->getName()+"_node_filtered", dl->node_node_scalar, "all",
                                       "scalar", initValue, /*registerOldState=*/ false, true);
    }

  }

 
}


/******************************************************************************/
void
ATO::OptimizationProblem::InitTopOpt()
/******************************************************************************/
{

  const Albany::WorksetArray<int>::type& wsPhysIndex = disc->getWsPhysIndex();
  const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type&
        coords = disc->getCoords();
  const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > > >::type&
    wsElNodeEqID = disc->getWsElNodeEqID();
  const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > >::type&
    wsElNodeID = disc->getWsElNodeID();

  Teuchos::RCP<const Epetra_BlockMap>
    overlapNodeMap = stateMgr->getNodalDataBlock()->getOverlapMap();



  const Albany::WorksetArray<std::string>::type& wsEBNames = disc->getWsEBNames();
  const Teuchos::Array<std::string>& fixedBlocks = topology->getFixedBlocks();


  Albany::StateArrays& stateArrays = stateMgr->getStateArrays();
  Albany::StateArrayVec& dest = stateArrays.elemStateArrays;

  int numWorksets = wsElNodeEqID.size();
  Intrepid::FieldContainer<double> jacobian;
  Intrepid::FieldContainer<double> jacobian_det;
  Intrepid::FieldContainer<double> coordCon;

  weighted_measure.resize(numWorksets);
  for(int ws=0; ws<numWorksets; ws++){

    int physIndex = wsPhysIndex[ws];
    int numCells  = wsElNodeEqID[ws].size();
    int numNodes  = wsElNodeEqID[ws][0].size();
    int numDims   = wsElNodeEqID[ws][0][0].size();
    int numQPs    = cubatures[physIndex]->getNumPoints();

    coordCon.resize(numCells, numNodes, numDims);
    jacobian.resize(numCells,numQPs,numDims,numDims);
    jacobian_det.resize(numCells,numQPs);
    weighted_measure[ws].resize(numCells,numQPs);

    for(int cell=0; cell<numCells; cell++)
      for(int node=0; node<numNodes; node++)
        for(int dim=0; dim<numDims; dim++)
          coordCon(cell,node,dim) = coords[ws][cell][node][dim];

    Intrepid::CellTools<double>::setJacobian(jacobian, refPoints[physIndex], 
                                             coordCon, *(cellTypes[physIndex]));
    Intrepid::CellTools<double>::setJacobianDet(jacobian_det, jacobian);
    Intrepid::FunctionSpaceTools::computeCellMeasure<double>
     (weighted_measure[ws], jacobian_det, refWeights[physIndex]);

    // initialize topology of fixed blocks to have material
    if( find(fixedBlocks.begin(), fixedBlocks.end(), wsEBNames[ws]) != fixedBlocks.end() ){
      double matVal = topology->getMaterialValue();
      Albany::MDArray& wsTopo = dest[ws][topology->getName()];
      if( topology->getCentering() == "Node" ){
        int numCells = wsTopo.dimension(0);
        int numNodes = wsTopo.dimension(1);
        for(int cell=0; cell<numCells; cell++)
          for(int node=0; node<numNodes; node++){
//            int gid = wsElNodeID[ws][cell][node];
//            int lid = overlapNodeMap->LID(gid);
            wsTopo(cell,node) = matVal;
          }
      } else
      if( topology->getCentering() == "Element" ){
        int wsSize = wsTopo.size();
        for(int i=0; i<wsSize; i++)
          wsTopo(i) = matVal;
      }
     
    }

  }
  if( topology->getCentering() == "Node" ){
    Teuchos::RCP<const Epetra_BlockMap>
      overlapNodeMap = stateMgr->getNodalDataBlock()->getOverlapMapE();
    Teuchos::RCP<const Epetra_BlockMap>
      localNodeMap = stateMgr->getNodalDataBlock()->getLocalMapE();

    overlapVec = Teuchos::rcp(new Epetra_Vector(*overlapNodeMap));
    localVec   = Teuchos::rcp(new Epetra_Vector(*localNodeMap));
    exporter   = Teuchos::rcp(new Epetra_Export(*overlapNodeMap, *localNodeMap));
  }

 
}


