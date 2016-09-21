//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Intrepid2_FunctionSpaceTools.hpp"
#include "ATO_OptimizationProblem.hpp"
#include "Albany_AbstractDiscretization.hpp"
#include "Epetra_Export.h"
#include "Adapt_NodalDataVector.hpp"
#include "ATO_TopoTools.hpp"
#include "ATO_Integrator.hpp"
#include "ATO_Types.hpp"
#include <functional>

#include <sstream>

/******************************************************************************/
ATO::OptimizationProblem::
OptimizationProblem( const Teuchos::RCP<Teuchos::ParameterList>& params,
                     const Teuchos::RCP<ParamLib>& paramLib,
                     const int numDim) :
Albany::AbstractProblem(params, paramLib, numDim)
/******************************************************************************/
{
  if( params->isSublist("Topologies Parameters") == false ){
    nTopologies = 0;
    return;
  }

  nTopologies = params->sublist("Topologies Parameters").get<int>("Number of Topologies");

  const Teuchos::ParameterList& configSpec = params->sublist("Configuration");

  if( configSpec.isSublist("Linear Measures") ){
    const Teuchos::ParameterList& measuresSpec = configSpec.sublist("Linear Measures");

    MeasureModelFactory modelFactory(configSpec);

    int nMeasures = measuresSpec.get<int>("Number of Linear Measures");
    for(int iMeasure=0; iMeasure<nMeasures; iMeasure++){
      const Teuchos::ParameterList& 
        measureSpec = measuresSpec.sublist(Albany::strint("Linear Measure", iMeasure));
      std::string name = measureSpec.get<std::string>("Linear Measure Name");
      TEUCHOS_TEST_FOR_EXCEPTION( measureModels.count(name) != 0,
        Teuchos::Exceptions::InvalidParameter, std::endl <<
        "Error in ATO::OptimizationProblem setup:  " << std::endl <<
        "  Names of Linear Measures must be unique." << std::endl);
      measureModels[name] = modelFactory.create(measureSpec);
    }

  } else {
    TEUCHOS_TEST_FOR_EXCEPTION( 
      true, Teuchos::Exceptions::InvalidParameter, std::endl <<
      "Error in ATO::OptimizationProblem setup:  " << std::endl <<
      "  'Linear Measures' section missing." << std::endl);
  }


}

/******************************************************************************/
void
ATO::OptimizationProblem::
ComputeMeasure(std::string measureType, double& measure)
/******************************************************************************/
{
  const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > >::type&
        wsElNodeID = disc->getWsElNodeID();
  int numWorksets = wsElNodeID.size();

  if(isNonconformal){
    for(int ws=0; ws<numWorksets; ws++){
      Albany::StateArray& 
      stateArrayRef = stateMgr->getStateArray(Albany::StateManager::ELEM, ws);
      Albany::MDArray savedWeights = stateArrayRef["Gauss Weights"];
      int numCells  = weighted_measure[ws].dimension(0);
      int numQPs    = weighted_measure[ws].dimension(1);
      for(int cell=0; cell<numCells; cell++)
        for(int qp=0; qp<numQPs; qp++)
          weighted_measure[ws](cell,qp) = savedWeights(cell,qp);
    }
  }
  
  if(measureType == "Volume"){
    double localm = 0.0;

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
          localm += weighted_measure[ws](cell,qp);
    }
  
    comm->SumAll(&localm, &measure, 1);
  } else

  if(measureType == "Mass"){

    // JR:  A reference mass is difficult to define.  return 1.0 so that
    // mass constraints are absolute, not relative.
    
    measure = 1.0;
  }
}

/******************************************************************************/
void
ATO::OptimizationProblem::
ComputeMeasure(std::string measureType, 
               std::vector<Teuchos::RCP<TopologyStruct> >& topologyStructs,
               double& measure, double* dmdp, 
               std::string strIntegrationMethod)
/******************************************************************************/
{
  if(strIntegrationMethod == "Conformal")
    if(measureType == "Volume")
      computeConformalVolume(topologyStructs, measure, dmdp);
    else
      computeConformalMeasure(measureType, topologyStructs, measure, dmdp);
  else
  if(strIntegrationMethod == "Gauss Quadrature")
    computeMeasure(measureType, topologyStructs, measure, dmdp);
  else
    TEUCHOS_TEST_FOR_EXCEPTION( true, Teuchos::Exceptions::InvalidParameter, std::endl <<
      "Error!  In ATO::OptimizationProblem setup:  Integration Method not recognized" << std::endl);
}

/******************************************************************************/
void
ATO::OptimizationProblem::
computeMeasure(std::string measureType, 
               std::vector<Teuchos::RCP<TopologyStruct> >& topologyStructs,
               double& measure, double* dmdp)
/******************************************************************************/
{

  const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > >::type&
        wsElNodeID = disc->getWsElNodeID();
  int numWorksets = wsElNodeID.size();

  if(isNonconformal){
    for(int ws=0; ws<numWorksets; ws++){
      Albany::StateArray& 
      stateArrayRef = stateMgr->getStateArray(Albany::StateManager::ELEM, ws);
      Albany::MDArray savedWeights = stateArrayRef["Gauss Weights"];
      int numCells  = weighted_measure[ws].dimension(0);
      int numQPs    = weighted_measure[ws].dimension(1);
      for(int cell=0; cell<numCells; cell++)
        for(int qp=0; qp<numQPs; qp++)
          weighted_measure[ws](cell,qp) = savedWeights(cell,qp);
    }
  }

  double localm = 0.0;
  const Albany::WorksetArray<int>::type& wsPhysIndex = disc->getWsPhysIndex();
  const Albany::WorksetArray<std::string>::type& wsEBNames = disc->getWsEBNames();

  std::vector<double*> topoValues(nTopologies);
  Teuchos::Array<Teuchos::RCP<Topology> > topologies(nTopologies);
  for(int itopo=0; itopo<nTopologies; itopo++){
    topologies[itopo] = topologyStructs[itopo]->topology;
    topologyStructs[itopo]->dataVector->ExtractView(&topoValues[itopo]);
  }

  std::vector<double*> odmdp(nTopologies);
  Teuchos::Array<double> drdz(nTopologies), pVals(nTopologies);
  if(dmdp != NULL){
    for(int i=0; i<nTopologies; i++){
      overlapVectors[i]->PutScalar(0.0);
      overlapVectors[i]->ExtractView(&odmdp[i]);
    }
  }

  Teuchos::RCP<BlockMeasureMap> measureModel = measureModels.at(measureType);

  for(int ws=0; ws<numWorksets; ws++){

    int physIndex = wsPhysIndex[ws];
    int numNodes  = basisAtQPs[physIndex].dimension(0);
    int numCells  = weighted_measure[ws].dimension(0);
    int numQPs    = weighted_measure[ws].dimension(1);

    std::string blockName = wsEBNames[ws];

    // not all blocks are required to have all measures 
    // (surface blocks, for example, don't have a mass)
    if(measureModel->find(blockName) == measureModel->end()) continue;

    Teuchos::RCP<MeasureModel> blockMeasureModel = measureModel->at(blockName);
      
    for(int cell=0; cell<numCells; cell++){
      double elMeasure = 0.0;
      for(int qp=0; qp<numQPs; qp++){

        // compute values of mixture topologies at the qp
        for(int itopo=0; itopo<nTopologies; itopo++) pVals[itopo]=0.0;
        for(int node=0; node<numNodes; node++){
          int gid = wsElNodeID[ws][cell][node];
          int lid = overlapNodeMap->LID(gid);
          for(int itopo=0; itopo<nTopologies; itopo++)
            pVals[itopo] += topoValues[itopo][lid]*basisAtQPs[physIndex](node,qp);
        }

        double qpMeasure = blockMeasureModel->Evaluate(pVals, topologies);

        elMeasure += qpMeasure*weighted_measure[ws](cell,qp);

        if(dmdp != NULL ){
          blockMeasureModel->Gradient(pVals, topologies, drdz);
          for(int node=0; node<numNodes; node++){
            int gid = wsElNodeID[ws][cell][node];
            int lid = overlapNodeMap->LID(gid);
            for(int itopo=0; itopo<nTopologies; itopo++)
              odmdp[itopo][lid] += drdz[itopo]
                                  *basisAtQPs[physIndex](node,qp)
                                  *weighted_measure[ws](cell,qp);
          }
        }
      }
      localm += elMeasure;
    }
  }

  comm->SumAll(&localm, &measure, 1);

  if( dmdp != NULL ){
    int numLocalNodes = localVec->MyLength();
    for(int itopo=0; itopo<nTopologies; itopo++){
      localVec->PutScalar(0.0);
      localVec->Export(*overlapVectors[itopo], *exporter, Add);
      double* lvec; localVec->ExtractView(&lvec);
      std::memcpy((void*)(dmdp+itopo*numLocalNodes), (void*)lvec, numLocalNodes*sizeof(double));
    }
  }
}

/******************************************************************************/
void
ATO::OptimizationProblem::
computeConformalMeasure(std::string measureType, 
                        std::vector<Teuchos::RCP<TopologyStruct> >& topologyStructs,
                        double& measure, double* dmdp)
/******************************************************************************/
{
TEUCHOS_TEST_FOR_EXCEPTION( isNonconformal, Teuchos::Exceptions::InvalidParameter, std::endl <<
"Error!  In ATO::OptimizationProblem setup: " << std::endl << 
"Conformal integration not implemented for non-conformal geometry. " << std::endl);

  double localm = 0.0;
  const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > >::type&
        wsElNodeID = disc->getWsElNodeID();
  const Albany::WorksetArray<int>::type& wsPhysIndex = disc->getWsPhysIndex();
  const Albany::WorksetArray<std::string>::type& wsEBNames = disc->getWsEBNames();
  int numWorksets = wsElNodeID.size();

  const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type&
        coords = disc->getCoords();

  Kokkos::DynRankView<RealType, PHX::Device> coordCon;
  Kokkos::DynRankView<RealType, PHX::Device> topoVals;
  Kokkos::DynRankView<RealType, PHX::Device> dMdtopo;

  std::vector<double*> topoValues(nTopologies);
  Teuchos::Array<Teuchos::RCP<Topology> > topologies(nTopologies);
  for(int itopo=0; itopo<nTopologies; itopo++){
    topologies[itopo] = topologyStructs[itopo]->topology;
    topologyStructs[itopo]->dataVector->ExtractView(&topoValues[itopo]);
  }

  std::vector<double*> odmdp(nTopologies);
  Teuchos::Array<double> drdz(nTopologies), pVals(nTopologies);
  if(dmdp != NULL){
    for(int i=0; i<nTopologies; i++){
      overlapVectors[i]->PutScalar(0.0);
      overlapVectors[i]->ExtractView(&odmdp[i]);
    }
  }

  Teuchos::RCP<BlockMeasureMap> measureModel = measureModels.at(measureType);

  for(int ws=0; ws<numWorksets; ws++){

    int physIndex = wsPhysIndex[ws];
    int numNodes  = basisAtQPs[physIndex].dimension(0);
    int numCells  = weighted_measure[ws].dimension(0);
    int numQPs    = weighted_measure[ws].dimension(1);
    int numDims   = cubatures[physIndex]->getDimension();

    SubIntegrator myDicer(cellTypes[physIndex],intrepidBasis[physIndex],/*maxRefs=*/1,/*maxErr=*/1e-5);

    coordCon = Kokkos::DynRankView<RealType, PHX::Device>("coordCon", numNodes, numDims);
    topoVals = Kokkos::DynRankView<RealType, PHX::Device>("topoVals", numNodes);
    dMdtopo = Kokkos::DynRankView<RealType, PHX::Device>("dMdtopo", numNodes);

    std::string blockName = wsEBNames[ws];

    // not all blocks are required to have all measures 
    // (surface blocks, for example, don't have a mass)
    if(measureModel->find(blockName) == measureModel->end()) continue;

    Teuchos::RCP<MeasureModel> blockMeasureModel = measureModel->at(blockName);

 
    int materialTopologyIndex = blockMeasureModel->getMaterialTopologyIndex();
    double* p; topologyStructs[materialTopologyIndex]->dataVector->ExtractView(&p);
    Teuchos::RCP<Topology> materialTopology = topologyStructs[materialTopologyIndex]->topology;
      
    for(int cell=0; cell<numCells; cell++){
      double elMass = 0.0;

      for(int node=0; node<numNodes; node++){
        for(int dim=0; dim<numDims; dim++)
          coordCon(node,dim) = coords[ws][cell][node][dim];
        int gid = wsElNodeID[ws][cell][node];
        int lid = overlapNodeMap->LID(gid);
        topoVals(node) = p[lid];
      }

      // JR:  Until this is done right ...

      double weight=0.0;
      if(dmdp != NULL ){
        myDicer.getMeasure(weight, topoVals, coordCon, 
                           materialTopology->getInterfaceValue(), Sense::Positive);
      } else {
        myDicer.getMeasure(weight, dMdtopo, topoVals, coordCon, 
                           materialTopology->getInterfaceValue(), Sense::Positive);
      }

      double totalWeight = 0.0;
      for(int qp=0; qp<numQPs; qp++) 
        totalWeight += weighted_measure[ws](cell,qp);

      double weightFraction = weight/totalWeight;

      double elMeasure = 0.0;
      for(int qp=0; qp<numQPs; qp++){

        // compute values of mixture topologies at the qp
        for(int itopo=0; itopo<nTopologies; itopo++) pVals[itopo]=0.0;
        for(int node=0; node<numNodes; node++){
          int gid = wsElNodeID[ws][cell][node];
          int lid = overlapNodeMap->LID(gid);
          for(int itopo=0; itopo<nTopologies; itopo++)
            pVals[itopo] += topoValues[itopo][lid]*basisAtQPs[physIndex](node,qp);
        }

        double qpMeasure = blockMeasureModel->Evaluate(pVals, topologies);

        elMeasure += qpMeasure*weightFraction*weighted_measure[ws](cell,qp);

        localm += weight;


        if(dmdp != NULL ){
          blockMeasureModel->Gradient(pVals, topologies, drdz);
          for(int node=0; node<numNodes; node++){
            int gid = wsElNodeID[ws][cell][node];
            int lid = overlapNodeMap->LID(gid);
            for(int itopo=0; itopo<nTopologies; itopo++){
              if(itopo == materialTopologyIndex){
                odmdp[itopo][lid] += drdz[itopo]*dMdtopo[node]*weightFraction
                                     *basisAtQPs[physIndex](node,qp)
                                     *weighted_measure[ws](cell,qp);
              } else {
                odmdp[itopo][lid] += drdz[itopo]*weightFraction
                                     *basisAtQPs[physIndex](node,qp)
                                     *weighted_measure[ws](cell,qp);
              }
            }
          }
        }
      }
      localm += elMeasure;
    }
  }

  comm->SumAll(&localm, &measure, 1);

  if( dmdp != NULL ){
    int numLocalNodes = localVec->MyLength();
    for(int itopo=0; itopo<nTopologies; itopo++){
      localVec->PutScalar(0.0);
      localVec->Export(*overlapVectors[itopo], *exporter, Add);
      double* lvec; localVec->ExtractView(&lvec);
      std::memcpy((void*)(dmdp+itopo*nTopologies), (void*)lvec, numLocalNodes*sizeof(double));
    }
  }
}
/******************************************************************************/
void
ATO::OptimizationProblem::
computeConformalVolume(std::vector<Teuchos::RCP<TopologyStruct> >& topologyStructs,
double& v, double* dvdp)
/******************************************************************************/
{

TEUCHOS_TEST_FOR_EXCEPTION( isNonconformal, Teuchos::Exceptions::InvalidParameter, std::endl <<
"Error!  In ATO::OptimizationProblem setup: " << std::endl << 
"Conformal integration not implemented for non-conformal geometry. " << std::endl);

  Teuchos::RCP<Topology> topology = topologyStructs[0]->topology;
  double* p; topologyStructs[0]->dataVector->ExtractView(&p);

  double localv = 0.0;
  const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > >::type&
        wsElNodeID = disc->getWsElNodeID();
  const Albany::WorksetArray<int>::type& wsPhysIndex = disc->getWsPhysIndex();
  int numWorksets = wsElNodeID.size();

  const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type&
        coords = disc->getCoords();

  Kokkos::DynRankView<RealType, PHX::Device> coordCon;
  Kokkos::DynRankView<RealType, PHX::Device> topoVals;
  Kokkos::DynRankView<RealType, PHX::Device> dMdtopo;

  double* odvdp = NULL;
  if( dvdp != NULL ){
    localVec->PutScalar(0.0);
    overlapVec->PutScalar(0.0);
    overlapVec->ExtractView(&odvdp);
  }

  for(int ws=0; ws<numWorksets; ws++){
  
    int physIndex = wsPhysIndex[ws];
    int numNodes  = basisAtQPs[physIndex].dimension(0);
    int numCells  = weighted_measure[ws].dimension(0);
    int numDims   = cubatures[physIndex]->getDimension();

    SubIntegrator myDicer(cellTypes[physIndex],intrepidBasis[physIndex],/*maxRefs=*/1,/*maxErr=*/1e-5);

    coordCon = Kokkos::DynRankView<RealType, PHX::Device>("coordCon", numNodes, numDims);  //inefficient, reallocating memory. 
    topoVals = Kokkos::DynRankView<RealType, PHX::Device>("topoVals", numNodes);   //inefficient, reallocating memory. 
    dMdtopo = Kokkos::DynRankView<RealType, PHX::Device>("dMdtopo", numNodes);   //inefficient, reallocating memory. 

    for(int cell=0; cell<numCells; cell++){
      for(int node=0; node<numNodes; node++){
        for(int dim=0; dim<numDims; dim++)
          coordCon(node,dim) = coords[ws][cell][node][dim];
        int gid = wsElNodeID[ws][cell][node];
        int lid = overlapNodeMap->LID(gid);
        topoVals(node) = p[lid];
      }

      if( dvdp == NULL ){

        double weight=0.0;
        myDicer.getMeasure(weight, topoVals, coordCon, 
                           topology->getInterfaceValue(), Sense::Positive);
        localv += weight;

      } else { 

        double weight=0.0;
        myDicer.getMeasure(weight, dMdtopo, topoVals, coordCon, 
                           topology->getInterfaceValue(), Sense::Positive);
        localv += weight;

        for(int node=0; node<numNodes; node++){
          int gid = wsElNodeID[ws][cell][node];
          int lid = overlapNodeMap->LID(gid);
          odvdp[lid] += dMdtopo(node);
        }
      }
    }
  }

  if( dvdp != NULL ){
    localVec->Export(*overlapVec, *exporter, Add);
    int numLocalNodes = localVec->MyLength();
    double* lvec; localVec->ExtractView(&lvec);
    std::memcpy((void*)dvdp, (void*)lvec, numLocalNodes*sizeof(double));
  }

  comm->SumAll(&localv, &v, 1);

}


/******************************************************************************/
void
ATO::OptimizationProblem::
ComputeVolume(double* p, const double* dfdp, 
              double& v, double threshhold, double minP)
/******************************************************************************/
{
  const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > >::type&
    wsElNodeID = disc->getWsElNodeID();
  int numWorksets = wsElNodeID.size();

  if(isNonconformal){
    for(int ws=0; ws<numWorksets; ws++){
      Albany::StateArray& 
      stateArrayRef = stateMgr->getStateArray(Albany::StateManager::ELEM, ws);
      Albany::MDArray savedWeights = stateArrayRef["Gauss Weights"];
      int numCells  = weighted_measure[ws].dimension(0);
      int numQPs    = weighted_measure[ws].dimension(1);
      for(int cell=0; cell<numCells; cell++)
        for(int qp=0; qp<numQPs; qp++)
          weighted_measure[ws](cell,qp) = savedWeights(cell,qp);
    }
  }

  double localv = 0.0;

  const Albany::WorksetArray<int>::type& wsPhysIndex = disc->getWsPhysIndex();



  for(int ws=0; ws<numWorksets; ws++){

    int physIndex = wsPhysIndex[ws];
    int numNodes  = basisAtQPs[physIndex].dimension(0);
    int numCells  = weighted_measure[ws].dimension(0);
    int numQPs    = weighted_measure[ws].dimension(1);
    
    for(int cell=0; cell<numCells; cell++){
      double elVol = 0.0;
      for(int node=0; node<numNodes; node++){
        int gid = wsElNodeID[ws][cell][node];
        int lid = overlapNodeMap->LID(gid);
        if(dfdp[lid] < threshhold) p[lid] = 1.0;
        else p[lid] = minP;
      }

      for(int node=0; node<numNodes; node++){
        int gid = wsElNodeID[ws][cell][node];
        int lid = overlapNodeMap->LID(gid);
        for(int qp=0; qp<numQPs; qp++)
          elVol += p[lid]*basisAtQPs[physIndex](node,qp)*weighted_measure[ws](cell,qp);
      }
      localv += elVol;
    }
  }
  comm->SumAll(&localv, &v, 1);
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

  Teuchos::RCP<TopologyArray> topologyArray = params->get<Teuchos::RCP<TopologyArray> >("Topologies");

  const Teuchos::ParameterList& configSpec = params->sublist("Configuration");
  isNonconformal = false;
  if(configSpec.isType<bool>("Nonconformal"))
    isNonconformal = configSpec.get<bool>("Nonconformal");

  Teuchos::ParameterList& aggParams = params->get<Teuchos::ParameterList>("Objective Aggregator");
  std::string derName = aggParams.get<std::string>("Output Derivative Name");
  std::string objName = aggParams.get<std::string>("Output Value Name");


  int numPhysSets = meshSpecs.size();

  cellTypes.resize(numPhysSets);
  cubatures.resize(numPhysSets);
  intrepidBasis.resize(numPhysSets);

  refPoints.resize(numPhysSets);
  refWeights.resize(numPhysSets);
  basisAtQPs.resize(numPhysSets);
  for(int i=0; i<numPhysSets; i++){

    cellTypes[i] = Teuchos::rcp(new shards::CellTopology (&meshSpecs[i]->ctd));
    Intrepid2::DefaultCubatureFactory cubFactory;

    // JR:  By incrementing the cubature degree by one, this creates a semantic coupling 
    // to the projected integrator in Cogent.

    int cubatureDegree = meshSpecs[i]->cubatureDegree;
    if(isNonconformal) cubatureDegree += 1;  // non-conformal is degree n (vs 2n-1)
   
    cubatures[i] = cubFactory.create<PHX::Device, RealType, RealType>(*(cellTypes[i]), cubatureDegree);
    intrepidBasis[i] = Albany::getIntrepid2Basis(meshSpecs[i]->ctd);

    int wsSize   = meshSpecs[i]->worksetSize;
    int numVerts = cellTypes[i]->getNodeCount();
    int numNodes = intrepidBasis[i]->getCardinality();
    int numQPs   = cubatures[i]->getNumPoints();
    int numDims  = cubatures[i]->getDimension();

    refPoints[i] = Kokkos::DynRankView<RealType, PHX::Device>("refPoints", numQPs, numDims);  //inefficient, reallocating memory.  
    refWeights[i] = Kokkos::DynRankView<RealType, PHX::Device>("refWeights", numQPs); //inefficient, reallocating memory. 
    basisAtQPs[i] = Kokkos::DynRankView<RealType, PHX::Device>("basisAtQPs", numNodes, numQPs); //inefficient, reallocating memory. 
    cubatures[i]->getCubature(refPoints[i],refWeights[i]);

    intrepidBasis[i]->getValues(basisAtQPs[i], refPoints[i], Intrepid2::OPERATOR_VALUE);

    Teuchos::RCP<Albany::Layouts> dl = 
      Teuchos::rcp( new Albany::Layouts(wsSize, numVerts, numNodes, numQPs, numDims));

    TopologyArray::iterator it;
    for(it=topologyArray->begin(); it!=topologyArray->end(); ++it){
      Teuchos::RCP<Topology> topology = *it;
      double initValue = topology->getInitialValue();
      Albany::StateStruct::MeshFieldEntity entity = Albany::StateStruct::NodalDataToElemNode;
      stateMgr->registerStateVariable(topology->getName()+"_node", dl->node_scalar, "all",true,&entity);
                                     
      if( topology->TopologyOutputFilter() >= 0 )
        stateMgr->registerStateVariable(topology->getName()+"_node_filtered", dl->node_node_scalar, "all",
                                       "scalar", initValue, /*registerOldState=*/ false, true);
  
      if( topology->getEntityType() == "State Variable" ){
        stateMgr->registerStateVariable(topology->getName(), dl->node_scalar, meshSpecs[i]->ebName, 
                                       "scalar", initValue, /*registerOldState=*/ false, false);
      } else if( topology->getEntityType() == "Distributed Parameter" ){
        Albany::StateStruct::MeshFieldEntity entity = Albany::StateStruct::NodalDistParameter;
        stateMgr->registerStateVariable(topology->getName(), dl->node_scalar, "all", true, &entity, "");
      } 
      else {
        TEUCHOS_TEST_FOR_EXCEPTION( true, Teuchos::Exceptions::InvalidParameter, std::endl <<
          "Error!  In ATO::OptimizationProblem setup:  Entity Type not recognized" << std::endl);
      }
//      strIntegrationMethod = topology->getIntegrationMethod();
    }
    
    //tpetra-conversion If registerOldState is ever made true, the code will
    // likely break.
    stateMgr->registerStateVariable(objName, dl->workset_scalar, meshSpecs[i]->ebName, 
                                   "scalar", 0.0, /*registerOldState=*/ false, true);

    stateMgr->registerStateVariable(derName, dl->node_scalar, meshSpecs[i]->ebName, 
                                   "scalar", 0.0, /*registerOldState=*/ false, false);

    int nTopos = topologyArray->size();
    for(int itopo=0; itopo<nTopos; itopo++){
      stateMgr->registerStateVariable(Albany::strint(derName+"_node",itopo), dl->node_node_scalar, "all",
                                     "scalar", 0.0, /*registerOldState=*/ false, true);
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
  const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<LO> > > >::type&
    wsElNodeEqID = disc->getWsElNodeEqID();
  const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > >::type&
    wsElNodeID = disc->getWsElNodeID();

  const Albany::WorksetArray<std::string>::type& wsEBNames = disc->getWsEBNames();

  int numWorksets = wsElNodeEqID.size();

  Kokkos::DynRankView<RealType, PHX::Device> jacobian;
  Kokkos::DynRankView<RealType, PHX::Device> jacobian_det;
  Kokkos::DynRankView<RealType, PHX::Device> coordCon;

  weighted_measure.resize(numWorksets);

  if(isNonconformal){
    for(int ws=0; ws<numWorksets; ws++){
      Albany::StateArray& 
      stateArrayRef = stateMgr->getStateArray(Albany::StateManager::ELEM, ws);
      Albany::MDArray savedWeights = stateArrayRef["Gauss Weights"];

      int physIndex = wsPhysIndex[ws];
      int numCells  = wsElNodeEqID[ws].size();
      int numQPs    = cubatures[physIndex]->getNumPoints();
      weighted_measure[ws] = Kokkos::DynRankView<RealType, PHX::Device>("weighted_measure", numCells,numQPs); //inefficient, reallocating memory. 
      for(int cell=0; cell<numCells; cell++)
        for(int qp=0; qp<numQPs; qp++)
          weighted_measure[ws](cell,qp) = savedWeights(cell,qp);
    }
  } else {
  
    for(int ws=0; ws<numWorksets; ws++){
  
      int physIndex = wsPhysIndex[ws];
      int numCells  = wsElNodeEqID[ws].size();
      int numNodes  = wsElNodeEqID[ws][0].size();
      int numDims   = cubatures[physIndex]->getDimension();
      int numQPs    = cubatures[physIndex]->getNumPoints();
  
      coordCon = Kokkos::DynRankView<RealType, PHX::Device>("coordCon", numCells, numNodes, numDims); //inefficient, reallocating memory. 
      jacobian = Kokkos::DynRankView<RealType, PHX::Device>("jacobian", numCells,numQPs,numDims,numDims); //inefficient, reallocating memory. 
      jacobian_det = Kokkos::DynRankView<RealType, PHX::Device>("jacobian_det", numCells,numQPs); //inefficient, reallocating memory. 
      weighted_measure[ws] = Kokkos::DynRankView<RealType, PHX::Device>("weighted_measure", numCells,numQPs); //inefficient, reallocating memory. 
  
      for(int cell=0; cell<numCells; cell++)
        for(int node=0; node<numNodes; node++)
          for(int dim=0; dim<numDims; dim++)
            coordCon(cell,node,dim) = coords[ws][cell][node][dim];
      Intrepid2::CellTools<PHX::Device>::setJacobian(jacobian, refPoints[physIndex], 
                                               coordCon, *(cellTypes[physIndex]));
      Intrepid2::CellTools<PHX::Device>::setJacobianDet(jacobian_det, jacobian);
      Intrepid2::FunctionSpaceTools<PHX::Device>::computeCellMeasure(weighted_measure[ws], jacobian_det, refWeights[physIndex]);
   
    }
  }

  overlapNodeMap = disc->getOverlapNodeMap();
  localNodeMap = disc->getNodeMap();

  overlapVec = Teuchos::rcp(new Epetra_Vector(*overlapNodeMap));
  localVec   = Teuchos::rcp(new Epetra_Vector(*localNodeMap));
  exporter   = Teuchos::rcp(new Epetra_Export(*overlapNodeMap, *localNodeMap));

  overlapVectors.resize(nTopologies);
  for(int i=0; i<nTopologies; i++)
    overlapVectors[i] = Teuchos::rcp(new Epetra_Vector(*overlapNodeMap));
}


/******************************************************************************/
ATO::TopologyBasedMixture::
TopologyBasedMixture(const Teuchos::ParameterList& blockSpec)
/******************************************************************************/
{
}

/******************************************************************************/
double 
ATO::TopologyBasedMixture::
Evaluate(const Teuchos::Array<double>& pVals,
         Teuchos::Array<Teuchos::RCP<Topology> >& topologies)
/******************************************************************************/
{
}

/******************************************************************************/
void
ATO::TopologyBasedMixture::
Gradient(const Teuchos::Array<double>& pVals,
         Teuchos::Array<Teuchos::RCP<Topology> >& topologies,
         Teuchos::Array<double>& outVals)
/******************************************************************************/
{
}

/******************************************************************************/
ATO::TopologyBasedMaterial::
TopologyBasedMaterial(const Teuchos::ParameterList& blockSpec)
/******************************************************************************/
{
}

/******************************************************************************/
double 
ATO::TopologyBasedMaterial::
Evaluate(const Teuchos::Array<double>& pVals,
         Teuchos::Array<Teuchos::RCP<Topology> >& topologies)
/******************************************************************************/
{
}

/******************************************************************************/
void
ATO::TopologyBasedMaterial::
Gradient(const Teuchos::Array<double>& pVals,
         Teuchos::Array<Teuchos::RCP<Topology> >& topologies,
         Teuchos::Array<double>& outVals)
/******************************************************************************/
{
}

/******************************************************************************/
ATO::MeasureModelFactory::
MeasureModelFactory( Teuchos::ParameterList _p ) : configParams(_p) {}
/******************************************************************************/

/******************************************************************************/
Teuchos::RCP<ATO::BlockMeasureMap>
ATO::MeasureModelFactory::
create(const Teuchos::ParameterList& measureParams )
/******************************************************************************/
{

  std::string measureType = measureParams.get<std::string>("Linear Measure Type");

  bool limitByBlock = false;
  Teuchos::Array<std::string> blocks;
  if(measureParams.isType<Teuchos::Array<std::string>>("Blocks")){
    blocks = measureParams.get<Teuchos::Array<std::string>>("Blocks");
    limitByBlock = true;
  }

  Teuchos::RCP<BlockMeasureMap> blockMeasureMap = Teuchos::rcp(new BlockMeasureMap);
  
  Teuchos::ParameterList& blocksSpec = configParams.sublist("Element Blocks");

  int nBlocks = blocksSpec.get<int>("Number of Element Blocks");
  
  for(int iblock=0; iblock<nBlocks; iblock++){
    const Teuchos::ParameterList& 
      blockSpec = blocksSpec.sublist(Albany::strint("Element Block", iblock));

    std::string name = blockSpec.get<std::string>("Name");

    if(limitByBlock){
      if( find(blocks.begin(), blocks.end(),name) == blocks.end() ) continue;
    }
    
    TEUCHOS_TEST_FOR_EXCEPTION( 
      blockSpec.isSublist("Material") && blockSpec.isSublist("Mixture"),
      Teuchos::Exceptions::InvalidParameter, std::endl <<
      "Error in ATO::OptimizationProblem setup:  " << std::endl <<
      "  Provide either 'Mixture' list or 'Material' list. Not both." << std::endl);

    std::string matType;
    if(blockSpec.isSublist("Material")) matType = "Material";
    else
    if(blockSpec.isSublist("Mixture")) matType = "Mixture";
    else
      TEUCHOS_TEST_FOR_EXCEPTION( 
        true, Teuchos::Exceptions::InvalidParameter, std::endl <<
        "Error in ATO::OptimizationProblem setup:  " << std::endl <<
        "  No 'Mixture' list or 'Material' list found." << std::endl);


    if( measureType == "Topology Weighted Integral" ){
      if( matType == "Mixture" ){
        (*blockMeasureMap)[name] = Teuchos::rcp(new TopologyWeightedIntegral_Mixture(blockSpec, measureParams));
      } else {
        (*blockMeasureMap)[name] = Teuchos::rcp(new TopologyWeightedIntegral_Material(blockSpec, measureParams));
      }
    } else 

    if( measureType == "Volume" ){
      (*blockMeasureMap)[name] = Teuchos::rcp(new VolumeMeasure(blockSpec, measureParams));
    } else 

//    if( measureType == "Center of Mass" ){
//      if( matType == "Mixture" ){
//        (*blockMeasureMap)[name] = Teuchos::rcp(new CenterOfMass_Mixture(blockSpec, measureParams));
//      } else {
//        (*blockMeasureMap)[name] = Teuchos::rcp(new CenterOfMass_Material(blockSpec, measureParams));
//      }
//    } else 

//    if( measureType == "Centroid" ){
//      (*blockMeasureMap)[name] = Teuchos::rcp(new Centroid(blockSpec, measureParams));
//    } else 

      TEUCHOS_TEST_FOR_EXCEPTION( 
        true, Teuchos::Exceptions::InvalidParameter, std::endl <<
        "Error in ATO::OptimizationProblem setup:  " << std::endl <<
        "  Unrecognized 'Linear Measure Type' requested." << std::endl);

  }

  return blockMeasureMap;

}

/******************************************************************************/
ATO::VolumeMeasure::
VolumeMeasure(const Teuchos::ParameterList& blockSpec, 
              const Teuchos::ParameterList& measureParams)
/******************************************************************************/
{
  const Teuchos::ParameterList& params = measureParams.sublist("Volume");

  materialTopologyIndex = params.get<int>("Topology Index");
  materialFunctionIndex = params.get<int>("Function Index");
}


/******************************************************************************/
ATO::TopologyWeightedIntegral_Material::
TopologyWeightedIntegral_Material(const Teuchos::ParameterList& blockSpec, 
                                  const Teuchos::ParameterList& measureParams)
/******************************************************************************/
{
  const Teuchos::ParameterList& materialSpec = blockSpec.sublist("Material");

  // get integrated parameter
  const Teuchos::ParameterList& params = measureParams.sublist("Topology Weighted Integral");
  std::string paramName = params.get<std::string>("Parameter Name");
  
  parameterValue = materialSpec.get<double>(paramName);

  materialTopologyIndex = params.get<int>("Topology Index");
  materialFunctionIndex = params.get<int>("Function Index");
}

/******************************************************************************/
ATO::TopologyWeightedIntegral_Mixture::
TopologyWeightedIntegral_Mixture(const Teuchos::ParameterList& blockSpec, 
                                 const Teuchos::ParameterList& measureParams)
/******************************************************************************/
{

  const Teuchos::ParameterList& mixtureSpec = blockSpec.sublist("Mixture");

  // get integrated parameter
  const Teuchos::ParameterList& params = measureParams.sublist("Topology Weighted Integral");
  std::string paramName = params.get<std::string>("Parameter Name");
  
  materialTopologyIndex = params.get<int>("Topology Index");
  materialFunctionIndex = params.get<int>("Function Index");


  // get parameter for each material
  int nMats = mixtureSpec.get<int>("Number of Materials");
  parameterValues.resize(nMats);
  for(int imat=0; imat<nMats; imat++)
    parameterValues[imat] = mixtureSpec.sublist(Albany::strint("Material", imat)).get<double>(paramName);

  const Teuchos::ParameterList& mixedParamsSpec = mixtureSpec.sublist("Mixed Parameters");
  int nMixedParams = mixedParamsSpec.get<int>("Number of Mixed Parameters");
  bool paramFound = false;
  for(int iParams=0; iParams<nMixedParams; iParams++){
    const Teuchos::ParameterList& 
      paramSpec = mixedParamsSpec.sublist(Albany::strint("Mixed Parameter", iParams));
    if(paramName == paramSpec.get<std::string>("Parameter Name")){
 
      paramFound = true;

      std::string rule = paramSpec.get<std::string>("Rule Type");
      const Teuchos::ParameterList& ruleSpec = paramSpec.sublist(rule);

      materialIndices = ruleSpec.get<Teuchos::Array<int> >("Material Indices");
      mixtureTopologyIndices = ruleSpec.get<Teuchos::Array<int> >("Topology Indices");
      mixtureFunctionIndices = ruleSpec.get<Teuchos::Array<int> >("Function Indices");
    }
  }
  TEUCHOS_TEST_FOR_EXCEPTION( 
    !paramFound, Teuchos::Exceptions::InvalidParameter, std::endl <<
    "Error in ATO::OptimizationProblem setup:  " << std::endl <<
    "  Requested parameter (" << paramName << ") not found." << std::endl);
}

/******************************************************************************/
double 
ATO::VolumeMeasure::
Evaluate(const Teuchos::Array<double>& pVals,
         Teuchos::Array<Teuchos::RCP<Topology> >& topologies)
/******************************************************************************/
{
  int matIdx = materialTopologyIndex;
  int fncIdx = materialFunctionIndex;
  return topologies[matIdx]->Penalize(fncIdx, pVals[matIdx]);
}

/******************************************************************************/
double 
ATO::TopologyWeightedIntegral_Material::
Evaluate(const Teuchos::Array<double>& pVals,
         Teuchos::Array<Teuchos::RCP<Topology> >& topologies)
/******************************************************************************/
{
  int matIdx = materialTopologyIndex;
  int fncIdx = materialFunctionIndex;
  return parameterValue*topologies[matIdx]->Penalize(fncIdx, pVals[matIdx]);
}

/******************************************************************************/
double 
ATO::TopologyWeightedIntegral_Mixture::
Evaluate(const Teuchos::Array<double>& pVals,
         Teuchos::Array<Teuchos::RCP<Topology> >& topologies)
/******************************************************************************/
{
  
  double mixtureValue = 0.0;
  double unityRemainder = 1.0;
  int ntopos = mixtureTopologyIndices.size();
  for(int i=0; i<ntopos; i++){
    int matIdx = materialIndices[i];
    int topoIdx = mixtureTopologyIndices[i];
    int fncIdx = mixtureFunctionIndices[i];
    double topoVal = pVals[topoIdx];
    double pVal = topologies[topoIdx]->Penalize(fncIdx,topoVal);
    unityRemainder -= pVal;
    mixtureValue += pVal*parameterValues[matIdx];
  }
  
  int lastMatIndex = materialIndices[ntopos];
  mixtureValue += unityRemainder*parameterValues[lastMatIndex];

  int topoIdx = materialTopologyIndex;
  int fncIdx = materialFunctionIndex;
  mixtureValue *= topologies[topoIdx]->Penalize(fncIdx, pVals[topoIdx]);

  return mixtureValue;

}

/******************************************************************************/
void
ATO::VolumeMeasure::
Gradient(const Teuchos::Array<double>& pVals,
         Teuchos::Array<Teuchos::RCP<Topology> >& topologies,
         Teuchos::Array<double>& outVals)
/******************************************************************************/
{
  int n = outVals.size();
  for(int i=0; i<n; i++) outVals[i]=0.0;
  
  int topoIdx = materialTopologyIndex;
  int fncIdx = materialFunctionIndex;
  outVals[topoIdx] = topologies[topoIdx]->dPenalize(fncIdx, pVals[topoIdx]);
}

/******************************************************************************/
void
ATO::TopologyWeightedIntegral_Material::
Gradient(const Teuchos::Array<double>& pVals,
         Teuchos::Array<Teuchos::RCP<Topology> >& topologies,
         Teuchos::Array<double>& outVals)
/******************************************************************************/
{
  int n = outVals.size();
  for(int i=0; i<n; i++) outVals[i]=0.0;
  
  int matIdx = materialTopologyIndex;
  int fncIdx = materialFunctionIndex;
  outVals[matIdx] = parameterValue * topologies[matIdx]->dPenalize(fncIdx, pVals[matIdx]);
}

/******************************************************************************/
void
ATO::TopologyWeightedIntegral_Mixture::
Gradient(const Teuchos::Array<double>& pVals,
         Teuchos::Array<Teuchos::RCP<Topology> >& topologies,
         Teuchos::Array<double>& outVals)
/******************************************************************************/
{
  int n = outVals.size();
  for(int i=0; i<n; i++) outVals[i]=0.0;
  
  double mixtureValue = 0.0;
  double unityRemainder = 1.0;
  int ntopos = mixtureTopologyIndices.size();
  for(int i=0; i<ntopos; i++){
    int imatIdx = materialIndices[i];
    int itopoIdx = mixtureTopologyIndices[i];
    int ifncIdx = mixtureFunctionIndices[i];
    double topoVal = pVals[itopoIdx];
    double pVal = topologies[itopoIdx]->Penalize(ifncIdx,topoVal);
    unityRemainder -= pVal;
    mixtureValue += pVal*parameterValues[imatIdx];
  }
  
  int lastMatIndex = materialIndices[ntopos];
  mixtureValue += unityRemainder*parameterValues[lastMatIndex];

  int topoIdx = materialTopologyIndex;
  int fncIdx = materialFunctionIndex;
  outVals[topoIdx] = mixtureValue * topologies[topoIdx]->dPenalize(fncIdx, pVals[topoIdx]);

  for(int i=0; i<ntopos; i++){
    int itopoIdx = mixtureTopologyIndices[i];
    int ifncIdx = mixtureFunctionIndices[i];
    int imatIdx = materialIndices[i];
    double dRi = topologies[itopoIdx]->dPenalize(ifncIdx, pVals[itopoIdx]);
    double R0 = topologies[topoIdx]->Penalize(fncIdx, pVals[topoIdx]);
    outVals[itopoIdx] = R0*dRi*(parameterValues[imatIdx] - parameterValues[lastMatIndex]);
  }

}
