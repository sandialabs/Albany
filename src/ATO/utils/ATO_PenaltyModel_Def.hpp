#include "Albany_Utils.hpp"

/******************************************************************************/
template<typename N>
ATO::PenaltyModel<N>::
PenaltyModel(Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl)
/******************************************************************************/
{
  Teuchos::ParameterList* responseParams = p.get<Teuchos::ParameterList*>("Parameter List");
  std::string gfLayout = responseParams->get<std::string>("Gradient Field Layout");
  
  Teuchos::RCP<PHX::DataLayout> layout;
  if(gfLayout == "QP Tensor3"){ layout = dl->qp_tensor3; rank = 3; }
  else
  if(gfLayout == "QP Tensor"){ layout = dl->qp_tensor; rank = 2; }
  else
  if(gfLayout == "QP Vector"){ layout = dl->qp_vector; rank = 1; }
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                               std::endl <<
                               "Error!  Unknown Gradient Field Layout " << gfLayout <<
                               "!" << std::endl << "Options are (QP Tensor3, QP Tensor, QP Vector)" <<
                               std::endl);

  PHX::MDField<N> _gradX(responseParams->get<std::string>("Gradient Field Name"), layout);
  gradX = _gradX;

  std::vector<int> dims;
  gradX.dimensions(dims);
  numDims = dims[2];
}

/******************************************************************************/
template<typename N>
ATO::PenaltyMixture<N>::
PenaltyMixture(Teuchos::ParameterList& blockParams,
               Teuchos::ParameterList& p, 
               const Teuchos::RCP<Albany::Layouts>& dl) :
  PenaltyModel<N>(p, dl)
/******************************************************************************/
{
  Teuchos::ParameterList* responseParams = p.get<Teuchos::ParameterList*>("Parameter List");
  std::string wcLayout = responseParams->get<std::string>("Work Conjugate Layout");
  
  Teuchos::RCP<PHX::DataLayout> layout;
  if(wcLayout == "QP Tensor3") layout = dl->qp_tensor3;
  else
  if(wcLayout == "QP Tensor") layout = dl->qp_tensor;
  else
  if(wcLayout == "QP Vector") layout = dl->qp_vector;
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                               std::endl <<
                               "Error!  Unknown Work Conjugate Layout " << wcLayout <<
                               "!" << std::endl << "Options are (QP Tensor3, QP Tensor, QP Vector)" <<
                               std::endl);

  std::string workConjBaseName = responseParams->get<std::string>("Work Conjugate Name");

  Teuchos::ParameterList& mixtureParams = blockParams.sublist("Mixture");
  int nMats = mixtureParams.get<int>("Number of Materials");

  workConj.resize(nMats);
  for(int imat=0; imat<nMats; imat++){
    PHX::MDField<N> _workConj(Albany::strint(workConjBaseName,imat), layout);
    workConj[imat] = _workConj;
  }

  bool fieldFound = false;
  Teuchos::ParameterList& fieldsParams = blockParams.sublist("Mixture").sublist("Mixed Fields");
  int nFields = fieldsParams.get<int>("Number of Mixed Fields");
  for(int ifield=0; ifield<nFields; ifield++){
    Teuchos::ParameterList& 
      fieldParams = fieldsParams.sublist(Albany::strint("Mixed Field", ifield));
    std::string fieldName = fieldParams.get<std::string>("Field Name");
    if( fieldName != workConjBaseName ) continue;
    fieldFound = true;
    
    TEUCHOS_TEST_FOR_EXCEPTION(
      fieldParams.get<std::string>("Field Layout") != wcLayout,
      Teuchos::Exceptions::InvalidParameter,
      std::endl << "Error!  Field layout mismatch." << std::endl);

    Teuchos::ParameterList& 
      ruleParams = fieldParams.sublist(fieldParams.get<std::string>("Rule Type"));

    materialIndices = ruleParams.get<Teuchos::Array<int> >("Material Indices");
    mixtureTopologyIndices = ruleParams.get<Teuchos::Array<int> >("Topology Indices");
    mixtureFunctionIndices = ruleParams.get<Teuchos::Array<int> >("Function Indices");
    
  }

  topologyIndex = responseParams->get<int>("Topology Index");
  functionIndex = responseParams->get<int>("Function Index");
  
}

/******************************************************************************/
template<typename N>
void ATO::PenaltyModel<N>::
getFieldDimensions(std::vector<int>& dims)
/******************************************************************************/
{
  gradX.dimensions(dims);
}

/******************************************************************************/
template<typename N>
void ATO::PenaltyMixture<N>::
getDependentFields(Teuchos::Array<PHX::MDField<N> >& depFields)
/******************************************************************************/
{
  int nWCs = workConj.size();
  depFields.resize(nWCs+1);
  for(int iwc=0; iwc<nWCs; iwc++)
    depFields[iwc] = workConj[iwc];

  depFields[nWCs] = gradX;
  
}
/******************************************************************************/
template<typename N>
void ATO::PenaltyMixture<N>::
getDependentFields(Teuchos::Array<PHX::MDField<N>* >& depFields)
/******************************************************************************/
{
  int nWCs = workConj.size();
  depFields.resize(nWCs+1);
  for(int iwc=0; iwc<nWCs; iwc++)
    depFields[iwc] = &workConj[iwc];

  depFields[nWCs] = &gradX;
  
}

/******************************************************************************/
template<typename N>
void ATO::PenaltyMaterial<N>::
getDependentFields(Teuchos::Array<PHX::MDField<N> >& depFields)
/******************************************************************************/
{
  depFields.resize(2);
  depFields[0] = workConj;
  depFields[1] = gradX;
}

/******************************************************************************/
template<typename N>
void ATO::PenaltyMaterial<N>::
getDependentFields(Teuchos::Array<PHX::MDField<N>* >& depFields)
/******************************************************************************/
{
  depFields.resize(2);
  depFields[0] = &workConj;
  depFields[1] = &gradX;
}

/******************************************************************************/
template<typename N>
ATO::PenaltyMaterial<N>::
PenaltyMaterial(Teuchos::ParameterList& blockParams,
               Teuchos::ParameterList& p,
               const Teuchos::RCP<Albany::Layouts>& dl) :
  PenaltyModel<N>(p, dl)
/******************************************************************************/
{
  Teuchos::ParameterList* responseParams = p.get<Teuchos::ParameterList*>("Parameter List");
  std::string wcLayout = responseParams->get<std::string>("Work Conjugate Layout");
  
  Teuchos::RCP<PHX::DataLayout> layout;
  if(wcLayout == "QP Tensor3") layout = dl->qp_tensor3;
  else
  if(wcLayout == "QP Tensor") layout = dl->qp_tensor;
  else
  if(wcLayout == "QP Vector") layout = dl->qp_vector;
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                               std::endl <<
                               "Error!  Unknown Work Conjugate Layout " << wcLayout <<
                               "!" << std::endl << "Options are (QP Tensor3, QP Tensor, QP Vector)" <<
                               std::endl);

  PHX::MDField<N> _workConj(responseParams->get<std::string>("Work Conjugate Name"), layout);
  workConj = _workConj;

  topologyIndex = responseParams->get<int>("Topology Index");
  functionIndex = responseParams->get<int>("Function Index");
  
}

/******************************************************************************/
template<typename N>
void ATO::PenaltyMixture<N>::
Evaluate(Teuchos::Array<N>& topoVals, Teuchos::RCP<TopologyArray>& topologies,
         int cell, int qp, N& response, Teuchos::Array<N>& dResponse)
/******************************************************************************/
{
  response = 0.0;
  N unityRemainder = 1.0;

  int nTopos = dResponse.size();
  for(int itopo=0; itopo<nTopos; itopo++)
    dResponse[itopo]=0.0;

  int nMats = mixtureTopologyIndices.size();
  int lastMatIndex = materialIndices[nMats];
  N lastMatdw = 0.0;
  if( rank == 1 ){
    for(int i=0; i<numDims; i++){
      lastMatdw += gradX(cell,qp,i)*workConj[lastMatIndex](cell,qp,i)/2.0;
    }
  } else
  if( rank == 2 ){
    for(int i=0; i<numDims; i++){
      for(int j=0; j<numDims; j++){
        lastMatdw += gradX(cell,qp,i,j)*workConj[lastMatIndex](cell,qp,i,j)/2.0;
      }
    }
  } else
  if( rank == 3 ){
    for(int i=0; i<numDims; i++){
      for(int j=0; j<numDims; j++){
        for(int k=0; k<numDims; k++){
          lastMatdw += gradX(cell,qp,i,j,k)*workConj[lastMatIndex](cell,qp,i,j,k)/2.0;
        }
      }
    }
  }
  N topoP = (*topologies)[topologyIndex]->Penalize(functionIndex, topoVals[topologyIndex]);
  for(int imat=0; imat<nMats; imat++){
    int matIdx = materialIndices[imat];
    int topoIdx = mixtureTopologyIndices[imat];
    int fncIdx = mixtureFunctionIndices[imat];
    N P = (*topologies)[topoIdx]->Penalize(fncIdx, topoVals[topoIdx]);
    N dP = (*topologies)[topoIdx]->dPenalize(fncIdx, topoVals[topoIdx]);
    unityRemainder -= P;
    N dw = 0.0;
    if( rank == 1 ){
      for(int i=0; i<numDims; i++){
        dw += gradX(cell,qp,i)*workConj[matIdx](cell,qp,i)/2.0;
      }
    } else
    if( rank == 2 ){
      for(int i=0; i<numDims; i++){
        for(int j=0; j<numDims; j++){
          dw += gradX(cell,qp,i,j)*workConj[matIdx](cell,qp,i,j)/2.0;
        }
      }
    } else
    if( rank == 3 ){
      for(int i=0; i<numDims; i++){
        for(int j=0; j<numDims; j++){
          for(int k=0; k<numDims; k++){
            dw += gradX(cell,qp,i,j,k)*workConj[matIdx](cell,qp,i,j,k)/2.0;
          }
        }
      }
    }
    response += P*dw;
    dResponse[topoIdx] = topoP*(dP*dw + (1.0-dP)*lastMatdw);
  }

  response += unityRemainder*lastMatdw;

  dResponse[topologyIndex] = 
    response * (*topologies)[topologyIndex]->dPenalize(functionIndex, topoVals[topologyIndex]);

  response *= (*topologies)[topologyIndex]->Penalize(functionIndex,topoVals[topologyIndex]);

}

/******************************************************************************/
template<typename N>
void ATO::PenaltyMaterial<N>::
Evaluate(Teuchos::Array<N>& topoVals, Teuchos::RCP<TopologyArray>& topologies,
         int cell, int qp, N& response, Teuchos::Array<N>& dResponse)
/******************************************************************************/
{
  response = 0.0;

  if( rank == 1 ){
    for(int i=0; i<numDims; i++)
      response += gradX(cell,qp,i)*workConj(cell,qp,i)/2.0;
  } else
  if( rank == 2 ){
    for(int i=0; i<numDims; i++)
      for(int j=0; j<numDims; j++)
        response += gradX(cell,qp,i,j)*workConj(cell,qp,i,j)/2.0;
  } else
  if( rank == 3 ){
    for(int i=0; i<numDims; i++)
      for(int j=0; j<numDims; j++)
        for(int k=0; k<numDims; k++)
          response += gradX(cell,qp,i,j,k)*workConj(cell,qp,i,j,k)/2.0;
  }
  
  dResponse[topologyIndex] = 
    response * (*topologies)[topologyIndex]->dPenalize(functionIndex, topoVals[topologyIndex]);

  response *= (*topologies)[topologyIndex]->Penalize(functionIndex,topoVals[topologyIndex]);
}



/******************************************************************************/
template<typename N>
Teuchos::RCP<ATO::PenaltyModel<N> > 
ATO::PenaltyModelFactory<N>::create(Teuchos::ParameterList& p,
                                    const Teuchos::RCP<Albany::Layouts>& dl,
                                    std::string elementBlockName)
/******************************************************************************/
{

  Teuchos::RCP<Teuchos::ParameterList> paramsFromProblem =
    p.get< Teuchos::RCP<Teuchos::ParameterList> >("Parameters From Problem");

  Teuchos::ParameterList& configParams = paramsFromProblem->sublist("Configuration");
  Teuchos::ParameterList& blocksParams = configParams.sublist("Element Blocks");
  int nblocks = blocksParams.get<int>("Number of Element Blocks");
  bool blockFound = false;
  for(int ib=0; ib<nblocks; ib++){
    Teuchos::ParameterList& blockParams = blocksParams.sublist(Albany::strint("Element Block", ib));
    std::string blockName = blockParams.get<std::string>("Name");
    if( blockName != elementBlockName) continue;
    blockFound = true;

   if( blockParams.isSublist("Material") )
     return Teuchos::rcp( new ATO::PenaltyMaterial<N>( blockParams, p, dl) );
   else
   if( blockParams.isSublist("Mixture") )
     return Teuchos::rcp( new ATO::PenaltyMixture<N>( blockParams, p, dl) );
   else
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter, std:: endl <<
      "Neither 'Material' spec nor 'Mixture' spec found." << std::endl);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(blockFound == false, Teuchos::Exceptions::InvalidParameter, std:: endl <<
    "Block spec not found." << std::endl);

}
