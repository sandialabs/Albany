//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace ATO {

//**********************************************************************
template<typename EvalT, typename Traits>
Mixture<EvalT, Traits>::
Mixture(const Teuchos::ParameterList& p,
        const Teuchos::RCP<Albany::Layouts>& dl) :
BF(p.get<std::string> ("BF Name"), dl->node_qp_scalar)
{

  Teuchos::RCP<TopologyArray> topologyArray = p.get< Teuchos::RCP<TopologyArray> >("Topologies");

  Teuchos::Array<int> topologyIndices = p.get< Teuchos::Array<int> >("Topology Indices");

  int nTopos = topologyIndices.size();
  topologies.resize(nTopos);
  for(int iTopo=0; iTopo<nTopos; iTopo++){
    topologies[iTopo] = (*topologyArray)[topologyIndices[iTopo]];
  }

  functionIndices = p.get< Teuchos::Array<int> >("Function Indices");

  std::string strLayout = p.get<std::string>("Field Layout");
 
  Teuchos::RCP<PHX::DataLayout> layout;
  if(strLayout == "QP Tensor3") layout = dl->qp_tensor3;
  else
  if(strLayout == "QP Tensor") layout = dl->qp_tensor;
  else
  if(strLayout == "QP Vector") layout = dl->qp_vector;
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                               std::endl <<
                               "Error!  Unknown variable layout " << strLayout <<
                               "!" << std::endl << "Options are (QP Tensor, QP Vector)" <<
                               std::endl);

  Teuchos::Array<std::string> inVarNames = p.get<Teuchos::Array<std::string> >("Constituent Variable Names");
  int nMats = inVarNames.size();
  constituentVar.resize(nMats);
  for(int i=0; i<nMats; i++){
    PHX::MDField<ScalarT> _constituentVar(inVarNames[i], layout);
    constituentVar[i] = _constituentVar;
    this->addDependentField(constituentVar[i]);
  }
  PHX::MDField<ScalarT> _mixtureVar(p.get<std::string>("Mixture Variable Name"), layout);
  mixtureVar = _mixtureVar;
  this->addEvaluatedField(mixtureVar);

  // Pull out numQPs and numDims from a Layout
  std::vector<PHX::Device::size_type> dims;
  layout->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  this->addDependentField(BF);

  this->setName("Topology Mixture"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void Mixture<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  int nMats = constituentVar.size();
  for(int i=0; i<nMats; i++) 
    this->utils.setFieldData(constituentVar[i],fm);
  this->utils.setFieldData(mixtureVar,fm);
  this->utils.setFieldData(BF,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void Mixture<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  std::vector<int> dims;
  constituentVar[0].dimensions(dims);
  int size = dims.size();

  int nTopos = topologies.size();
  std::vector<Albany::MDArray> topos(nTopos);
  for(int i=0; i<nTopos; i++)
   topos[i] = (*workset.stateArrayPtr)[topologies[i]->getName()];

  int numCells = dims[0];
  int numQPs   = dims[1];
  int numDims  = dims[2];
  int numNodes = topos[0].dimension(1);

  ScalarT unityRemainder(0);
  std::vector<ScalarT> P(nTopos);
  int nMats = constituentVar.size();

  int firstMat = 0;
  int lastMat = nMats-1;

  if( size == 3 ){
    for(int cell=0; cell<numCells; cell++){
      for(int qp=0; qp<numQPs; qp++){

        for(int i=0; i<nTopos; i++){
          ScalarT topoVal = 0.0;
          for(int node=0; node<numNodes; node++)
            topoVal += topos[i](cell,node)*BF(cell,node,qp);
          P[i] = topologies[i]->Penalize(functionIndices[i],topoVal);
        }

        for(int i=0; i<numDims; i++)
          mixtureVar(cell,qp,i) = P[firstMat]*constituentVar[firstMat](cell,qp,i);

        for(int k=1; k<nTopos; k++)
          for(int i=0; i<numDims; i++)
            mixtureVar(cell,qp,i) += P[k]*constituentVar[k](cell,qp,i);

        unityRemainder = 1.0;
        for(int k=0; k<nTopos; k++)
            unityRemainder -= P[k];
        for(int i=0; i<numDims; i++)
           mixtureVar(cell,qp,i) += unityRemainder*constituentVar[lastMat](cell,qp,i);

      }
    }
  } else
  if( size == 4 ){
    for(int cell=0; cell<numCells; cell++){
      for(int qp=0; qp<numQPs; qp++){

        for(int i=0; i<nTopos; i++){
          ScalarT topoVal = 0.0;
          for(int node=0; node<numNodes; node++)
            topoVal += topos[i](cell,node)*BF(cell,node,qp);
          P[i] = topologies[i]->Penalize(functionIndices[i],topoVal);
        }

        for(int i=0; i<numDims; i++)
          for(int j=0; j<numDims; j++)
            mixtureVar(cell,qp,i,j) = P[firstMat]*constituentVar[firstMat](cell,qp,i,j);

        for(int k=1; k<nTopos; k++)
          for(int i=0; i<numDims; i++)
            for(int j=0; j<numDims; j++)
              mixtureVar(cell,qp,i,j) += P[k]*constituentVar[k](cell,qp,i,j);

        unityRemainder = 1.0;
        for(int k=0; k<nTopos; k++)
            unityRemainder -= P[k];
        for(int i=0; i<numDims; i++)
          for(int j=0; j<numDims; j++)
            mixtureVar(cell,qp,i,j) += unityRemainder*constituentVar[lastMat](cell,qp,i,j);

      }
    }
  } else
  if( size == 5 ){
    for(int cell=0; cell<numCells; cell++){
      for(int qp=0; qp<numQPs; qp++){

        for(int i=0; i<nTopos; i++){
          ScalarT topoVal = 0.0;
          for(int node=0; node<numNodes; node++)
            topoVal += topos[i](cell,node)*BF(cell,node,qp);
          P[i] = topologies[i]->Penalize(functionIndices[i],topoVal);
        }

        for(int i=0; i<numDims; i++)
          for(int j=0; j<numDims; j++)
            for(int k=0; k<numDims; k++)
              mixtureVar(cell,qp,i,j,k) = P[firstMat]*constituentVar[firstMat](cell,qp,i,j,k);

        for(int m=1; m<nTopos; m++)
          for(int i=0; i<numDims; i++)
            for(int j=0; j<numDims; j++)
              for(int k=0; k<numDims; k++)
                mixtureVar(cell,qp,i,j,k) += P[m]*constituentVar[m](cell,qp,i,j,k);

        unityRemainder = 1.0;
        for(int m=0; m<nTopos; m++)
            unityRemainder -= P[m];
        for(int i=0; i<numDims; i++)
          for(int j=0; j<numDims; j++)
            for(int k=0; k<numDims; k++)
              mixtureVar(cell,qp,i,j,k) += unityRemainder*constituentVar[lastMat](cell,qp,i,j,k);

      }
    }
  } else {
     TEUCHOS_TEST_FOR_EXCEPTION(size<3||size>5, Teuchos::Exceptions::InvalidParameter,
       "Unexpected array dimensions in Mixture:" << size << std::endl);
  }
}

//**********************************************************************
}

