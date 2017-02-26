//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace ATO {

//**********************************************************************
template<typename EvalT, typename Traits>
TopologyWeighting<EvalT, Traits>::
TopologyWeighting(const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Albany::Layouts>& dl) :
BF(p.get<std::string> ("BF Name"), dl->node_qp_scalar)
{

  topology = p.get<Teuchos::RCP<Topology> >("Topology");
  topoName = topology->getName();
  functionIndex = p.get<int>("Function Index");

  std::string strLayout = p.get<std::string>("Variable Layout");
 
  Teuchos::RCP<PHX::DataLayout> layout;
  if(strLayout == "QP Tensor3") layout = dl->qp_tensor3;
  else
  if(strLayout == "QP Tensor") layout = dl->qp_tensor;
  else
  if(strLayout == "QP Vector") layout = dl->qp_vector;
  else
  if(strLayout == "QP Scalar") layout = dl->qp_scalar;
  else
    TEUCHOS_TEST_FOR_EXCEPTION(
      true, Teuchos::Exceptions::InvalidParameter,
      std::endl << "Error!  Unknown variable layout " << strLayout << "!" << 
      std::endl << "Options are (QP Tensor3, QP Tensor, QP Vector, QP Scalar)" <<
      std::endl);

  unWeightedVar = decltype(unWeightedVar)(p.get<std::string>("Unweighted Variable Name"), layout);
  weightedVar = decltype(weightedVar)(p.get<std::string>("Weighted Variable Name"), layout);

  this->addDependentField(unWeightedVar);
  this->addDependentField(BF);
  this->addEvaluatedField(weightedVar);

  this->setName("Topology Weighting"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void TopologyWeighting<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(unWeightedVar,fm);
  this->utils.setFieldData(weightedVar,fm);
  this->utils.setFieldData(BF,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void TopologyWeighting<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  std::vector<int> dims;
  unWeightedVar.dimensions(dims);
  int size = dims.size();

  Albany::MDArray topo = (*workset.stateArrayPtr)[this->topoName];

  int numCells = dims[0];
  int numQPs   = dims[1];
  int numDims  = 0.0;
  if( size > 2 ) numDims = dims[2];
  int numNodes = topo.dimension(1);

  if( size == 2 ){
    for(int cell=0; cell<numCells; cell++){
      for(int qp=0; qp<numQPs; qp++){
        double topoVal = 0.0;
        for(int node=0; node<numNodes; node++)
          topoVal += topo(cell,node)*BF(cell,node,qp);
        ScalarT P = topology->Penalize(functionIndex,topoVal);
        weightedVar(cell,qp) = P*unWeightedVar(cell,qp);
      }
    }
  } else
  if( size == 3 ){
    for(int cell=0; cell<numCells; cell++){
      for(int qp=0; qp<numQPs; qp++){
        double topoVal = 0.0;
        for(int node=0; node<numNodes; node++)
          topoVal += topo(cell,node)*BF(cell,node,qp);
        ScalarT P = topology->Penalize(functionIndex,topoVal);
        for(int i=0; i<numDims; i++)
          weightedVar(cell,qp,i) = P*unWeightedVar(cell,qp,i);
      }
    }
  } else
  if( size == 4 ){
    for(int cell=0; cell<numCells; cell++){
      for(int qp=0; qp<numQPs; qp++){
        double topoVal = 0.0;
        for(int node=0; node<numNodes; node++)
          topoVal += topo(cell,node)*BF(cell,node,qp);
        ScalarT P = topology->Penalize(functionIndex,topoVal);
        for(int i=0; i<numDims; i++)
          for(int j=0; j<numDims; j++)
            weightedVar(cell,qp,i,j) = P*unWeightedVar(cell,qp,i,j);
      }
    }
  } else
  if( size == 5 ){
    for(int cell=0; cell<numCells; cell++){
      for(int qp=0; qp<numQPs; qp++){
        double topoVal = 0.0;
        for(int node=0; node<numNodes; node++)
          topoVal += topo(cell,node)*BF(cell,node,qp);
        ScalarT P = topology->Penalize(functionIndex,topoVal);
        for(int i=0; i<numDims; i++)
          for(int j=0; j<numDims; j++)
            for(int k=0; k<numDims; k++)
              weightedVar(cell,qp,i,j,k) = P*unWeightedVar(cell,qp,i,j,k);
      }
    }
  } else {
     TEUCHOS_TEST_FOR_EXCEPTION(size<2||size>5, Teuchos::Exceptions::InvalidParameter,
       "Unexpected array dimensions in TopologyWeighting:" << size << std::endl);
  }
}

//**********************************************************************
}

