//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Albany_StateManager.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace ATO {
template<typename EvalT, typename Traits>
ComputeBasisFunctions<EvalT, Traits>::
ComputeBasisFunctions(const Teuchos::ParameterList& p,
                      const Teuchos::RCP<Albany::Layouts>& dl,
                      const Albany::MeshSpecsStruct* meshSpecs) :
  coordVec      (p.get<std::string>  ("Coordinate Vector Name"), dl->vertices_vector ),
  cubature      (p.get<Teuchos::RCP <Cogent::Integrator> >("Cubature")),
  weighted_measure (p.get<std::string>  ("Weights Name"), dl->qp_scalar ),
  jacobian_det (p.get<std::string>  ("Jacobian Det Name"), dl->qp_scalar ),
  BF            (p.get<std::string>  ("BF Name"), dl->node_qp_scalar),
  wBF           (p.get<std::string>  ("Weighted BF Name"), dl->node_qp_scalar),
  GradBF        (p.get<std::string>  ("Gradient BF Name"), dl->node_qp_gradient),
  wGradBF       (p.get<std::string>  ("Weighted Gradient BF Name"), dl->node_qp_gradient)
{
  elementBlockName = meshSpecs->ebName;
  
  m_isStatic = false;
  if(p.isType<bool>("Static Topology")) m_isStatic = p.get<bool>("Static Topology");

  if(m_isStatic){
    Albany::StateManager* stateMgr = p.get<Albany::StateManager*>("State Manager Ptr");
    stateMgr->registerStateVariable("Gauss Weights", dl->qp_scalar, dl->dummy, 
                                    "all", "scalar", 0.0, false, false);

    // this should be registered as a workset_scalar, but the workset stateArrays all
    // point to a single value.  that is, workset_scalar behaves more like 'global_scalar'.
    // After some snooping around in Albany_StateManager, I can't figure out how to
    // register workset_scalars so that they actually store a separate value for each
    // workset, not a single value for all worksets.  
    stateMgr->registerStateVariable("isSet", dl->cell_scalar, dl->dummy, 
                                    "all", "scalar", 0.0, false, false);
  }


  this->addDependentField(coordVec);
  this->addEvaluatedField(weighted_measure);
  this->addEvaluatedField(jacobian_det);
  this->addEvaluatedField(BF);
  this->addEvaluatedField(wBF);
  this->addEvaluatedField(GradBF);
  this->addEvaluatedField(wGradBF);

  // Get Dimensions
  std::vector<PHX::DataLayout::size_type> dim;
  dl->node_qp_gradient->dimensions(dim);

  numCells = dim[0];
  numNodes = dim[1];
  numQPs   = dim[2];
  numDims  = dim[3];

  std::vector<PHX::DataLayout::size_type> dims;
  dl->vertices_vector->dimensions(dims);
  numVertices = dims[1];

  topoNames = cubature->getFieldNames();
  numTopos = topoNames.size();
  topoVals = Kokkos::DynRankView<RealType, PHX::Device>("XXX",numNodes,numTopos);
  coordVals = Kokkos::DynRankView<RealType, PHX::Device>("XXX",numNodes,numDims);

  // Allocate Temporary FieldContainers
  val_at_cub_points = Kokkos::DynRankView<RealType, PHX::Device>("XXX", numNodes, numQPs);
  grad_at_cub_points = Kokkos::DynRankView<RealType, PHX::Device>("XXX", numNodes, numQPs, numDims);
  refPoints = Kokkos::DynRankView<RealType, PHX::Device>("XXX", numQPs, numDims);
  weights = Kokkos::DynRankView<RealType, PHX::Device>("XXX", numQPs);

  cubature->getStandardPoints(refPoints);

  cubature->getBasis()->getValues(val_at_cub_points, refPoints, Intrepid2::OPERATOR_VALUE);
  cubature->getBasis()->getValues(grad_at_cub_points, refPoints, Intrepid2::OPERATOR_GRAD);

  this->setName("ComputeBasisFunctions"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ComputeBasisFunctions<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(coordVec,fm);
  this->utils.setFieldData(weighted_measure,fm);
  this->utils.setFieldData(jacobian_det,fm);
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(GradBF,fm);
  this->utils.setFieldData(wGradBF,fm);

  jacobian = Kokkos::createDynRankView(jacobian_det.get_view(),"XXX", numCells, numQPs, numDims, numDims);
  jacobian_inv = Kokkos::createDynRankView(jacobian_det.get_view(),"XXX", numCells, numQPs, numDims, numDims);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ComputeBasisFunctions<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

  if( elementBlockName != workset.EBName ) return;

  /** The allocated size of the Field Containers must currently
    * match the full workset size of the allocated PHX Fields,
    * this is the size that is used in the computation. There is
    * wasted effort computing on zeroes for the padding on the
    * final workset. Ideally, these are size numCells.
  //int numCells = workset.numCells;
    */

  typedef typename Intrepid2::CellTools<PHX::Device>   ICT;
  typedef Intrepid2::FunctionSpaceTools<PHX::Device>   IFST;

  ICT::setJacobian(jacobian, refPoints, coordVec.get_view(), cubature->getBasis());
  ICT::setJacobianInv (jacobian_inv, jacobian);
  ICT::setJacobianDet (jacobian_det.get_view(), jacobian);

  bool isSet = false;
  Albany::MDArray savedWeights;
  if(m_isStatic){
    savedWeights = (*workset.stateArrayPtr)["Gauss Weights"];
    Albany::MDArray wsSet = (*workset.stateArrayPtr)["isSet"];
    isSet = (wsSet(0,0) == 0) ? false : true;
  }

  if(m_isStatic && isSet){
    for(int cell=0; cell<workset.numCells; cell++){
      for(int qp=0; qp<numQPs; qp++)
        weighted_measure(cell, qp) = savedWeights(cell,qp);
    }
  } else {

    for(int cell=0; cell<workset.numCells; cell++){
      for(int node=0; node<numNodes; node++)
        for(int dim=0; dim<numDims; dim++)
          coordVals(node,dim) = coordVec(cell,node,dim);
  
      Teuchos::Array<Albany::MDArray> topo(numTopos);
      for(int itopo=0; itopo<numTopos; itopo++){
        topo[itopo] = (*workset.stateArrayPtr)[topoNames[itopo]];
      }
  
      for(int itopo=0; itopo<numTopos; itopo++)
        for(int node=0; node<numNodes; node++)
          topoVals(node,itopo) = topo[itopo](cell,node);
  
      cubature->getCubatureWeights(weights, topoVals, coordVals);

      for(int qp=0; qp<numQPs; qp++)
        weighted_measure(cell, qp) = weights(qp);

      if(m_isStatic && !isSet){
        for(int qp=0; qp<numQPs; qp++)
          savedWeights(cell,qp) = weighted_measure(cell, qp);
      }
    }
    (*workset.stateArrayPtr)["isSet"](0,0) = 1;
  }

  IFST::HGRADtransformVALUE(BF.get_view(), val_at_cub_points);
  IFST::multiplyMeasure    (wBF.get_view(), weighted_measure.get_view(), BF.get_view());
  IFST::HGRADtransformGRAD (GradBF.get_view(), jacobian_inv, grad_at_cub_points);
  IFST::multiplyMeasure    (wGradBF.get_view(), weighted_measure.get_view(), GradBF.get_view());
}

//**********************************************************************
}
