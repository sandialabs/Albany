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
AddForce<EvalT, Traits>::
AddForce(const Teuchos::ParameterList& p) :
  add_force   (p.get<std::string>                   ("Force Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Force Data Layout") ),
  inResidual  (p.get<std::string>                   ("In Residual Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node Vector Data Layout") ),
  outResidual (p.get<std::string>                   ("Out Residual Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node Vector Data Layout") )
{
  this->addDependentField(add_force);
  this->addDependentField(inResidual);

  this->addEvaluatedField(outResidual);

  this->setName("AddForce"+PHX::typeAsString<EvalT>());

  std::vector<PHX::Device::size_type> dims;
  add_force.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numDims  = dims[2];
  
  if(p.isType<std::string>("Weighted BF Name")){
    w_bf = PHX::MDField<MeshScalarT,Cell,Node,QuadPoint>(p.get<std::string>("Weighted BF Name"),
                                    p.get<Teuchos::RCP<PHX::DataLayout> >("Weighted BF Data Layout") );
    this->addDependentField(w_bf);
    projectFromQPs = true;
    w_bf.fieldTag().dataLayout().dimensions(dims);
    numQPs   = dims[2];
  } else 
    projectFromQPs = false;


  if(p.isType<bool>("Negative"))
    negative = p.get<bool>("Negative");
  else
    negative = false;
  
}

//**********************************************************************
template<typename EvalT, typename Traits>
void AddForce<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(add_force,fm);
  this->utils.setFieldData(inResidual,fm);
  this->utils.setFieldData(outResidual,fm);
  if( projectFromQPs )
    this->utils.setFieldData(w_bf,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void AddForce<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  if( projectFromQPs ){
    if( negative ){
      for (std::size_t cell=0; cell < workset.numCells; ++cell) {
        for (std::size_t node=0; node < numNodes; ++node) {
          for (std::size_t qp=0; qp < numQPs; ++qp)
            for (std::size_t dim=0; dim<numDims; dim++)
              outResidual(cell,node,dim) = inResidual(cell,node,dim) -
                w_bf(cell, node, qp) * add_force(cell, qp, dim);
        }     
      }
    } else {
      for (std::size_t cell=0; cell < workset.numCells; ++cell) {
        for (std::size_t node=0; node < numNodes; ++node) {
          for (std::size_t qp=0; qp < numQPs; ++qp)
            for (std::size_t dim=0; dim<numDims; dim++)
              outResidual(cell,node,dim) = inResidual(cell,node,dim) +
                w_bf(cell, node, qp) * add_force(cell, qp, dim);
        }   
      }
    }
  } else {
    if( negative ){
      for (std::size_t cell=0; cell < workset.numCells; ++cell) {
        for (std::size_t node=0; node < numNodes; ++node) {
          for (std::size_t dim=0; dim<numDims; dim++)
            outResidual(cell,node,dim) = inResidual(cell,node,dim) -
              add_force(cell, node, dim);
        }     
      }
    } else {
      for (std::size_t cell=0; cell < workset.numCells; ++cell) {
        for (std::size_t node=0; node < numNodes; ++node) {
          for (std::size_t dim=0; dim<numDims; dim++)
            outResidual(cell,node,dim) = inResidual(cell,node,dim) +
              add_force(cell, node, dim);
        }   
      }
    }
  }
}

//**********************************************************************
}

