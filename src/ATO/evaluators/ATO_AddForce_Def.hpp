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
AddVector<EvalT, Traits>::
AddVector(const Teuchos::ParameterList& p) :
  add_vector  (p.get<std::string>                   ("Vector Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Vector Data Layout") ),
  outResidual (p.get<std::string>                   ("Out Residual Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node Vector Data Layout") )
{
  this->addDependentField(add_vector);

  this->addEvaluatedField(outResidual);

  this->setName("AddVector"+PHX::typeAsString<EvalT>());


  if(p.isType<std::string>("In Residual Name")){
    inResidual = decltype(inResidual)(p.get<std::string>("In Residual Name"),
                                    p.get<Teuchos::RCP<PHX::DataLayout> >("Node Vector Data Layout") );
    this->addDependentField(inResidual);
    plusEquals = true;
  } else 
    plusEquals = false;
  
  if(p.isType<std::string>("Weighted BF Name")){
    w_bf = decltype(w_bf)(p.get<std::string>("Weighted BF Name"),
                                    p.get<Teuchos::RCP<PHX::DataLayout> >("Weighted BF Data Layout") );
    this->addDependentField(w_bf);
    projectFromQPs = true;
  } else 
    projectFromQPs = false;


  if(p.isType<bool>("Negative"))
    negative = p.get<bool>("Negative");
  else
    negative = false;
  
}

//**********************************************************************
template<typename EvalT, typename Traits>
void AddVector<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  std::vector<PHX::Device::size_type> dims;
  this->utils.setFieldData(add_vector,fm);
  this->utils.setFieldData(outResidual,fm);
  outResidual.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numDims  = dims[2];
  if( projectFromQPs ) {
    this->utils.setFieldData(w_bf,fm);
    w_bf.fieldTag().dataLayout().dimensions(dims);
    numQPs = dims[2];
  }
  if( plusEquals )
    this->utils.setFieldData(inResidual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void AddVector<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  if( projectFromQPs ){
    if( negative ){
      for (std::size_t cell=0; cell < workset.numCells; ++cell) {
        for (std::size_t node=0; node < numNodes; ++node) {
          for (std::size_t dim=0; dim<numDims; dim++){
            if( plusEquals ) outResidual(cell,node,dim) = inResidual(cell,node,dim);
            else outResidual(cell,node,dim) = 0.0;
            for (std::size_t qp=0; qp < numQPs; ++qp)
              outResidual(cell,node,dim) -= w_bf(cell, node, qp) * add_vector(cell, qp, dim);
          }
        }     
      }
    } else { // positive
      for (std::size_t cell=0; cell < workset.numCells; ++cell) {
        for (std::size_t node=0; node < numNodes; ++node) {
          for (std::size_t dim=0; dim<numDims; dim++){
            if( plusEquals ) outResidual(cell,node,dim) = inResidual(cell,node,dim);
            else outResidual(cell,node,dim) = 0.0;
            for (std::size_t qp=0; qp < numQPs; ++qp)
              outResidual(cell,node,dim) += w_bf(cell, node, qp) * add_vector(cell, qp, dim);
          }
        }   
      }
    }
  } else {
    if( negative ){
      if( plusEquals ){
        for (std::size_t cell=0; cell < workset.numCells; ++cell) {
          for (std::size_t node=0; node < numNodes; ++node) {
            for (std::size_t dim=0; dim<numDims; dim++)
              outResidual(cell,node,dim) = inResidual(cell,node,dim) -
                add_vector(cell, node, dim);
          }     
        }
      } else { // equals
        for (std::size_t cell=0; cell < workset.numCells; ++cell) {
          for (std::size_t node=0; node < numNodes; ++node) {
            for (std::size_t dim=0; dim<numDims; dim++)
              outResidual(cell,node,dim) = -add_vector(cell, node, dim);
          }     
        }
      }
    } else {
      if( plusEquals ){
        for (std::size_t cell=0; cell < workset.numCells; ++cell) {
          for (std::size_t node=0; node < numNodes; ++node) {
            for (std::size_t dim=0; dim<numDims; dim++)
              outResidual(cell,node,dim) = inResidual(cell,node,dim) +
                add_vector(cell, node, dim);
          }   
        }
      } else { // equals
        for (std::size_t cell=0; cell < workset.numCells; ++cell) {
          for (std::size_t node=0; node < numNodes; ++node) {
            for (std::size_t dim=0; dim<numDims; dim++)
              outResidual(cell,node,dim) = add_vector(cell, node, dim);
          }   
        }
      }
    }
  }
}

//**********************************************************************
template<typename EvalT, typename Traits>
AddScalar<EvalT, Traits>::
AddScalar(const Teuchos::ParameterList& p) :
  add_scalar  (p.get<std::string>                   ("Scalar Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Scalar Data Layout") ),
  outResidual (p.get<std::string>                   ("Out Residual Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node Scalar Data Layout") )
{
  this->addDependentField(add_scalar);

  this->addEvaluatedField(outResidual);

  this->setName("AddScalar"+PHX::typeAsString<EvalT>());

  if(p.isType<std::string>("In Residual Name")){
    inResidual = decltype(inResidual)(p.get<std::string>("In Residual Name"),
                                    p.get<Teuchos::RCP<PHX::DataLayout> >("Node Scalar Data Layout") );
    this->addDependentField(inResidual);
    plusEquals = true;
  } else 
    plusEquals = false;
  
  if(p.isType<std::string>("Weighted BF Name")){
    w_bf = decltype(w_bf)(p.get<std::string>("Weighted BF Name"),
                                    p.get<Teuchos::RCP<PHX::DataLayout> >("Weighted BF Data Layout") );
    this->addDependentField(w_bf);
    projectFromQPs = true;
  } else 
    projectFromQPs = false;


  if(p.isType<bool>("Negative"))
    negative = p.get<bool>("Negative");
  else
    negative = false;
  
}

//**********************************************************************
template<typename EvalT, typename Traits>
void AddScalar<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(add_scalar,fm);
  this->utils.setFieldData(outResidual,fm);
  std::vector<PHX::Device::size_type> dims;
  outResidual.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  if( projectFromQPs ) {
    this->utils.setFieldData(w_bf,fm);
    w_bf.fieldTag().dataLayout().dimensions(dims);
    numQPs = dims[2];
  }
  if( plusEquals )
    this->utils.setFieldData(inResidual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void AddScalar<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  if( projectFromQPs ){
    if( negative ){
      for (std::size_t cell=0; cell < workset.numCells; ++cell) {
        for (std::size_t node=0; node < numNodes; ++node) {
          if( plusEquals ) outResidual(cell,node) = inResidual(cell,node);
          else outResidual(cell,node) = 0.0;
          for (std::size_t qp=0; qp < numQPs; ++qp)
            outResidual(cell,node) -= w_bf(cell, node, qp) * add_scalar(cell, qp);
        }     
      }
    } else { // positive
      for (std::size_t cell=0; cell < workset.numCells; ++cell) {
        for (std::size_t node=0; node < numNodes; ++node) {
          if( plusEquals ) outResidual(cell,node) = inResidual(cell,node);
          else outResidual(cell,node) = 0.0;
          for (std::size_t qp=0; qp < numQPs; ++qp)
            outResidual(cell,node) += w_bf(cell, node, qp) * add_scalar(cell, qp);
        }   
      }
    }
  } else {
    if( negative ){
      if( plusEquals ){
        for (std::size_t cell=0; cell < workset.numCells; ++cell) {
          for (std::size_t node=0; node < numNodes; ++node) {
            outResidual(cell,node) = inResidual(cell,node) - add_scalar(cell,node);
          }     
        }
      } else { // equals
        for (std::size_t cell=0; cell < workset.numCells; ++cell) {
          for (std::size_t node=0; node < numNodes; ++node) {
            outResidual(cell,node) = -add_scalar(cell,node);
          }     
        }
      }
    } else {
      if( plusEquals ){
        for (std::size_t cell=0; cell < workset.numCells; ++cell) {
          for (std::size_t node=0; node < numNodes; ++node) {
            outResidual(cell,node) = inResidual(cell,node) + add_scalar(cell, node);
          }   
        }
      } else { // equals
        for (std::size_t cell=0; cell < workset.numCells; ++cell) {
          for (std::size_t node=0; node < numNodes; ++node) {
            outResidual(cell,node) = add_scalar(cell, node);
          }   
        }
      }
    }
  }
}

//**********************************************************************
}

