//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits>
GPAMResid<EvalT, Traits>::
GPAMResid(const Teuchos::ParameterList& p,
          const Teuchos::RCP<Albany::Layouts>& dl) :
  wBF     (p.get<std::string> ("Weighted BF Name"), dl->node_qp_scalar ),
  wGradBF    (p.get<std::string>  ("Weighted Gradient BF Name"), dl->node_qp_gradient),
  C          (p.get<std::string>  ("QP Variable Name"), dl->qp_vector),
  Cgrad      (p.get<std::string>  ("Gradient QP Variable Name"), dl->qp_vecgradient),
  CDot       (p.get<std::string>  ("QP Time Derivative Variable Name"), dl->qp_vector),
  Residual   (p.get<std::string>  ("Residual Name"), dl->node_vector )
{

  if (p.isType<bool>("Disable Transient"))
    enableTransient = !p.get<bool>("Disable Transient");
  else enableTransient = true;

  this->addDependentField(C);
  this->addDependentField(Cgrad);
  if (enableTransient) this->addDependentField(CDot);
  this->addDependentField(wBF);
  this->addDependentField(wGradBF);

  this->addEvaluatedField(Residual);


  this->setName("GPAMResid" );

  std::vector<PHX::DataLayout::size_type> dims;
  wGradBF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
  numDims  = dims[3];

  C.fieldTag().dataLayout().dimensions(dims);
  vecDim  = dims[2];


  //
  convectionTerm = true;
  u.resize(numDims,0.0);
  u[0] = 1.0;

  std::cout << "GPAM Constructor vecDim = " << vecDim << " numDims = " << numDims << std::endl;
}

//**********************************************************************
template<typename EvalT, typename Traits>
void GPAMResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(C,fm);
  this->utils.setFieldData(Cgrad,fm);
  if (enableTransient) this->utils.setFieldData(CDot,fm);
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(wGradBF,fm);

  this->utils.setFieldData(Residual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void GPAMResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  typedef Intrepid2::FunctionSpaceTools FST;

    //Set Redidual to 0, add Diffusion Term
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t node=0; node < numNodes; ++node) {
              for (std::size_t i=0; i<vecDim; i++)  Residual(cell,node,i)=0.0;
          for (std::size_t qp=0; qp < numQPs; ++qp) {
            for (std::size_t i=0; i<vecDim; i++) {
              for (std::size_t dim=0; dim<numDims; dim++) {
                Residual(cell,node,i) += Cgrad(cell, qp, i, dim) * wGradBF(cell, node, qp, dim);
    } } } } }

  // These both should always be true if transient is enabled
  if (workset.transientTerms && enableTransient) {
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t node=0; node < numNodes; ++node) {
          for (std::size_t qp=0; qp < numQPs; ++qp) {
            for (std::size_t i=0; i<vecDim; i++) {
                Residual(cell,node,i) += CDot(cell, qp, i) * wBF(cell, node, qp);
    } } } } }

  if (convectionTerm) {
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t node=0; node < numNodes; ++node) {
          for (std::size_t qp=0; qp < numQPs; ++qp) {
            for (std::size_t i=0; i<vecDim; i++) {
              for (std::size_t dim=0; dim<numDims; dim++) {
                Residual(cell,node,i) += u[dim]*Cgrad(cell, qp, i, dim) * wBF(cell, node, qp);
    } } } } } }

}

//**********************************************************************
}

