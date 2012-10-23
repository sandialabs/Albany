//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits>
CahnHillElasResid<EvalT, Traits>::
CahnHillElasResid(const Teuchos::ParameterList& p) :
  Stress      (p.get<std::string>                   ("Stress Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  wGradBF     (p.get<std::string>                   ("Weighted Gradient BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout") ),
  ExResidual   (p.get<std::string>                   ("Residual Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node Vector Data Layout") )
{
  this->addDependentField(Stress);
  this->addDependentField(wGradBF);

  this->addEvaluatedField(ExResidual);

  this->setName("CahnHillElasResid"+PHX::TypeString<EvalT>::value);

  std::vector<PHX::DataLayout::size_type> dims;
  wGradBF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
  numDims  = dims[3];
}

//**********************************************************************
template<typename EvalT, typename Traits>
void CahnHillElasResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(Stress,fm);
  this->utils.setFieldData(wGradBF,fm);

  this->utils.setFieldData(ExResidual,fm);

}

//**********************************************************************
template<typename EvalT, typename Traits>
void CahnHillElasResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  typedef Intrepid::FunctionSpaceTools FST;

    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t node=0; node < numNodes; ++node) {
              for (std::size_t dim=0; dim<numDims; dim++)  ExResidual(cell,node,dim)=0.0;
          for (std::size_t qp=0; qp < numQPs; ++qp) {
            for (std::size_t i=0; i<numDims; i++) {
              for (std::size_t dim=0; dim<numDims; dim++) {
                ExResidual(cell,node,i) += Stress(cell, qp, i, dim) * wGradBF(cell, node, qp, dim);
    } } } } }

// Inner product of stress (transpose) and strain gives strain energy density


//  FST::integrate<ScalarT>(ExResidual, Stress, wGradBF, Intrepid::COMP_CPP, false); // "false" overwrites

}

//**********************************************************************
}

