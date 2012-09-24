/********************************************************************\
*            Albany, Copyright (2012) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Glen Hansen, gahanse@sandia.gov                    *
\********************************************************************/


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

