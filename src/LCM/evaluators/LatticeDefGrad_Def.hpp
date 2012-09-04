/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
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
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"
#include "Intrepid_RealSpaceTools.hpp"

namespace LCM {

//**********************************************************************
template<typename EvalT, typename Traits>
LatticeDefGrad<EvalT, Traits>::
LatticeDefGrad(const Teuchos::ParameterList& p) :
  weights       (p.get<std::string>                   ("Weights Name"),
	         p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  defgrad       (p.get<std::string>                  ("DefGrad Name"),
	         p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  J             (p.get<std::string>                   ("DetDefGrad Name"),
	         p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  CtotalRef     (p.get<std::string>                   ("Stress Free Total Concentration Name"),
	         p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  Ctotal     (p.get<std::string>                   ("Total Concentration Name"),
	         p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  VH     (p.get<std::string>                   ("Partial Molar Volume Name"),
	         p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  VM     (p.get<std::string>                   ("Molar Volume Name"),
	     p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  latticeDefGrad       (p.get<std::string>                  ("Lattice Deformation Gradient Name"),
	         	         p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  weightedAverage(false),
  alpha(0.05)
{
  if ( p.isType<string>("Weighted Volume Average J Name") )
    weightedAverage = p.get<bool>("Weighted Volume Average J");
  if ( p.isType<double>("Average J Stabilization Parameter Name") )
    alpha = p.get<double>("Average J Stabilization Parameter");

  Teuchos::RCP<PHX::DataLayout> tensor_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout");

  std::vector<PHX::DataLayout::size_type> dims;
  tensor_dl->dimensions(dims);
  worksetSize  = dims[0];
  numQPs  = dims[1];
  numDims = dims[2];


  this->addDependentField(weights);
  this->addDependentField(CtotalRef);
  this->addDependentField(Ctotal);
  this->addDependentField(VH);
  this->addDependentField(VM);
  this->addDependentField(defgrad);
  this->addDependentField(J);


  this->addEvaluatedField(latticeDefGrad);

  this->setName("Lattice Deformation Gradient"+PHX::TypeString<EvalT>::value);

}

//**********************************************************************
template<typename EvalT, typename Traits>
void LatticeDefGrad<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(weights,fm);
  this->utils.setFieldData(defgrad,fm);
  this->utils.setFieldData(J,fm);
  this->utils.setFieldData(CtotalRef,fm);
  this->utils.setFieldData(Ctotal,fm);
  this->utils.setFieldData(VH,fm);
  this->utils.setFieldData(VM,fm);
  this->utils.setFieldData(latticeDefGrad,fm);

}

//**********************************************************************
template<typename EvalT, typename Traits>
void LatticeDefGrad<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Compute LatticeDefGrad tensor from displacement gradient
  for (std::size_t cell=0; cell < workset.numCells; ++cell)
  {
    for (std::size_t qp=0; qp < numQPs; ++qp)
    {
      for (std::size_t i=0; i < numDims; ++i)
      {
        for (std::size_t j=0; j < numDims; ++j)
	{
        	latticeDefGrad(cell,qp,i,j) = defgrad(cell,qp,i,j);
        }
      }
    }
  }
  // Since Intrepid will later perform calculations on the entire workset size
  // and not just the used portion, we must fill the excess with reasonable 
  // values. Leaving this out leads to inversion of 0 tensors.
  for (std::size_t cell=workset.numCells; cell < worksetSize; ++cell) 
    for (std::size_t qp=0; qp < numQPs; ++qp) 
      for (std::size_t i=0; i < numDims; ++i)
	defgrad(cell,qp,i,i) = 1.0;

  Intrepid::RealSpaceTools<ScalarT>::det(J, defgrad);

  if (weightedAverage)
  {
    ScalarT Jbar, wJbar, vol;
    for (std::size_t cell=0; cell < workset.numCells; ++cell)
    {
      Jbar = 0.0;
      vol = 0.0;
      for (std::size_t qp=0; qp < numQPs; ++qp)
      {
  	Jbar += weights(cell,qp) * std::log( 1 + VH(cell,qp)/VM(cell,qp)*(Ctotal(cell,qp) - CtotalRef(cell,qp)) );
  	vol  += weights(cell,qp);
      }
      Jbar /= vol;

      // Jbar = std::exp(Jbar);
      for (std::size_t qp=0; qp < numQPs; ++qp)
      {
  	for (std::size_t i=0; i < numDims; ++i)
  	{
  	  for (std::size_t j=0; j < numDims; ++j)
  	  {
            wJbar = std::exp( (1-alpha) * Jbar + alpha * std::log( 1 + VH(cell,qp)/VM(cell,qp)*(Ctotal(cell,qp) - CtotalRef(cell,qp)) ) );
  	    latticeDefGrad(cell,qp,i,j) *= std::pow( wJbar ,1./3. );
  	  }
  	}
//
      }
    }
  }
}

//**********************************************************************
}
