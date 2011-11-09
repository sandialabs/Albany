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

namespace LCM {

//**********************************************************************
template<typename EvalT, typename Traits>
DiffusionCoefficient<EvalT, Traits>::
DiffusionCoefficient(const Teuchos::ParameterList& p) :
  Rideal       (p.get<std::string>                   ("Ideal Gas Constant Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  temperature       (p.get<std::string>                   ("Temperature Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  Qdiff       (p.get<std::string>                   ("Diffusion Activation Enthalpy Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  Dpre       (p.get<std::string>                   ("Pre Exponential Factor Name"),
	       	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  diffusionCoefficient      (p.get<std::string>                   ("Diffusion Coefficient Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") )
{
  this->addDependentField(Rideal);
  this->addDependentField(temperature);
  this->addDependentField(Qdiff);
  this->addDependentField(Dpre);

  this->addEvaluatedField(diffusionCoefficient);

  this->setName("Diffusion Coefficient"+PHX::TypeString<EvalT>::value);

  Teuchos::RCP<PHX::DataLayout> scalar_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  scalar_dl->dimensions(dims);
  numQPs  = dims[1];
}

//**********************************************************************
template<typename EvalT, typename Traits>
void DiffusionCoefficient<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(diffusionCoefficient,fm);
  this->utils.setFieldData(Rideal,fm);
  this->utils.setFieldData(temperature,fm);
  this->utils.setFieldData(Qdiff,fm);
  this->utils.setFieldData(Dpre,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void DiffusionCoefficient<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Compute Strain tensor from displacement gradient
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {

        	diffusionCoefficient(cell,qp) = Dpre(cell,qp)*exp(-1.0*Qdiff(cell,qp)/Rideal(cell,qp)/temperature(cell,qp));


    }
  }

}

//**********************************************************************
}

