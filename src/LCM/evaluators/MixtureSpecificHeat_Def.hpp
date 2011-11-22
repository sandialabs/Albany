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
MixtureSpecificHeat<EvalT, Traits>::
MixtureSpecificHeat(const Teuchos::ParameterList& p) :
  porosity       (p.get<std::string>                   ("Porosity Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  gammaSkeleton  (p.get<std::string>            ("Skeleton Specific Heat Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  gammaPoreFluid       (p.get<std::string>      ("Pore-Fluid Specific Heat Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  densitySkeleton  (p.get<std::string>            ("Skeleton Density Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  densityPoreFluid       (p.get<std::string>      ("Pore-Fluid Density Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  mixtureSpecificHeat      (p.get<std::string>    ("Mixture Specific Heat Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") )
{
  this->addDependentField(porosity);
  this->addDependentField(gammaPoreFluid);
  this->addDependentField(gammaSkeleton);
  this->addDependentField(densityPoreFluid);
  this->addDependentField(densitySkeleton);

  this->addEvaluatedField(mixtureSpecificHeat);

  this->setName("Mixture Specific Heat"+PHX::TypeString<EvalT>::value);

  Teuchos::RCP<PHX::DataLayout> scalar_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  scalar_dl->dimensions(dims);
  numQPs  = dims[1];
}

//**********************************************************************
template<typename EvalT, typename Traits>
void MixtureSpecificHeat<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(mixtureSpecificHeat,fm);
  this->utils.setFieldData(porosity,fm);
  this->utils.setFieldData(gammaSkeleton,fm);
  this->utils.setFieldData(gammaPoreFluid,fm);
  this->utils.setFieldData(densitySkeleton,fm);
  this->utils.setFieldData(densityPoreFluid,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void MixtureSpecificHeat<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Compute Strain tensor from displacement gradient
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {

        mixtureSpecificHeat(cell,qp) = (1-porosity(cell,qp))
        		                           *gammaSkeleton(cell,qp)*densitySkeleton(cell,qp) +
        		                           porosity(cell,qp)*gammaPoreFluid(cell,qp)*
        		                           densityPoreFluid(cell,qp);

    }
  }

}

//**********************************************************************
}

