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
MixtureThermalExpansion<EvalT, Traits>::
MixtureThermalExpansion(const Teuchos::ParameterList& p) :
  biotCoefficient       (p.get<std::string>     ("Biot Coefficient Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  porosity       (p.get<std::string>                   ("Porosity Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  alphaSkeleton  (p.get<std::string>                   ("Skeleton Thermal Expansion Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  alphaPoreFluid       (p.get<std::string>      ("Pore-Fluid Thermal Expansion Name"),
	       	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  mixtureThermalExpansion      (p.get<std::string>    ("Mixture Thermal Expansion Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") )
{
  this->addDependentField(biotCoefficient);
  this->addDependentField(porosity);
  this->addDependentField(alphaPoreFluid);
  this->addDependentField(alphaSkeleton);

  this->addEvaluatedField(mixtureThermalExpansion);

  this->setName("Mixture Thermal Expansion"+PHX::TypeString<EvalT>::value);

  Teuchos::RCP<PHX::DataLayout> scalar_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  scalar_dl->dimensions(dims);
  numQPs  = dims[1];
}

//**********************************************************************
template<typename EvalT, typename Traits>
void MixtureThermalExpansion<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(mixtureThermalExpansion,fm);
  this->utils.setFieldData(biotCoefficient,fm);
  this->utils.setFieldData(porosity,fm);
  this->utils.setFieldData(alphaSkeleton,fm);
  this->utils.setFieldData(alphaPoreFluid,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void MixtureThermalExpansion<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Compute Strain tensor from displacement gradient
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {

        mixtureThermalExpansion(cell,qp) = (biotCoefficient(cell,qp)-porosity(cell,qp))
        		                           *alphaSkeleton(cell,qp) +
        		                           porosity(cell,qp)*alphaPoreFluid(cell,qp);

    }
  }

}

//**********************************************************************
}

