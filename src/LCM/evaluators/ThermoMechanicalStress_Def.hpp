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
#include "LCM/utils/Tensor.h"

namespace LCM {

//**********************************************************************
template<typename EvalT, typename Traits>
ThermoMechanicalStress<EvalT, Traits>::
ThermoMechanicalStress(const Teuchos::ParameterList& p) :
  F                (p.get<std::string>                   ("DefGrad Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  J                (p.get<std::string>                   ("DetDefGrad Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  shearModulus     (p.get<std::string>                   ("Shear Modulus Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  bulkModulus      (p.get<std::string>                   ("Bulk Modulus Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  temperature      (p.get<std::string>                   ("Temperature Name"),
		    p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  stress           (p.get<std::string>                   ("Stress Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  thermalExpansionCoeff (p.get<RealType>("Thermal Expansion Coefficient") ),
  refTemperature (p.get<RealType>("Reference Temperature") )
{
  // Pull out numQPs and numDims from a Layout
  Teuchos::RCP<PHX::DataLayout> tensor_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  tensor_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  this->addDependentField(F);
  this->addDependentField(J);
  this->addDependentField(shearModulus);
  this->addDependentField(bulkModulus);
  this->addDependentField(temperature);

  this->addEvaluatedField(stress);

  this->setName("ThermoMechanical Stress"+PHX::TypeString<EvalT>::value);

}

//**********************************************************************
template<typename EvalT, typename Traits>
void ThermoMechanicalStress<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(stress,fm);
  this->utils.setFieldData(F,fm);
  this->utils.setFieldData(J,fm);
  this->utils.setFieldData(shearModulus,fm);
  this->utils.setFieldData(bulkModulus,fm);
  this->utils.setFieldData(temperature,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ThermoMechanicalStress<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  ScalarT Jm53;
  ScalarT trace;
  ScalarT deltaTemp;

  Intrepid::FieldContainer<ScalarT> C(workset.numCells, numQPs, numDims, numDims);
  Intrepid::FunctionSpaceTools::tensorMultiplyDataData<ScalarT> (C, F, F, 'T');
  
  for (std::size_t cell=0; cell < workset.numCells; ++cell) 
  {
    for (std::size_t qp=0; qp < numQPs; ++qp) 
    {
      Jm53  = std::pow(J(cell,qp), -5./3.);
      trace = 0.0;
      deltaTemp = temperature(cell,qp) - refTemperature;
      for (std::size_t i=0; i < numDims; ++i) 
	trace += (1./numDims) * C(cell,qp,i,i);
      for (std::size_t i=0; i < numDims; ++i) 
      {
	for (std::size_t j=0; j < numDims; ++j) 
	{
	  stress(cell,qp,i,j) = shearModulus(cell,qp) * Jm53 * ( C(cell,qp,i,j) );
	}
	stress(cell,qp,i,i) += 0.5 * bulkModulus(cell,qp)
	  * ( J(cell,qp) - 1. / J(cell,qp) - 6. * thermalExpansionCoeff * deltaTemp )
	  - shearModulus(cell,qp) * Jm53 * trace;
      }
    }
  }
}
//**********************************************************************

}
