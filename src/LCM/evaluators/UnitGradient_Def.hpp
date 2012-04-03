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
UnitGradient<EvalT, Traits>::
UnitGradient(const Teuchos::ParameterList& p) :
  scalarGrad       (p.get<std::string>               ("Gradient QP Variable Name"),
	 p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  unitScalarGradient       (p.get<std::string>       ("Unit Gradient QP Variable Name"),
	 p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") )
{
  this->addDependentField(scalarGrad);

  this->addEvaluatedField(unitScalarGradient);

  this->setName("Unit Gradient QP Variable"+PHX::TypeString<EvalT>::value);

  Teuchos::RCP<PHX::DataLayout> scalar_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  scalar_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];
}

//**********************************************************************
template<typename EvalT, typename Traits>
void UnitGradient<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(scalarGrad,fm);
  this->utils.setFieldData(unitScalarGradient,fm);

}

//**********************************************************************
template<typename EvalT, typename Traits>
void UnitGradient<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  ScalarT scalarMag(0.0);

  // Compute Strain tensor from displacement gradient
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {

    for (std::size_t qp=0; qp < numQPs; ++qp) {

    	 scalarMag = 0.0;
    	 for (std::size_t j=0; j<numDims; j++)
    	 {
              unitScalarGradient(cell,qp,j) = scalarGrad(cell,qp,j);
              scalarMag += scalarGrad(cell,qp,j)*scalarGrad(cell,qp,j);
    	 }

    	 scalarMag = std::sqrt(scalarMag);

    	 if (scalarMag > 0) {
    		 for (std::size_t i=0; i<numDims; i++)
    		 {
    			 //unitScalarGradient(cell,qp,i) = scalarMag;
    			  unitScalarGradient(cell,qp,i) = unitScalarGradient(cell,qp,i)/scalarMag;
    		 }
    	 }
		 else {
			 for (std::size_t i=0; i<numDims; i++){
			     unitScalarGradient(cell,qp,i) = 1/std::sqrt(numDims);
			 }
		 }


    }


  }

}

//**********************************************************************
}

