//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace LCM {

//**********************************************************************
template<typename EvalT, typename Traits>
GradientElementLength<EvalT, Traits>::
GradientElementLength(const Teuchos::ParameterList& p) :
  GradBF      (p.get<std::string>                   ("Gradient BF Name"),
	 p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout") ),
  unitScalarGradient       (p.get<std::string>       ("Unit Gradient QP Variable Name"),
	 p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  elementLength  (p.get<std::string>                 ("Element Length Name"),
	 p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") )
{
  this->addDependentField(unitScalarGradient);
  this->addDependentField(GradBF);

  this->addEvaluatedField(elementLength);

  this->setName("Element Length"+PHX::TypeString<EvalT>::value);

  Teuchos::RCP<PHX::DataLayout> node_dl =
     p.get< Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout");
     std::vector<PHX::DataLayout::size_type> dims;
     node_dl->dimensions(dims);
     worksetSize = dims[0];
     numNodes = dims[1];
     numQPs  = dims[2];
     numDims = dims[3];


}

//**********************************************************************
template<typename EvalT, typename Traits>
void GradientElementLength<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(GradBF,fm);
  this->utils.setFieldData(unitScalarGradient,fm);
  this->utils.setFieldData(elementLength,fm);

}

//**********************************************************************
template<typename EvalT, typename Traits>
void GradientElementLength<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  ScalarT scalarH(0.0);

  // Compute Strain tensor from displacement gradient
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {

    for (std::size_t qp=0; qp < numQPs; ++qp) {
    	scalarH = 0.0;
    	for (std::size_t j=0; j<numDims; j++)
    	{
    		for (std::size_t node=0; node < numNodes; ++node)
    		{

                scalarH += std::abs(
                	//	unitScalarGradient(cell, qp, j)*GradBF(cell,node,qp,j)
                		GradBF(cell,node,qp,j)/std::sqrt(numDims)
                		);
    	     }
    	 }
    	 elementLength(cell,qp) = 2.0/scalarH;

    }
  }
}




//**********************************************************************
}

