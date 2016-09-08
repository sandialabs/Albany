//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Intrepid2_RealSpaceTools.hpp"


namespace LCM {

//**********************************************************************
template<typename EvalT, typename Traits>
TLPoroStress<EvalT, Traits>::
TLPoroStress(const Teuchos::ParameterList& p) :
  stress           (p.get<std::string>                   ("Stress Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout") ),
  defGrad           (p.get<std::string>                   ("DefGrad Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout") ),
  J                 (p.get<std::string>                   ("DetDefGrad Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout") ),
  biotCoefficient  (p.get<std::string>                   ("Biot Coefficient Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout") ),
  porePressure    (p.get<std::string>                   ("QP Variable Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout") ),
  totstress        (p.get<std::string>                   ("Total Stress Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout") )
{
  // Pull out numQPs and numDims from a Layout
  Teuchos::RCP<PHX::DataLayout> tensor_dl =
    p.get< Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  tensor_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  // Works space FCs
  int worksetSize = dims[0];
  F_inv.resize(worksetSize, numQPs, numDims, numDims);
  F_invT.resize(worksetSize, numQPs, numDims, numDims);
  JF_invT.resize(worksetSize, numQPs, numDims, numDims);
  JpF_invT.resize(worksetSize, numQPs, numDims, numDims);
  JBpF_invT.resize(worksetSize, numQPs, numDims, numDims);

  this->addDependentField(stress);
  this->addDependentField(defGrad);
  this->addDependentField(J);
  this->addDependentField(biotCoefficient);
 // this->addDependentField(porePressure);

  this->addEvaluatedField(porePressure);
  this->addEvaluatedField(totstress);

  this->setName("TLPoroStress"+PHX::typeAsString<EvalT>());

}

//**********************************************************************
template<typename EvalT, typename Traits>
void TLPoroStress<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(stress,fm);
  this->utils.setFieldData(defGrad,fm);
  this->utils.setFieldData(J,fm);
  this->utils.setFieldData(biotCoefficient,fm);
  this->utils.setFieldData(porePressure,fm);
  this->utils.setFieldData(totstress,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void TLPoroStress<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

  typedef Intrepid2::FunctionSpaceTools FST;
  typedef Intrepid2::RealSpaceTools<ScalarT> RST;

  if (numDims == 1) {
    Intrepid2::FunctionSpaceTools::scalarMultiplyDataData<ScalarT>(totstress, J, stress);
    for (int cell=0; cell < workset.numCells; ++cell) {
          for (int qp=0; qp < numQPs; ++qp) {
              for (int dim=0; dim<numDims; ++ dim) {
                  for (int j=0; j<numDims; ++ j) {
        	    totstress(cell, qp, dim, j) = stress(cell, qp, dim, j) - biotCoefficient(cell,qp)*porePressure(cell,qp);
                 }
              }
          }
    }
  }
  else
    {

	   RST::inverse(F_inv, defGrad);
	   RST::transpose(F_invT, F_inv);
	   FST::scalarMultiplyDataData<ScalarT>(JF_invT, J, F_invT);
	   FST::scalarMultiplyDataData<ScalarT>(JpF_invT, porePressure,JF_invT);
	   FST::scalarMultiplyDataData<ScalarT>(JBpF_invT, biotCoefficient, JpF_invT);
  	   FST::tensorMultiplyDataData<ScalarT>(totstress, stress,JF_invT); // Cauchy to 1st PK

    // Compute Stress

    for (int cell=0; cell < workset.numCells; ++cell) {
      for (int qp=0; qp < numQPs; ++qp) {
    	  for (int dim=0; dim<numDims; ++ dim) {
    		  for (int j=0; j<numDims; ++ j) {
	         totstress(cell,qp,dim,j) -= JBpF_invT(cell,qp, dim,j);
    		  }
    	  }
      }
    }


  }
}

//**********************************************************************
}

