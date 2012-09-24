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
TLPoroStress<EvalT, Traits>::
TLPoroStress(const Teuchos::ParameterList& p) :
  stress           (p.get<std::string>                   ("Stress Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  defGrad           (p.get<std::string>                   ("DefGrad Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  J                 (p.get<std::string>                   ("DetDefGrad Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  biotCoefficient  (p.get<std::string>                   ("Biot Coefficient Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  porePressure    (p.get<std::string>                   ("QP Variable Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  totstress        (p.get<std::string>                   ("Total Stress Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") )
{
  // Pull out numQPs and numDims from a Layout
  Teuchos::RCP<PHX::DataLayout> tensor_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout");
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

  this->setName("TLPoroStress"+PHX::TypeString<EvalT>::value);

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

  typedef Intrepid::FunctionSpaceTools FST;
  typedef Intrepid::RealSpaceTools<ScalarT> RST;

  if (numDims == 1) {
    Intrepid::FunctionSpaceTools::scalarMultiplyDataData<ScalarT>(totstress, J, stress);
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
          for (std::size_t qp=0; qp < numQPs; ++qp) {
        	  totstress(cell, qp) = stress(cell, qp) - biotCoefficient(cell,qp)*porePressure(cell,qp);
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

    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
    	  for (std::size_t dim=0; dim<numDims; ++ dim) {
    		  for (std::size_t j=0; j<numDims; ++ j) {
	         totstress(cell,qp,dim,j) -= JBpF_invT(cell,qp, dim,j);
    		  }
    	  }
      }
    }


  }
}

//**********************************************************************
}

