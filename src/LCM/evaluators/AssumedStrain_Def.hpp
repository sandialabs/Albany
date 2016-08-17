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
AssumedStrain<EvalT, Traits>::
AssumedStrain(const Teuchos::ParameterList& p) :
  GradU         (p.get<std::string>                   ("Gradient QP Variable Name"),
	         p.get<Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout") ),
  weights       (p.get<std::string>                   ("Weights Name"),
	         p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout") ),
  defgrad       (p.get<std::string>                  ("DefGrad Name"),
	         p.get<Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout") ),
  assumedStrain       (p.get<std::string>            ("Assumed Strain Name"),
	         p.get<Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout") ),
  J             (p.get<std::string>                   ("DetDefGrad Name"),
	         p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout") ),
  avgJ          (p.get<bool> ("avgJ Name")),
  volavgJ       (p.get<bool> ("volavgJ Name")),
  weighted_Volume_Averaged_J      (p.get<bool> ("weighted_Volume_Averaged_J Name"))
{
  Teuchos::RCP<PHX::DataLayout> tensor_dl =
    p.get< Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout");

  std::vector<PHX::DataLayout::size_type> dims;
  tensor_dl->dimensions(dims);
  worksetSize  = dims[0];
  numQPs  = dims[1];
  numDims = dims[2];

  this->addDependentField(GradU);
  this->addDependentField(weights);

  this->addEvaluatedField(defgrad);
  this->addEvaluatedField(assumedStrain);
  this->addEvaluatedField(J);

  this->setName("Assumed Strain"+PHX::typeAsString<EvalT>());

}

//**********************************************************************
template<typename EvalT, typename Traits>
void AssumedStrain<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(weights,fm);
  this->utils.setFieldData(assumedStrain,fm);
  this->utils.setFieldData(J,fm);
  this->utils.setFieldData(GradU,fm);
  this->utils.setFieldData(defgrad,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void AssumedStrain<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Compute AssumedStrain tensor from displacement gradient
  for (int cell=0; cell < workset.numCells; ++cell)
  {
    for (int qp=0; qp < numQPs; ++qp)
    {
      for (int i=0; i < numDims; ++i)
      {
        for (int j=0; j < numDims; ++j)
	{
          defgrad(cell,qp,i,j) = GradU(cell,qp,i,j);
        }
	defgrad(cell,qp,i,i) += 1.0;
      }
    }
  }

  Intrepid2::RealSpaceTools<PHX::Device>::det(J.get_view(), defgrad.get_view());

  if (avgJ)
  {
    ScalarT Jbar;
    for (int cell=0; cell < workset.numCells; ++cell)
    {
      Jbar = 0.0;
      for (int qp=0; qp < numQPs; ++qp)
      {
        //TEUCHOS_TEST_FOR_EXCEPTION(J(cell,qp) < 0, std::runtime_error,
        //    " negative volume detected in avgJ routine");
	Jbar += std::log(J(cell,qp));
        //Jbar += J(cell,qp);
      }
      Jbar /= numQPs;
      Jbar = std::exp(Jbar);
      for (int qp=0; qp < numQPs; ++qp)
      {
	for (int i=0; i < numDims; ++i)
	{
	  for (int j=0; j < numDims; ++j)
	  {
	    defgrad(cell,qp,i,j) *= std::pow(Jbar/J(cell,qp),1./3.);
	  }
	}
	J(cell,qp) = Jbar;
      }
    }
  }
  else if (volavgJ)
  {
    ScalarT Jbar, vol;
    for (int cell=0; cell < workset.numCells; ++cell)
    {
      Jbar = 0.0;
      vol = 0.0;
      for (int qp=0; qp < numQPs; ++qp)
      {
        //TEUCHOS_TEST_FOR_EXCEPTION(J(cell,qp) < 0, std::runtime_error,
        //    " negative volume detected in volavgJ routine");
	Jbar += weights(cell,qp) * std::log( J(cell,qp) );
	vol  += weights(cell,qp);
      }
      Jbar /= vol;
      Jbar = std::exp(Jbar);
      for (int qp=0; qp < numQPs; ++qp)
      {
	for (int i=0; i < numDims; ++i)
	{
	  for (int j=0; j < numDims; ++j)
	  {
	    defgrad(cell,qp,i,j) *= std::pow(Jbar/J(cell,qp),1./3.);
	  }
	}
	J(cell,qp) = Jbar;
      }
    }
  }
  else if (weighted_Volume_Averaged_J)
    {
      ScalarT Jbar, wJbar, vol;
      ScalarT StabAlpha = 0.5; // This setting need to change later..
      for (int cell=0; cell < workset.numCells; ++cell)
      {
        Jbar = 0.0;
        vol = 0.0;
        for (int qp=0; qp < numQPs; ++qp)
        {
          //TEUCHOS_TEST_FOR_EXCEPTION(J(cell,qp) < 0, std::runtime_error,
          //    " negative volume detected in volavgJ routine");
  	Jbar += weights(cell,qp) * std::log( J(cell,qp) );
  	vol  += weights(cell,qp);

        }
        Jbar /= vol;

       // Jbar = std::exp(Jbar);
        for (int qp=0; qp < numQPs; ++qp)
        {
  	for (int i=0; i < numDims; ++i)
  	{
  	  for (int j=0; j < numDims; ++j)
  	  {
  		wJbar =   std::exp( (1-StabAlpha)*Jbar+
  		          	        		  StabAlpha*std::log(J(cell,qp)));

  	    defgrad(cell,qp,i,j) *= std::pow(wJbar /J(cell,qp),1./3.);
  	  }
  	}
  	J(cell,qp) = wJbar;
        }
      }
    }

  // Since Intrepid2 will later perform calculations on the entire workset size
  // and not just the used portion, we must fill the excess with reasonable 
  // values. Leaving this out leads to inversion of 0 tensors.
  for (int cell=workset.numCells; cell < worksetSize; ++cell) 
    for (int qp=0; qp < numQPs; ++qp) 
      for (int i=0; i < numDims; ++i)
	defgrad(cell,qp,i,i) = 1.0;


  // assembly assumed strain
  for (int cell=0; cell < workset.numCells; ++cell) {
      for (int qp=0; qp < numQPs; ++qp) {
        for (int i=0; i < numDims; ++i){
        	for (int j=0; j < numDims; ++j){
  	            assumedStrain(cell,qp,i,j) =0.5*(defgrad(cell,qp,i,j) + defgrad(cell,qp,j,i));
  	            if (i==j) assumedStrain(cell,qp,i,j) = assumedStrain(cell,qp,i,j) - 1.0;
        	}
        }
      }
  }





}

//**********************************************************************
}
