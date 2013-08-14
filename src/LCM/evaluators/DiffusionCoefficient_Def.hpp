//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Intrepid_FunctionSpaceTools.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "Albany_Utils.hpp"

namespace LCM {

//**********************************************************************
template<typename EvalT, typename Traits>
DiffusionCoefficient<EvalT, Traits>::
DiffusionCoefficient(const Teuchos::ParameterList& p) :
  temperature       (p.get<std::string>                   ("Temperature Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  Qdiff       (p.get<std::string>                   ("Diffusion Activation Enthalpy Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  Dpre       (p.get<std::string>                   ("Pre Exponential Factor Name"),
	       	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  diffusionCoefficient      (p.get<std::string>                   ("Diffusion Coefficient Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  Rideal(p.get<RealType>("Ideal Gas Constant"))
{


  //Rideal = p.get<RealType>("Ideal Gas Constant J/K/mol", 8.3144621);
 // this->addDependentField(Rideal);
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
  this->utils.setFieldData(temperature,fm);
  this->utils.setFieldData(Qdiff,fm);
  this->utils.setFieldData(Dpre,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void DiffusionCoefficient<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Diffusion Coefficient
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {

        	diffusionCoefficient(cell,qp) = Dpre(cell,qp)*
        			                                          std::exp(-1.0*Qdiff(cell,qp)/
        			                                         Rideal/temperature(cell,qp));


    }
  }

}

//**********************************************************************
}

