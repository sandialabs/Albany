//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits>
NSPermeabilityTerm<EvalT, Traits>::
NSPermeabilityTerm(const Teuchos::ParameterList& p) :
  V           (p.get<std::string>                   ("Velocity QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  mu         (p.get<std::string>                   ("Viscosity QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  phi         (p.get<std::string>                   ("Porosity QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  K         (p.get<std::string>                   ("Permeability QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  permTerm   (p.get<std::string>                ("Permeability Term"),
 	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") )
 
{
  if (p.isType<bool>("Disable Transient"))
    enableTransient = !p.get<bool>("Disable Transient");
  else enableTransient = true;

  this->addDependentField(V.fieldTag());
  this->addDependentField(mu.fieldTag());
  this->addDependentField(phi.fieldTag());
  this->addDependentField(K.fieldTag());

  this->addEvaluatedField(permTerm);

  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numNodes = dims[1];
  numQPs  = dims[2];
  numDims = dims[3];

  this->setName("NSPermeabilityTerm" );
}

//**********************************************************************
template<typename EvalT, typename Traits>
void NSPermeabilityTerm<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(V,fm);
  this->utils.setFieldData(mu,fm);
  this->utils.setFieldData(phi,fm);
  this->utils.setFieldData(K,fm);

  this->utils.setFieldData(permTerm,fm); 
}

//**********************************************************************
template<typename EvalT, typename Traits>
void NSPermeabilityTerm<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {      
      for (std::size_t i=0; i < numDims; ++i) {
          permTerm(cell,qp,i) = phi(cell,qp)*mu(cell,qp)*V(cell,qp,i)/K(cell,qp);
      } 
    }
  }
}

}

