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
NSForchheimerTerm<EvalT, Traits>::
NSForchheimerTerm(const Teuchos::ParameterList& p) :
  V           (p.get<std::string>                   ("Velocity QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  rho         (p.get<std::string>                   ("Density QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  phi         (p.get<std::string>                   ("Porosity QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  K         (p.get<std::string>                   ("Permeability QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  F         (p.get<std::string>                   ("Forchheimer QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  ForchTerm   (p.get<std::string>                ("Forchheimer Term"),
 	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") )
 
{
  if (p.isType<bool>("Disable Transient"))
    enableTransient = !p.get<bool>("Disable Transient");
  else enableTransient = true;

  this->addDependentField(V);
  this->addDependentField(rho);
  this->addDependentField(phi);
  this->addDependentField(K);
  this->addDependentField(F);

  this->addEvaluatedField(ForchTerm);

  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numNodes = dims[1];
  numQPs  = dims[2];
  numDims = dims[3];

  // Allocate workspace
  normV.resize(dims[0], numQPs);

  this->setName("NSForchheimerTerm" );
}

//**********************************************************************
template<typename EvalT, typename Traits>
void NSForchheimerTerm<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(V,fm);
  this->utils.setFieldData(rho,fm);
  this->utils.setFieldData(phi,fm);
  this->utils.setFieldData(K,fm);
  this->utils.setFieldData(F,fm);

  this->utils.setFieldData(ForchTerm,fm); 
}

//**********************************************************************
template<typename EvalT, typename Traits>
void NSForchheimerTerm<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {     
      normV(cell,qp) = 0.0; 
      for (std::size_t i=0; i < numDims; ++i) {
          normV(cell,qp) += V(cell,qp,i)*V(cell,qp,i); 
      } 
      if (normV(cell,qp) > 0)
        normV(cell,qp) = std::sqrt(normV(cell,qp));
      else
        normV(cell,qp) = 0.0;
      for (std::size_t i=0; i < numDims; ++i) {
          ForchTerm(cell,qp,i) = phi(cell,qp)*rho(cell,qp)*F(cell,qp)*normV(cell,qp)*V(cell,qp,i)/std::sqrt(K(cell,qp));
      } 
    }
  }
}

}

