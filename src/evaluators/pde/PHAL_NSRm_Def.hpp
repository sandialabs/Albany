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
NSRm<EvalT, Traits>::
NSRm(const Teuchos::ParameterList& p) :
  pGrad       (p.get<std::string>                   ("Pressure Gradient QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  VGrad       (p.get<std::string>                   ("Velocity Gradient QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  V           (p.get<std::string>                   ("Velocity QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  V_Dot       (p.get<std::string>                   ("Velocity Dot QP Variable Name"),
               p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  rho         (p.get<std::string>                   ("Density QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  phi         (p.get<std::string>                   ("Porosity QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  force       (p.get<std::string>                   ("Body Force QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  permTerm   (p.get<std::string>                ("Permeability Term"),
 	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  ForchTerm   (p.get<std::string>                ("Forchheimer Term"),
 	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  Rm   (p.get<std::string>                ("Rm Name"),
 	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  porousMedia  (p.get<bool>("Porous Media"))
 
{
  if (p.isType<bool>("Disable Transient"))
    enableTransient = !p.get<bool>("Disable Transient");
  else enableTransient = true;

  this->addDependentField(pGrad.fieldTag());
  this->addDependentField(VGrad.fieldTag());
  this->addDependentField(V.fieldTag());
  if (enableTransient) this->addDependentField(V_Dot.fieldTag());
  this->addDependentField(force.fieldTag()); 
  this->addDependentField(rho.fieldTag());
  if (porousMedia) {
   this->addDependentField(phi.fieldTag());   
   this->addDependentField(permTerm.fieldTag());
   this->addDependentField(ForchTerm.fieldTag());
  }
  this->addEvaluatedField(Rm);

  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numNodes = dims[1];
  numQPs  = dims[2];
  numDims = dims[3];

  this->setName("NSRm" );
}

//**********************************************************************
template<typename EvalT, typename Traits>
void NSRm<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(pGrad,fm);
  this->utils.setFieldData(VGrad,fm);
  this->utils.setFieldData(V,fm);
  if (enableTransient) this->utils.setFieldData(V_Dot,fm);
  this->utils.setFieldData(force,fm);
  this->utils.setFieldData(rho,fm);
  if (porousMedia) {
   this->utils.setFieldData(phi,fm);
   this->utils.setFieldData(permTerm,fm);
   this->utils.setFieldData(ForchTerm,fm); 
  }

  this->utils.setFieldData(Rm,fm); 
}

//**********************************************************************
template<typename EvalT, typename Traits>
void NSRm<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {      
      for (std::size_t i=0; i < numDims; ++i) {
        if (workset.transientTerms && enableTransient) 
          Rm(cell,qp,i) = rho(cell,qp)*V_Dot(cell,qp,i);
        else
          Rm(cell,qp,i) = 0;
        if (!porousMedia) // Navier-Stokes
          Rm(cell,qp,i) += pGrad(cell,qp,i)+force(cell,qp,i);
        else              // Porous Media
          Rm(cell,qp,i) += phi(cell,qp)*pGrad(cell,qp,i)+phi(cell,qp)*force(cell,qp,i);
        if (porousMedia) { //permeability and Forchheimer terms 
         Rm(cell,qp,i) += -permTerm(cell,qp,i)+ForchTerm(cell,qp,i);
        }
        for (std::size_t j=0; j < numDims; ++j) {
          if (!porousMedia) // Navier-Stokes
            Rm(cell,qp,i) += rho(cell,qp)*V(cell,qp,j)*VGrad(cell,qp,i,j);
          else              // Porous Media 
            Rm(cell,qp,i) += rho(cell,qp)*V(cell,qp,j)*VGrad(cell,qp,i,j)/phi(cell,qp);
        }
      } 
    }
  }
}

}

