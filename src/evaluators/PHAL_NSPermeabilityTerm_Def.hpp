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

  this->addDependentField(V);
  this->addDependentField(mu);
  this->addDependentField(phi);
  this->addDependentField(K);

  this->addEvaluatedField(permTerm);

  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numNodes = dims[1];
  numQPs  = dims[2];
  numDims = dims[3];

  this->setName("NSPermeabilityTerm"+PHX::TypeString<EvalT>::value);
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

