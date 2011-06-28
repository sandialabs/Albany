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
NSRmEqResid<EvalT, Traits>::
NSRmEqResid(const Teuchos::ParameterList& p) :
  pGrad       (p.get<std::string>                   ("Pressure Gradient QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  VGrad       (p.get<std::string>                   ("Velocity Gradient QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  V       (p.get<std::string>                   ("Velocity QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  V_Dot       (p.get<std::string>                   ("Velocity Dot QP Variable Name"),
               p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  T       (p.get<std::string>                   ("Temperature QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  rho       (p.get<std::string>                   ("Density QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  force      (p.get<std::string>                   ("Body Force QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  acceleration   (p.get<std::string>                ("Acceleration Residual Name"),
 	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") )
 
{
  if (p.isType<bool>("Disable Transient"))
    enableTransient = !p.get<bool>("Disable Transient");
  else enableTransient = true;

  this->addDependentField(pGrad);
  this->addDependentField(VGrad);
  this->addDependentField(V);
  if (enableTransient) this->addDependentField(V_Dot);
  this->addDependentField(T);
  this->addDependentField(rho);
  this->addDependentField(force);

  this->addEvaluatedField(acceleration);

  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numNodes = dims[1];
  numQPs  = dims[2];
  numDims = dims[3];

  this->setName("NSRmEqResid"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void NSRmEqResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(pGrad,fm);
  this->utils.setFieldData(VGrad,fm);
  this->utils.setFieldData(V,fm);
  if (enableTransient) this->utils.setFieldData(V_Dot,fm);
  this->utils.setFieldData(T,fm);
  this->utils.setFieldData(rho,fm);
  this->utils.setFieldData(force,fm);

  this->utils.setFieldData(acceleration,fm); 
}

//**********************************************************************
template<typename EvalT, typename Traits>
void NSRmEqResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  typedef Intrepid::FunctionSpaceTools FST;

 for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {      
      for (std::size_t i=0; i < numDims; ++i) {
        if (workset.transientTerms && enableTransient) 
          acceleration(cell,qp,i) = rho(cell,qp)*V_Dot(cell,qp,i);
        else
          acceleration(cell,qp,i) = 0;
        acceleration(cell,qp,i) += pGrad(cell,qp,i)+force(cell,qp,i);
        for (std::size_t j=0; j < numDims; ++j) {
          acceleration(cell,qp,i) += rho(cell,qp)*V(cell,qp,j)*VGrad(cell,qp,i,j);
        }
      } 
    }
  }

}

}

