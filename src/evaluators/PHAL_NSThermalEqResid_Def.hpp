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
NSThermalEqResid<EvalT, Traits>::
NSThermalEqResid(const Teuchos::ParameterList& p) :
  wBF         (p.get<std::string>                   ("Weighted BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Scalar Data Layout") ),
  Temperature (p.get<std::string>                   ("QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  Tdot        (p.get<std::string>                   ("QP Time Derivative Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  ThermalCond (p.get<std::string>                   ("Thermal Conductivity Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  wGradBF     (p.get<std::string>                   ("Weighted Gradient BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout") ),
  TGrad       (p.get<std::string>                   ("Gradient QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  V           (p.get<std::string>                   ("Velocity QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  Source      (p.get<std::string>                   ("Source Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  TauT            (p.get<std::string>                 ("Tau T Name"),  
                 p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  rho       (p.get<std::string>                   ("Density QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  Cp       (p.get<std::string>                  ("Specific Heat QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),

  TResidual   (p.get<std::string>                   ("Residual Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node Scalar Data Layout") ),
  haveSource  (p.get<bool>("Have Source"))
{
  if (p.isType<bool>("Disable Transient"))
    enableTransient = !p.get<bool>("Disable Transient");
  else enableTransient = true;

  this->addDependentField(wBF);
  this->addDependentField(Temperature);
  this->addDependentField(ThermalCond);
  if (enableTransient) this->addDependentField(Tdot);
  this->addDependentField(TGrad);
  this->addDependentField(V);
  this->addDependentField(wGradBF);
  if (haveSource) this->addDependentField(Source);
  this->addDependentField(TauT);
  this->addDependentField(rho);
  this->addDependentField(Cp);

  this->addEvaluatedField(TResidual);

  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  // Allocate workspace
  flux.resize(dims[0], numQPs, numDims);
 
  this->setName("NSThermalEqResid"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void NSThermalEqResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(Temperature,fm);
  this->utils.setFieldData(ThermalCond,fm);
  this->utils.setFieldData(TGrad,fm);
  this->utils.setFieldData(V,fm);
  this->utils.setFieldData(wGradBF,fm);
  if (haveSource)  this->utils.setFieldData(Source,fm);
  if (enableTransient) this->utils.setFieldData(Tdot,fm);
  this->utils.setFieldData(TauT,fm);
  this->utils.setFieldData(rho,fm);
  this->utils.setFieldData(Cp,fm);

  this->utils.setFieldData(TResidual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void NSThermalEqResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  typedef Intrepid::FunctionSpaceTools FST;

  FST::scalarMultiplyDataData<ScalarT> (flux, ThermalCond, TGrad);

  FST::integrate<ScalarT>(TResidual, flux, wGradBF, Intrepid::COMP_CPP, false); // "false" overwrites
  

  if (workset.transientTerms && enableTransient) 
    FST::integrate<ScalarT>(TResidual, Tdot, wBF, Intrepid::COMP_CPP, true); // "true" sums into

  //if (haveConvection)  {
    Intrepid::FieldContainer<ScalarT> convection(workset.worksetSize, numQPs);

    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
        convection(cell,qp) = 0.0;
	if (haveSource) convection(cell,qp) -= Source(cell,qp);
        if (workset.transientTerms && enableTransient) 
          convection(cell,qp) += Tdot(cell,qp);
        for (std::size_t i=0; i < numDims; ++i) { 
            convection(cell,qp) += rho(cell,qp) * Cp(cell,qp) * V(cell,qp,i) * TGrad(cell,qp,i);        
        }
      }
    }

    FST::integrate<ScalarT>(TResidual, convection, wBF, Intrepid::COMP_CPP, true); // "true" sums into
//  }

   for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t node=0; node < numNodes; ++node) {          
         for (std::size_t qp=0; qp < numQPs; ++qp) {               
            for (std::size_t j=0; j < numDims; ++j) { 
              TResidual(cell,node) += rho(cell,qp)*Cp(cell,qp)*TauT(cell,qp)*convection(cell,qp)*V(cell,qp,j)*wGradBF(cell,node,qp,j);
            }  
         }
       }
    }

}

//**********************************************************************
}

