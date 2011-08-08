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
NSNeutronEqResid<EvalT, Traits>::
NSNeutronEqResid(const Teuchos::ParameterList& p) :
  wBF         (p.get<std::string>                   ("Weighted BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Scalar Data Layout") ),
  Neutron      (p.get<std::string>                   ("QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  NeutronDiff  (p.get<std::string>                   ("Neutron Diffusion Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  wGradBF     (p.get<std::string>                   ("Weighted Gradient BF Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout") ),
  NGrad       (p.get<std::string>                   ("Gradient QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  Absorp      (p.get<std::string>                   ("Neutron Absorption Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  Fission     (p.get<std::string>                  ("Neutron Fission Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  Tref        (p.get<std::string>                  ("Reference Temperature Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  NResidual   (p.get<std::string>                   ("Residual Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("Node Scalar Data Layout") ),
  haveNeutSource  (p.get<bool>("Have Neutron Source")),
  haveFlow    (p.get<bool>("Have Flow")),
  haveHeat    (p.get<bool>("Have Heat"))
{

  this->addDependentField(wBF);
  this->addDependentField(wGradBF);
  this->addDependentField(Neutron);
  this->addDependentField(NGrad);
  this->addDependentField(NeutronDiff);
  this->addDependentField(Absorp);
  this->addDependentField(Fission);
  this->addDependentField(Tref);
  
  if (haveNeutSource) {
    Source = PHX::MDField<ScalarT,Cell,QuadPoint>(
      p.get<std::string>("Source Name"),
      p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") );
    this->addDependentField(Source);
  }

  if (haveHeat) {
    T = PHX::MDField<ScalarT,Cell,QuadPoint>(
      p.get<std::string>("Temperature QP Variable Name"),
      p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") );
    this->addDependentField(T);
  }

  this->addEvaluatedField(NResidual);

  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  // Allocate workspace
  flux.resize(dims[0], numQPs, numDims);
  abscoeff.resize(dims[0], numQPs);
 
  this->setName("NSNeutronEqResid"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void NSNeutronEqResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(wGradBF,fm);
  this->utils.setFieldData(Neutron,fm);
  this->utils.setFieldData(NGrad,fm);
  this->utils.setFieldData(NeutronDiff,fm);
  this->utils.setFieldData(Absorp,fm);
  this->utils.setFieldData(Fission,fm);
  this->utils.setFieldData(Tref,fm);
  if (haveNeutSource)  this->utils.setFieldData(Source,fm);
  if (haveHeat) this->utils.setFieldData(T,fm);

  this->utils.setFieldData(NResidual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void NSNeutronEqResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  typedef Intrepid::FunctionSpaceTools FST;

  FST::scalarMultiplyDataData<ScalarT> (flux, NeutronDiff, NGrad);

  FST::integrate<ScalarT>(NResidual, flux, wGradBF, Intrepid::COMP_CPP, false); // "false" overwrites
  
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {
      abscoeff(cell,qp) = 0.0;
      if (haveNeutSource) abscoeff(cell,qp) -= Source(cell,qp);
         abscoeff(cell,qp) += 
            //(Absorp(cell,qp) - Fission(cell,qp)) * sqrt(Tref(cell,qp)) / sqrt(T(cell,qp)) * Neutron(cell,qp);
            (Absorp(cell,qp) - Fission(cell,qp)) * Neutron(cell,qp);
    }
  }

  FST::integrate<ScalarT>(NResidual, abscoeff, wBF, Intrepid::COMP_CPP, true); // "true" sums into

}

//**********************************************************************
}

