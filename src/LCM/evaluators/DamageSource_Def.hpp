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


#include <fstream>
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "Albany_Utils.hpp"

namespace LCM {

template<typename EvalT, typename Traits>
DamageSource<EvalT, Traits>::
DamageSource(Teuchos::ParameterList& p) :
  bulkModulus (p.get<std::string>("Bulk Modulus Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  dp          (p.get<std::string>("DP Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  J           (p.get<std::string>("DetDefGrad Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  damage      (p.get<std::string>("Damage Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  source      (p.get<std::string>("Damage Source Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") )
{
  Teuchos::RCP<PHX::DataLayout> scalar_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  scalar_dl->dimensions(dims);
  numQPs  = dims[1];

  Teuchos::RCP<ParamLib> paramLib = 
    p.get< Teuchos::RCP<ParamLib> >("Parameter Library", Teuchos::null);

  // add dependent fields
  this->addDependentField(bulkModulus);
  this->addDependentField(dp);
  this->addDependentField(J);
  this->addDependentField(damage);

  this->addEvaluatedField(source);
  this->setName("Damage Source"+PHX::TypeString<EvalT>::value);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void DamageSource<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(bulkModulus,fm);
  this->utils.setFieldData(dp,fm);
  this->utils.setFieldData(J,fm);
  this->utils.setFieldData(damage,fm);
  this->utils.setFieldData(source,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void DamageSource<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  ScalarT exphsq, Jsq;

  for (unsigned int cell=0; cell < workset.numCells; ++cell) 
  {
    for (std::size_t qp=0; qp < numQPs; ++qp) 
    {
      exphsq = std::exp( 2. * damage(cell,qp) );
      Jsq    = J(cell,qp) * J(cell,qp);
      source(cell,qp) = 0.5 * bulkModulus(cell,qp) * ( exphsq * Jsq - 1. );
    }
  }
}

// **********************************************************************
}

