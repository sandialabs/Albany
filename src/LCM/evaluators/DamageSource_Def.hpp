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

#include <typeinfo>

namespace LCM {

template<typename EvalT, typename Traits>
DamageSource<EvalT, Traits>::
DamageSource(Teuchos::ParameterList& p) :
  bulkModulus (p.get<std::string>("Bulk Modulus Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  dp          (p.get<std::string>("DP Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  seff        (p.get<std::string>("Effective Stress Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  energy      (p.get<std::string>("Energy Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  J           (p.get<std::string>("DetDefGrad Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  damageLS    (p.get<std::string>("Damage Length Scale Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  gc          (p.get<double>("gc Name")),
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
  this->addDependentField(seff);
  this->addDependentField(energy);
  this->addDependentField(damageLS);

  sourceName = p.get<string>("Damage Source Name");
  damageName = p.get<string>("Damage Name");
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
  this->utils.setFieldData(damageLS,fm);
  this->utils.setFieldData(seff,fm);
  this->utils.setFieldData(energy,fm);
  this->utils.setFieldData(source,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void DamageSource<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  bool print = false;
  //if (typeid(ScalarT) == typeid(RealType)) print = true;

  Albany::StateVariables  oldState = *workset.oldState;
  Intrepid::FieldContainer<RealType>& source_old_FC = *oldState[sourceName];
  Intrepid::FieldContainer<RealType>& damage_old_FC = *oldState[damageName];

  ScalarT p, triax, source_new, source_old, term;
  ScalarT damage_old;

  for (unsigned int cell=0; cell < workset.numCells; ++cell) 
  {
    for (std::size_t qp=0; qp < numQPs; ++qp) 
    {
      source_old = source_old_FC(cell,qp);
      damage_old = damage_old_FC(cell,qp);
      ScalarT fac = 1.0 / ( 1.0 - damage(cell,qp) );
      p = 0.5 * bulkModulus(cell,qp) * (J(cell,qp) - (1.0/J(cell,qp)));
      triax = 0.0;
      if (seff(cell,qp) > 0.0) triax = p/seff(cell,qp);
      //source(cell,qp) = p * Je + expz * triax * dp(cell,qp);
      source_new = - (damage(cell,qp) * gc) / damageLS(cell,qp) 
	+ 2.0 * (1.0 - damage(cell,qp)) * energy(cell,qp)
	+ fac * triax * dp(cell,qp);
      term = 1.0*(std::abs(damage(cell,qp)-damage_old) 
		  - (damage(cell,qp)-damage_old) );
      source_new += term;
      if (print)
      {
        cout << "!*********!" << endl;
	cout << "damage    : " << damage(cell,qp) << endl;
	cout << "damage_old: " << damage_old << endl;
	cout << "energy    : " << energy(cell,qp) << endl;
	cout << "J         : " << J(cell,qp) << endl;      
	cout << "seff      : " << seff(cell,qp) << endl;
	cout << "p         : " << p << endl;
	cout << "triax     : " << triax << endl;
	cout << "dp        : " << dp(cell,qp) << endl;      
	cout << "term      : " << term << endl;      
	cout << "source_old: " << source_old << endl;
	cout << "source_new: " << source_new << endl;
      }
      //source(cell,qp) = std::max(source_old, source_new);
      source(cell,qp) = source_new;
    }
  }
}

// **********************************************************************
}

