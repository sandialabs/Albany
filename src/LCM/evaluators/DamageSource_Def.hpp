//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

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
	       p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout") ),
  dp          (p.get<std::string>("DP Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout") ),
  seff        (p.get<std::string>("Effective Stress Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout") ),
  energy      (p.get<std::string>("Energy Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout") ),
  J           (p.get<std::string>("DetDefGrad Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout") ),
  damageLS    (p.get<std::string>("Damage Length Scale Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout") ),
  gc          (p.get<double>("gc Name")),
  damage      (p.get<std::string>("Damage Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout") ),
  source      (p.get<std::string>("Damage Source Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout") )
{
  Teuchos::RCP<PHX::DataLayout> scalar_dl =
    p.get< Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  scalar_dl->dimensions(dims);
  numQPs  = dims[1];

  Teuchos::RCP<ParamLib> paramLib = 
    p.get< Teuchos::RCP<ParamLib>>("Parameter Library", Teuchos::null);

  // add dependent fields
  this->addDependentField(bulkModulus);
  this->addDependentField(dp);
  this->addDependentField(J);
  this->addDependentField(damage);
  this->addDependentField(seff);
  this->addDependentField(energy);
  this->addDependentField(damageLS);

  sourceName = p.get<std::string>("Damage Source Name")+"_old";
  damageName = p.get<std::string>("Damage Name")+"_old";
  this->addEvaluatedField(source);
  this->setName("Damage Source"+PHX::typeAsString<EvalT>());
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

//  Albany::StateVariables  oldState = *workset.oldState;
//  Intrepid::FieldContainer<RealType>& source_old_FC = *oldState[sourceName];
//  Intrepid::FieldContainer<RealType>& damage_old_FC = *oldState[damageName];
  Albany::MDArray source_old_FC = (*workset.stateArrayPtr)[sourceName];
  Albany::MDArray damage_old_FC = (*workset.stateArrayPtr)[damageName];


  ScalarT p, triax, source_new, term;
  RealType damage_old, source_old;

  for (int  cell=0; cell < workset.numCells; ++cell) 
  {
    for (int qp=0; qp < numQPs; ++qp) 
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
        std::cout << "!*********!" << std::endl;
	std::cout << "damage    : " << damage(cell,qp) << std::endl;
	std::cout << "damage_old: " << damage_old << std::endl;
	std::cout << "energy    : " << energy(cell,qp) << std::endl;
	std::cout << "J         : " << J(cell,qp) << std::endl;      
	std::cout << "seff      : " << seff(cell,qp) << std::endl;
	std::cout << "p         : " << p << std::endl;
	std::cout << "triax     : " << triax << std::endl;
	std::cout << "dp        : " << dp(cell,qp) << std::endl;      
	std::cout << "term      : " << term << std::endl;      
	std::cout << "source_old: " << source_old << std::endl;
	std::cout << "source_new: " << source_new << std::endl;
      }
      //source(cell,qp) = std::max(source_old, source_new);
      source(cell,qp) = source_new;
    }
  }
}

// **********************************************************************
}

