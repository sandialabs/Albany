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

namespace LCM {

template<typename EvalT, typename Traits>
DamageLS<EvalT, Traits>::
DamageLS(Teuchos::ParameterList& p) :
  damageLS(p.get<std::string>("QP Variable Name"),
	   p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout"))
{
  Teuchos::ParameterList* dls_list = 
    p.get<Teuchos::ParameterList*>("Parameter List");

  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout>>("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  Teuchos::RCP<ParamLib> paramLib = 
    p.get< Teuchos::RCP<ParamLib>>("Parameter Library", Teuchos::null);

  std::string type = dls_list->get("Damage Length Scale Type", "Constant");
  if (type == "Constant") {
    is_constant = true;
    constant_value = dls_list->get("Value", 1.0);

    // Add Damage Length Scale as a Sacado-ized parameter
    this->registerSacadoParameter("Damage Length Scale", paramLib);
  }
  else if (type == "Truncated KL Expansion") {
    is_constant = false;
    PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>
      fx(p.get<std::string>("QP Coordinate Vector Name"), vector_dl);
    coordVec = fx;
    this->addDependentField(coordVec);

    exp_rf_kl = 
      Teuchos::rcp(new Stokhos::KL::ExponentialRandomField<RealType>(*dls_list));
    int num_KL = exp_rf_kl->stochasticDimension();

    // Add KL random variables as Sacado-ized parameters
    rv.resize(num_KL);
    for (int i=0; i<num_KL; i++) {
      std::string ss = Albany::strint("Damage Length Scale KL Random Variable",i);
      this->registerSacadoParameter(ss, paramLib);
      rv[i] = dls_list->get(ss, 0.0);
    }
  }
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
			       "Invalid damage length scale type " << type);
  } 

  this->addEvaluatedField(damageLS);
  this->setName("Damage Length Scale"+PHX::typeAsString<EvalT>());
}

// **********************************************************************
template<typename EvalT, typename Traits>
void DamageLS<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(damageLS,fm);
  if (!is_constant) this->utils.setFieldData(coordVec,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void DamageLS<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  int numCells = workset.numCells;

  if (is_constant) {
    for (int cell=0; cell < numCells; ++cell) {
      for (int qp=0; qp < numQPs; ++qp) {
	damageLS(cell,qp) = constant_value;
      }
    }
  }
  else {
    for (int cell=0; cell < numCells; ++cell) {
      for (int qp=0; qp < numQPs; ++qp) {
	Teuchos::Array<MeshScalarT> point(numDims);
	for (int i=0; i<numDims; i++)
	  point[i] = Sacado::ScalarValue<MeshScalarT>::eval(coordVec(cell,qp,i));
	damageLS(cell,qp) = exp_rf_kl->evaluate(point, rv);
      }
    }
  }
}

// **********************************************************************
template<typename EvalT,typename Traits>
typename DamageLS<EvalT,Traits>::ScalarT& 
DamageLS<EvalT,Traits>::getValue(const std::string &n)
{
  if (n == "Damage Length Scale")
    return constant_value;
  for (int i=0; i<rv.size(); i++) {
    if (n == Albany::strint("Damage Length Scale KL Random Variable",i))
      return rv[i];
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
			     std::endl <<
			     "Error! Logic error in getting paramter " << n
			     << " in DamageLS::getValue()" << std::endl);
  return constant_value;
}

// **********************************************************************
// **********************************************************************
}

