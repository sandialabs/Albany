//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
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
RecoveryModulus<EvalT, Traits>::
RecoveryModulus(Teuchos::ParameterList& p) :
  recoveryModulus(p.get<std::string>("QP Variable Name"),
      p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout"))
{
  Teuchos::ParameterList* elmd_list = 
    p.get<Teuchos::ParameterList*>("Parameter List");

  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout>>("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  Teuchos::RCP<ParamLib> paramLib = 
    p.get< Teuchos::RCP<ParamLib>>("Parameter Library", Teuchos::null);

  std::string type = elmd_list->get("Recovery Modulus Type", "Constant");
  if (type == "Constant") {
    is_constant = true;
    constant_value = elmd_list->get("Value", 1.0);

    // Add Recovery Modulus as a Sacado-ized parameter
    this->registerSacadoParameter("Recovery Modulus", paramLib);
  }
#ifdef ALBANY_STOKHOS
  else if (type == "Truncated KL Expansion") {
    is_constant = false;
    PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>
      fx(p.get<std::string>("QP Coordinate Vector Name"), vector_dl);
    coordVec = fx;
    this->addDependentField(coordVec);

    exp_rf_kl = 
      Teuchos::rcp(new Stokhos::KL::ExponentialRandomField<RealType>(*elmd_list));
    int num_KL = exp_rf_kl->stochasticDimension();

    // Add KL random variables as Sacado-ized parameters
    rv.resize(num_KL);
    for (int i=0; i<num_KL; i++) {
      std::string ss = Albany::strint("Recovery Modulus KL Random Variable",i);
      this->registerSacadoParameter(ss, paramLib);
      rv[i] = elmd_list->get(ss, 0.0);
    }
  }
#endif
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
			       "Invalid recovery modulus type " << type);
  } 

  // Optional dependence on Temperature
  // Switched ON by sending Temperature field in p
  if ( p.isType<std::string>("QP Temperature Name") ) {
    Teuchos::RCP<PHX::DataLayout> scalar_dl =
      p.get< Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout");
    PHX::MDField<ScalarT,Cell,QuadPoint>
      tmp(p.get<std::string>("QP Temperature Name"), scalar_dl);
    Temperature = tmp;
    this->addDependentField(Temperature);
    isThermoElastic = true;
    c1 = elmd_list->get("c1 Value", 0.0);
    c2 = elmd_list->get("c2 Value", 0.0);
    refTemp = p.get<RealType>("Reference Temperature", 0.0);
    this->registerSacadoParameter("c1 Value", paramLib);
    this->registerSacadoParameter("c2 Value", paramLib);
  }
  else {
    isThermoElastic=false;
    c1=0.0;
    c2=0.0;
  }

  this->addEvaluatedField(recoveryModulus);
  this->setName("Recovery Modulus"+PHX::typeAsString<EvalT>());
}

// **********************************************************************
template<typename EvalT, typename Traits>
void RecoveryModulus<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(recoveryModulus,fm);
  if (!is_constant) this->utils.setFieldData(coordVec,fm);
  if (isThermoElastic) this->utils.setFieldData(Temperature,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void RecoveryModulus<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  int numCells = workset.numCells;

  if (is_constant) {
    for (int cell=0; cell < numCells; ++cell) {
      for (int qp=0; qp < numQPs; ++qp) {
    	  recoveryModulus(cell,qp) = constant_value;
      }
    }
  }
#ifdef ALBANY_STOKHOS
  else {
    for (int cell=0; cell < numCells; ++cell) {
      for (int qp=0; qp < numQPs; ++qp) {
    	  Teuchos::Array<MeshScalarT> point(numDims);
    	  for (int i=0; i<numDims; i++)
    		  point[i] = Sacado::ScalarValue<MeshScalarT>::eval(coordVec(cell,qp,i));
    	  	  recoveryModulus(cell,qp) = exp_rf_kl->evaluate(point, rv);
      }
    }
  }
#endif
  if (isThermoElastic) {
    for (int cell=0; cell < numCells; ++cell) {
      for (int qp=0; qp < numQPs; ++qp) {
    	  if (Temperature(cell,qp) != 0)
    		  recoveryModulus(cell,qp) = c1 * std::exp(c2 / Temperature(cell,qp));
    	  else
    		  recoveryModulus(cell,qp) = 0.0;
      }
    }
  }
}

// **********************************************************************
template<typename EvalT,typename Traits>
typename RecoveryModulus<EvalT,Traits>::ScalarT&
RecoveryModulus<EvalT,Traits>::getValue(const std::string &n)
{
  if (n == "Recovery Modulus")
    return constant_value;
  else if (n == "c1 Value")
    return c1;
  else if (n == "c2 Value")
	return c2;

#ifdef ALBANY_STOKHOS
  for (int i=0; i<rv.size(); i++) {
    if (n == Albany::strint("Recovery Modulus KL Random Variable",i))
      return rv[i];
  }
#endif
  TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
			     std::endl <<
			     "Error! Logic error in getting paramter " << n
			     << " in RecoveryModulus::getValue()" << std::endl);
  return constant_value;
}

// **********************************************************************
// **********************************************************************
}

