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
ShearModulus<EvalT, Traits>::
ShearModulus(Teuchos::ParameterList& p) :
  shearModulus(p.get<std::string>("QP Variable Name"),
	       p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout"))
{
  Teuchos::ParameterList* shmd_list = 
    p.get<Teuchos::ParameterList*>("Parameter List");

  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout>>("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  Teuchos::RCP<ParamLib> paramLib = 
    p.get< Teuchos::RCP<ParamLib>>("Parameter Library", Teuchos::null);

  std::string type = shmd_list->get("Shear Modulus Type", "Constant");
  if (type == "Constant") {
    is_constant = true;
    constant_value = shmd_list->get("Value", 1.0);

    // Add Shear Modulus as a Sacado-ized parameter
    this->registerSacadoParameter("Shear Modulus", paramLib);
  }
#ifdef ALBANY_STOKHOS
  else if (type == "Truncated KL Expansion") {
    is_constant = false;
    PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>
      fx(p.get<std::string>("QP Coordinate Vector Name"), vector_dl);
    coordVec = fx;
    this->addDependentField(coordVec);

    exp_rf_kl = 
      Teuchos::rcp(new Stokhos::KL::ExponentialRandomField<RealType>(*shmd_list));
    int num_KL = exp_rf_kl->stochasticDimension();

    // Add KL random variables as Sacado-ized parameters
    rv.resize(num_KL);
    for (int i=0; i<num_KL; i++) {
      std::string ss = Albany::strint("Shear Modulus KL Random Variable",i);
      this->registerSacadoParameter(ss, paramLib);
      rv[i] = shmd_list->get(ss, 0.0);
    }
  }
#endif
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
			       "Invalid shear modulus type " << type);
  } 

  if ( p.isType<std::string>("QP Temperature Name") ) {
    Teuchos::RCP<PHX::DataLayout> scalar_dl =
      p.get< Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout");
    PHX::MDField<ScalarT,Cell,QuadPoint>
      tmp(p.get<std::string>("QP Temperature Name"), scalar_dl);
    Temperature = tmp;
    this->addDependentField(Temperature);
    isThermoElastic = true;
    dmudT_value = shmd_list->get("dmudT Value", 0.0);
    refTemp = p.get<RealType>("Reference Temperature", 0.0);
    this->registerSacadoParameter("dmudT Value", paramLib);
  }
  else {
    isThermoElastic=false;
    dmudT_value=0.0;
  }

  this->addEvaluatedField(shearModulus);
  this->setName("Shear Modulus"+PHX::typeAsString<EvalT>());
}

// **********************************************************************
template<typename EvalT, typename Traits>
void ShearModulus<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(shearModulus,fm);
  if (!is_constant) this->utils.setFieldData(coordVec,fm);
  if (isThermoElastic) this->utils.setFieldData(Temperature,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void ShearModulus<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  int numCells = workset.numCells;

  if (is_constant) {
    for (int cell=0; cell < numCells; ++cell) {
      for (int qp=0; qp < numQPs; ++qp) {
	shearModulus(cell,qp) = constant_value;
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
	shearModulus(cell,qp) = exp_rf_kl->evaluate(point, rv);
      }
    }
  }
#endif
  if (isThermoElastic) {
    for (int cell=0; cell < numCells; ++cell) {
      for (int qp=0; qp < numQPs; ++qp) {
	shearModulus(cell,qp) -= dmudT_value * (Temperature(cell,qp) - refTemp);
      }
    }
  }
}

// **********************************************************************
template<typename EvalT,typename Traits>
typename ShearModulus<EvalT,Traits>::ScalarT& 
ShearModulus<EvalT,Traits>::getValue(const std::string &n)
{
  if (n == "Shear Modulus")
    return constant_value;
  else if (n == "dmudT Value")
    return dmudT_value;
#ifdef ALBANY_STOKHOS
  for (int i=0; i<rv.size(); i++) {
    if (n == Albany::strint("Shear Modulus KL Random Variable",i))
      return rv[i];
  }
#endif
  TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
			     std::endl <<
			     "Error! Logic error in getting paramter " << n
			     << " in ShearModulus::getValue()" << std::endl);
  return constant_value;
}

// **********************************************************************
// **********************************************************************
}

