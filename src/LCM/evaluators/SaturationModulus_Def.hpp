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
SaturationModulus<EvalT, Traits>::
SaturationModulus(Teuchos::ParameterList& p) :
  satMod(p.get<std::string>("Saturation Modulus Name"),
	 p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout"))
{
  Teuchos::ParameterList* satmod_list = 
    p.get<Teuchos::ParameterList*>("Parameter List");

  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout>>("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  Teuchos::RCP<ParamLib> paramLib = 
    p.get< Teuchos::RCP<ParamLib>>("Parameter Library", Teuchos::null);

  std::string type = satmod_list->get("Saturation Modulus Type", "Constant");
  if (type == "Constant") {
    is_constant = true;
    constant_value = satmod_list->get("Value", 0.0);

    // Add Saturation Modulus as a Sacado-ized parameter
    this->registerSacadoParameter("Saturation Modulus", paramLib);
  }
#ifdef ALBANY_STOKHOS
  else if (type == "Truncated KL Expansion") {
    is_constant = false;
    PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>
      fx(p.get<std::string>("QP Coordinate Vector Name"), vector_dl);
    coordVec = fx;
    this->addDependentField(coordVec);

    exp_rf_kl = 
      Teuchos::rcp(new Stokhos::KL::ExponentialRandomField<RealType>(*satmod_list));
    int num_KL = exp_rf_kl->stochasticDimension();

    // Add KL random variables as Sacado-ized parameters
    rv.resize(num_KL);
    for (int i=0; i<num_KL; i++) {
      std::string ss = Albany::strint("Saturation Modulus KL Random Variable",i);
      this->registerSacadoParameter(ss, paramLib);
      rv[i] = satmod_list->get(ss, 0.0);
    }
  }
#endif
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
			       "Invalid saturation modulus type " << type);
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
    dSdT_value = satmod_list->get("dSdT Value", 0.0);
    refTemp = p.get<RealType>("Reference Temperature", 0.0);
    this->registerSacadoParameter("dSdT Value", paramLib);
  }
  else {
    isThermoElastic=false;
    dSdT_value=0.0;
  }

  this->addEvaluatedField(satMod);
  this->setName("Saturation Modulus"+PHX::typeAsString<EvalT>());
}

// **********************************************************************
template<typename EvalT, typename Traits>
void SaturationModulus<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(satMod,fm);
  if (!is_constant) this->utils.setFieldData(coordVec,fm);
  if (isThermoElastic) this->utils.setFieldData(Temperature,fm);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void SaturationModulus<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  bool print = false;
  //if (typeid(ScalarT) == typeid(RealType)) print = true;

  if (print)
    std::cout << " *** SaturatioModulus *** " << std::endl;

  int numCells = workset.numCells;

  if (is_constant) {
    for (int cell=0; cell < numCells; ++cell) {
      for (int qp=0; qp < numQPs; ++qp) {
	satMod(cell,qp) = constant_value;
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
	satMod(cell,qp) = exp_rf_kl->evaluate(point, rv);
      }
    }
  }
#endif
  if (isThermoElastic) {
    for (int cell=0; cell < numCells; ++cell) {
      for (int qp=0; qp < numQPs; ++qp) {
	satMod(cell,qp) -= dSdT_value * (Temperature(cell,qp) - refTemp);
        if (print)
        {
          std::cout << "    S   : " << satMod(cell,qp) << std::endl;
          std::cout << "    temp: " << Temperature(cell,qp) << std::endl;
          std::cout << "    dSdT: " << dSdT_value << std::endl;
          std::cout << "    refT: " << refTemp << std::endl;
        }

      }
    }
  }
}

// **********************************************************************
template<typename EvalT,typename Traits>
typename SaturationModulus<EvalT,Traits>::ScalarT& 
SaturationModulus<EvalT,Traits>::getValue(const std::string &n)
{
  if (n == "Saturation Modulus")
    return constant_value;
  else if (n == "dSdT Value")
    return dSdT_value;
#ifdef ALBANY_STKHOS
  for (int i=0; i<rv.size(); i++) {
    if (n == Albany::strint("Saturation Modulus KL Random Variable",i))
      return rv[i];
  }
#endif
  TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
			     std::endl <<
			     "Error! Logic error in getting paramter " << n
			     << " in SaturationModulus::getValue()" << std::endl);
  return constant_value;
}

// **********************************************************************
// **********************************************************************
}

