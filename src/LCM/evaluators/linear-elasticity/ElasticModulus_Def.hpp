//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <fstream>
#include "Albany_Utils.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "Teuchos_TestForException.hpp"

namespace LCM {

template <typename EvalT, typename Traits>
ElasticModulus<EvalT, Traits>::ElasticModulus(Teuchos::ParameterList& p)
    : elasticModulus(
          p.get<std::string>("QP Variable Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout"))
{
  Teuchos::ParameterList* elmd_list =
      p.get<Teuchos::ParameterList*>("Parameter List");

  Teuchos::RCP<PHX::DataLayout> vector_dl =
      p.get<Teuchos::RCP<PHX::DataLayout>>("QP Vector Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  vector_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  Teuchos::RCP<ParamLib> paramLib =
      p.get<Teuchos::RCP<ParamLib>>("Parameter Library", Teuchos::null);

  std::string type = elmd_list->get("Elastic Modulus Type", "Constant");
  if (type == "Constant") {
    is_constant    = true;
    constant_value = elmd_list->get<double>("Value");

    // Add Elastic Modulus as a Sacado-ized parameter
    this->registerSacadoParameter("Elastic Modulus", paramLib);
  }
//  else if (type == 'Variable') {
//	  is_constant = true; // this means no stochastic nature
//	  is_field = true;
//	  constant_value = elmd_list->get("Value", 1.0);
//
//	  // Add Elastic Modulus as a Sacado-ized parameter
//	  new Sacado::ParameterRegistration<EvalT, SPL_Traits>(
//	  	"Elastic Modulus", this, paramLib);
//
//  }
#ifdef ALBANY_STOKHOS
  else if (type == "Truncated KL Expansion") {
    is_constant = false;
    coordVec    = decltype(coordVec)(
        p.get<std::string>("QP Coordinate Vector Name"), vector_dl);
    this->addDependentField(coordVec);

    exp_rf_kl = Teuchos::rcp(
        new Stokhos::KL::ExponentialRandomField<RealType>(*elmd_list));
    int num_KL = exp_rf_kl->stochasticDimension();

    // Add KL random variables as Sacado-ized parameters
    rv.resize(num_KL);
    for (int i = 0; i < num_KL; i++) {
      std::string ss = Albany::strint("Elastic Modulus KL Random Variable", i);
      this->registerSacadoParameter(ss, paramLib);
      rv[i] = elmd_list->get(ss, 0.0);
    }
  }
#endif
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        Teuchos::Exceptions::InvalidParameter,
        "Invalid elastic modulus type " << type);
  }

  // Optional dependence on Temperature (E = E_ + dEdT * T)
  // Switched ON by sending Temperature field in p

  if (p.isType<std::string>("QP Temperature Name")) {
    Teuchos::RCP<PHX::DataLayout> scalar_dl =
        p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout");
    Temperature = decltype(Temperature)(
        p.get<std::string>("QP Temperature Name"), scalar_dl);
    this->addDependentField(Temperature);
    isThermoElastic = true;
    dEdT_value      = elmd_list->get("dEdT Value", 0.0);
    refTemp         = p.get<RealType>("Reference Temperature", 0.0);
    this->registerSacadoParameter("dEdT Value", paramLib);
  } else {
    isThermoElastic = false;
    dEdT_value      = 0.0;
  }

  // Optional dependence on porosity (E = E_ + dEdT * T)
  // Switched ON by sending Temperature field in p

  if (p.isType<std::string>("Porosity Name")) {
    Teuchos::RCP<PHX::DataLayout> scalar_dl =
        p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout");
    porosity =
        decltype(porosity)(p.get<std::string>("Porosity Name"), scalar_dl);
    this->addDependentField(porosity);
    isPoroElastic = true;

  } else {
    isPoroElastic = false;
  }

  this->addEvaluatedField(elasticModulus);
  this->setName("Elastic Modulus" + PHX::typeAsString<EvalT>());
}

// **********************************************************************
template <typename EvalT, typename Traits>
void
ElasticModulus<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(elasticModulus, fm);
  if (!is_constant) this->utils.setFieldData(coordVec, fm);
  if (isThermoElastic) this->utils.setFieldData(Temperature, fm);
  if (isPoroElastic) this->utils.setFieldData(porosity, fm);
}

// **********************************************************************
template <typename EvalT, typename Traits>
void
ElasticModulus<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  int numCells = workset.numCells;

  if (is_constant) {
    for (int cell = 0; cell < numCells; ++cell) {
      for (int qp = 0; qp < numQPs; ++qp) {
        elasticModulus(cell, qp) = constant_value;
      }
    }
  }
#ifdef ALBANY_STOKHOS
  else {
    for (int cell = 0; cell < numCells; ++cell) {
      for (int qp = 0; qp < numQPs; ++qp) {
        Teuchos::Array<MeshScalarT> point(numDims);
        for (int i = 0; i < numDims; i++)
          point[i] =
              Sacado::ScalarValue<MeshScalarT>::eval(coordVec(cell, qp, i));
        elasticModulus(cell, qp) = exp_rf_kl->evaluate(point, rv);
      }
    }
  }
#endif
  if (isThermoElastic) {
    for (int cell = 0; cell < numCells; ++cell) {
      for (int qp = 0; qp < numQPs; ++qp) {
        elasticModulus(cell, qp) +=
            dEdT_value * (Temperature(cell, qp) - refTemp);
      }
    }
  }
  if (isPoroElastic) {
    for (int cell = 0; cell < numCells; ++cell) {
      for (int qp = 0; qp < numQPs; ++qp) {
        // porosity dependent Young's Modulus. It will be replaced by
        // the hyperelasticity model in (Borja, Tamagnini and Amorosi, ASCE JGGE
        // 1997).
        elasticModulus(cell, qp) = constant_value;
        // 			*sqrt(2.0 - porosity(cell,qp));
      }
    }
  }
}

// **********************************************************************
template <typename EvalT, typename Traits>
typename ElasticModulus<EvalT, Traits>::ScalarT&
ElasticModulus<EvalT, Traits>::getValue(const std::string& n)
{
  if (n == "Elastic Modulus")
    return constant_value;
  else if (n == "dEdT Value")
    return dEdT_value;
#ifdef ALBANY_STOKHOS
  for (int i = 0; i < rv.size(); i++) {
    if (n == Albany::strint("Elastic Modulus KL Random Variable", i))
      return rv[i];
  }
#endif
  TEUCHOS_TEST_FOR_EXCEPTION(
      true,
      Teuchos::Exceptions::InvalidParameter,
      std::endl
          << "Error! Logic error in getting paramter " << n
          << " in ElasticModulus::getValue()" << std::endl);
  return constant_value;
}

// **********************************************************************
// **********************************************************************
}  // namespace LCM
