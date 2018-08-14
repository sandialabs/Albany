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
KCPermeability<EvalT, Traits>::KCPermeability(Teuchos::ParameterList& p)
    : kcPermeability(
          p.get<std::string>("Kozeny-Carman Permeability Name"),
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

  std::string type =
      elmd_list->get("Kozeny-Carman Permeability Type", "Constant");
  if (type == "Constant") {
    is_constant    = true;
    constant_value = elmd_list->get(
        "Value", 1.0e-5);  // default value=1, identical to Terzaghi stress

    // Add Kozeny-Carman Permeability as a Sacado-ized parameter
    this->registerSacadoParameter("Kozeny-Carman Permeability", paramLib);
  }
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
      std::string ss =
          Albany::strint("Kozeny-Carman Permeability KL Random Variable", i);
      this->registerSacadoParameter(ss, paramLib);
      rv[i] = elmd_list->get(ss, 0.0);
    }
  }
#endif
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        Teuchos::Exceptions::InvalidParameter,
        "Invalid Kozeny-Carman Permeability type " << type);
  }

  // Optional dependence on Temperature (E = E_ + dEdT * T)
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

  this->addEvaluatedField(kcPermeability);
  this->setName("Kozeny-Carman Permeability" + PHX::typeAsString<EvalT>());
}

// **********************************************************************
template <typename EvalT, typename Traits>
void
KCPermeability<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(kcPermeability, fm);
  if (!is_constant) this->utils.setFieldData(coordVec, fm);
  if (isPoroElastic) this->utils.setFieldData(porosity, fm);
}

// **********************************************************************
template <typename EvalT, typename Traits>
void
KCPermeability<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  int numCells = workset.numCells;

  if (is_constant) {
    for (int cell = 0; cell < numCells; ++cell) {
      for (int qp = 0; qp < numQPs; ++qp) {
        kcPermeability(cell, qp) = constant_value;
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
        kcPermeability(cell, qp) = exp_rf_kl->evaluate(point, rv);
      }
    }
  }
#endif
  if (isPoroElastic) {
    for (int cell = 0; cell < numCells; ++cell) {
      for (int qp = 0; qp < numQPs; ++qp) {
        // Cozeny Karman permeability equation
        kcPermeability(cell, qp) =
            constant_value * porosity(cell, qp) * porosity(cell, qp) *
            porosity(cell, qp) /
            ((1.0 - porosity(cell, qp)) * (1.0 - porosity(cell, qp)));
      }
    }
  }
}

// **********************************************************************
template <typename EvalT, typename Traits>
typename KCPermeability<EvalT, Traits>::ScalarT&
KCPermeability<EvalT, Traits>::getValue(const std::string& n)
{
  if (n == "Kozeny-Carman Permeability") return constant_value;
#ifdef ALBAY_STOKHOS
  for (int i = 0; i < rv.size(); i++) {
    if (n == Albany::strint("Kozeny-Carman Permeability KL Random Variable", i))
      return rv[i];
  }
#endif
  TEUCHOS_TEST_FOR_EXCEPTION(
      true,
      Teuchos::Exceptions::InvalidParameter,
      std::endl
          << "Error! Logic error in getting paramter " << n
          << " in KCPermeability::getValue()" << std::endl);
  return constant_value;
}

// **********************************************************************
// **********************************************************************
}  // namespace LCM
