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
BiotCoefficient<EvalT, Traits>::BiotCoefficient(Teuchos::ParameterList& p)
    : biotCoefficient(
          p.get<std::string>("Biot Coefficient Name"),
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

  std::string type = elmd_list->get("Biot Coefficient Type", "Constant");
  if (type == "Constant") {
    is_constant    = true;
    constant_value = elmd_list->get(
        "Value", 1.0);  // default value=1, identical to Terzaghi stress

    // Add Biot Coefficient as a Sacado-ized parameter
    this->registerSacadoParameter("Biot Coefficient", paramLib);
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
      std::string ss = Albany::strint("Biot Coefficient KL Random Variable", i);
      this->registerSacadoParameter(ss, paramLib);
      rv[i] = elmd_list->get(ss, 0.0);
    }
  }
#endif
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        Teuchos::Exceptions::InvalidParameter,
        "Invalid Biot coefficient type " << type);
  }

  // Optional dependence on Temperature (E = E_ + dEdT * T)
  // Switched ON by sending Temperature field in p

  isPoroElastic = true;
  Kskeleton_value =
      elmd_list->get("Skeleton Bulk Modulus Parameter Value", 10.0e5);
  this->registerSacadoParameter(
      "Skeleton Bulk Modulus Parameter Value", paramLib);
  Kgrain_value = elmd_list->get(
      "Grain Bulk Modulus Value", 10.0e12);  // typically Kgrain >> Kskeleton
  this->registerSacadoParameter("Grain Bulk Modulus Value", paramLib);

  this->addEvaluatedField(biotCoefficient);
  this->setName("Biot Coefficient" + PHX::typeAsString<EvalT>());
}

// **********************************************************************
template <typename EvalT, typename Traits>
void
BiotCoefficient<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(biotCoefficient, fm);
  if (!is_constant) this->utils.setFieldData(coordVec, fm);
  //  if (isPoroElastic) this->utils.setFieldData(porosity,fm);
}

// **********************************************************************
template <typename EvalT, typename Traits>
void
BiotCoefficient<EvalT, Traits>::evaluateFields(
    typename Traits::EvalData workset)
#ifdef __clang__
    // GAH - clang++ causes an FPE here when optimizing. Turn off opt for this
    // function.
    __attribute__((optnone))
#endif
{
  int numCells = workset.numCells;

  if (is_constant) {
    for (int cell = 0; cell < numCells; ++cell) {
      for (int qp = 0; qp < numQPs; ++qp) {
        biotCoefficient(cell, qp) = constant_value;
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
        biotCoefficient(cell, qp) = exp_rf_kl->evaluate(point, rv);
      }
    }
  }
#endif
  if (isPoroElastic) {
    for (int cell = 0; cell < numCells; ++cell) {
      for (int qp = 0; qp < numQPs; ++qp) {
        // assume that bulk modulus is linear with respect to porosity
        biotCoefficient(cell, qp) = 1.0 - Kskeleton_value / Kgrain_value;
      }
    }
  }
}

// **********************************************************************
template <typename EvalT, typename Traits>
typename BiotCoefficient<EvalT, Traits>::ScalarT&
BiotCoefficient<EvalT, Traits>::getValue(const std::string& n)
{
  if (n == "Biot Coefficient")
    return constant_value;
  else if (n == "Skeleton Bulk Modulus Parameter Value")
    return Kskeleton_value;
  else if (n == "Grain Bulk Modulus Value")
    return Kgrain_value;
#ifdef ALBANY_STOKHOS
  for (int i = 0; i < rv.size(); i++) {
    if (n == Albany::strint("Biot Coefficient KL Random Variable", i))
      return rv[i];
  }
#endif
  TEUCHOS_TEST_FOR_EXCEPTION(
      true,
      Teuchos::Exceptions::InvalidParameter,
      std::endl
          << "Error! Logic error in getting paramter " << n
          << " in BiotCoefficient::getValue()" << std::endl);
  return constant_value;
}

// **********************************************************************
// **********************************************************************
}  // namespace LCM
