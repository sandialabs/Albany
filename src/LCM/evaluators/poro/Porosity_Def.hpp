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

//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
Porosity<EvalT, Traits>::Porosity(
    Teuchos::ParameterList&              p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : porosity(p.get<std::string>("Porosity Name"), dl->qp_scalar),
      is_constant(true),
      isCompressibleSolidPhase(false),
      isCompressibleFluidPhase(false),
      isPoroElastic(false),
      hasStrain(false),
      hasJ(false),
      hasTemp(false)
{
  Teuchos::ParameterList* porosity_list =
      p.get<Teuchos::ParameterList*>("Parameter List");

  std::vector<PHX::DataLayout::size_type> dims;
  dl->qp_vector->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  Teuchos::RCP<ParamLib> paramLib =
      p.get<Teuchos::RCP<ParamLib>>("Parameter Library", Teuchos::null);

  std::string type = porosity_list->get("Porosity Type", "Constant");
  if (type == "Constant") {
    is_constant = true;
    // Default value = 0 means no pores in the material
    constant_value = porosity_list->get("Value", 0.0);
    // Add Porosity as a Sacado-ized parameter
    this->registerSacadoParameter("Porosity", paramLib);
  }
#ifdef ALBANY_STOKHOS
  else if (type == "Truncated KL Expansion") {
    is_constant = false;
    coordVec    = decltype(coordVec)(
        p.get<std::string>("QP Coordinate Vector Name"), dl->qp_vector);
    this->addDependentField(coordVec);

    exp_rf_kl = Teuchos::rcp(
        new Stokhos::KL::ExponentialRandomField<RealType>(*porosity_list));
    int num_KL = exp_rf_kl->stochasticDimension();

    // Add KL random variables as Sacado-ized parameters
    rv.resize(num_KL);
    for (int i = 0; i < num_KL; i++) {
      std::string ss = Albany::strint("Porosity KL Random Variable", i);
      this->registerSacadoParameter(ss, paramLib);
      rv[i] = porosity_list->get(ss, 0.0);
    }
  }
#endif
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        Teuchos::Exceptions::InvalidParameter,
        "Invalid porosity type " << type);
  }

  // Optional dependence on porePressure and Biot coefficient
  // Switched ON by sending porePressure field in p
  if (p.isType<std::string>("Strain Name")) {
    hasStrain = true;

    strain = decltype(strain)(p.get<std::string>("Strain Name"), dl->qp_tensor);
    this->addDependentField(strain);

    isCompressibleSolidPhase = true;
    isCompressibleFluidPhase = true;
    isPoroElastic            = true;
    initialPorosityValue = porosity_list->get("Initial Porosity Value", 0.0);
    this->registerSacadoParameter("Initial Porosity Value", paramLib);
  } else if (p.isType<std::string>("DetDefGrad Name")) {
    hasJ = true;
    J    = decltype(J)(p.get<std::string>("DetDefGrad Name"), dl->qp_scalar);
    this->addDependentField(J);
    isPoroElastic        = true;
    initialPorosityValue = porosity_list->get("Initial Porosity Value", 0.0);
    this->registerSacadoParameter("Initial Porosity Value", paramLib);
  } else {
    // porosity will not change in this case.
    isPoroElastic        = false;
    initialPorosityValue = 0.0;
  }

  if (p.isType<std::string>("Biot Coefficient Name")) {
    biotCoefficient = decltype(biotCoefficient)(
        p.get<std::string>("Biot Coefficient Name"), dl->qp_scalar);
    isCompressibleSolidPhase = true;
    isCompressibleFluidPhase = true;
    isPoroElastic            = true;
    this->addDependentField(biotCoefficient);
  }

  if (p.isType<std::string>("QP Pore Pressure Name")) {
    porePressure = decltype(porePressure)(
        p.get<std::string>("QP Pore Pressure Name"), dl->qp_scalar);
    isCompressibleSolidPhase = true;
    isCompressibleFluidPhase = true;
    isPoroElastic            = true;
    this->addDependentField(porePressure);

    // typically Kgrain >> Kskeleton
    GrainBulkModulus = porosity_list->get("Grain Bulk Modulus Value", 10.0e12);
    this->registerSacadoParameter("Grain Bulk Modulus Value", paramLib);
  }

  if (p.isType<std::string>("QP Temperature Name")) {
    Temperature = decltype(Temperature)(
        p.get<std::string>("QP Temperature Name"), dl->qp_scalar);
    this->addDependentField(Temperature);

    if (p.isType<std::string>("Skeleton Thermal Expansion Name")) {
      skeletonThermalExpansion = decltype(skeletonThermalExpansion)(
          p.get<std::string>("Skeleton Thermal Expansion Name"), dl->qp_scalar);
      this->addDependentField(skeletonThermalExpansion);

      if (p.isType<std::string>("Reference Temperature Name")) {
        refTemperature = decltype(refTemperature)(
            p.get<std::string>("Reference Temperature Name"), dl->qp_scalar);
        hasTemp = true;
        this->addDependentField(refTemperature);
      }
    }
  }

  this->addEvaluatedField(porosity);
  this->setName("Porosity" + PHX::typeAsString<EvalT>());
}

//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void
Porosity<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(porosity, fm);
  if (!is_constant) this->utils.setFieldData(coordVec, fm);
  if (isPoroElastic && hasStrain) this->utils.setFieldData(strain, fm);
  if (isPoroElastic && hasJ) this->utils.setFieldData(J, fm);
  if (isPoroElastic && hasTemp) this->utils.setFieldData(Temperature, fm);
  if (isPoroElastic && hasTemp) this->utils.setFieldData(refTemperature, fm);
  if (isPoroElastic && hasTemp)
    this->utils.setFieldData(skeletonThermalExpansion, fm);
  if (isCompressibleSolidPhase) this->utils.setFieldData(biotCoefficient, fm);
  if (isCompressibleFluidPhase) this->utils.setFieldData(porePressure, fm);
}

//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void
Porosity<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  int numCells = workset.numCells;

  ScalarT temp;

  if (is_constant) {
    for (int cell = 0; cell < numCells; ++cell) {
      for (int qp = 0; qp < numQPs; ++qp) {
        porosity(cell, qp) = constant_value;
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
        porosity(cell, qp) = exp_rf_kl->evaluate(point, rv);
      }
    }
  }
#endif

  // if the porous media is deforming
  if ((isPoroElastic) && (isCompressibleSolidPhase) &&
      (isCompressibleFluidPhase)) {
    if (hasStrain) {
      for (int cell = 0; cell < numCells; ++cell) {
        for (int qp = 0; qp < numQPs; ++qp) {
          // small deformation; only valid for small porosity changes
          porosity(cell, qp) = initialPorosityValue;

          Teuchos::Array<MeshScalarT> point(numDims);

          for (int i = 0; i < numDims; i++) {
            porosity(cell, qp) =
                initialPorosityValue +
                biotCoefficient(cell, qp) * strain(cell, qp, i, i) +
                porePressure(cell, qp) *
                    (biotCoefficient(cell, qp) - initialPorosityValue) /
                    GrainBulkModulus;
          }
          // Set Warning message
          if (porosity(cell, qp) < 0) {
            std::cout << "negative porosity detected. Error! \n";
          }
          // // for debug
          // std::cout << "initial Porosity: " << initialPorosity_value << endl;
          // std::cout << "Pore Pressure: " << porePressure << endl;
          // std::cout << "Biot Coefficient: " << biotCoefficient << endl;
          // std::cout << "Grain Bulk Modulus " << GrainBulkModulus << endl;

          // porosity(cell,qp) += (1.0 - initialPorosity_value)
          //   /GrainBulkModulus*porePressure(cell,qp);
          // // for large deformation, \phi = J \dot \phi_{o}
        }
      }
    } else if (hasJ)
      for (int cell = 0; cell < numCells; ++cell) {
        for (int qp = 0; qp < numQPs; ++qp) {
          if (hasTemp == false) {
            porosity(cell, qp) =
                initialPorosityValue *
                std::exp(
                    GrainBulkModulus /
                        (porePressure(cell, qp) + GrainBulkModulus) *
                        biotCoefficient(cell, qp) * std::log(J(cell, qp)) +
                    biotCoefficient(cell, qp) /
                        (porePressure(cell, qp) + GrainBulkModulus) *
                        porePressure(cell, qp));
          } else {
            temp = 1.0 + porePressure(cell, qp) / GrainBulkModulus -
                   3.0 * skeletonThermalExpansion(cell, qp) *
                       (Temperature(cell, qp) - refTemperature(cell, qp));

            //          TEUCHOS_TEST_FOR_EXCEPTION(J(cell,qp) <= 0,
            //          std::runtime_error,
            //              " negative / zero volume detected in
            //              Porosity_def.hpp line " + __LINE__);
            // Note - J(cell, qp) equal to zero causes an FPE (GAH)

            porosity(cell, qp) =
                initialPorosityValue *
                std::exp(
                    biotCoefficient(cell, qp) * std::log(J(cell, qp)) +
                    biotCoefficient(cell, qp) / GrainBulkModulus *
                        porePressure(cell, qp) -
                    3.0 * J(cell, qp) * skeletonThermalExpansion(cell, qp) *
                        (Temperature(cell, qp) - refTemperature(cell, qp)) /
                        temp);
          }

          // Set Warning message
          if (porosity(cell, qp) < 0) {
            std::cout << "negative porosity detected. Error! \n";
          }
        }
      }
  } else {
    for (int cell = 0; cell < numCells; ++cell) {
      for (int qp = 0; qp < numQPs; ++qp) {
        porosity(cell, qp) = initialPorosityValue;
        /*
        if ( hasStrain ) {
          Teuchos::Array<MeshScalarT> point(numDims);
          for (int i=0; i<numDims; i++) {
            porosity(cell,qp) += strain(cell,qp,i,i);
          }
        }
        */
      }
    }
  }
}

//----------------------------------------------------------------------------
template <typename EvalT, typename Traits>
typename Porosity<EvalT, Traits>::ScalarT&
Porosity<EvalT, Traits>::getValue(const std::string& n)
{
  if (n == "Porosity")
    return constant_value;
  else if (n == "Initial Porosity Value")
    return initialPorosityValue;
  else if (n == "Grain Bulk Modulus Value")
    return GrainBulkModulus;
#ifdef ALBANY_STOKHOS
  for (int i = 0; i < rv.size(); i++) {
    if (n == Albany::strint("Porosity KL Random Variable", i)) return rv[i];
  }
#endif
  TEUCHOS_TEST_FOR_EXCEPTION(
      true,
      Teuchos::Exceptions::InvalidParameter,
      std::endl
          << "Error! Logic error in getting parameter " << n
          << " in Porosity::getValue()" << std::endl);
  return constant_value;
}

//----------------------------------------------------------------------------
}  // namespace LCM
