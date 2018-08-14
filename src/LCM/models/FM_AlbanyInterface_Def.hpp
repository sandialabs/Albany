//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include <MiniTensor.h>
#include "Albany_Utils.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"

namespace LCM {

/******************************************************************************/
template <typename EvalT, typename Traits>
FerroicDriver<EvalT, Traits>::FerroicDriver(
    Teuchos::ParameterList*              p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : LCM::ConstitutiveModel<EvalT, Traits>(p, dl)
/******************************************************************************/
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      num_dims_ != FM::THREE_D,
      std::invalid_argument,
      ">>> ERROR (FerroicModel): Only valid for 3D.");

  ferroicModel = Teuchos::rcp(new FM::FerroicModel<EvalT>());

  // PARSE MATERIAL BASIS
  //
  minitensor::Tensor<RealType, FM::THREE_D>& R = ferroicModel->getBasis();
  R.set_dimension(FM::THREE_D);
  R.clear();
  if (p->isType<Teuchos::ParameterList>("Material Basis")) {
    const Teuchos::ParameterList& pBasis =
        p->get<Teuchos::ParameterList>("Material Basis");
    LCM::parseBasis(pBasis, R);
  } else {
    R(0, 0) = 1.0;
    R(1, 1) = 1.0;
    R(2, 2) = 1.0;
  }

  // PARSE INITIAL BIN FRACTIONS
  //
  Teuchos::Array<RealType>& initialBinFractions =
      ferroicModel->getInitialBinFractions();
  if (p->isType<Teuchos::Array<RealType>>("Bin Fractions"))
    initialBinFractions = p->get<Teuchos::Array<RealType>>("Bin Fractions");
  else
    initialBinFractions.resize(0);

  // PARSE PHASES
  //
  Teuchos::Array<Teuchos::RCP<FM::CrystalPhase>>& crystalPhases =
      ferroicModel->getCrystalPhases();
  int nphases = p->get<int>("Number of Phases");
  for (int i = 0; i < nphases; i++) {
    Teuchos::ParameterList& pParam =
        p->get<Teuchos::ParameterList>(Albany::strint("Phase", i + 1));
    minitensor::Tensor4<RealType, FM::THREE_D> C;
    LCM::parseTensor4(pParam, C);
    minitensor::Tensor3<RealType, FM::THREE_D> h;
    LCM::parseTensor3(pParam, h);
    minitensor::Tensor<RealType, FM::THREE_D> eps;
    LCM::parseTensor(pParam, eps);
    crystalPhases.push_back(Teuchos::rcp(new FM::CrystalPhase(R, C, h, eps)));
  }

  // PARSE VARIANTS
  //
  Teuchos::Array<FM::CrystalVariant>& crystalVariants =
      ferroicModel->getCrystalVariants();
  if (initialBinFractions.size() > 0) {
    Teuchos::ParameterList& vParams =
        p->get<Teuchos::ParameterList>("Variants");
    int nvars = vParams.get<int>("Number of Variants");
    crystalVariants.resize(nvars);
    TEUCHOS_TEST_FOR_EXCEPTION(
        initialBinFractions.size() != nvars,
        std::invalid_argument,
        ">>> ERROR (FerroicModel): 'Number of Variants' must equal length of "
        "'Bin Fractions' array");
    for (int i = 0; i < nvars; i++) {
      Teuchos::ParameterList& vParam =
          vParams.get<Teuchos::ParameterList>(Albany::strint("Variant", i + 1));
      crystalVariants[i] = parseCrystalVariant(crystalPhases, vParam);
    }
  } else {
    // no variants specified.  Create single dummy variant.
    initialBinFractions.resize(1);
    initialBinFractions[0] = 1.0;
  }

  // PARSE CRITICAL ENERGIES
  //
  int                       nVariants = crystalVariants.size();
  Teuchos::Array<RealType>& tBarrier  = ferroicModel->getTransitionBarrier();
  tBarrier.resize(nVariants * nVariants);
  if (p->isType<Teuchos::ParameterList>("Critical Values")) {
    const Teuchos::ParameterList& cParams =
        p->get<Teuchos::ParameterList>("Critical Values");
    int transitionIndex = 0;
    for (int i = 0; i < nVariants; i++) {
      Teuchos::Array<RealType> array = cParams.get<Teuchos::Array<RealType>>(
          Albany::strint("Variant", i + 1));
      TEUCHOS_TEST_FOR_EXCEPTION(
          array.size() != nVariants,
          std::invalid_argument,
          ">>> ERROR (FerroicModel): List of critical values for variant "
              << i + 1 << " is wrong length");
      for (int j = 0; j < nVariants; j++) {
        tBarrier[transitionIndex] = array[j];
        transitionIndex++;
      }
    }
  }

  // DEFINE THE EVALUATED FIELDS
  //
  stressName = "Stress";
  this->eval_field_map_.insert(std::make_pair(stressName, dl->qp_tensor));

  edispName = "Electric Displacement";
  this->eval_field_map_.insert(std::make_pair(edispName, dl->qp_vector));

  // DEFINE THE DEPENDENT FIELDS
  //
  strainName = "Strain";
  this->dep_field_map_.insert(std::make_pair(strainName, dl->qp_tensor));

  efieldName = "Electric Potential Gradient";
  this->dep_field_map_.insert(std::make_pair(efieldName, dl->qp_vector));

  // DEFINE STATE VARIABLES (output)
  //
  for (int i = 0; i < nVariants; i++) {
    std::string binName = Albany::strint("Bin Fraction", i + 1);
    binNames.push_back(binName);
    this->eval_field_map_.insert(std::make_pair(binName, dl->qp_scalar));
  }

  // bin fractions
  for (int i = 0; i < nVariants; i++) {
    this->num_state_variables_++;
    this->state_var_names_.push_back(binNames[i]);
    this->state_var_layouts_.push_back(dl->qp_scalar);
    this->state_var_init_types_.push_back("scalar");
    this->state_var_init_values_.push_back(initialBinFractions[i]);
    this->state_var_old_state_flags_.push_back(true);
    this->state_var_output_flags_.push_back(true);
  }

  // stress
  this->num_state_variables_++;
  this->state_var_names_.push_back(stressName);
  this->state_var_layouts_.push_back(dl->qp_tensor);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(true);
  // strain
  this->num_state_variables_++;
  this->state_var_names_.push_back(strainName);
  this->state_var_layouts_.push_back(dl->qp_tensor);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(true);

  // edisp
  this->num_state_variables_++;
  this->state_var_names_.push_back(edispName);
  this->state_var_layouts_.push_back(dl->qp_vector);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(true);
  // efield
  this->num_state_variables_++;
  this->state_var_names_.push_back(efieldName);
  this->state_var_layouts_.push_back(dl->qp_vector);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(true);

  ferroicModel->PostParseInitialize();
}

/******************************************************************************/
template <typename EvalT, typename Traits>
void
FerroicDriver<EvalT, Traits>::computeState(
    typename Traits::EvalData workset,
    DepFieldMap               dep_fields,
    FieldMap                  eval_fields)
/******************************************************************************/
{
  auto strain = *dep_fields[strainName];
  auto Gradp  = *dep_fields[efieldName];

  auto stress = *eval_fields[stressName];
  auto edisp  = *eval_fields[edispName];

  int                                   nVariants = binNames.size();
  Teuchos::Array<PHX::MDField<ScalarT>> newBinFractions(nVariants);
  Teuchos::Array<Albany::MDArray>       oldBinFractions(nVariants);
  for (int i = 0; i < nVariants; i++) {
    oldBinFractions[i] = (*workset.stateArrayPtr)[binNames[i] + "_old"];
    newBinFractions[i] = *eval_fields[binNames[i]];
  }

  int numCells = workset.numCells;

  minitensor::Tensor<ScalarT, FM::THREE_D> X, x;
  minitensor::Vector<ScalarT, FM::THREE_D> E, D;
  Teuchos::Array<RealType>                 oldfractions(nVariants);
  Teuchos::Array<ScalarT>                  newfractions(nVariants);

  for (int cell = 0; cell < numCells; ++cell) {
    for (int qp = 0; qp < num_pts_; ++qp) {
      x.fill(strain, cell, qp, 0, 0);
      E.fill(Gradp, cell, qp, 0);
      E *= -1.0;

      for (int vnt = 0; vnt < nVariants; vnt++) {
        oldfractions[vnt] = oldBinFractions[vnt](cell, qp);
        //        newfractions[vnt] = newBinFractions[vnt](cell,qp);
      }

      ferroicModel->computeState(
          /* In:  strain, Efield, ...  */ x,
          E,
          oldfractions,
          /* Out: stress, Edisp,  ...  */ X,
          D,
          newfractions);

      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) stress(cell, qp, i, j) = X(i, j);

      for (int i = 0; i < 3; i++) edisp(cell, qp, i) = D(i);

      for (int vnt = 0; vnt < nVariants; vnt++)
        newBinFractions[vnt](cell, qp) = newfractions[vnt];
    }
  }
}

/******************************************************************************/
template <typename EvalT, typename Traits>
void
FerroicDriver<EvalT, Traits>::computeStateParallel(
    typename Traits::EvalData workset,
    DepFieldMap               dep_fields,
    FieldMap                  eval_fields)
/******************************************************************************/
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      true,
      std::invalid_argument,
      ">>> ERROR (FerroicDriver): computeStateParallel not implemented");
}

/******************************************************************************/
void
parseBasis(
    const Teuchos::ParameterList&              pBasis,
    minitensor::Tensor<RealType, FM::THREE_D>& R)
/******************************************************************************/
{
  if (pBasis.isType<Teuchos::Array<RealType>>("X axis")) {
    Teuchos::Array<RealType> Xhat =
        pBasis.get<Teuchos::Array<RealType>>("X axis");
    R(0, 0) = Xhat[0];
    R(0, 1) = Xhat[1];
    R(0, 2) = Xhat[2];
  }
  if (pBasis.isType<Teuchos::Array<RealType>>("Y axis")) {
    Teuchos::Array<RealType> Yhat =
        pBasis.get<Teuchos::Array<RealType>>("Y axis");
    R(1, 0) = Yhat[0];
    R(1, 1) = Yhat[1];
    R(1, 2) = Yhat[2];
  }
  if (pBasis.isType<Teuchos::Array<RealType>>("Z axis")) {
    Teuchos::Array<RealType> Zhat =
        pBasis.get<Teuchos::Array<RealType>>("Z axis");
    R(2, 0) = Zhat[0];
    R(2, 1) = Zhat[1];
    R(2, 2) = Zhat[2];
  }
}

/******************************************************************************/
void
parseTensor4(
    const Teuchos::ParameterList&               cParam,
    minitensor::Tensor4<RealType, FM::THREE_D>& C)
/******************************************************************************/
{
  // JR:  This should be generalized to read stiffness tensors of various
  // symmetries.

  // parse
  //
  RealType C11 = cParam.get<RealType>("C11");
  RealType C33 = cParam.get<RealType>("C33");
  RealType C12 = cParam.get<RealType>("C12");
  RealType C23 = cParam.get<RealType>("C23");
  RealType C44 = cParam.get<RealType>("C44");
  RealType C66 = cParam.get<RealType>("C66");

  C.clear();

  C(0, 0, 0, 0) = C11;
  C(0, 0, 1, 1) = C12;
  C(0, 0, 2, 2) = C23;
  C(1, 1, 0, 0) = C12;
  C(1, 1, 1, 1) = C11;
  C(1, 1, 2, 2) = C23;
  C(2, 2, 0, 0) = C23;
  C(2, 2, 1, 1) = C23;
  C(2, 2, 2, 2) = C33;
  C(0, 1, 0, 1) = C66 / 2.0;
  C(1, 0, 1, 0) = C66 / 2.0;
  C(0, 2, 0, 2) = C44 / 2.0;
  C(2, 0, 2, 0) = C44 / 2.0;
  C(1, 2, 1, 2) = C44 / 2.0;
  C(2, 1, 2, 1) = C44 / 2.0;
}
/******************************************************************************/
void
parseTensor3(
    const Teuchos::ParameterList&               cParam,
    minitensor::Tensor3<RealType, FM::THREE_D>& h)
/******************************************************************************/
{
  // JR:  This should be generalized to read piezoelectric tensors of various
  // symmetries.

  // parse
  //
  RealType h31 = cParam.get<RealType>("h31");
  RealType h33 = cParam.get<RealType>("h33");
  RealType h15 = cParam.get<RealType>("h15");

  h.clear();
  h(0, 0, 2) = h15 / 2.0;
  h(0, 2, 0) = h15 / 2.0;
  h(1, 1, 2) = h15 / 2.0;
  h(1, 2, 1) = h15 / 2.0;
  h(2, 0, 0) = h31;
  h(2, 1, 1) = h31;
  h(2, 2, 2) = h33;
}
/******************************************************************************/
void
parseTensor(
    const Teuchos::ParameterList&              cParam,
    minitensor::Tensor<RealType, FM::THREE_D>& e)
/******************************************************************************/
{
  // JR:  This should be generalized to read permittivity tensors of various
  // symmetries.

  // parse
  //
  RealType E11 = cParam.get<RealType>("Eps11");
  RealType E33 = cParam.get<RealType>("Eps33");

  e.clear();
  e(0, 0) = E11;
  e(1, 1) = E11;
  e(2, 2) = E33;
}

/******************************************************************************/
FM::CrystalVariant
parseCrystalVariant(
    const Teuchos::Array<Teuchos::RCP<FM::CrystalPhase>>& phases,
    const Teuchos::ParameterList&                         vParam)
/******************************************************************************/
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      phases.size() == 0,
      std::invalid_argument,
      ">>> ERROR (FerroicModel): CrystalVariant constructor passed empty list "
      "of Phases.");

  FM::CrystalVariant cv;

  int phaseIndex;
  if (vParam.isType<int>("Phase")) {
    phaseIndex = vParam.get<int>("Phase");
    phaseIndex--;  // Ids are one-based.  Indices are zero-based.
  } else
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        std::invalid_argument,
        ">>> ERROR (FerroicModel): Crystal variants require a phase.");

  TEUCHOS_TEST_FOR_EXCEPTION(
      phaseIndex < 0 || phaseIndex >= phases.size(),
      std::invalid_argument,
      ">>> ERROR (FerroicModel): Requested phase has not been defined.");

  if (vParam.isType<Teuchos::ParameterList>("Crystallographic Basis")) {
    cv.R.set_dimension(phases[phaseIndex]->C.get_dimension());
    const Teuchos::ParameterList& pBasis =
        vParam.get<Teuchos::ParameterList>("Crystallographic Basis");
    LCM::parseBasis(pBasis, cv.R);
  } else
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        std::invalid_argument,
        ">>> ERROR (FerroicModel): Crystal variants require a crystallograph "
        "basis.");

  if (vParam.isType<Teuchos::Array<RealType>>("Spontaneous Polarization")) {
    Teuchos::Array<RealType> inVals =
        vParam.get<Teuchos::Array<RealType>>("Spontaneous Polarization");
    TEUCHOS_TEST_FOR_EXCEPTION(
        inVals.size() != FM::THREE_D,
        std::invalid_argument,
        ">>> ERROR (FerroicModel): Expected 3 terms 'Spontaneous Polarization' "
        "vector.");
    cv.spontEDisp.set_dimension(FM::THREE_D);
    for (int i = 0; i < FM::THREE_D; i++) cv.spontEDisp(i) = inVals[i];
  } else
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        std::invalid_argument,
        ">>> ERROR (FerroicModel): Crystal variants require a 'Spontaneous "
        "Polarization'.");

  if (vParam.isType<Teuchos::Array<RealType>>("Spontaneous Strain")) {
    Teuchos::Array<RealType> inVals =
        vParam.get<Teuchos::Array<RealType>>("Spontaneous Strain");
    TEUCHOS_TEST_FOR_EXCEPTION(
        inVals.size() != 6,
        std::invalid_argument,
        ">>> ERROR (FerroicModel): Expected 6 voigt terms 'Spontaneous Strain' "
        "tensor.");
    cv.spontStrain.set_dimension(FM::THREE_D);
    cv.spontStrain(0, 0) = inVals[0];
    cv.spontStrain(1, 1) = inVals[1];
    cv.spontStrain(2, 2) = inVals[2];
    cv.spontStrain(1, 2) = inVals[3];
    cv.spontStrain(0, 2) = inVals[4];
    cv.spontStrain(0, 1) = inVals[5];
    cv.spontStrain(2, 1) = inVals[3];
    cv.spontStrain(2, 0) = inVals[4];
    cv.spontStrain(1, 0) = inVals[5];
  } else
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        std::invalid_argument,
        ">>> ERROR (FerroicModel): Crystal variants require a 'Spontaneous "
        "Strain'.");

  cv.C.set_dimension(phases[phaseIndex]->C.get_dimension());
  cv.C.clear();
  FM::changeBasis(cv.C, phases[phaseIndex]->C, cv.R);

  cv.h.set_dimension(phases[phaseIndex]->h.get_dimension());
  cv.h.clear();
  FM::changeBasis(cv.h, phases[phaseIndex]->h, cv.R);

  cv.b.set_dimension(phases[phaseIndex]->b.get_dimension());
  cv.b.clear();
  FM::changeBasis(cv.b, phases[phaseIndex]->b, cv.R);

  return cv;
}
}  // namespace LCM
