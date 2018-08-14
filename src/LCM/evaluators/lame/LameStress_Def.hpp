//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <MiniTensor.h>
#include "Albany_MaterialDatabase.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"

namespace LCM {

template <typename EvalT, typename Traits>
LameStressBase<EvalT, Traits>::LameStressBase(Teuchos::ParameterList& p)
    : defGradField(
          p.get<std::string>("DefGrad Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout")),
      stressField(
          p.get<std::string>("Stress Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout")),
      lameMaterialModel(Teuchos::RCP<LameMaterial>())
{
  // Pull out numQPs and numDims from a Layout
  tensor_dl = p.get<Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  tensor_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  TEUCHOS_TEST_FOR_EXCEPTION(
      this->numDims != 3,
      Teuchos::Exceptions::InvalidParameter,
      " LAME materials enabled only for three-dimensional analyses.");

  defGradName = p.get<std::string>("DefGrad Name") + "_old";
  this->addDependentField(defGradField);

  stressName = p.get<std::string>("Stress Name") + "_old";
  this->addEvaluatedField(stressField);

  this->setName("LameStress" + PHX::typeAsString<EvalT>());

  // Default to getting material info form base input file (possibley
  // overwritten later)
  lameMaterialModelName = p.get<std::string>("Lame Material Model", "Elastic");
  Teuchos::ParameterList& lameMaterialParameters =
      p.sublist("Lame Material Parameters");

  // Code to allow material data to come from materials.xml data file
  int haveMatDB = p.get<bool>("Have MatDB", false);

  std::string ebName = p.get<std::string>("Element Block Name", "Missing");

  // Check for material database file
  if (haveMatDB) {
    // Check if material database will be supplying the data
    bool dataFromDatabase = lameMaterialParameters.get<bool>(
        "Material Dependent Data Source", false);

    // If so, overwrite material model and data from database file
    if (dataFromDatabase) {
      Teuchos::RCP<Albany::MaterialDatabase> materialDB =
          p.get<Teuchos::RCP<Albany::MaterialDatabase>>("MaterialDB");

      lameMaterialModelName = materialDB->getElementBlockParam<std::string>(
          ebName, "Lame Material Model");
      lameMaterialParameters = materialDB->getElementBlockSublist(
          ebName, "Lame Material Parameters");
    }
  }

  // Initialize the LAME material model
  // This assumes that there is a single material model associated with this
  // evaluator and that the material properties are constant (read directly
  // from input deck parameter list)
  lameMaterialModel = LameUtils::constructLameMaterialModel(
      lameMaterialModelName, lameMaterialParameters);

  // Get a list of the LAME material model state variable names
  lameMaterialModelStateVariableNames = LameUtils::getStateVariableNames(
      lameMaterialModelName, lameMaterialParameters);

  // Declare the state variables as evaluated fields (type is always double)
  Teuchos::RCP<PHX::DataLayout> dataLayout =
      p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout");
  for (unsigned int i = 0; i < lameMaterialModelStateVariableNames.size();
       ++i) {
    PHX::MDField<ScalarT, Cell, QuadPoint> lameMaterialModelStateVariableField(
        lameMaterialModelStateVariableNames[i], dataLayout);
    this->addEvaluatedField(lameMaterialModelStateVariableField);
    lameMaterialModelStateVariableFields.push_back(
        lameMaterialModelStateVariableField);
  }
}

template <typename EvalT, typename Traits>
void
LameStressBase<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(defGradField, fm);
  this->utils.setFieldData(stressField, fm);
  for (unsigned int i = 0; i < lameMaterialModelStateVariableFields.size(); ++i)
    this->utils.setFieldData(lameMaterialModelStateVariableFields[i], fm);

  typedef PHX::KokkosViewFactory<RealType, PHX::Device> ViewFactory;
  this->stressFieldRealType = PHX::MDField<RealType, Cell, QuadPoint, Dim, Dim>(
      "stress_RealType", this->tensor_dl);
  this->defGradFieldRealType =
      PHX::MDField<RealType, Cell, QuadPoint, Dim, Dim>(
          "defGrad_RealType", this->tensor_dl);
  this->stressFieldRealType.setFieldData(
      ViewFactory::buildView(this->stressFieldRealType.fieldTag()));
  this->defGradFieldRealType.setFieldData(
      ViewFactory::buildView(this->defGradFieldRealType.fieldTag()));
}

template <typename EvalT, typename Traits>
void
LameStressBase<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
      "LameStressBase::evaluateFields not implemented for this template type",
      Teuchos::Exceptions::InvalidParameter,
      "Need specialization.");
}

template <typename Traits>
void
LameStress<PHAL::AlbanyTraits::Residual, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  Teuchos::RCP<LameMatParams> matp = Teuchos::rcp(new LameMatParams());
  this->setMatP(matp, workset);

  this->calcStressRealType(
      this->stressField, this->defGradField, workset, matp);

  this->freeMatP(matp);
}

template <typename Traits>
void
LameStress<PHAL::AlbanyTraits::Jacobian, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  // This block forces stressField Fad to allocate deriv array --is ther a
  // better way?
  PHAL::AlbanyTraits::Jacobian::ScalarT scalarToForceAllocation =
      this->defGradField(0, 0, 0, 0);
  for (int cell = 0; cell < (int)workset.numCells; ++cell)
    for (int qp = 0; qp < (int)this->numQPs; ++qp)
      for (int i = 0; i < (int)this->numDims; ++i)
        for (int j = 0; j < (int)this->numDims; ++j)
          this->stressField(cell, qp, i, j) = scalarToForceAllocation;

  // Allocate double arrays in matp
  Teuchos::RCP<LameMatParams> matp = Teuchos::rcp(new LameMatParams());
  this->setMatP(matp, workset);

  // Begin Finite Difference
  // Do Base unperturbed case
  for (int cell = 0; cell < (int)workset.numCells; ++cell)
    for (int qp = 0; qp < (int)this->numQPs; ++qp)
      for (int i = 0; i < (int)this->numDims; ++i)
        for (int j = 0; j < (int)this->numDims; ++j)
          this->defGradFieldRealType(cell, qp, i, j) =
              this->defGradField(cell, qp, i, j).val();

  this->calcStressRealType(
      this->stressFieldRealType, this->defGradFieldRealType, workset, matp);

  for (int cell = 0; cell < (int)workset.numCells; ++cell)
    for (int qp = 0; qp < (int)this->numQPs; ++qp)
      for (int i = 0; i < (int)this->numDims; ++i)
        for (int j = 0; j < (int)this->numDims; ++j)
          this->stressField(cell, qp, i, j).val() =
              this->stressFieldRealType(cell, qp, i, j);

  // Do Perturbations
  double pert   = 1.0e-6;
  int    numIVs = this->defGradField(0, 0, 0, 0).size();
  for (int iv = 0; iv < numIVs; ++iv) {
    for (int cell = 0; cell < (int)workset.numCells; ++cell)
      for (int qp = 0; qp < (int)this->numQPs; ++qp)
        for (int i = 0; i < (int)this->numDims; ++i)
          for (int j = 0; j < (int)this->numDims; ++j)
            this->defGradFieldRealType(cell, qp, i, j) =
                this->defGradField(cell, qp, i, j).val() +
                pert * this->defGradField(cell, qp, i, j).fastAccessDx(iv);

    this->calcStressRealType(
        this->stressFieldRealType, this->defGradFieldRealType, workset, matp);

    for (int cell = 0; cell < (int)workset.numCells; ++cell)
      for (int qp = 0; qp < (int)this->numQPs; ++qp)
        for (int i = 0; i < (int)this->numDims; ++i)
          for (int j = 0; j < (int)this->numDims; ++j)
            this->stressField(cell, qp, i, j).fastAccessDx(iv) =
                (this->stressFieldRealType(cell, qp, i, j) -
                 this->stressField(cell, qp, i, j).val()) /
                pert;
  }

  // Free double arrays allocated in matp
  this->freeMatP(matp);
}

// Tangent implementation is Identical to Jacobian
template <typename Traits>
void
LameStress<PHAL::AlbanyTraits::Tangent, Traits>::evaluateFields(
    typename Traits::EvalData workset)
{
  // This block forces stressField Fad to allocate deriv array
  PHAL::AlbanyTraits::Tangent::ScalarT scalarToForceAllocation =
      this->defGradField(0, 0, 0, 0);
  for (int cell = 0; cell < (int)workset.numCells; ++cell)
    for (int qp = 0; qp < (int)this->numQPs; ++qp)
      for (int i = 0; i < (int)this->numDims; ++i)
        for (int j = 0; j < (int)this->numDims; ++j)
          this->stressField(cell, qp, i, j) = scalarToForceAllocation;

  // Allocate double arrays in matp
  Teuchos::RCP<LameMatParams> matp = Teuchos::rcp(new LameMatParams());
  this->setMatP(matp, workset);

  // Begin Finite Difference
  // Do Base unperturbed case
  for (int cell = 0; cell < (int)workset.numCells; ++cell)
    for (int qp = 0; qp < (int)this->numQPs; ++qp)
      for (int i = 0; i < (int)this->numDims; ++i)
        for (int j = 0; j < (int)this->numDims; ++j)
          this->defGradFieldRealType(cell, qp, i, j) =
              this->defGradField(cell, qp, i, j).val();

  this->calcStressRealType(
      this->stressFieldRealType, this->defGradFieldRealType, workset, matp);

  for (int cell = 0; cell < (int)workset.numCells; ++cell)
    for (int qp = 0; qp < (int)this->numQPs; ++qp)
      for (int i = 0; i < (int)this->numDims; ++i)
        for (int j = 0; j < (int)this->numDims; ++j)
          this->stressField(cell, qp, i, j).val() =
              this->stressFieldRealType(cell, qp, i, j);

  // Do Perturbations
  double pert   = 1.0e-8;
  int    numIVs = this->defGradField(0, 0, 0, 0).size();
  for (int iv = 0; iv < numIVs; ++iv) {
    for (int cell = 0; cell < (int)workset.numCells; ++cell)
      for (int qp = 0; qp < (int)this->numQPs; ++qp)
        for (int i = 0; i < (int)this->numDims; ++i)
          for (int j = 0; j < (int)this->numDims; ++j)
            this->defGradFieldRealType(cell, qp, i, j) =
                this->defGradField(cell, qp, i, j).val() +
                pert * this->defGradField(cell, qp, i, j).fastAccessDx(iv);

    this->calcStressRealType(
        this->stressFieldRealType, this->defGradFieldRealType, workset, matp);

    for (int cell = 0; cell < (int)workset.numCells; ++cell)
      for (int qp = 0; qp < (int)this->numQPs; ++qp)
        for (int i = 0; i < (int)this->numDims; ++i)
          for (int j = 0; j < (int)this->numDims; ++j)
            this->stressField(cell, qp, i, j).fastAccessDx(iv) =
                (this->stressFieldRealType(cell, qp, i, j) -
                 this->stressField(cell, qp, i, j).val()) /
                pert;
  }

  // Free double arrays allocated in matp
  this->freeMatP(matp);
}

template <typename EvalT, typename Traits>
void
LameStressBase<EvalT, Traits>::setMatP(
    Teuchos::RCP<LameMatParams>& matp,
    typename Traits::EvalData    workset)
{
  // \todo Get actual time step for calls to LAME materials.
  RealType deltaT = 1.0;

  int numStateVariables =
      (int)(this->lameMaterialModelStateVariableNames.size());

  // Allocate workset space
  // Lame is called one time (called for all material points in the workset at
  // once)
  int numMaterialEvaluations = workset.numCells * this->numQPs;

  double* strainRate =
      new double[6 * numMaterialEvaluations];  // symmetric tensor5
  double* spin =
      new double[3 * numMaterialEvaluations];  // skew-symmetric tensor
  double* leftStretch =
      new double[6 * numMaterialEvaluations];  // symmetric tensor
  double* rotation = new double[9 * numMaterialEvaluations];  // full tensor
  double* stressOld =
      new double[6 * numMaterialEvaluations];  // symmetric tensor
  double* stressNew =
      new double[6 * numMaterialEvaluations];  // symmetric tensor
  double* stateOld = new double
      [numStateVariables *
       numMaterialEvaluations];  // a single double for each state variable
  double* stateNew = new double
      [numStateVariables *
       numMaterialEvaluations];  // a single double for each state variable

  // \todo Set up scratch space for material models using getNumScratchVars()
  // and setScratchPtr().

  // Create the matParams structure, which is passed to Lame
  matp->nelements    = numMaterialEvaluations;
  matp->dt           = deltaT;
  matp->time         = 0.0;
  matp->strain_rate  = strainRate;
  matp->spin         = spin;
  matp->left_stretch = leftStretch;
  matp->rotation     = rotation;
  matp->state_old    = stateOld;
  matp->state_new    = stateNew;
  matp->stress_old   = stressOld;
  matp->stress_new   = stressNew;
  //   matp->dt_mat = std::numeric_limits<double>::max();

  // matParams that still need to be added:
  // matp->temp_old  (temperature)
  // matp->temp_new
  // matp->sound_speed_old
  // matp->sound_speed_new
  // matp->volume
  // scratch pointer
  // function pointers (lots to be done here)
}

template <typename EvalT, typename Traits>
void
LameStressBase<EvalT, Traits>::freeMatP(Teuchos::RCP<LameMatParams>& matp)
{
  delete[] matp->strain_rate;
  delete[] matp->spin;
  delete[] matp->left_stretch;
  delete[] matp->rotation;
  delete[] matp->state_old;
  delete[] matp->state_new;
  delete[] matp->stress_old;
  delete[] matp->stress_new;
}

template <typename EvalT, typename Traits>
void
LameStressBase<EvalT, Traits>::calcStressRealType(
    PHX::MDField<RealType, Cell, QuadPoint, Dim, Dim>& stressFieldRef,
    PHX::MDField<RealType, Cell, QuadPoint, Dim, Dim>& defGradFieldRef,
    typename Traits::EvalData                          workset,
    Teuchos::RCP<LameMatParams>&                       matp)
{
  // Get the old state data
  Albany::MDArray oldDefGrad = (*workset.stateArrayPtr)[defGradName];
  Albany::MDArray oldStress  = (*workset.stateArrayPtr)[stressName];

  int numStateVariables =
      (int)(this->lameMaterialModelStateVariableNames.size());

  // Pointers used for filling the matParams structure
  double* strainRatePtr  = matp->strain_rate;
  double* spinPtr        = matp->spin;
  double* leftStretchPtr = matp->left_stretch;
  double* rotationPtr    = matp->rotation;
  double* stateOldPtr    = matp->state_old;
  double* stressOldPtr   = matp->stress_old;

  double deltaT = matp->dt;

  for (int cell = 0; cell < (int)workset.numCells; ++cell) {
    for (int qp = 0; qp < (int)numQPs; ++qp) {
      // Fill the following entries in matParams for call to LAME
      //
      // nelements     - number of elements
      // dt            - time step, this one is tough because Albany does not
      // currently have a concept of time step for implicit integration time -
      // current time, again Albany does not currently have a concept of time
      // for implicit integration strain_rate   - what Sierra calls the rate of
      // deformation, it is the symmetric part of the velocity gradient spin -
      // anti-symmetric part of the velocity gradient left_stretch  - found as V
      // in the polar decomposition of the deformation gradient F = VR rotation
      // - found as R in the polar decomposition of the deformation gradient F =
      // VR state_old     - material state data for previous time step (material
      // dependent, none for lame(nt)::Elastic) state_new     - material state
      // data for current time step (material dependent, none for
      // lame(nt)::Elastic) stress_old    - stress at previous time step
      // stress_new    - stress at current time step, filled by material model
      //
      // The total deformation gradient is available as field data
      //
      // The velocity gradient is not available but can be computed at the
      // logarithm of the incremental deformation gradient divided by deltaT The
      // incremental deformation gradient is computed as F_new F_old^-1

      // JTO:  here is how I think this will go (of course the first two lines
      // won't work as is...) minitensor::Tensor<RealType> F = newDefGrad;
      // minitensor::Tensor<RealType> Fn = oldDefGrad;
      // minitensor::Tensor<RealType> f = F*minitensor::inverse(Fn);
      // minitensor::Tensor<RealType> V;
      // minitensor::Tensor<RealType> R;
      // boost::tie(V,R) = minitensor::polar_left(F);
      // minitensor::Tensor<RealType> Vinc;
      // minitensor::Tensor<RealType> Rinc;
      // minitensor::Tensor<RealType> logVinc;
      // boost::tie(Vinc,Rinc,logVinc) = minitensor::polar_left_logV(f)
      // minitensor::Tensor<RealType> logRinc = minitensor::log_rotation(Rinc);
      // minitensor::Tensor<RealType> logf = Intrepid2::bch(logVinc,logRinc);
      // minitensor::Tensor<RealType> L = (1.0/deltaT)*logf;
      // minitensor::Tensor<RealType> D = minitensor::sym(L);
      // minitensor::Tensor<RealType> W = minitensor::skew(L);
      // and then fill data into the vectors below

      // new deformation gradient (the current deformation gradient as computed
      // in the current configuration)
      minitensor::Tensor<RealType> Fnew(
          defGradFieldRef(cell, qp, 0, 0),
          defGradFieldRef(cell, qp, 0, 1),
          defGradFieldRef(cell, qp, 0, 2),
          defGradFieldRef(cell, qp, 1, 0),
          defGradFieldRef(cell, qp, 1, 1),
          defGradFieldRef(cell, qp, 1, 2),
          defGradFieldRef(cell, qp, 2, 0),
          defGradFieldRef(cell, qp, 2, 1),
          defGradFieldRef(cell, qp, 2, 2));

      // old deformation gradient (deformation gradient at previous load step)
      minitensor::Tensor<RealType> Fold(
          oldDefGrad(cell, qp, 0, 0),
          oldDefGrad(cell, qp, 0, 1),
          oldDefGrad(cell, qp, 0, 2),
          oldDefGrad(cell, qp, 1, 0),
          oldDefGrad(cell, qp, 1, 1),
          oldDefGrad(cell, qp, 1, 2),
          oldDefGrad(cell, qp, 2, 0),
          oldDefGrad(cell, qp, 2, 1),
          oldDefGrad(cell, qp, 2, 2));

      // incremental deformation gradient
      minitensor::Tensor<RealType> Finc = Fnew * minitensor::inverse(Fold);

      // left stretch V, and rotation R, from left polar decomposition of new
      // deformation gradient
      minitensor::Tensor<RealType> V(3), R(3);
      boost::tie(V, R) = minitensor::polar_left_eig(Fnew);

      // incremental left stretch Vinc, incremental rotation Rinc, and log of
      // incremental left stretch, logVinc
      minitensor::Tensor<RealType> Vinc(3), Rinc(3), logVinc(3);
      boost::tie(Vinc, Rinc, logVinc) = minitensor::polar_left_logV_lame(Finc);

      // log of incremental rotation
      minitensor::Tensor<RealType> logRinc = minitensor::log_rotation(Rinc);

      // log of incremental deformation gradient
      minitensor::Tensor<RealType> logFinc = Intrepid2::bch(logVinc, logRinc);

      // velocity gradient
      minitensor::Tensor<RealType> L = RealType(1.0 / deltaT) * logFinc;

      // strain rate (a.k.a rate of deformation)
      minitensor::Tensor<RealType> D = minitensor::sym(L);

      // spin
      minitensor::Tensor<RealType> W = minitensor::skew(L);

      // load everything into the Lame data structure

      strainRatePtr[0] = (D(0, 0));
      strainRatePtr[1] = (D(1, 1));
      strainRatePtr[2] = (D(2, 2));
      strainRatePtr[3] = (D(0, 1));
      strainRatePtr[4] = (D(1, 2));
      strainRatePtr[5] = (D(0, 2));

      spinPtr[0] = (W(0, 1));
      spinPtr[1] = (W(1, 2));
      spinPtr[2] = (W(0, 2));

      leftStretchPtr[0] = (V(0, 0));
      leftStretchPtr[1] = (V(1, 1));
      leftStretchPtr[2] = (V(2, 2));
      leftStretchPtr[3] = (V(0, 1));
      leftStretchPtr[4] = (V(1, 2));
      leftStretchPtr[5] = (V(0, 2));

      rotationPtr[0] = (R(0, 0));
      rotationPtr[1] = (R(1, 1));
      rotationPtr[2] = (R(2, 2));
      rotationPtr[3] = (R(0, 1));
      rotationPtr[4] = (R(1, 2));
      rotationPtr[5] = (R(0, 2));
      rotationPtr[6] = (R(1, 0));
      rotationPtr[7] = (R(2, 1));
      rotationPtr[8] = (R(2, 0));

      stressOldPtr[0] = oldStress(cell, qp, 0, 0);
      stressOldPtr[1] = oldStress(cell, qp, 1, 1);
      stressOldPtr[2] = oldStress(cell, qp, 2, 2);
      stressOldPtr[3] = oldStress(cell, qp, 0, 1);
      stressOldPtr[4] = oldStress(cell, qp, 1, 2);
      stressOldPtr[5] = oldStress(cell, qp, 0, 2);

      // increment the pointers
      strainRatePtr += 6;
      spinPtr += 3;
      leftStretchPtr += 6;
      rotationPtr += 9;
      stressOldPtr += 6;

      // copy data from the state manager to the LAME data structure
      for (int iVar = 0; iVar < numStateVariables; iVar++, stateOldPtr++) {
        // std::string& variableName =
        // this->lameMaterialModelStateVariableNames[iVar]; const
        // Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device>&
        // stateVar = *oldState[variableName];
        const std::string& variableName =
            this->lameMaterialModelStateVariableNames[iVar] + "_old";
        Albany::MDArray stateVar = (*workset.stateArrayPtr)[variableName];
        *stateOldPtr             = stateVar(cell, qp);
      }
    }
  }

  // Make a call to the LAME material model to initialize the load step
  this->lameMaterialModel->loadStepInit(matp.get());

  // Get the stress from the LAME material
  this->lameMaterialModel->getStress(matp.get());

  double* stressNewPtr = matp->stress_new;

  // Post-process data from Lame call
  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int qp = 0; qp < numQPs; ++qp) {
      // Copy the new stress into the stress field
      stressFieldRef(cell, qp, 0, 0) = stressNewPtr[0];
      stressFieldRef(cell, qp, 1, 1) = stressNewPtr[1];
      stressFieldRef(cell, qp, 2, 2) = stressNewPtr[2];
      stressFieldRef(cell, qp, 0, 1) = stressNewPtr[3];
      stressFieldRef(cell, qp, 1, 2) = stressNewPtr[4];
      stressFieldRef(cell, qp, 0, 2) = stressNewPtr[5];
      stressFieldRef(cell, qp, 1, 0) = stressNewPtr[3];
      stressFieldRef(cell, qp, 2, 1) = stressNewPtr[4];
      stressFieldRef(cell, qp, 2, 0) = stressNewPtr[5];

      stressNewPtr += 6;
    }
  }

  // !!!!! When should this be done???
  double* stateNewPtr = matp->state_new;
  for (int cell = 0; cell < workset.numCells; ++cell) {
    for (int qp = 0; qp < numQPs; ++qp) {
      // copy state_new data from the LAME data structure to the corresponding
      // state variable field
      for (int iVar = 0; iVar < numStateVariables; iVar++, stateNewPtr++)
        this->lameMaterialModelStateVariableFields[iVar](cell, qp) =
            *stateNewPtr;
    }
  }
}

}  // namespace LCM
