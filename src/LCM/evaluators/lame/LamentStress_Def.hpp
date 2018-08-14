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

using namespace std;

namespace LCM {

template <typename EvalT, typename Traits>
LamentStress<EvalT, Traits>::LamentStress(Teuchos::ParameterList& p)
    : defGradField(
          p.get<std::string>("DefGrad Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout")),
      stressField(
          p.get<std::string>("Stress Name"),
          p.get<Teuchos::RCP<PHX::DataLayout>>("QP Tensor Data Layout")),
      lamentMaterialModel(Teuchos::RCP<lament::Material<ScalarT>>())
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
      " LAMENT materials enabled only for three-dimensional analyses.");

  defGradName = p.get<std::string>("DefGrad Name") + "_old";
  this->addDependentField(defGradField);

  stressName = p.get<std::string>("Stress Name") + "_old";
  this->addEvaluatedField(stressField);

  this->setName("LamentStress" + PHX::typeAsString<EvalT>());

  // Default to getting material info form base input file (possibley
  // overwritten later)
  lamentMaterialModelName = p.get<string>("Lame Material Model", "Elastic");
  std::cout << "Material Model Name : " << lamentMaterialModelName << std::endl;
  Teuchos::ParameterList& lamentMaterialParameters =
      p.sublist("Lame Material Parameters");

  // Code to allow material data to come from materials.xml data file
  int haveMatDB = p.get<bool>("Have MatDB", false);

  std::string ebName = p.get<std::string>("Element Block Name", "Missing");

  // Check for material database file
  if (haveMatDB) {
    // Check if material database will be supplying the data
    bool dataFromDatabase = lamentMaterialParameters.get<bool>(
        "Material Dependent Data Source", false);

    // If so, overwrite material model and data from database file
    if (dataFromDatabase) {
      Teuchos::RCP<Albany::MaterialDatabase> materialDB =
          p.get<Teuchos::RCP<Albany::MaterialDatabase>>("MaterialDB");

      lamentMaterialModelName = materialDB->getElementBlockParam<std::string>(
          ebName, "Lame Material Model");
      lamentMaterialParameters = materialDB->getElementBlockSublist(
          ebName, "Lame Material Parameters");
    }
  }

  // Initialize the LAMENT material model
  // This assumes that there is a single material model associated with this
  // evaluator and that the material properties are constant (read directly
  // from input deck parameter list)
  lamentMaterialModel = LameUtils::constructLamentMaterialModel<ScalarT>(
      lamentMaterialModelName, lamentMaterialParameters);

  // Get a list of the LAMENT material model state variable names
  lamentMaterialModelStateVariableNames = LameUtils::getStateVariableNames(
      lamentMaterialModelName, lamentMaterialParameters);

  // Declare the state variables as evaluated fields (type is always double)
  Teuchos::RCP<PHX::DataLayout> dataLayout =
      p.get<Teuchos::RCP<PHX::DataLayout>>("QP Scalar Data Layout");
  for (unsigned int i = 0; i < lamentMaterialModelStateVariableNames.size();
       ++i) {
    PHX::MDField<ScalarT, Cell, QuadPoint>
        lamentMaterialModelStateVariableField(
            lamentMaterialModelStateVariableNames[i], dataLayout);
    this->addEvaluatedField(lamentMaterialModelStateVariableField);
    lamentMaterialModelStateVariableFields.push_back(
        lamentMaterialModelStateVariableField);
  }
}

template <typename EvalT, typename Traits>
void
LamentStress<EvalT, Traits>::postRegistrationSetup(
    typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(defGradField, fm);
  this->utils.setFieldData(stressField, fm);
  for (unsigned int i = 0; i < lamentMaterialModelStateVariableFields.size();
       ++i)
    this->utils.setFieldData(lamentMaterialModelStateVariableFields[i], fm);
}

template <typename EvalT, typename Traits>
void
LamentStress<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::RCP<lament::matParams<ScalarT>> matp =
      Teuchos::rcp(new lament::matParams<ScalarT>());

  // Get the old state data
  Albany::MDArray oldDefGrad = (*workset.stateArrayPtr)[defGradName];
  Albany::MDArray oldStress  = (*workset.stateArrayPtr)[stressName];

  int numStateVariables =
      (int)(this->lamentMaterialModelStateVariableNames.size());

  // \todo Get actual time step for calls to LAMENT materials.
  double deltaT = 1.0;

  vector<ScalarT> strainRate(6);   // symmetric tensor
  vector<ScalarT> spin(3);         // skew-symmetric tensor
  vector<ScalarT> defGrad(9);      // symmetric tensor
  vector<ScalarT> leftStretch(6);  // symmetric tensor
  vector<ScalarT> rotation(9);     // full tensor
  vector<double>  stressOld(6);    // symmetric tensor
  vector<ScalarT> stressNew(6);    // symmetric tensor
  vector<double>  stateOld(
      numStateVariables);  // a single scalar for each state variable
  vector<double> stateNew(
      numStateVariables);  // a single scalar for each state variable

  // \todo Set up scratch space for material models using getNumScratchVars()
  // and setScratchPtr().

  // Create the matParams structure, which is passed to Lament
  matp->nelements            = 1;
  matp->dt                   = deltaT;
  matp->time                 = 0.0;
  matp->strain_rate          = &strainRate[0];
  matp->spin                 = &spin[0];
  matp->deformation_gradient = &defGrad[0];
  matp->left_stretch         = &leftStretch[0];
  matp->rotation             = &rotation[0];
  matp->state_old            = &stateOld[0];
  matp->state_new            = &stateNew[0];
  matp->stress_old           = &stressOld[0];
  matp->stress_new           = &stressNew[0];
  //   matp->dt_mat = std::numeric_limits<double>::max();

  // matParams that still need to be added:
  // matp->temp_old  (temperature)
  // matp->temp_new
  // matp->sound_speed_old
  // matp->sound_speed_new
  // matp->volume
  // scratch pointer
  // function pointers (lots to be done here)

  for (int cell = 0; cell < (int)workset.numCells; ++cell) {
    for (int qp = 0; qp < (int)numQPs; ++qp) {
      // std::cout << "QP: " << qp << std::endl;

      // Fill the following entries in matParams for call to LAMENT
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
      // dependent, none for lament::Elastic) state_new     - material state
      // data for current time step (material dependent, none for
      // lament::Elastic) stress_old    - stress at previous time step
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
      minitensor::Tensor<ScalarT> Fnew(3, defGradField, cell, qp, 0, 0);

      // old deformation gradient (deformation gradient at previous load step)
      minitensor::Tensor<ScalarT> Fold(
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
      minitensor::Tensor<ScalarT> Finc = Fnew * minitensor::inverse(Fold);

      // DEBUGGING //
      // if(cell==0 && qp==0){
      // std::cout << "Fnew(0,0) " << Fnew(0,0) << endl;
      // std::cout << "Fnew(1,0) " << Fnew(1,0) << endl;
      // std::cout << "Fnew(2,0) " << Fnew(2,0) << endl;
      // std::cout << "Fnew(0,1) " << Fnew(0,1) << endl;
      // std::cout << "Fnew(1,1) " << Fnew(1,1) << endl;
      // std::cout << "Fnew(2,1) " << Fnew(2,1) << endl;
      // std::cout << "Fnew(0,2) " << Fnew(0,2) << endl;
      // std::cout << "Fnew(1,2) " << Fnew(1,2) << endl;
      // std::cout << "Fnew(2,2) " << Fnew(2,2) << endl;
      //}
      // END DEBUGGING //

      // left stretch V, and rotation R, from left polar decomposition of new
      // deformation gradient
      minitensor::Tensor<ScalarT> V(3), R(3), U(3);
      boost::tie(V, R) = minitensor::polar_left(Fnew);
      // V = R * U * transpose(R);

      // DEBUGGING //
      // if(cell==0 && qp==0){
      // std::cout << "U(0,0) " << U(0,0) << endl;
      // std::cout << "U(1,0) " << U(1,0) << endl;
      // std::cout << "U(2,0) " << U(2,0) << endl;
      // std::cout << "U(0,1) " << U(0,1) << endl;
      // std::cout << "U(1,1) " << U(1,1) << endl;
      // std::cout << "U(2,1) " << U(2,1) << endl;
      // std::cout << "U(0,2) " << U(0,2) << endl;
      // std::cout << "U(1,2) " << U(1,2) << endl;
      // std::cout << "U(2,2) " << U(2,2) << endl;
      // std::cout << "========\n";
      // std::cout << "V(0,0) " << V(0,0) << endl;
      // std::cout << "V(1,0) " << V(1,0) << endl;
      // std::cout << "V(2,0) " << V(2,0) << endl;
      // std::cout << "V(0,1) " << V(0,1) << endl;
      // std::cout << "V(1,1) " << V(1,1) << endl;
      // std::cout << "V(2,1) " << V(2,1) << endl;
      // std::cout << "V(0,2) " << V(0,2) << endl;
      // std::cout << "V(1,2) " << V(1,2) << endl;
      // std::cout << "V(2,2) " << V(2,2) << endl;
      // std::cout << "========\n";
      // std::cout << "R(0,0) " << R(0,0) << endl;
      // std::cout << "R(1,0) " << R(1,0) << endl;
      // std::cout << "R(2,0) " << R(2,0) << endl;
      // std::cout << "R(0,1) " << R(0,1) << endl;
      // std::cout << "R(1,1) " << R(1,1) << endl;
      // std::cout << "R(2,1) " << R(2,1) << endl;
      // std::cout << "R(0,2) " << R(0,2) << endl;
      // std::cout << "R(1,2) " << R(1,2) << endl;
      // std::cout << "R(2,2) " << R(2,2) << endl;
      //}
      // END DEBUGGING //

      // incremental left stretch Vinc, incremental rotation Rinc, and log of
      // incremental left stretch, logVinc

      minitensor::Tensor<ScalarT> Uinc(3), Vinc(3), Rinc(3), logVinc(3);
      // boost::tie(Vinc,Rinc,logVinc) = minitensor::polar_left_logV(Finc);
      boost::tie(Vinc, Rinc) = minitensor::polar_left(Finc);
      // Vinc = Rinc * Uinc * transpose(Rinc);
      logVinc = minitensor::log(Vinc);

      // log of incremental rotation
      minitensor::Tensor<ScalarT> logRinc = minitensor::log_rotation(Rinc);

      // log of incremental deformation gradient
      minitensor::Tensor<ScalarT> logFinc = Intrepid2::bch(logVinc, logRinc);

      // velocity gradient
      minitensor::Tensor<ScalarT> L = (1.0 / deltaT) * logFinc;

      // strain rate (a.k.a rate of deformation)
      minitensor::Tensor<ScalarT> D = minitensor::sym(L);

      // spin
      minitensor::Tensor<ScalarT> W = minitensor::skew(L);

      // load everything into the Lament data structure

      strainRate[0] = (D(0, 0));
      strainRate[1] = (D(1, 1));
      strainRate[2] = (D(2, 2));
      strainRate[3] = (D(0, 1));
      strainRate[4] = (D(1, 2));
      strainRate[5] = (D(2, 0));

      spin[0] = (W(0, 1));
      spin[1] = (W(1, 2));
      spin[2] = (W(2, 0));

      leftStretch[0] = (V(0, 0));
      leftStretch[1] = (V(1, 1));
      leftStretch[2] = (V(2, 2));
      leftStretch[3] = (V(0, 1));
      leftStretch[4] = (V(1, 2));
      leftStretch[5] = (V(2, 0));

      rotation[0] = (R(0, 0));
      rotation[1] = (R(1, 1));
      rotation[2] = (R(2, 2));
      rotation[3] = (R(0, 1));
      rotation[4] = (R(1, 2));
      rotation[5] = (R(2, 0));
      rotation[6] = (R(1, 0));
      rotation[7] = (R(2, 1));
      rotation[8] = (R(0, 2));

      defGrad[0] = (Fnew(0, 0));
      defGrad[1] = (Fnew(1, 1));
      defGrad[2] = (Fnew(2, 2));
      defGrad[3] = (Fnew(0, 1));
      defGrad[4] = (Fnew(1, 2));
      defGrad[5] = (Fnew(2, 0));
      defGrad[6] = (Fnew(1, 0));
      defGrad[7] = (Fnew(2, 1));
      defGrad[8] = (Fnew(0, 2));

      stressOld[0] = oldStress(cell, qp, 0, 0);
      stressOld[1] = oldStress(cell, qp, 1, 1);
      stressOld[2] = oldStress(cell, qp, 2, 2);
      stressOld[3] = oldStress(cell, qp, 0, 1);
      stressOld[4] = oldStress(cell, qp, 1, 2);
      stressOld[5] = oldStress(cell, qp, 2, 0);

      // copy data from the state manager to the LAMENT data structure
      for (int iVar = 0; iVar < numStateVariables; iVar++) {
        const std::string& variableName =
            this->lamentMaterialModelStateVariableNames[iVar] + "_old";
        Albany::MDArray stateVar = (*workset.stateArrayPtr)[variableName];
        stateOld[iVar]           = stateVar(cell, qp);
      }

      // Make a call to the LAMENT material model to initialize the load step
      this->lamentMaterialModel->loadStepInit(matp.get());

      // Get the stress from the LAMENT material

      // std::cout << "about to call lament->getStress()" << std::endl;

      this->lamentMaterialModel->getStress(matp.get());

      // std::cout << "after calling lament->getStress() 2" << std::endl;

      // rotate to get the Cauchy Stress
      minitensor::Tensor<ScalarT> lameStress(
          stressNew[0],
          stressNew[3],
          stressNew[5],
          stressNew[3],
          stressNew[1],
          stressNew[4],
          stressNew[5],
          stressNew[4],
          stressNew[2]);
      minitensor::Tensor<ScalarT> cauchy = R * lameStress * transpose(R);

      // DEBUGGING //
      // if(cell==0 && qp==0){
      // std::cout << "check strainRate[0] " << strainRate[0] << endl;
      // std::cout << "check strainRate[1] " << strainRate[1] << endl;
      // std::cout << "check strainRate[2] " << strainRate[2] << endl;
      // std::cout << "check strainRate[3] " << strainRate[3] << endl;
      // std::cout << "check strainRate[4] " << strainRate[4] << endl;
      // std::cout << "check strainRate[5] " << strainRate[5] << endl;
      //}
      // END DEBUGGING //

      // Copy the new stress into the stress field
      for (int i(0); i < 3; ++i)
        for (int j(0); j < 3; ++j) stressField(cell, qp, i, j) = cauchy(i, j);

      // stressField(cell,qp,0,0) = stressNew[0];
      // stressField(cell,qp,1,1) = stressNew[1];
      // stressField(cell,qp,2,2) = stressNew[2];
      // stressField(cell,qp,0,1) = stressNew[3];
      // stressField(cell,qp,1,2) = stressNew[4];
      // stressField(cell,qp,2,0) = stressNew[5];
      // stressField(cell,qp,1,0) = stressNew[3];
      // stressField(cell,qp,2,1) = stressNew[4];
      // stressField(cell,qp,0,2) = stressNew[5];

      // copy state_new data from the LAMENT data structure to the corresponding
      // state variable field
      for (int iVar = 0; iVar < numStateVariables; iVar++)
        this->lamentMaterialModelStateVariableFields[iVar](cell, qp) =
            stateNew[iVar];

      // DEBUGGING //
      // if(cell==0 && qp==0){
      //   std::cout << "stress(0,0) " << this->stressField(cell,qp,0,0) <<
      //   endl; std::cout << "stress(1,1) " << this->stressField(cell,qp,1,1)
      //   << endl; std::cout << "stress(2,2) " <<
      //   this->stressField(cell,qp,2,2) << endl; std::cout << "stress(0,1) "
      //   << this->stressField(cell,qp,0,1) << endl; std::cout << "stress(1,2)
      //   " << this->stressField(cell,qp,1,2) << endl; std::cout <<
      //   "stress(0,2) " << this->stressField(cell,qp,0,2) << endl; std::cout
      //   << "stress(1,0) " << this->stressField(cell,qp,1,0) << endl;
      //   std::cout << "stress(2,1) " << this->stressField(cell,qp,2,1) <<
      //   endl; std::cout << "stress(2,0) " << this->stressField(cell,qp,2,0)
      //   << endl;
      //   //}
      // // END DEBUGGING //
    }
  }
}

}  // namespace LCM
