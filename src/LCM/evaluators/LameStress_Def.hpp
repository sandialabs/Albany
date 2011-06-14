/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Intrepid_FunctionSpaceTools.hpp"
#include "Tensor.h"

namespace LCM {

template<typename EvalT, typename Traits>
LameStress<EvalT, Traits>::
LameStress(const Teuchos::ParameterList& p) :
  defGradField(p.get<std::string>("DefGrad Name"),
               p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout")),
  stressField(p.get<std::string>("Stress Name"),
              p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  lameMaterialModel(Teuchos::RCP<lame::Material>())
{
  // Pull out numQPs and numDims from a Layout
  Teuchos::RCP<PHX::DataLayout> tensor_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  tensor_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  defGradName = p.get<std::string>("DefGrad Name");
  this->addDependentField(defGradField);

  stressName = p.get<std::string>("Stress Name");
  this->addEvaluatedField(stressField);

  this->setName("LameStress"+PHX::TypeString<EvalT>::value);

  string inputLameMaterialModelName = p.get<string>("Lame Material Model");
  const Teuchos::ParameterList& inputLameMaterialParameters = p.sublist("Lame Material Parameters");

  // Initialize the LAME material model
  // This assumes that there is a single material model associated with this
  // evaluator and that the material properties are constant (read directly
  // from input deck parameter list)
  lameMaterialModel = LameUtils::constructLameMaterialModel(inputLameMaterialModelName, inputLameMaterialParameters);

  // Query the material model for its name and the list of state variables
  lameMaterialModel->getStateVarListAndName(lameMaterialModelStateVariables, lameMaterialModelName);

  // Declare the state variables as evaluated fields (type is always double)
  Teuchos::RCP<PHX::DataLayout> dataLayout = p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
  for(unsigned int i=0 ; i<lameMaterialModelStateVariables.size() ; ++i){
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> lameMaterialModelStateVariableField(lameMaterialModelStateVariables[i], dataLayout);
    this->addEvaluatedField(lameMaterialModelStateVariableField);
    lameMaterialModelStateVariableFields.push_back(lameMaterialModelStateVariableField);
  }
}

template<typename EvalT, typename Traits>
void LameStress<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(defGradField,fm);
  this->utils.setFieldData(stressField,fm);
}

template<typename EvalT, typename Traits>
void LameStress<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  TEST_FOR_EXCEPTION(numDims != 3, Teuchos::Exceptions::InvalidParameter, " LAME materials enabled only for three-dimensional analyses.");

  // Get the old and new state data
  // StateVariables is:  typedef std::map<std::string, Teuchos::RCP<Intrepid::FieldContainer<RealType> > >
  Albany::StateVariables oldState = *workset.oldState;
  const Intrepid::FieldContainer<RealType>& oldDefGrad  = *oldState[defGradName];
  const Intrepid::FieldContainer<RealType>& oldStress  = *oldState[stressName];

  // \todo Get actual time step for calls to LAME materials.
  RealType deltaT = 1.0;

  int numStateVariables = (int)(lameMaterialModelStateVariables.size());

  // Allocate workset space
  // Lame is called one time (called for all material points in the workset at once)
  int numMaterialEvaluations = workset.numCells * numQPs;
  std::vector<RealType> strainRate(6*numMaterialEvaluations);   // symmetric tensor5
  std::vector<RealType> spin(3*numMaterialEvaluations);         // skew-symmetric tensor
  std::vector<RealType> leftStretch(6*numMaterialEvaluations);  // symmetric tensor
  std::vector<RealType> rotation(9*numMaterialEvaluations);     // full tensor
  std::vector<RealType> stressOld(6*numMaterialEvaluations);    // symmetric tensor
  std::vector<RealType> stressNew(6*numMaterialEvaluations);    // symmetric tensor
  std::vector<RealType> stateOld(numStateVariables*numMaterialEvaluations);  // a single double for each state variable
  std::vector<RealType> stateNew(numStateVariables*numMaterialEvaluations);  // a single double for each state variable

  // \todo Set up scratch space for material models using getNumScratchVars() and setScratchPtr().

  // Create the matParams structure, which is passed to Lame
  Teuchos::RCP<lame::matParams> matp = Teuchos::rcp(new lame::matParams());
  matp->nelements = numMaterialEvaluations;
  matp->dt = deltaT;
  matp->time = 0.0;
  matp->strain_rate = &strainRate[0];
  matp->spin = &spin[0];
  matp->left_stretch = &leftStretch[0];
  matp->rotation = &rotation[0];
  matp->state_old = &stateOld[0];
  matp->state_new = &stateNew[0];
  matp->stress_old = &stressOld[0];
  matp->stress_new = &stressNew[0];

  // Pointers used for filling the matParams structure
  double* strainRatePtr = matp->strain_rate;
  double* spinPtr = matp->spin;
  double* leftStretchPtr = matp->left_stretch;
  double* rotationPtr = matp->rotation;
  double* stateOldPtr = matp->state_old;
  double* stressOldPtr = matp->stress_old;

  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {

      // Fill the following entries in matParams for call to LAME
      //
      // nelements     - number of elements 
      // dt            - time step, this one is tough because Albany does not currently have a concept of time step for implicit integration
      // time          - current time, again Albany does not currently have a concept of time for implicit integration
      // strain_rate   - what Sierra calls the rate of deformation, it is the symmetric part of the velocity gradient
      // spin          - anti-symmetric part of the velocity gradient
      // left_stretch  - found as V in the polar decomposition of the deformation gradient F = VR
      // rotation      - found as R in the polar decomposition of the deformation gradient F = VR
      // state_old     - material state data for previous time step (material dependent, none for lame::Elastic)
      // state_new     - material state data for current time step (material dependent, none for lame::Elastic)
      // stress_old    - stress at previous time step
      // stress_new    - stress at current time step, filled by material model
      //
      // The total deformation gradient is available as field data
      // 
      // The velocity gradient is not available but can be computed at the logarithm of the incremental deformation gradient divided by deltaT
      // The incremental deformation gradient is computed as F_new F_old^-1

      // JTO:  here is how I think this will go (of course the first two lines won't work as is...)
      // LCM::Tensor<ScalarT> F = newDefGrad;
      // LCM::Tensor<ScalarT> Fn = oldDefGrad;
      // LCM::Tensor<ScalarT> f = F*LCM::inverse(Fn);
      // LCM::Tensor<ScalarT> V;
      // LCM::Tensor<ScalarT> R;
      // boost::tie(V,R) = LCM::polar_left(F);
      // LCM::Tensor<ScalarT> Vinc;
      // LCM::Tensor<ScalarT> Rinc;
      // LCM::Tensor<ScalarT> logVinc;
      // boost::tie(Vinc,Rinc,logVinc) = LCM::polar_left_logV(f)
      // LCM::Tensor<ScalarT> logRinc = LCM::log_rotation(Rinc);
      // LCM::Tensor<ScalarT> logf = LCM::bch(logVinc,logRinc);
      // LCM::Tensor<ScalarT> L = (1.0/deltaT)*logf;
      // LCM::Tensor<ScalarT> D = LCM::sym(L);
      // LCM::Tensor<ScalarT> W = LCM::skew(L);
      // and then fill data into the vectors below

      // new deformation gradient (the current deformation gradient as computed in the current configuration)
      LCM::Tensor<ScalarT> Fnew(
       Sacado::ScalarValue<ScalarT>::eval(defGradField(cell,qp,0,0)),
       Sacado::ScalarValue<ScalarT>::eval(defGradField(cell,qp,0,1)),
       Sacado::ScalarValue<ScalarT>::eval(defGradField(cell,qp,0,2)),
       Sacado::ScalarValue<ScalarT>::eval(defGradField(cell,qp,1,0)),
       Sacado::ScalarValue<ScalarT>::eval(defGradField(cell,qp,1,1)),
       Sacado::ScalarValue<ScalarT>::eval(defGradField(cell,qp,1,2)),
       Sacado::ScalarValue<ScalarT>::eval(defGradField(cell,qp,2,0)),
       Sacado::ScalarValue<ScalarT>::eval(defGradField(cell,qp,2,1)),
       Sacado::ScalarValue<ScalarT>::eval(defGradField(cell,qp,2,2)) );


      // old deformation gradient (deformation gradient at previous load step)
      LCM::Tensor<ScalarT> Fold( oldDefGrad(cell,qp,0,0), oldDefGrad(cell,qp,0,1), oldDefGrad(cell,qp,0,2),
                                 oldDefGrad(cell,qp,1,0), oldDefGrad(cell,qp,1,1), oldDefGrad(cell,qp,1,2),
                                 oldDefGrad(cell,qp,2,0), oldDefGrad(cell,qp,2,1), oldDefGrad(cell,qp,2,2) );

      // incremental deformation gradient
      LCM::Tensor<ScalarT> Finc = Fnew * LCM::inverse(Fold);

      // left stretch V, and rotation R, from left polar decomposition of new deformation gradient
      LCM::Tensor<ScalarT> V, R;
      boost::tie(V,R) = LCM::polar_left(Fnew);

      // incremental left stretch Vinc, incremental rotation Rinc, and log of incremental left stretch, logVinc
      LCM::Tensor<ScalarT> Vinc, Rinc, logVinc;
      boost::tie(Vinc,Rinc,logVinc) = LCM::polar_left_logV(Finc);

      // log of incremental rotation
      LCM::Tensor<ScalarT> logRinc = LCM::log_rotation(Rinc);

      // log of incremental deformation gradient
      LCM::Tensor<ScalarT> logFinc = LCM::bch(logVinc, logRinc);

      // velocity gradient
      LCM::Tensor<ScalarT> L = ScalarT(1.0/deltaT)*logFinc;

      // strain rate (a.k.a rate of deformation)
      LCM::Tensor<ScalarT> D = LCM::symm(L);

      // spin
      LCM::Tensor<ScalarT> W = LCM::skew(L);

      // load everything into the Lame data structure

      strainRatePtr[0] = Sacado::ScalarValue<ScalarT>::eval( D(0,0) );
      strainRatePtr[1] = Sacado::ScalarValue<ScalarT>::eval( D(1,1) );
      strainRatePtr[2] = Sacado::ScalarValue<ScalarT>::eval( D(2,2) );
      strainRatePtr[3] = Sacado::ScalarValue<ScalarT>::eval( D(0,1) );
      strainRatePtr[4] = Sacado::ScalarValue<ScalarT>::eval( D(1,2) );
      strainRatePtr[5] = Sacado::ScalarValue<ScalarT>::eval( D(0,2) );

      spinPtr[0] = Sacado::ScalarValue<ScalarT>::eval( W(0,1) );
      spinPtr[1] = Sacado::ScalarValue<ScalarT>::eval( W(1,2) );
      spinPtr[2] = Sacado::ScalarValue<ScalarT>::eval( W(0,2) );

      leftStretchPtr[0] = Sacado::ScalarValue<ScalarT>::eval( V(0,0) );
      leftStretchPtr[1] = Sacado::ScalarValue<ScalarT>::eval( V(1,1) );
      leftStretchPtr[2] = Sacado::ScalarValue<ScalarT>::eval( V(2,2) );
      leftStretchPtr[3] = Sacado::ScalarValue<ScalarT>::eval( V(0,1) );
      leftStretchPtr[4] = Sacado::ScalarValue<ScalarT>::eval( V(1,2) );
      leftStretchPtr[5] = Sacado::ScalarValue<ScalarT>::eval( V(0,2) );

      rotationPtr[0] = Sacado::ScalarValue<ScalarT>::eval( R(0,0) );
      rotationPtr[1] = Sacado::ScalarValue<ScalarT>::eval( R(1,1) );
      rotationPtr[2] = Sacado::ScalarValue<ScalarT>::eval( R(2,2) );
      rotationPtr[3] = Sacado::ScalarValue<ScalarT>::eval( R(0,1) );
      rotationPtr[4] = Sacado::ScalarValue<ScalarT>::eval( R(1,2) );
      rotationPtr[5] = Sacado::ScalarValue<ScalarT>::eval( R(0,2) );
      rotationPtr[6] = Sacado::ScalarValue<ScalarT>::eval( R(1,0) );
      rotationPtr[7] = Sacado::ScalarValue<ScalarT>::eval( R(2,1) );
      rotationPtr[8] = Sacado::ScalarValue<ScalarT>::eval( R(2,0) );

      stressOldPtr[0] = oldStress(cell,qp,0,0);
      stressOldPtr[1] = oldStress(cell,qp,1,1);
      stressOldPtr[2] = oldStress(cell,qp,2,2);
      stressOldPtr[3] = oldStress(cell,qp,0,1);
      stressOldPtr[4] = oldStress(cell,qp,1,2);
      stressOldPtr[5] = oldStress(cell,qp,0,2);

      // increment the pointers
      strainRatePtr += 6;
      spinPtr += 3;
      leftStretchPtr += 6;
      rotationPtr += 9;
      stressOldPtr += 6;

      // copy data from the state manager to the LAME data structure
      for(int iVar=0 ; iVar<numStateVariables ; iVar++, stateOldPtr++){
        std::string& variableName = lameMaterialModelStateVariables[iVar];
        const Intrepid::FieldContainer<RealType>& stateVar = *oldState[variableName];
        *stateOldPtr = stateVar(cell,qp,0,0);
      }
    }
  }

  // \todo Call loadStepInit();

  // Get the stress from the LAME material
  lameMaterialModel->getStress(matp.get());

  double* stateNewPtr = matp->state_new;
  double* stressNewPtr = matp->stress_new;

  // Post-process data from Lame call
  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {

      // Copy the new stress into the stress field
      stressField(cell,qp,0,0) = stressNewPtr[0];
      stressField(cell,qp,1,1) = stressNewPtr[1];
      stressField(cell,qp,2,2) = stressNewPtr[2];
      stressField(cell,qp,0,1) = stressNewPtr[3];
      stressField(cell,qp,1,2) = stressNewPtr[4];
      stressField(cell,qp,0,2) = stressNewPtr[5];
      stressField(cell,qp,1,0) = stressField(cell,qp,0,1); 
      stressField(cell,qp,2,1) = stressField(cell,qp,1,2); 
      stressField(cell,qp,2,0) = stressField(cell,qp,0,2);

      stressNewPtr += 6;

      // copy state_new data from the LAME data structure to the corresponding state variable field
      for(int iVar=0 ; iVar<numStateVariables ; iVar++, stateNewPtr++)
        lameMaterialModelStateVariableFields[iVar](cell,qp,0,0) = *stateNewPtr;

    }
  }
}

}

