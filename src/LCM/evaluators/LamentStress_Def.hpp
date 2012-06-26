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
#include "QCAD_MaterialDatabase.hpp"
#include "Tensor.h"

using namespace std;

namespace LCM {

template<typename EvalT, typename Traits>
LamentStress<EvalT, Traits>::
LamentStress(Teuchos::ParameterList& p) :
  defGradField(p.get<std::string>("DefGrad Name"),
               p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout")),
  stressField(p.get<std::string>("Stress Name"),
              p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  lamentMaterialModel(Teuchos::RCP<lament::Material<ScalarT> >())
{
  // Pull out numQPs and numDims from a Layout
  tensor_dl = p.get< Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  tensor_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  TEUCHOS_TEST_FOR_EXCEPTION(this->numDims != 3, Teuchos::Exceptions::InvalidParameter, " LAMENT materials enabled only for three-dimensional analyses.");

  defGradName = p.get<std::string>("DefGrad Name")+"_old";
  this->addDependentField(defGradField);

  stressName = p.get<std::string>("Stress Name")+"_old";
  this->addEvaluatedField(stressField);

  this->setName("LamentStress"+PHX::TypeString<EvalT>::value);

  // Default to getting material info form base input file (possibley overwritten later)
  lamentMaterialModelName = p.get<string>("Lame Material Model", "Elastic");
  Teuchos::ParameterList& lamentMaterialParameters = p.sublist("Lame Material Parameters");

  // Code to allow material data to come from materials.xml data file
  int haveMatDB = p.get<bool>("Have MatDB", false);

  std::string ebName = p.get<std::string>("Element Block Name", "Missing");

  // Check for material database file
  if (haveMatDB) {
    // Check if material database will be supplying the data
    bool dataFromDatabase = lamentMaterialParameters.get<bool>("Material Dependent Data Source",false);

    // If so, overwrite material model and data from database file
    if (dataFromDatabase) {
       Teuchos::RCP<QCAD::MaterialDatabase> materialDB = p.get< Teuchos::RCP<QCAD::MaterialDatabase> >("MaterialDB");

       lamentMaterialModelName = materialDB->getElementBlockParam<std::string>(ebName, "Lame Material Model");
       lamentMaterialParameters = materialDB->getElementBlockSublist(ebName, "Lame Material Parameters");
     }
  }

  // Initialize the LAMENT material model
  // This assumes that there is a single material model associated with this
  // evaluator and that the material properties are constant (read directly
  // from input deck parameter list)
  lamentMaterialModel = LameUtils::constructLamentMaterialModel<ScalarT>(lamentMaterialModelName, lamentMaterialParameters);

  // Get a list of the LAMENT material model state variable names
  lamentMaterialModelStateVariableNames = LameUtils::getStateVariableNames(lamentMaterialModelName, lamentMaterialParameters);

  // Declare the state variables as evaluated fields (type is always double)
  Teuchos::RCP<PHX::DataLayout> dataLayout = p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
  for(unsigned int i=0 ; i<lamentMaterialModelStateVariableNames.size() ; ++i){
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> lamentMaterialModelStateVariableField(lamentMaterialModelStateVariableNames[i], dataLayout);
    this->addEvaluatedField(lamentMaterialModelStateVariableField);
    lamentMaterialModelStateVariableFields.push_back(lamentMaterialModelStateVariableField);
  }
}

template<typename EvalT, typename Traits>
void LamentStress<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(defGradField,fm);
  this->utils.setFieldData(stressField,fm);
  for(unsigned int i=0 ; i<lamentMaterialModelStateVariableFields.size() ; ++i)
    this->utils.setFieldData(lamentMaterialModelStateVariableFields[i],fm);
}

template<typename EvalT, typename Traits>
void LamentStress<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::RCP<lament::matParams<ScalarT> > matp = Teuchos::rcp(new lament::matParams<ScalarT>());

  // Get the old state data
  Albany::MDArray oldDefGrad = (*workset.stateArrayPtr)[defGradName];
  Albany::MDArray oldStress = (*workset.stateArrayPtr)[stressName];

  int numStateVariables = (int)(this->lamentMaterialModelStateVariableNames.size());

  // \todo Get actual time step for calls to LAMENT materials.
  double deltaT = 1.0;

  vector<ScalarT> strainRate(6);              // symmetric tensor
  vector<ScalarT> spin(3);                     // skew-symmetric tensor
  vector<ScalarT> leftStretch(6);              // symmetric tensor
  vector<ScalarT> rotation(9);                 // full tensor
  vector<double> stressOld(6);                // symmetric tensor
  vector<ScalarT> stressNew(6);               // symmetric tensor
  vector<double> stateOld(numStateVariables); // a single scalar for each state variable
  vector<double> stateNew(numStateVariables); // a single scalar for each state variable

  // \todo Set up scratch space for material models using getNumScratchVars() and setScratchPtr().

  // Create the matParams structure, which is passed to Lament
  matp->nelements = 1;
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
//   matp->dt_mat = std::numeric_limits<double>::max();
  
  // matParams that still need to be added:
  // matp->temp_old  (temperature)
  // matp->temp_new
  // matp->sound_speed_old
  // matp->sound_speed_new
  // matp->volume
  // scratch pointer
  // function pointers (lots to be done here)

  for (int cell=0; cell < (int)workset.numCells; ++cell) {
    for (int qp=0; qp < (int)numQPs; ++qp) {

      // Fill the following entries in matParams for call to LAMENT
      //
      // nelements     - number of elements 
      // dt            - time step, this one is tough because Albany does not currently have a concept of time step for implicit integration
      // time          - current time, again Albany does not currently have a concept of time for implicit integration
      // strain_rate   - what Sierra calls the rate of deformation, it is the symmetric part of the velocity gradient
      // spin          - anti-symmetric part of the velocity gradient
      // left_stretch  - found as V in the polar decomposition of the deformation gradient F = VR
      // rotation      - found as R in the polar decomposition of the deformation gradient F = VR
      // state_old     - material state data for previous time step (material dependent, none for lament::Elastic)
      // state_new     - material state data for current time step (material dependent, none for lament::Elastic)
      // stress_old    - stress at previous time step
      // stress_new    - stress at current time step, filled by material model
      //
      // The total deformation gradient is available as field data
      // 
      // The velocity gradient is not available but can be computed at the logarithm of the incremental deformation gradient divided by deltaT
      // The incremental deformation gradient is computed as F_new F_old^-1

      // JTO:  here is how I think this will go (of course the first two lines won't work as is...)
      // LCM::Tensor<RealType> F = newDefGrad;
      // LCM::Tensor<RealType> Fn = oldDefGrad;
      // LCM::Tensor<RealType> f = F*LCM::inverse(Fn);
      // LCM::Tensor<RealType> V;
      // LCM::Tensor<RealType> R;
      // boost::tie(V,R) = LCM::polar_left(F);
      // LCM::Tensor<RealType> Vinc;
      // LCM::Tensor<RealType> Rinc;
      // LCM::Tensor<RealType> logVinc;
      // boost::tie(Vinc,Rinc,logVinc) = LCM::polar_left_logV(f)
      // LCM::Tensor<RealType> logRinc = LCM::log_rotation(Rinc);
      // LCM::Tensor<RealType> logf = LCM::bch(logVinc,logRinc);
      // LCM::Tensor<RealType> L = (1.0/deltaT)*logf;
      // LCM::Tensor<RealType> D = LCM::sym(L);
      // LCM::Tensor<RealType> W = LCM::skew(L);
      // and then fill data into the vectors below

      // new deformation gradient (the current deformation gradient as computed in the current configuration)
      LCM::Tensor<ScalarT, 3> Fnew(
       defGradField(cell,qp,0,0), defGradField(cell,qp,0,1), defGradField(cell,qp,0,2),
       defGradField(cell,qp,1,0), defGradField(cell,qp,1,1), defGradField(cell,qp,1,2),
       defGradField(cell,qp,2,0), defGradField(cell,qp,2,1), defGradField(cell,qp,2,2) );

      // old deformation gradient (deformation gradient at previous load step)
      LCM::Tensor<ScalarT, 3> Fold( oldDefGrad(cell,qp,0,0), oldDefGrad(cell,qp,0,1), oldDefGrad(cell,qp,0,2),
				 oldDefGrad(cell,qp,1,0), oldDefGrad(cell,qp,1,1), oldDefGrad(cell,qp,1,2),
				 oldDefGrad(cell,qp,2,0), oldDefGrad(cell,qp,2,1), oldDefGrad(cell,qp,2,2) );

      // incremental deformation gradient
      LCM::Tensor<ScalarT, 3> Finc = Fnew * LCM::inverse(Fold);

      // DEBUGGING //
      if(cell==0 && qp==0){
	std::cout << "Fnew(0,0) " << Fnew(0,0) << endl;
	std::cout << "Fnew(1,0) " << Fnew(1,0) << endl;
	std::cout << "Fnew(2,0) " << Fnew(2,0) << endl;
	std::cout << "Fnew(0,1) " << Fnew(0,1) << endl;
	std::cout << "Fnew(1,1) " << Fnew(1,1) << endl;
	std::cout << "Fnew(2,1) " << Fnew(2,1) << endl;
	std::cout << "Fnew(0,2) " << Fnew(0,2) << endl;
	std::cout << "Fnew(1,2) " << Fnew(1,2) << endl;
	std::cout << "Fnew(2,2) " << Fnew(2,2) << endl;
      }
      // END DEBUGGING //

      // left stretch V, and rotation R, from left polar decomposition of new deformation gradient
      LCM::Tensor<ScalarT, 3> V, R;
      boost::tie(V,R) = LCM::polar_left(Fnew);

      // DEBUGGING //
      if(cell==0 && qp==0){
	std::cout << "V(0,0) " << V(0,0) << endl;
	std::cout << "V(1,0) " << V(1,0) << endl;
	std::cout << "V(2,0) " << V(2,0) << endl;
	std::cout << "V(0,1) " << V(0,1) << endl;
	std::cout << "V(1,1) " << V(1,1) << endl;
	std::cout << "V(2,1) " << V(2,1) << endl;
	std::cout << "V(0,2) " << V(0,2) << endl;
	std::cout << "V(1,2) " << V(1,2) << endl;
	std::cout << "V(2,2) " << V(2,2) << endl;
	std::cout << "R(0,0) " << R(0,0) << endl;
	std::cout << "R(1,0) " << R(1,0) << endl;
	std::cout << "R(2,0) " << R(2,0) << endl;
	std::cout << "R(0,1) " << R(0,1) << endl;
	std::cout << "R(1,1) " << R(1,1) << endl;
	std::cout << "R(2,1) " << R(2,1) << endl;
	std::cout << "R(0,2) " << R(0,2) << endl;
	std::cout << "R(1,2) " << R(1,2) << endl;
	std::cout << "R(2,2) " << R(2,2) << endl;
      }
      // END DEBUGGING //

      // incremental left stretch Vinc, incremental rotation Rinc, and log of incremental left stretch, logVinc
      LCM::Tensor<ScalarT, 3> Vinc, Rinc, logVinc;
      boost::tie(Vinc,Rinc,logVinc) = LCM::polar_left_logV(Finc);

      // log of incremental rotation
      LCM::Tensor<ScalarT, 3> logRinc = LCM::log_rotation(Rinc);

      // log of incremental deformation gradient
      LCM::Tensor<ScalarT, 3> logFinc = LCM::bch(logVinc, logRinc);

      // velocity gradient
      LCM::Tensor<ScalarT, 3> L = RealType(1.0/deltaT)*logFinc;

      // strain rate (a.k.a rate of deformation)
      LCM::Tensor<ScalarT, 3> D = LCM::symm(L);

      // spin
      LCM::Tensor<ScalarT, 3> W = LCM::skew(L);

      // load everything into the Lament data structure

      strainRate[0] = ( D(0,0) );
      strainRate[1] = ( D(1,1) );
      strainRate[2] = ( D(2,2) );
      strainRate[3] = ( D(0,1) );
      strainRate[4] = ( D(1,2) );
      strainRate[5] = ( D(0,2) );

      spin[0] = ( W(0,1) );
      spin[1] = ( W(1,2) );
      spin[2] = ( W(0,2) );

      leftStretch[0] = ( V(0,0) );
      leftStretch[1] = ( V(1,1) );
      leftStretch[2] = ( V(2,2) );
      leftStretch[3] = ( V(0,1) );
      leftStretch[4] = ( V(1,2) );
      leftStretch[5] = ( V(0,2) );

      rotation[0] = ( R(0,0) );
      rotation[1] = ( R(1,1) );
      rotation[2] = ( R(2,2) );
      rotation[3] = ( R(0,1) );
      rotation[4] = ( R(1,2) );
      rotation[5] = ( R(0,2) );
      rotation[6] = ( R(1,0) );
      rotation[7] = ( R(2,1) );
      rotation[8] = ( R(2,0) );

      stressOld[0] = oldStress(cell,qp,0,0);
      stressOld[1] = oldStress(cell,qp,1,1);
      stressOld[2] = oldStress(cell,qp,2,2);
      stressOld[3] = oldStress(cell,qp,0,1);
      stressOld[4] = oldStress(cell,qp,1,2);
      stressOld[5] = oldStress(cell,qp,0,2);

      // copy data from the state manager to the LAMENT data structure
      for(int iVar=0 ; iVar<numStateVariables ; iVar++){
        const std::string& variableName = this->lamentMaterialModelStateVariableNames[iVar]+"_old";
        Albany::MDArray stateVar = (*workset.stateArrayPtr)[variableName];
        stateOld[iVar] = stateVar(cell,qp);
      }

      // Make a call to the LAMENT material model to initialize the load step
      this->lamentMaterialModel->loadStepInit(matp.get());

      // Get the stress from the LAMENT material
      this->lamentMaterialModel->getStress(matp.get());

      // DEBUGGING //
      if(cell==0 && qp==0){
	std::cout << "check strainRate[0] " << strainRate[0] << endl;
	std::cout << "check strainRate[1] " << strainRate[1] << endl;
	std::cout << "check strainRate[2] " << strainRate[2] << endl;
	std::cout << "check strainRate[3] " << strainRate[3] << endl;
	std::cout << "check strainRate[4] " << strainRate[4] << endl;
	std::cout << "check strainRate[5] " << strainRate[5] << endl;
      }
      // END DEBUGGING //

      // Copy the new stress into the stress field
      stressField(cell,qp,0,0) = stressNew[0];
      stressField(cell,qp,1,1) = stressNew[1];
      stressField(cell,qp,2,2) = stressNew[2];
      stressField(cell,qp,0,1) = stressNew[3];
      stressField(cell,qp,1,2) = stressNew[4];
      stressField(cell,qp,0,2) = stressNew[5];
      stressField(cell,qp,1,0) = stressNew[3]; 
      stressField(cell,qp,2,1) = stressNew[4]; 
      stressField(cell,qp,2,0) = stressNew[5];

      // copy state_new data from the LAMENT data structure to the corresponding state variable field
      for(int iVar=0 ; iVar<numStateVariables ; iVar++)
	this->lamentMaterialModelStateVariableFields[iVar](cell,qp) = stateNew[iVar];

      // DEBUGGING //
      if(cell==0 && qp==0){
	std::cout << "stress(0,0) " << this->stressField(cell,qp,0,0) << endl;
	std::cout << "stress(1,1) " << this->stressField(cell,qp,1,1) << endl;
	std::cout << "stress(2,2) " << this->stressField(cell,qp,2,2) << endl;
	std::cout << "stress(0,1) " << this->stressField(cell,qp,0,1) << endl;
	std::cout << "stress(1,2) " << this->stressField(cell,qp,1,2) << endl;
	std::cout << "stress(0,2) " << this->stressField(cell,qp,0,2) << endl;
	std::cout << "stress(1,0) " << this->stressField(cell,qp,1,0) << endl;
	std::cout << "stress(2,1) " << this->stressField(cell,qp,2,1) << endl;
	std::cout << "stress(2,0) " << this->stressField(cell,qp,2,0) << endl;
      }
      // END DEBUGGING //

    }
  }
}

}

