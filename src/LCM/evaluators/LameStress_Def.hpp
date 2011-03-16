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

#ifdef ENABLE_LAME
#include <models/Material.h>
#include <models/Elastic.h>
#endif

namespace LCM {

//**********************************************************************
template<typename EvalT, typename Traits>
LameStress<EvalT, Traits>::
LameStress(const Teuchos::ParameterList& p) :
  defGradField(p.get<std::string>("DefGrad Name"),
                           p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout")),
  elasticModulusField(p.get<std::string>("Elastic Modulus Name"),
                      p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  poissonsRatioField(p.get<std::string>("Poissons Ratio Name"),
                     p.get<Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout") ),
  stressField(p.get<std::string>("Stress Name"),
              p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") )
{
  // Pull out numQPs and numDims from a Layout
  Teuchos::RCP<PHX::DataLayout> tensor_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  tensor_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  this->addDependentField(defGradField);
  this->addDependentField(elasticModulusField);
  this->addDependentField(poissonsRatioField);

  this->addEvaluatedField(stressField);

  this->setName("LameStress"+PHX::TypeString<EvalT>::value);

  cout << "\nUSING LIBRARY OF ADVANCED MATERIALS FOR ENGINEERING (LAME)\n" << endl;
}

//**********************************************************************
template<typename EvalT, typename Traits>
void LameStress<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(elasticModulusField,fm);
  this->utils.setFieldData(poissonsRatioField,fm);
  this->utils.setFieldData(defGradField,fm);
  this->utils.setFieldData(stressField,fm);
}

//**********************************************************************
#ifndef ENABLE_LAME

template<typename EvalT, typename Traits>
void LameStress<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  TEST_FOR_EXCEPTION(true, std::runtime_error, " LAME materials not enabled, recompile with -DENABLE_LAME");
}

#else

template<typename EvalT, typename Traits>
void LameStress<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  if(numDims == 1 || numDims == 2)
    TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter, " LAME materials enabled only for three-dimensional analyses.");

  // Get the old and new state data
  // StateVariables is:  typedef std::map<std::string, Teuchos::RCP<Intrepid::FieldContainer<RealType> > >
  Albany::StateVariables& newState = *workset.newState;
  Albany::StateVariables oldState = *workset.oldState;
  const Intrepid::FieldContainer<RealType>& oldDefGrad  = *oldState["def_grad"];
  Intrepid::FieldContainer<RealType>& newDefGrad  = *newState["def_grad"];
  const Intrepid::FieldContainer<RealType>& oldStress  = *oldState["stress"];
  Intrepid::FieldContainer<RealType>& newStress  = *newState["stress"];

  // \todo Get actual time step for calls to LAME materials.
  RealType deltaT = 1.0e-3;

  for (std::size_t cell=0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp < numQPs; ++qp) {

      // \todo Optimize calls to LAME such that we're not creating a new material instance for each evaluation; also, call on block of elements with same material properties.
      
      Teuchos::RCP<lame::MatProps> props = Teuchos::rcp(new lame::MatProps());
      std::vector<RealType> elasticModulusVector;
      RealType elasticModulusValue = Sacado::ScalarValue<ScalarT>::eval(elasticModulusField(cell,qp));
      elasticModulusVector.push_back(elasticModulusValue);
      props->insert(std::string("YOUNGS_MODULUS"), elasticModulusVector);
      std::vector<RealType> poissonsRatioVector;
      poissonsRatioVector.push_back( Sacado::ScalarValue<ScalarT>::eval(poissonsRatioField(cell,qp)) );
      props->insert(std::string("POISSONS_RATIO"), poissonsRatioVector);
      
      Teuchos::RCP<lame::Material> elasticMat = Teuchos::rcp(new lame::Elastic(*props));

      // \todo Call initialize() on material.

      Teuchos::RCP<lame::matParams> matp = Teuchos::rcp(new lame::matParams());

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
      LCM::Tensor<ScalarT> Fnew( newDefGrad(cell,qp,0,0), newDefGrad(cell,qp,0,1), newDefGrad(cell,qp,0,2),
                                 newDefGrad(cell,qp,1,0), newDefGrad(cell,qp,1,1), newDefGrad(cell,qp,1,2),
                                 newDefGrad(cell,qp,2,0), newDefGrad(cell,qp,2,1), newDefGrad(cell,qp,2,2) );

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
      boost::tie(Vinc,Rinc,logVinc) = LCM::polar_left_logV(Fnew);

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
     
      // load data into standard arrays for LAME
      std::vector<RealType> strainRate(6);   // symmetric tensor
      std::vector<RealType> spin(3);         // skew-symmetric tensor
      std::vector<RealType> leftStretch(6);  // symmetric tensor
      std::vector<RealType> rotation(9);     // full tensor
      std::vector<RealType> stressOld(6);    // symmetric tensor
      std::vector<RealType> stressNew(6);    // symmetric tensor

      matp->nelements = 1;
      matp->dt = deltaT;
      matp->time = 0.0;
      matp->strain_rate = &strainRate[0];
      matp->spin = &spin[0];
      matp->left_stretch = &leftStretch[0];
      matp->rotation = &rotation[0];
      matp->state_old = 0;
      matp->state_new = 0;
      matp->stress_old = &stressOld[0];
      matp->stress_new = &stressNew[0];

      strainRate[0] = Sacado::ScalarValue<ScalarT>::eval( D(0,0) );
      strainRate[1] = Sacado::ScalarValue<ScalarT>::eval( D(1,1) );
      strainRate[2] = Sacado::ScalarValue<ScalarT>::eval( D(2,2) );
      strainRate[3] = Sacado::ScalarValue<ScalarT>::eval( D(0,1) );
      strainRate[4] = Sacado::ScalarValue<ScalarT>::eval( D(1,2) );
      strainRate[5] = Sacado::ScalarValue<ScalarT>::eval( D(0,2) );

      spin[0] = Sacado::ScalarValue<ScalarT>::eval( W(0,1) );
      spin[1] = Sacado::ScalarValue<ScalarT>::eval( W(1,2) );
      spin[2] = Sacado::ScalarValue<ScalarT>::eval( W(0,2) );

      leftStretch[0] = Sacado::ScalarValue<ScalarT>::eval( V(0,0) );
      leftStretch[1] = Sacado::ScalarValue<ScalarT>::eval( V(1,1) );
      leftStretch[2] = Sacado::ScalarValue<ScalarT>::eval( V(2,2) );
      leftStretch[3] = Sacado::ScalarValue<ScalarT>::eval( V(0,1) );
      leftStretch[4] = Sacado::ScalarValue<ScalarT>::eval( V(1,2) );
      leftStretch[5] = Sacado::ScalarValue<ScalarT>::eval( V(0,2) );

      rotation[0] = Sacado::ScalarValue<ScalarT>::eval( R(0,0) );
      rotation[1] = Sacado::ScalarValue<ScalarT>::eval( R(1,1) );
      rotation[2] = Sacado::ScalarValue<ScalarT>::eval( R(2,2) );
      rotation[3] = Sacado::ScalarValue<ScalarT>::eval( R(0,1) );
      rotation[4] = Sacado::ScalarValue<ScalarT>::eval( R(1,2) );
      rotation[5] = Sacado::ScalarValue<ScalarT>::eval( R(0,2) );
      rotation[6] = Sacado::ScalarValue<ScalarT>::eval( R(1,0) );
      rotation[7] = Sacado::ScalarValue<ScalarT>::eval( R(2,1) );
      rotation[8] = Sacado::ScalarValue<ScalarT>::eval( R(2,0) );

      stressOld[0] = oldStress(cell,qp,0,0);
      stressOld[1] = oldStress(cell,qp,1,1);
      stressOld[2] = oldStress(cell,qp,2,2);
      stressOld[3] = oldStress(cell,qp,0,1);
      stressOld[4] = oldStress(cell,qp,1,2);
      stressOld[5] = oldStress(cell,qp,0,2);

      // \todo Call loadStepInit();

      // Get the stress from the LAME material
      elasticMat->getStress(matp.get());
        
      // Copy the new stress into the stress field
      stressField(cell,qp,0,0) = stressNew[0];
      stressField(cell,qp,1,1) = stressNew[1];
      stressField(cell,qp,2,2) = stressNew[2];
      stressField(cell,qp,0,1) = stressNew[3];
      stressField(cell,qp,1,2) = stressNew[4];
      stressField(cell,qp,0,2) = stressNew[5];
      stressField(cell,qp,1,0) = stressField(cell,qp,0,1); 
      stressField(cell,qp,2,1) = stressField(cell,qp,1,2); 
      stressField(cell,qp,2,0) = stressField(cell,qp,0,2);

      // Copy the new stress into the "stress" component of StateVariables
      newStress(cell,qp,0,0) = stressNew[0];
      newStress(cell,qp,1,1) = stressNew[1];
      newStress(cell,qp,2,2) = stressNew[2];
      newStress(cell,qp,0,1) = stressNew[3];
      newStress(cell,qp,1,2) = stressNew[4];
      newStress(cell,qp,0,2) = stressNew[5];
      newStress(cell,qp,1,0) = newStress(cell,qp,0,1); 
      newStress(cell,qp,2,1) = newStress(cell,qp,1,2); 
      newStress(cell,qp,2,0) = newStress(cell,qp,0,2); 

      // Copy the new deformation gradient into the "def_grad" component of StateVariables
      // Need to do this so we can get oldDefGrad next time around
      // \todo Should this be done in the defGrad evaluator?
      newDefGrad(cell,qp,0,0) = Sacado::ScalarValue<ScalarT>::eval(defGradField[0]);
      newDefGrad(cell,qp,1,1) = Sacado::ScalarValue<ScalarT>::eval(defGradField[1]);
      newDefGrad(cell,qp,2,2) = Sacado::ScalarValue<ScalarT>::eval(defGradField[2]);
      newDefGrad(cell,qp,0,1) = Sacado::ScalarValue<ScalarT>::eval(defGradField[3]);
      newDefGrad(cell,qp,1,2) = Sacado::ScalarValue<ScalarT>::eval(defGradField[4]);
      newDefGrad(cell,qp,0,2) = Sacado::ScalarValue<ScalarT>::eval(defGradField[5]);
      newDefGrad(cell,qp,1,0) = Sacado::ScalarValue<ScalarT>::eval(defGradField[6]);
      newDefGrad(cell,qp,2,1) = Sacado::ScalarValue<ScalarT>::eval(defGradField[7]);
      newDefGrad(cell,qp,2,0) = Sacado::ScalarValue<ScalarT>::eval(defGradField[8]);
    }
  }
}

#endif
//**********************************************************************
}

