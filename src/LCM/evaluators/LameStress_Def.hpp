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

#ifdef ENABLE_LAME
#include <models/Material.h>
#include <models/Elastic.h>
#endif

namespace LCM {

//**********************************************************************
template<typename EvalT, typename Traits>
LameStress<EvalT, Traits>::
LameStress(const Teuchos::ParameterList& p) :
  strainField(p.get<std::string>("Strain Name"),
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

  this->addDependentField(strainField);
  this->addDependentField(elasticModulusField);
  if (numDims>1) this->addDependentField(poissonsRatioField);

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
  this->utils.setFieldData(stressField,fm);
  this->utils.setFieldData(strainField,fm);
  this->utils.setFieldData(elasticModulusField,fm);
  if (numDims>1) this->utils.setFieldData(poissonsRatioField,fm);
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
  switch (numDims) {
  case 1:
    TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter, " LAME materials enabled only for three-dimensional analyses.");
    break;
  case 2:
    TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter, " LAME materials enabled only for three-dimensional analyses.");
    break;
  case 3:

    // Get the old and new state data
    // StateVariables is:  typedef std::map<std::string, Teuchos::RCP<Intrepid::FieldContainer<RealType> > >
    Albany::StateVariables& newState = *workset.newState;
    Albany::StateVariables oldState = *workset.oldState;
    const Intrepid::FieldContainer<RealType>& oldStrain  = *oldState["strain"];
    Intrepid::FieldContainer<RealType>& newStrain  = *newState["strain"];
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
        Teuchos::RCP<lame::matParams> matp = Teuchos::rcp(new lame::matParams());
        matp->nelements = 1;
        matp->dt = deltaT;

        std::vector<RealType> strainRate(6);
        strainRate[0] = ( newStrain(cell,qp,0,0) - oldStrain(cell,qp,0,0) ) / deltaT; // xx
        strainRate[1] = ( newStrain(cell,qp,1,1) - oldStrain(cell,qp,1,1) ) / deltaT; // yy
        strainRate[2] = ( newStrain(cell,qp,2,2) - oldStrain(cell,qp,2,2) ) / deltaT; // zz
        strainRate[3] = ( newStrain(cell,qp,0,1) - oldStrain(cell,qp,0,1) ) / deltaT; // xy
        strainRate[4] = ( newStrain(cell,qp,1,2) - oldStrain(cell,qp,1,2) ) / deltaT; // yz
        strainRate[5] = ( newStrain(cell,qp,0,2) - oldStrain(cell,qp,0,2) ) / deltaT; // xz
        matp->strain_rate = &strainRate[0];

        std::vector<RealType> stressOld(6);
        stressOld[0] = oldStress(cell,qp,0,0);
        stressOld[1] = oldStress(cell,qp,1,1);
        stressOld[2] = oldStress(cell,qp,2,2);
        stressOld[3] = oldStress(cell,qp,0,1);
        stressOld[4] = oldStress(cell,qp,1,2);
        stressOld[5] = oldStress(cell,qp,0,2);
        matp->stress_old = &stressOld[0];

        std::vector<RealType> stressNew(6);
        matp->stress_new = &stressNew[0];

        // Get the stress from the LAME material
        elasticMat->getStress(matp.get());

        // Copy the stress into both the state variable and the field
        newStress(cell,qp,0,0) = stressNew[0];
        newStress(cell,qp,1,1) = stressNew[1];
        newStress(cell,qp,2,2) = stressNew[2];
        newStress(cell,qp,0,1) = stressNew[3];
        newStress(cell,qp,1,2) = stressNew[4];
        newStress(cell,qp,0,2) = stressNew[5];
        newStress(cell,qp,1,0) = newStress(cell,qp,0,1); 
        newStress(cell,qp,2,1) = newStress(cell,qp,1,2); 
        newStress(cell,qp,2,0) = newStress(cell,qp,0,2); 
        
        stressField(cell,qp,0,0) = stressNew[0];
        stressField(cell,qp,1,1) = stressNew[1];
        stressField(cell,qp,2,2) = stressNew[2];
        stressField(cell,qp,0,1) = stressNew[3];
        stressField(cell,qp,1,2) = stressNew[4];
        stressField(cell,qp,0,2) = stressNew[5];
        stressField(cell,qp,1,0) = stressField(cell,qp,0,1); 
        stressField(cell,qp,2,1) = stressField(cell,qp,1,2); 
        stressField(cell,qp,2,0) = stressField(cell,qp,0,2); 
      }
    }

    break;
  }
}

#endif
//**********************************************************************
}

