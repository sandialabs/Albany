//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Albany_Layouts.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Albany_StateManager.hpp"

namespace ATO {

//**********************************************************************
template<typename EvalT, typename Traits>
Stress<EvalT, Traits>::
Stress(const Teuchos::ParameterList& p) :
  strain           (p.get<std::string>                   ("Strain Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") ),
  stress           (p.get<std::string>                   ("Stress Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout") )
{
  // Pull out numQPs and numDims from a Layout
  Teuchos::RCP<PHX::DataLayout> tensor_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Tensor Data Layout");
  std::vector<PHX::Device::size_type> dims;
  tensor_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  this->addDependentField(strain);
  this->addEvaluatedField(stress);

  if(p.isSublist("Homogenized Constants")){
    homogenizedConstantsName = p.sublist("Homogenized Constants").get<std::string>("Stiffness Name");
    useHomogenizedConstants = true;
    Albany::StateManager* stateMgr = p.get<Albany::StateManager*>("State Manager");
    Teuchos::RCP<Albany::Layouts> dl = p.get<Teuchos::RCP<Albany::Layouts> >("Data Layout");
    int nVoigt = 0;
    for(int i=1; i<=numDims; i++)
      nVoigt += i;
    for(int i=1; i<=nVoigt; i++){
      for(int j=i; j<=nVoigt; j++){
        std::stringstream valname;
        valname << homogenizedConstantsName << " " << i << j;
        stateMgr->registerStateVariable(valname.str(),dl->workset_scalar, dl->dummy, "all", "scalar",0.0, false, false);
      }
    }
  } else {
    useHomogenizedConstants = false;
    elasticModulus = p.get<double>("Elastic Modulus");
    poissonsRatio = p.get<double>("Poissons Ratio");
  }

  this->setName("Stress"+PHX::typeAsString<EvalT>());

  if(p.isType<int>("Cell Forcing Column")){
    addCellForcing = true;
    cellForcingColumn = p.get<int>("Cell Forcing Column");

    int nVoigt = 0;
    for(int i=1; i<=numDims; i++)
      nVoigt += i;
    TEUCHOS_TEST_FOR_EXCEPTION( 
      cellForcingColumn < 0 || cellForcingColumn >= nVoigt, std::logic_error,
      "Add Cell Problem Forcing: invalid column index")

    RealType lambda = ( elasticModulus * poissonsRatio ) / ( ( 1 + poissonsRatio ) * ( 1 - 2 * poissonsRatio ) );
    RealType mu = elasticModulus / ( 2 * ( 1 + poissonsRatio ) );

    subTensor = Kokkos::DynRankView<RealType, PHX::Device>("SUB", numDims,numDims);

    switch (numDims) {
    case 1:
      subTensor(0,0) = lambda + 2.0*mu;
    break;

    case 2:
      if(cellForcingColumn==0){ subTensor(0,0)=lambda + 2.0*mu; subTensor(1,1)=lambda; } else 
      if(cellForcingColumn==1){ subTensor(0,0)=lambda; subTensor(1,1)=lambda + 2.0*mu; } else
      if(cellForcingColumn==2){ subTensor(0,1)=2.0*mu; subTensor(1,0)=2.0*mu; }
    break;

    case 3:
      if(cellForcingColumn==0){ subTensor(0,0)=lambda + 2.0*mu; subTensor(1,1)=lambda; subTensor(2,2)=lambda; } else 
      if(cellForcingColumn==1){ subTensor(0,0)=lambda; subTensor(1,1)=lambda + 2.0*mu; subTensor(2,2)=lambda; } else
      if(cellForcingColumn==2){ subTensor(0,0)=lambda; subTensor(1,1)=lambda; subTensor(2,2)=lambda + 2.0*mu; } else
      if(cellForcingColumn==3){ subTensor(1,2)=2.0*mu; subTensor(2,1)=2.0*mu; } else
      if(cellForcingColumn==4){ subTensor(0,2)=2.0*mu; subTensor(2,0)=2.0*mu; } else
      if(cellForcingColumn==5){ subTensor(0,1)=2.0*mu; subTensor(1,0)=2.0*mu; }
    break;
    }

  } else 
    addCellForcing = false;

}

//**********************************************************************
template<typename EvalT, typename Traits>
void Stress<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(stress,fm);
  this->utils.setFieldData(strain,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void Stress<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  if(useHomogenizedConstants){

    std::string C(homogenizedConstantsName);

    if(numDims == 1) {
      RealType C11; Albany::MDArray wsC = (*workset.stateArrayPtr)[C+" 11"]; C = wsC(0);
      for (std::size_t cell=0; cell < workset.numCells; ++cell) {
        for (std::size_t qp=0; qp < numQPs; ++qp) {
          stress(cell,qp,0,0) = C11 * strain(cell,qp,0,0);
        }
      }
    } else
    if(numDims == 2) {
      RealType C11; Albany::MDArray wsC11 = (*workset.stateArrayPtr)[C+" 11"]; C11 = wsC11(0);
      RealType C12; Albany::MDArray wsC12 = (*workset.stateArrayPtr)[C+" 12"]; C12 = wsC12(0);
      RealType C13; Albany::MDArray wsC13 = (*workset.stateArrayPtr)[C+" 13"]; C13 = wsC13(0);
      RealType C22; Albany::MDArray wsC22 = (*workset.stateArrayPtr)[C+" 22"]; C22 = wsC22(0);
      RealType C23; Albany::MDArray wsC23 = (*workset.stateArrayPtr)[C+" 23"]; C23 = wsC23(0);
      RealType C33; Albany::MDArray wsC33 = (*workset.stateArrayPtr)[C+" 33"]; C33 = wsC33(0);
      for (std::size_t cell=0; cell < workset.numCells; ++cell) {
        for (std::size_t qp=0; qp < numQPs; ++qp) {
          stress(cell,qp,0,0) = C11 * ( strain(cell,qp,0,0) ) + C12 * strain(cell,qp,1,1) + C13 * strain(cell,qp,0,1);
          stress(cell,qp,1,1) = C12 * ( strain(cell,qp,0,0) ) + C22 * strain(cell,qp,1,1) + C23 * strain(cell,qp,0,1);
          stress(cell,qp,0,1) = C13 * ( strain(cell,qp,0,0) ) + C23 * strain(cell,qp,1,1) + C33 * strain(cell,qp,0,1);
          stress(cell,qp,1,0) = stress(cell,qp,0,1); 
        }
      }
    } else
    if(numDims == 3) {
      RealType C11; Albany::MDArray wsC11 = (*workset.stateArrayPtr)[C+" 11"]; C11 = wsC11(0);
      RealType C12; Albany::MDArray wsC12 = (*workset.stateArrayPtr)[C+" 12"]; C12 = wsC12(0);
      RealType C13; Albany::MDArray wsC13 = (*workset.stateArrayPtr)[C+" 13"]; C13 = wsC13(0);
      RealType C14; Albany::MDArray wsC14 = (*workset.stateArrayPtr)[C+" 14"]; C14 = wsC14(0);
      RealType C15; Albany::MDArray wsC15 = (*workset.stateArrayPtr)[C+" 15"]; C15 = wsC15(0);
      RealType C16; Albany::MDArray wsC16 = (*workset.stateArrayPtr)[C+" 16"]; C16 = wsC16(0);

      RealType C22; Albany::MDArray wsC22 = (*workset.stateArrayPtr)[C+" 22"]; C22 = wsC22(0);
      RealType C23; Albany::MDArray wsC23 = (*workset.stateArrayPtr)[C+" 23"]; C23 = wsC23(0);
      RealType C24; Albany::MDArray wsC24 = (*workset.stateArrayPtr)[C+" 24"]; C24 = wsC24(0);
      RealType C25; Albany::MDArray wsC25 = (*workset.stateArrayPtr)[C+" 25"]; C25 = wsC25(0);
      RealType C26; Albany::MDArray wsC26 = (*workset.stateArrayPtr)[C+" 26"]; C26 = wsC26(0);

      RealType C33; Albany::MDArray wsC33 = (*workset.stateArrayPtr)[C+" 33"]; C33 = wsC33(0);
      RealType C34; Albany::MDArray wsC34 = (*workset.stateArrayPtr)[C+" 34"]; C34 = wsC34(0);
      RealType C35; Albany::MDArray wsC35 = (*workset.stateArrayPtr)[C+" 35"]; C35 = wsC35(0);
      RealType C36; Albany::MDArray wsC36 = (*workset.stateArrayPtr)[C+" 36"]; C36 = wsC36(0);

      RealType C44; Albany::MDArray wsC44 = (*workset.stateArrayPtr)[C+" 44"]; C44 = wsC44(0);
      RealType C45; Albany::MDArray wsC45 = (*workset.stateArrayPtr)[C+" 45"]; C45 = wsC45(0);
      RealType C46; Albany::MDArray wsC46 = (*workset.stateArrayPtr)[C+" 46"]; C46 = wsC46(0);

      RealType C55; Albany::MDArray wsC55 = (*workset.stateArrayPtr)[C+" 55"]; C55 = wsC55(0);
      RealType C56; Albany::MDArray wsC56 = (*workset.stateArrayPtr)[C+" 56"]; C56 = wsC56(0);

      RealType C66; Albany::MDArray wsC66 = (*workset.stateArrayPtr)[C+" 66"]; C66 = wsC66(0);

      for (std::size_t cell=0; cell < workset.numCells; ++cell) {
        for (std::size_t qp=0; qp < numQPs; ++qp) {
          stress(cell,qp,0,0) = C11*strain(cell,qp,0,0)+C12*strain(cell,qp,1,1)+C13*strain(cell,qp,2,2)+C14*strain(cell,qp,1,2)+C15*strain(cell,qp,0,2)+C16*strain(cell,qp,0,1);
          stress(cell,qp,1,1) = C12*strain(cell,qp,0,0)+C22*strain(cell,qp,1,1)+C23*strain(cell,qp,2,2)+C24*strain(cell,qp,1,2)+C25*strain(cell,qp,0,2)+C26*strain(cell,qp,0,1);
          stress(cell,qp,2,2) = C13*strain(cell,qp,0,0)+C23*strain(cell,qp,1,1)+C33*strain(cell,qp,2,2)+C34*strain(cell,qp,1,2)+C35*strain(cell,qp,0,2)+C36*strain(cell,qp,0,1);
          stress(cell,qp,1,2) = C14*strain(cell,qp,0,0)+C24*strain(cell,qp,1,1)+C34*strain(cell,qp,2,2)+C44*strain(cell,qp,1,2)+C45*strain(cell,qp,0,2)+C46*strain(cell,qp,0,1);
          stress(cell,qp,0,2) = C15*strain(cell,qp,0,0)+C25*strain(cell,qp,1,1)+C35*strain(cell,qp,2,2)+C45*strain(cell,qp,1,2)+C55*strain(cell,qp,0,2)+C56*strain(cell,qp,0,1);
          stress(cell,qp,0,1) = C16*strain(cell,qp,0,0)+C26*strain(cell,qp,1,1)+C36*strain(cell,qp,2,2)+C46*strain(cell,qp,1,2)+C56*strain(cell,qp,0,2)+C66*strain(cell,qp,0,1);
          stress(cell,qp,2,1) = stress(cell,qp,1,2); 
          stress(cell,qp,2,0) = stress(cell,qp,0,2); 
          stress(cell,qp,1,0) = stress(cell,qp,0,1); 
        }
      }
    }
  } else {
    RealType lambda = ( elasticModulus * poissonsRatio ) / ( ( 1 + poissonsRatio ) * ( 1 - 2 * poissonsRatio ) );
    RealType mu = elasticModulus / ( 2 * ( 1 + poissonsRatio ) );
  
    switch (numDims) {
    case 1:
      for (std::size_t cell=0; cell < workset.numCells; ++cell) {
        for (std::size_t qp=0; qp < numQPs; ++qp) {
  	stress(cell,qp,0,0) = (lambda + 2.0 * mu) * strain(cell,qp,0,0);
        }
      }
      break;
    case 2:
      // Compute Stress (with the plane strain assumption for now)
      for (std::size_t cell=0; cell < workset.numCells; ++cell) {
        for (std::size_t qp=0; qp < numQPs; ++qp) {
          stress(cell,qp,0,0) = 2.0 * mu * ( strain(cell,qp,0,0) ) + lambda * ( strain(cell,qp,0,0) + strain(cell,qp,1,1) );
          stress(cell,qp,1,1) = 2.0 * mu * ( strain(cell,qp,1,1) ) + lambda * ( strain(cell,qp,0,0) + strain(cell,qp,1,1) );
          stress(cell,qp,0,1) = 2.0 * mu * ( strain(cell,qp,0,1) );
          stress(cell,qp,1,0) = stress(cell,qp,0,1); 
        }
      }
      break;
    case 3:
      // Compute Stress
      for (std::size_t cell=0; cell < workset.numCells; ++cell) {
        for (std::size_t qp=0; qp < numQPs; ++qp) {
          stress(cell,qp,0,0) = 2.0 * mu * ( strain(cell,qp,0,0) ) + lambda * ( strain(cell,qp,0,0) + strain(cell,qp,1,1) + strain(cell,qp,2,2) );
          stress(cell,qp,1,1) = 2.0 * mu * ( strain(cell,qp,1,1) ) + lambda * ( strain(cell,qp,0,0) + strain(cell,qp,1,1) + strain(cell,qp,2,2) );
          stress(cell,qp,2,2) = 2.0 * mu * ( strain(cell,qp,2,2) ) + lambda * ( strain(cell,qp,0,0) + strain(cell,qp,1,1) + strain(cell,qp,2,2) );
          stress(cell,qp,0,1) = 2.0 * mu * ( strain(cell,qp,0,1) );
          stress(cell,qp,1,2) = 2.0 * mu * ( strain(cell,qp,1,2) );
          stress(cell,qp,2,0) = 2.0 * mu * ( strain(cell,qp,2,0) );
          stress(cell,qp,1,0) = stress(cell,qp,0,1); 
          stress(cell,qp,2,1) = stress(cell,qp,1,2); 
          stress(cell,qp,0,2) = stress(cell,qp,2,0); 
        }
      }
      break;
    }
  } 

  if(addCellForcing){
    for (std::size_t cell=0; cell < workset.numCells; ++cell)
      for (std::size_t qp=0; qp < numQPs; ++qp)
        for (std::size_t i=0; i < numDims; ++i)
          for (std::size_t j=0; j < numDims; ++j)
            stress(cell,qp,i,j) -= subTensor(i,j);
  }
}

//**********************************************************************
}

