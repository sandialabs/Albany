//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

namespace Tsunami {

//**********************************************************************
template<typename EvalT, typename Traits>
BoussinesqBodyForce<EvalT, Traits>::
BoussinesqBodyForce(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl) :
  out              (Teuchos::VerboseObjectBase::getDefaultOStream()),
  waterDepthQP     (p.get<std::string> ("Water Depth QP Name"), dl->qp_scalar), 
  betaQP           (p.get<std::string> ("Beta QP Name"), dl->qp_scalar), 
  zalphaQP         (p.get<std::string> ("z_alpha QP Name"), dl->qp_scalar), 
  muSqr            (p.get<double>("Mu Squared")), 
  epsilon          (p.get<double>("Epsilon")), 
  coordVec         (p.get<std::string>("Coordinate Vector Name"), dl->qp_gradient), 
  force            (p.get<std::string> ("Body Force Name"), dl->qp_vector)
{

  this->addDependentField(waterDepthQP);
  this->addDependentField(betaQP);
  this->addDependentField(zalphaQP);
  this->addDependentField(coordVec);

  this->addEvaluatedField(force);

  std::vector<PHX::DataLayout::size_type> dims;
  dl->qp_gradient->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  force.fieldTag().dataLayout().dimensions(dims);
  vecDim  = dims[2];
   
 
  Teuchos::ParameterList* bf_list =
    p.get<Teuchos::ParameterList*>("Parameter List");

  std::string type = bf_list->get("Type", "None");
  
  if (type == "None") {
    bf_type = NONE;
  }
  else if (type == "1D Solitary Wave") {
    bf_type = ONED_SOL_WAVE; 
  }
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true, std::logic_error,
        "Error in Tsunami::Boussinesq: Invalid Body Force Type = "
            << type << "!  Valid types are: 'None, 1D Solitary Wave'.");
  }

  this->setName("BoussinesqBodyForce"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void BoussinesqBodyForce<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(waterDepthQP,fm);
  this->utils.setFieldData(betaQP,fm);
  this->utils.setFieldData(zalphaQP,fm);
  this->utils.setFieldData(coordVec,fm);
  this->utils.setFieldData(force,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void BoussinesqBodyForce<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  //Zero out body force
 if (bf_type == NONE) {
   for (int cell=0; cell < workset.numCells; ++cell) 
     for (int qp=0; qp < numQPs; ++qp) 
       for (int i=0; i<vecDim; i++) 
           force(cell,qp,i) = 0.0; 
 }
//ZW: manufactured solution in form of Gaussian
 else if (bf_type == ONED_SOL_WAVE) {    
   const RealType time = workset.current_time; //ZW added time 
   const double A1 = 1.0;//ZW added constant A1(a)
   const double c = 2.0;//ZW added constant speed c

   for (std::size_t cell=0; cell < workset.numCells; ++cell) {
     for (std::size_t qp=0; qp < numQPs; ++qp) {
        //ScalarT* f = &force(cell,qp,0);
        //ZW: need to modify from dim to non-dim
        MeshScalarT X0 = coordVec(cell,qp,0);
        force(cell,qp,0) = A1*c*exp(-0.5*(X0-c*time)*(X0-c*time))*(X0-c*time)
			 + (waterDepthQP(cell,qp)+epsilon*A1*exp(-0.5*(X0-c*time)*(X0-c*time)))*(-0.5*A1*exp(-0.25*(X0-c*time)*(X0-c*time))*(X0-c*time));
			 + muSqr*(0.5*(betaQP(cell,qp)*betaQP(cell,qp)-1/3)+betaQP(cell,qp)+0.5)*waterDepthQP(cell,qp)*waterDepthQP(cell,qp)*waterDepthQP(cell,qp)*(-0.125*A1*exp(-0.25*(X0-c*time)*(X0-c*time))*(X0-c*time)*(c*c*time*time-2*c*time*X0+X0*X0-6));
        force(cell,qp,1) = (-0.5*A1*c*exp(-0.25*(X0-c*time)*(X0-c*time))*(X0-c*time))
			 + A1*exp(-0.5*(X0-c*time)*(X0-c*time))*(X0-c*time)
			 + (0.5*epsilon*(-A1*A1*exp(-0.5*(X0-c*time)*(X0-c*time))*(X0-c*time)))
			 + muSqr*waterDepthQP(cell,qp)*waterDepthQP(cell,qp)*(betaQP(cell,qp)*betaQP(cell,qp)*0.5+betaQP(cell,qp))*(0.125*A1*c*exp(-0.25*(X0-c*time)*(X0-c*time))*(X0-c*time)*(c*c*time*time-2*c*time*X0+X0*X0-6));
        force(cell,qp,2) = 0.0; 
    }
  }
 }
}


}

