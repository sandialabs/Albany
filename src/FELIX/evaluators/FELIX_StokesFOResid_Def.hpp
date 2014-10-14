//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN



namespace FELIX {

//**********************************************************************
template<typename EvalT, typename Traits>
StokesFOResid<EvalT, Traits>::
StokesFOResid(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl) :
  wBF      (p.get<std::string> ("Weighted BF Name"), dl->node_qp_scalar),
  wGradBF  (p.get<std::string> ("Weighted Gradient BF Name"),dl->node_qp_gradient),
  U        (p.get<std::string> ("QP Variable Name"), dl->qp_vector),
  Ugrad    (p.get<std::string> ("Gradient QP Variable Name"), dl->qp_vecgradient),
  force    (p.get<std::string> ("Body Force Name"), dl->qp_vector),
  muFELIX  (p.get<std::string> ("FELIX Viscosity QP Variable Name"), dl->qp_scalar),
  Residual (p.get<std::string> ("Residual Name"), dl->node_vector)
{

  Teuchos::ParameterList* list = 
    p.get<Teuchos::ParameterList*>("Parameter List");

  std::string type = list->get("Type", "FELIX");

  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());
  if (type == "FELIX") {
#ifdef OUTPUT_TO_SCREEN
    *out << "setting FELIX FO model physics" << std::endl;
#endif 
    eqn_type = FELIX;
  }
  else if (type == "Poisson") { //temporary addition of Poisson operator for debugging of Neumann BC
#ifdef OUTPUT_TO_SCREEN
    *out << "setting Poisson (Laplace) operator" << std::endl; 
#endif
    eqn_type = POISSON;
  }

  this->addDependentField(U);
  this->addDependentField(Ugrad);
  this->addDependentField(force);
  //this->addDependentField(UDot);
  this->addDependentField(wBF);
  this->addDependentField(wGradBF);
  this->addDependentField(muFELIX);

  this->addEvaluatedField(Residual);


  this->setName("StokesFOResid"+PHX::TypeString<EvalT>::value);

  std::vector<PHX::DataLayout::size_type> dims;
  wGradBF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
  numDims  = dims[3];

  U.fieldTag().dataLayout().dimensions(dims);
  vecDim  = dims[2];

//*out << " in FELIX Stokes FO residual! " << std::endl;
//*out << " vecDim = " << vecDim << std::endl;
//*out << " numDims = " << numDims << std::endl;
//*out << " numQPs = " << numQPs << std::endl; 
//*out << " numNodes = " << numNodes << std::endl; 

if (vecDim != 2 & eqn_type == FELIX)  {TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
				  std::endl << "Error in FELIX::StokesFOResid constructor:  " <<
				  "Invalid Parameter vecDim.  Problem implemented for 2 dofs per node only (u and v). " << std::endl);}
if (vecDim != 1 & eqn_type == POISSON)  {TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
				  std::endl << "Error in FELIX::StokesFOResid constructor:  " <<
				  "Invalid Parameter vecDim.  Poisson problem implemented for 1 dof per node only. " << std::endl);}

}

//**********************************************************************
template<typename EvalT, typename Traits>
void StokesFOResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(U,fm);
  this->utils.setFieldData(Ugrad,fm);
  this->utils.setFieldData(force,fm);
  //this->utils.setFieldData(UDot,fm);
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(wGradBF,fm);
  this->utils.setFieldData(muFELIX,fm);

  this->utils.setFieldData(Residual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void StokesFOResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  typedef Intrepid::FunctionSpaceTools FST; 

  for (std::size_t i=0; i < Residual.size(); ++i) Residual(i)=0.0;

  if (numDims == 3) { //3D case
    if (eqn_type == FELIX) {
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
        ScalarT& mu = muFELIX(cell,qp);
        ScalarT strs00 = 2.0*mu*(2.0*Ugrad(cell,qp,0,0) + Ugrad(cell,qp,1,1));
        ScalarT strs11 = 2.0*mu*(2.0*Ugrad(cell,qp,1,1) + Ugrad(cell,qp,0,0));
        ScalarT strs01 = mu*(Ugrad(cell,qp,1,0)+ Ugrad(cell,qp,0,1));
        ScalarT strs02 = mu*Ugrad(cell,qp,0,2);
        ScalarT strs12 = mu*Ugrad(cell,qp,1,2);
        for (std::size_t node=0; node < numNodes; ++node) {
             Residual(cell,node,0) += strs00*wGradBF(cell,node,qp,0) + 
                                      strs01*wGradBF(cell,node,qp,1) + 
                                      strs02*wGradBF(cell,node,qp,2);
             Residual(cell,node,1) += strs01*wGradBF(cell,node,qp,0) +
                                      strs11*wGradBF(cell,node,qp,1) + 
                                      strs12*wGradBF(cell,node,qp,2); 
        }
      }
      for (std::size_t qp=0; qp < numQPs; ++qp) {
        ScalarT& frc0 = force(cell,qp,0);
        ScalarT& frc1 = force(cell,qp,1);
        for (std::size_t node=0; node < numNodes; ++node) {
             Residual(cell,node,0) += frc0*wBF(cell,node,qp);
             Residual(cell,node,1) += frc1*wBF(cell,node,qp); 
        }
      }
    } }
    else if (eqn_type == POISSON) { //Laplace (Poisson) operator
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t node=0; node < numNodes; ++node) {
          for (std::size_t qp=0; qp < numQPs; ++qp) {
             Residual(cell,node,0) += Ugrad(cell,qp,0,0)*wGradBF(cell,node,qp,0) + 
                                      Ugrad(cell,qp,0,1)*wGradBF(cell,node,qp,1) + 
                                      Ugrad(cell,qp,0,2)*wGradBF(cell,node,qp,2) +  
                                      force(cell,qp,0)*wBF(cell,node,qp);
              }
           
    } } }
   }
   else { //2D case
   if (eqn_type == FELIX) { 
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t node=0; node < numNodes; ++node) {
          for (std::size_t qp=0; qp < numQPs; ++qp) {
             Residual(cell,node,0) += 2.0*muFELIX(cell,qp)*((2.0*Ugrad(cell,qp,0,0) + Ugrad(cell,qp,1,1))*wGradBF(cell,node,qp,0) + 
                                      0.5*(Ugrad(cell,qp,0,1) + Ugrad(cell,qp,1,0))*wGradBF(cell,node,qp,1)) + 
                                      force(cell,qp,0)*wBF(cell,node,qp);
             Residual(cell,node,1) += 2.0*muFELIX(cell,qp)*(0.5*(Ugrad(cell,qp,0,1) + Ugrad(cell,qp,1,0))*wGradBF(cell,node,qp,0) +
                                      (Ugrad(cell,qp,0,0) + 2.0*Ugrad(cell,qp,1,1))*wGradBF(cell,node,qp,1)) + force(cell,qp,1)*wBF(cell,node,qp); 
              }
           
    } } }
    else if (eqn_type == POISSON) { //Laplace (Poisson) operator
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t node=0; node < numNodes; ++node) {
          for (std::size_t qp=0; qp < numQPs; ++qp) {
             Residual(cell,node,0) += Ugrad(cell,qp,0,0)*wGradBF(cell,node,qp,0) + 
                                      Ugrad(cell,qp,0,1)*wGradBF(cell,node,qp,1) + 
                                      force(cell,qp,0)*wBF(cell,node,qp);
              }
           
    } } }
   }
}

//**********************************************************************
}

