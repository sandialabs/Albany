//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

#include "LandIce_StokesFOResid.hpp"

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN

namespace LandIce {

//**********************************************************************
template<typename EvalT, typename Traits>
StokesFOResid<EvalT, Traits>::
StokesFOResid(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl) :
  numNodes  (dl->node_qp_gradient->dimension(1)),
  numQPs    (dl->node_qp_gradient->dimension(2)),
  numDims   (dl->node_qp_gradient->dimension(3)),
  useStereographicMap (p.get<Teuchos::ParameterList*>("Stereographic Map")->get("Use Stereographic Map", false)),
  R2 (std::pow(p.get<Teuchos::ParameterList*>("Stereographic Map")->get<RealType>("Earth Radius", 6371),2)),
  x_0 (p.get<Teuchos::ParameterList*>("Stereographic Map")->get<RealType>("X_0", 0)),//-136)),
  y_0 (p.get<Teuchos::ParameterList*>("Stereographic Map")->get<RealType>("Y_0", 0)),//-2040)),
  wBF       (p.get<std::string> ("Weighted BF Variable Name"), dl->node_qp_scalar),
  wGradBF   (p.get<std::string> ("Weighted Gradient BF Variable Name"),dl->node_qp_gradient),
  U         (p.get<std::string> ("Velocity QP Variable Name"), dl->qp_vector),
  Ugrad     (p.get<std::string> ("Velocity Gradient QP Variable Name"), dl->qp_vecgradient),
  force     (p.get<std::string> ("Body Force Variable Name"), dl->qp_vector),
  muLandIce (p.get<std::string> ("Viscosity QP Variable Name"), dl->qp_scalar),
  Residual  (p.get<std::string> ("Residual Variable Name"), dl->node_vector)
{
#ifdef OUTPUT_TO_SCREEN
  Teuchos::RCP<Teuchos::FancyOStream> output(Teuchos::VerboseObjectBase::getDefaultOStream());

  int procRank = Teuchos::GlobalMPISession::getRank();
  int numProcs = Teuchos::GlobalMPISession::getNProc();
  output->setProcRankAndSize (procRank, numProcs);
  output->setOutputToRootOnly (0);
#endif

  Teuchos::ParameterList* list = p.get<Teuchos::ParameterList*>("Parameter List");

  std::string type = list->get("Type", "LandIce");

  if (type == "LandIce") {
#ifdef OUTPUT_TO_SCREEN
    *output << "setting LandIce FO model physics" << std::endl;
#endif
    eqn_type = LandIce;
  }
  //LandIce FO x-z MMS test case
  else if (type == "LandIce X-Z") {
#ifdef OUTPUT_TO_SCREEN
    *output << "setting LandIce FO X-Z model physics" << std::endl;
#endif
  eqn_type = LandIce_XZ;
  }
  else if (type == "Poisson") { //temporary addition of Poisson operator for debugging of Neumann BC
#ifdef OUTPUT_TO_SCREEN
    *output << "setting Poisson (Laplace) operator" << std::endl;
#endif
    eqn_type = POISSON;
  }

  this->addDependentField(U);
  this->addDependentField(Ugrad);
  this->addDependentField(force);
  this->addDependentField(wBF);
  this->addDependentField(wGradBF);
  this->addDependentField(muLandIce);

  if(useStereographicMap)
  {
    coordVec = decltype(coordVec)(p.get<std::string>("Coordinate Vector Name"),dl->qp_gradient);
    this->addDependentField(coordVec);
  }

  this->addEvaluatedField(Residual);

  this->setName("StokesFOResid"+PHX::typeAsString<EvalT>());

  int vecDimFO = (numDims < 2) ? numDims : 2;

#ifdef OUTPUT_TO_SCREEN
  *output << " in LandIce Stokes FO residual! " << std::endl;
  *output << " vecDimFO = " << vecDimFO << std::endl;
  *output << " numDims = " << numDims << std::endl;
  *output << " numQPs = " << numQPs << std::endl;
  *output << " numNodes = " << numNodes << std::endl;
#endif

  TEUCHOS_TEST_FOR_EXCEPTION (vecDimFO != 2 && eqn_type == LandIce, Teuchos::Exceptions::InvalidParameter,
                              std::endl << "Error in LandIce::StokesFOResid constructor:  " <<
                              "Invalid Parameter vecDim.  Problem implemented for at least 2 dofs per node (u and v). " << std::endl);

  TEUCHOS_TEST_FOR_EXCEPTION (vecDimFO != 1 && eqn_type == POISSON, Teuchos::Exceptions::InvalidParameter,
                              std::endl << "Error in LandIce::StokesFOResid constructor:  " <<
                              "Invalid Parameter vecDim.  Poisson problem implemented for 1 dof per node only. " << std::endl);

  TEUCHOS_TEST_FOR_EXCEPTION (vecDimFO != 1 && eqn_type == LandIce_XZ, Teuchos::Exceptions::InvalidParameter,
                              std::endl << "Error in LandIce::StokesFOResid constructor:  " <<
                              "Invalid Parameter vecDim.  LandIce XZ problem implemented for 1 dof per node only. " << std::endl);

  TEUCHOS_TEST_FOR_EXCEPTION (numDims != 2 && eqn_type == LandIce_XZ, Teuchos::Exceptions::InvalidParameter,
                              std::endl << "Error in LandIce::StokesFOResid constructor:  " <<
                              "Invalid Parameter numDims.  LandIce XZ problem is 2D. " << std::endl);
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
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(wGradBF,fm);
  this->utils.setFieldData(muLandIce,fm);
  if(useStereographicMap) {
    this->utils.setFieldData(coordVec, fm);
  }

  this->utils.setFieldData(Residual,fm);
}

//**********************************************************************
//Kokkos functors
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void StokesFOResid<EvalT, Traits>::
operator() (const LandIce_3D_Tag& tag, const int& cell) const{

  for (int node=0; node<numNodes; ++node){
    Residual(cell,node,0)=0.;
    Residual(cell,node,1)=0.;
  }

  if(useStereographicMap) {
    for (int qp=0; qp < numQPs; ++qp) {
      //evaluate non-linear viscosity, given by Glen's law, at quadrature points
      ScalarT mu = muLandIce(cell,qp);
      MeshScalarT x = coordVec(cell,qp,0)-x_0;
      MeshScalarT y = coordVec(cell,qp,1)-y_0;
      MeshScalarT h = 4.0*R2/(4.0*R2 + x*x + y*y);
      MeshScalarT h2 = h*h;
      MeshScalarT invh_x = x/2.0/R2;
      MeshScalarT invh_y = y/2.0/R2;

      ScalarT strs00 = 2*mu*(Ugrad(cell,qp,0,0)/h-invh_y*U(cell,qp,1)); //epsilon_xx
      ScalarT strs01 = mu*(Ugrad(cell,qp,0,1)/h+invh_x*U(cell,qp,0)+Ugrad(cell,qp,1,0)/h+invh_y*U(cell,qp,1)); //epsilon_xy
      ScalarT strs02 = mu*Ugrad(cell,qp,0,2); //epsilon_xz
      ScalarT strs11 = 2*mu*(Ugrad(cell,qp,1,1)/h-invh_x*U(cell,qp,0)); //epsilon_yy
      ScalarT strs12 = mu*Ugrad(cell,qp,1,2); //epsilon_yz

      for (int node=0; node < numNodes; ++node) {
        ScalarT epsb00 = wGradBF(cell,node,qp,0)/h; //epsilon_xx
        ScalarT epsb01 = (wGradBF(cell,node,qp,1)/h+invh_x*wBF(cell,node,qp))/2.0; //epsilon_xy
        ScalarT epsb02 = wGradBF(cell,node,qp,2)/2.0; //epsilon_xz
        ScalarT epsb11 = -invh_x*wBF(cell,node,qp); //epsilon_yy
        ScalarT epsb12 = 0;
        Residual(cell,node,0) +=  strs00*epsb00*h2 +
                                  strs11 * epsb11*h2 +
                                  2*strs01*epsb01*h2 +
                                  2*strs02*epsb02*h2 +
                                  2*strs12 * epsb12*h2 +
                                  (strs00+strs11)*(epsb00+epsb11)*h2;

        epsb00 = -invh_y*wBF(cell,node,qp); //epsilon_xx
        epsb01 = (wGradBF(cell,node,qp,0)/h+invh_y*wBF(cell,node,qp))/2.0; //epsilon_xy
        epsb02 = 0;
        epsb11 = wGradBF(cell,node,qp,1)/h; //epsilon_yy
        epsb12 = wGradBF(cell,node,qp,2)/2.0; //epsilon_yz

        Residual(cell,node,1) +=  strs00*epsb00*h2 +
                                  strs11 * epsb11*h2 +
                                  2*strs01*epsb01*h2 +
                                  2*strs02*epsb02*h2 +
                                  2*strs12 * epsb12*h2 +
                                  (strs00+strs11)*(epsb00+epsb11)*h2;
      }
    }
  } else {
    for (int qp=0; qp < numQPs; ++qp) {
      ScalarT mu = muLandIce(cell,qp);
      ScalarT strs00 = 2.0*mu*(2.0*Ugrad(cell,qp,0,0) + Ugrad(cell,qp,1,1));
      ScalarT strs11 = 2.0*mu*(2.0*Ugrad(cell,qp,1,1) + Ugrad(cell,qp,0,0));
      ScalarT strs01 = mu*(Ugrad(cell,qp,1,0)+ Ugrad(cell,qp,0,1));
      ScalarT strs02 = mu*Ugrad(cell,qp,0,2);
      ScalarT strs12 = mu*Ugrad(cell,qp,1,2);
      for (int node=0; node < numNodes; ++node) {
        Residual(cell,node,0) += strs00*wGradBF(cell,node,qp,0) +
                                 strs01*wGradBF(cell,node,qp,1) +
                                 strs02*wGradBF(cell,node,qp,2);
        Residual(cell,node,1) += strs01*wGradBF(cell,node,qp,0) +
                                 strs11*wGradBF(cell,node,qp,1) +
                                 strs12*wGradBF(cell,node,qp,2);
      }
    }
  }

  for (int qp=0; qp < numQPs; ++qp) {
    ScalarT frc0 = force(cell,qp,0);
    ScalarT frc1 = force(cell,qp,1);
    for (int node=0; node < numNodes; ++node) {
         Residual(cell,node,0) += frc0*wBF(cell,node,qp);
         Residual(cell,node,1) += frc1*wBF(cell,node,qp);
    }
  }
}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void StokesFOResid<EvalT, Traits>::
operator() (const POISSON_3D_Tag& tag, const int& cell) const{

  for (int node=0; node<numNodes; ++node){
    Residual(cell,node,0)=0.;
    Residual(cell,node,1)=0.;
  }

  for (int node=0; node < numNodes; ++node) {
    for (int qp=0; qp < numQPs; ++qp) {
       Residual(cell,node,0) += Ugrad(cell,qp,0,0)*wGradBF(cell,node,qp,0) +
                                Ugrad(cell,qp,0,1)*wGradBF(cell,node,qp,1) +
                                Ugrad(cell,qp,0,2)*wGradBF(cell,node,qp,2) +
                                force(cell,qp,0)*wBF(cell,node,qp);
    }
  }
}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void StokesFOResid<EvalT, Traits>::
operator() (const LandIce_2D_Tag& tag, const int& cell) const{

  for (int node=0; node<numNodes; ++node){
    Residual(cell,node,0)=0.;
    Residual(cell,node,1)=0.;
  }

  for (int node=0; node < numNodes; ++node) {
    for (int qp=0; qp < numQPs; ++qp) {
       Residual(cell,node,0) += 2.0*muLandIce(cell,qp)*((2.0*Ugrad(cell,qp,0,0) + Ugrad(cell,qp,1,1))*wGradBF(cell,node,qp,0) +
                                0.5*(Ugrad(cell,qp,0,1) + Ugrad(cell,qp,1,0))*wGradBF(cell,node,qp,1)) +
                                force(cell,qp,0)*wBF(cell,node,qp);
       Residual(cell,node,1) += 2.0*muLandIce(cell,qp)*(0.5*(Ugrad(cell,qp,0,1) + Ugrad(cell,qp,1,0))*wGradBF(cell,node,qp,0) +
                                (Ugrad(cell,qp,0,0) + 2.0*Ugrad(cell,qp,1,1))*wGradBF(cell,node,qp,1)) + force(cell,qp,1)*wBF(cell,node,qp);
    }
  }
}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void StokesFOResid<EvalT, Traits>::
operator() (const LandIce_XZ_2D_Tag& tag, const int& cell) const{

  for (int node=0; node<numNodes; ++node){
    Residual(cell,node,0)=0.;
    Residual(cell,node,1)=0.;
  }

  for (int node=0; node < numNodes; ++node) {
    for (int qp=0; qp < numQPs; ++qp) {
       //z dimension is treated as 2nd dimension
       //PDEs is: -d/dx(4*mu*du/dx) - d/dz(mu*du/dz) - f1 0
       Residual(cell,node,0) += 4.0*muLandIce(cell,qp)*Ugrad(cell,qp,0,0)*wGradBF(cell,node,qp,0)
                             + muLandIce(cell,qp)*Ugrad(cell,qp,0,1)*wGradBF(cell,node,qp,1)+force(cell,qp,0)*wBF(cell,node,qp);
    }
  }

}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void StokesFOResid<EvalT, Traits>::
operator() (const POISSON_2D_Tag& tag, const int& cell) const{

  for (int node=0; node<numNodes; ++node){
    Residual(cell,node,0)=0.;
    Residual(cell,node,1)=0.;
  }

  for (int node=0; node < numNodes; ++node) {
    for (int qp=0; qp < numQPs; ++qp) {
      Residual(cell,node,0) += Ugrad(cell,qp,0,0)*wGradBF(cell,node,qp,0) +
                               Ugrad(cell,qp,0,1)*wGradBF(cell,node,qp,1) +
                               force(cell,qp,0)*wBF(cell,node,qp);
    }
  }
}

//**********************************************************************
template<typename EvalT, typename Traits>
void StokesFOResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
#ifdef OUTPUT_TO_SCREEN
  Teuchos::RCP<Teuchos::FancyOStream> output(Teuchos::VerboseObjectBase::getDefaultOStream());

  int procRank = Teuchos::GlobalMPISession::getRank();
  int numProcs = Teuchos::GlobalMPISession::getNProc();
  output->setProcRankAndSize (procRank, numProcs);
  output->setOutputToRootOnly (0);
#endif
  if (numDims == 3) { //3D case
    if (eqn_type == LandIce) {
     Kokkos::parallel_for(LandIce_3D_Policy(0,workset.numCells), *this);
    }
    else if (eqn_type == POISSON) {
      Kokkos::parallel_for(POISSON_3D_Policy(0,workset.numCells), *this);
    }
  }
  else { //2D case
   if (eqn_type == LandIce) {
     Kokkos::parallel_for(LandIce_2D_Policy(0,workset.numCells), *this);
   }
   if (eqn_type == LandIce_XZ) {
     Kokkos::parallel_for(LandIce_XZ_2D_Policy(0,workset.numCells), *this);
   }
   else if (eqn_type == POISSON) {
    Kokkos::parallel_for(POISSON_2D_Policy(0,workset.numCells), *this);
   }
  }
}

//**********************************************************************
} // namespace LandIce
