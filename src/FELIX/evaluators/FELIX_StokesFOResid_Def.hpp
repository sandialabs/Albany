//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN



namespace FELIX {

//**********************************************************************
template<typename EvalT, typename Traits>
StokesFOResid<EvalT, Traits>::
StokesFOResid(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl) :
  wBF      (p.get<std::string> ("Weighted BF Variable Name"), dl->node_qp_scalar),
  wGradBF  (p.get<std::string> ("Weighted Gradient BF Variable Name"),dl->node_qp_gradient),
  U        (p.get<std::string> ("Velocity QP Variable Name"), dl->qp_vector),
  Ugrad    (p.get<std::string> ("Velocity Gradient QP Variable Name"), dl->qp_vecgradient),
  force    (p.get<std::string> ("Body Force Variable Name"), dl->qp_vector),
  muFELIX  (p.get<std::string> ("Viscosity QP Variable Name"), dl->qp_scalar),
  Residual (p.get<std::string> ("Residual Variable Name"), dl->node_vector)
{
  Teuchos::ParameterList* list = p.get<Teuchos::ParameterList*>("Parameter List");

  std::string type = list->get("Type", "FELIX");

  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());
  if (type == "FELIX") {
#ifdef OUTPUT_TO_SCREEN
    *out << "setting FELIX FO model physics" << std::endl;
#endif
    eqn_type = FELIX;
  }
  //FELIX FO x-z MMS test case
  else if (type == "FELIX X-Z") {
#ifdef OUTPUT_TO_SCREEN
    *out << "setting FELIX FO X-Z model physics" << std::endl;
#endif
  eqn_type = FELIX_XZ;
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
  this->addDependentField(wBF);
  this->addDependentField(wGradBF);
  this->addDependentField(muFELIX);

  needsBasalResidual = p.get<bool>("Needs Basal Residual");
  if (needsBasalResidual)
  {
    basalRes  = PHX::MDField<ScalarT,Cell,Node,VecDim>(p.get<std::string> ("Basal Residual Variable Name"), dl->node_vector);
    this->addDependentField(basalRes);
  }

  stereographicMapList = p.get<Teuchos::ParameterList*>("Stereographic Map");
  useStereographicMap = stereographicMapList->get("Use Stereographic Map", false);
  if(useStereographicMap)
  {
    coordVec = PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>(p.get<std::string>("Coordinate Vector Name"),dl->qp_gradient);
    this->addDependentField(coordVec);
  }

  this->addEvaluatedField(Residual);

  this->setName("StokesFOResid"+PHX::typeAsString<EvalT>());

  std::vector<PHX::Device::size_type> dims;
  wGradBF.fieldTag().dataLayout().dimensions(dims);
  numNodes = dims[1];
  numQPs   = dims[2];
  numDims  = dims[3];

  U.fieldTag().dataLayout().dimensions(dims);
  vecDimFO = (numDims < 2) ? numDims : 2;

#ifdef OUTPUT_TO_SCREEN
  *out << " in FELIX Stokes FO residual! " << std::endl;
  *out << " vecDimFO = " << vecDimFO << std::endl;
  *out << " numDims = " << numDims << std::endl;
  *out << " numQPs = " << numQPs << std::endl;
  *out << " numNodes = " << numNodes << std::endl;
#endif

  TEUCHOS_TEST_FOR_EXCEPTION (vecDimFO != 2 && eqn_type == FELIX, Teuchos::Exceptions::InvalidParameter,
                              std::endl << "Error in FELIX::StokesFOResid constructor:  " <<
                              "Invalid Parameter vecDim.  Problem implemented for at least 2 dofs per node (u and v). " << std::endl);

  TEUCHOS_TEST_FOR_EXCEPTION (vecDimFO != 1 && eqn_type == POISSON, Teuchos::Exceptions::InvalidParameter,
                              std::endl << "Error in FELIX::StokesFOResid constructor:  " <<
                              "Invalid Parameter vecDim.  Poisson problem implemented for 1 dof per node only. " << std::endl);

  TEUCHOS_TEST_FOR_EXCEPTION (vecDimFO != 1 && eqn_type == FELIX_XZ, Teuchos::Exceptions::InvalidParameter,
                              std::endl << "Error in FELIX::StokesFOResid constructor:  " <<
                              "Invalid Parameter vecDim.  FELIX XZ problem implemented for 1 dof per node only. " << std::endl);

  TEUCHOS_TEST_FOR_EXCEPTION (numDims != 2 && eqn_type == FELIX_XZ, Teuchos::Exceptions::InvalidParameter,
                              std::endl << "Error in FELIX::StokesFOResid constructor:  " <<
                              "Invalid Parameter numDims.  FELIX XZ problem is 2D. " << std::endl);
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
  this->utils.setFieldData(muFELIX,fm);
  if (needsBasalResidual)
    this->utils.setFieldData(basalRes,fm);
  if(useStereographicMap)
    this->utils.setFieldData(coordVec, fm);

  this->utils.setFieldData(Residual,fm);
}
//**********************************************************************
//Kokkos functors
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void StokesFOResid<EvalT, Traits>::
operator() (const FELIX_3D_Tag& tag, const int& cell) const{

  if (needsBasalResidual)
  {
    for (int node=0; node<numNodes; ++node){
      Residual(cell,node,0)=basalRes(cell,node,0);
      Residual(cell,node,1)=basalRes(cell,node,1);
    }
  }
  else
  {
    for (int node=0; node<numNodes; ++node){
      Residual(cell,node,0)=0.;
      Residual(cell,node,1)=0.;
    }
  }

  if(useStereographicMap) {
          double R = stereographicMapList->get<double>("Earth Radius", 6371);
          double x_0 = stereographicMapList->get<double>("X_0", 0);//-136);
          double y_0 = stereographicMapList->get<double>("Y_0", 0);//-2040);
          double R2 = std::pow(R,2);
          for (int qp=0; qp < numQPs; ++qp) {
            //evaluate non-linear viscosity, given by Glen's law, at quadrature points
            ScalarT mu = muFELIX(cell,qp);
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
        }
        else {
           for (int qp=0; qp < numQPs; ++qp) {
            ScalarT mu = muFELIX(cell,qp);
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

  if (needsBasalResidual)
  {
    for (int node=0; node<numNodes; ++node){
      Residual(cell,node,0)=basalRes(cell,node,0);
      Residual(cell,node,1)=basalRes(cell,node,1);
    }
  }
  else
  {
    for (int node=0; node<numNodes; ++node){
      Residual(cell,node,0)=0.;
      Residual(cell,node,1)=0.;
    }
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
operator() (const FELIX_2D_Tag& tag, const int& cell) const{

  if (needsBasalResidual)
  {
    for (int node=0; node<numNodes; ++node){
      Residual(cell,node,0)=basalRes(cell,node,0);
      Residual(cell,node,1)=basalRes(cell,node,1);
    }
  }
  else
  {
    for (int node=0; node<numNodes; ++node){
      Residual(cell,node,0)=0.;
      Residual(cell,node,1)=0.;
    }
  }

  for (int node=0; node < numNodes; ++node) {
          for (int qp=0; qp < numQPs; ++qp) {
             Residual(cell,node,0) += 2.0*muFELIX(cell,qp)*((2.0*Ugrad(cell,qp,0,0) + Ugrad(cell,qp,1,1))*wGradBF(cell,node,qp,0) +
                                      0.5*(Ugrad(cell,qp,0,1) + Ugrad(cell,qp,1,0))*wGradBF(cell,node,qp,1)) +
                                      force(cell,qp,0)*wBF(cell,node,qp);
             Residual(cell,node,1) += 2.0*muFELIX(cell,qp)*(0.5*(Ugrad(cell,qp,0,1) + Ugrad(cell,qp,1,0))*wGradBF(cell,node,qp,0) +
                                      (Ugrad(cell,qp,0,0) + 2.0*Ugrad(cell,qp,1,1))*wGradBF(cell,node,qp,1)) + force(cell,qp,1)*wBF(cell,node,qp);
              }
   }
}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void StokesFOResid<EvalT, Traits>::
operator() (const FELIX_XZ_2D_Tag& tag, const int& cell) const{

  if (needsBasalResidual)
  {
    for (int node=0; node<numNodes; ++node){
      Residual(cell,node,0)=basalRes(cell,node,0);
      Residual(cell,node,1)=basalRes(cell,node,1);
    }
  }
  else
  {
    for (int node=0; node<numNodes; ++node){
      Residual(cell,node,0)=0.;
      Residual(cell,node,1)=0.;
    }
  }

  for (int node=0; node < numNodes; ++node) {
          for (int qp=0; qp < numQPs; ++qp) {
             //z dimension is treated as 2nd dimension
             //PDEs is: -d/dx(4*mu*du/dx) - d/dz(mu*du/dz) - f1 0
             Residual(cell,node,0) += 4.0*muFELIX(cell,qp)*Ugrad(cell,qp,0,0)*wGradBF(cell,node,qp,0)
                                   + muFELIX(cell,qp)*Ugrad(cell,qp,0,1)*wGradBF(cell,node,qp,1)+force(cell,qp,0)*wBF(cell,node,qp);
          }
       }

}

template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void StokesFOResid<EvalT, Traits>::
operator() (const POISSON_2D_Tag& tag, const int& cell) const{

  if (needsBasalResidual)
  {
    for (int node=0; node<numNodes; ++node){
      Residual(cell,node,0)=basalRes(cell,node,0);
      Residual(cell,node,1)=basalRes(cell,node,1);
    }
  }
  else
  {
    for (int node=0; node<numNodes; ++node){
      Residual(cell,node,0)=0.;
      Residual(cell,node,1)=0.;
    }
  }

  for (int node=0; node < numNodes; ++node) {
          for (int qp=0; qp < numQPs; ++qp) {
             Residual(cell,node,0) += Ugrad(cell,qp,0,0)*wGradBF(cell,node,qp,0) +
                                      Ugrad(cell,qp,0,1)*wGradBF(cell,node,qp,1) +
                                      force(cell,qp,0)*wBF(cell,node,qp);
              }
   }
}


template <typename ScalarType, class DeviceType, class MDFieldType1, class MDFieldType2, class MDFieldType3, class MDFieldType4, class MDFieldType5, class MDFieldType6, class MDFieldType7 >
class StokesFOResid_3D_FELIX  {
 MDFieldType1 Residual_;
 MDFieldType2 wGradBF_;
 MDFieldType3 force_;
 MDFieldType4 Ugrad_;
 MDFieldType5 mu_;
 MDFieldType6 wBF_;
 MDFieldType7 basalRes_;
 const int numNodes_;
 const int numQPs_;
 bool needsBasalResidual_;

 public:
 typedef DeviceType device_type;

 StokesFOResid_3D_FELIX (MDFieldType1 &Residual,
                         MDFieldType2 &wGradBF,
                         MDFieldType3 &force,
                         MDFieldType4 &Ugrad,
                         MDFieldType5 &mu,
                         MDFieldType6 &wBF,
                         MDFieldType7 &basalRes,
                         int numNodes,
                         int numQPs,
                         bool needsBasalResidual)
  : Residual_(Residual)
  , wGradBF_(wGradBF)
  , force_(force)
  , Ugrad_(Ugrad)
  , mu_(mu)
  , wBF_(wBF)
  , basalRes_(basalRes)
  , numNodes_(numNodes)
  , numQPs_(numQPs)
  , needsBasalResidual_(needsBasalResidual){}

 KOKKOS_INLINE_FUNCTION
 void operator () (const int i) const
 {
  if (needsBasalResidual_)
  {
    for (int node=0; node<numNodes_; ++node){
      Residual_(i,node,0)=basalRes_(i,node,0);
      Residual_(i,node,1)=basalRes_(i,node,1);
    }
  }
  else
  {
    for (int node=0; node<numNodes_; ++node){
      Residual_(i,node,0)=0.;
      Residual_(i,node,1)=0.;
    }
  }

  for (int j=0; j<numQPs_; j++)
  {
   const ScalarType strs00 = 2.0*mu_(i,j)*(2.0*Ugrad_(i,j,0,0)+Ugrad_(i,j,1,1));
   const ScalarType strs11 = 2.0*mu_(i,j)*(2.0*Ugrad_(i,j,1,1)+Ugrad_(i,j,0,0));
   const ScalarType strs01 = mu_(i,j)*(Ugrad_(i,j,1,0)+Ugrad_(i,j,0,1));
   const ScalarType strs02 = mu_(i,j)*Ugrad_(i,j,0,2);
   const ScalarType strs12 = mu_(i,j)*Ugrad_(i,j,1,2);
   for (int node=0; node<numNodes_; ++node){
     Residual_(i,node,0) +=strs00*wGradBF_(i,node,j,0) +
                        strs01*wGradBF_(i,node,j,1) +
                        strs02*wGradBF_(i,node,j,2);
     Residual_(i,node,1) +=strs01*wGradBF_(i,node,j,0) +
                       strs11*wGradBF_(i,node,j,1) +
                        strs12*wGradBF_(i,node,j,2);
   }

  }

  for (int qp=0; qp < numQPs_; ++qp) {
        const ScalarType frc0 = force_(i,qp,0);
        const ScalarType frc1 = force_(i,qp,1);
        for (int node=0; node < numNodes_; ++node) {
             Residual_(i,node,0) += frc0*wBF_(i,node,qp);
             Residual_(i,node,1) += frc1*wBF_(i,node,qp);
        }
      }
 }
};

#endif
//**********************************************************************
template<typename EvalT, typename Traits>
void StokesFOResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  typedef Intrepid2::FunctionSpaceTools FST;

#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT

  // Initialize residual to 0.0

//  Kokkos::deep_copy(Residual.get_kokkos_view(), ScalarT(0.0));

  if (needsBasalResidual)
  {
    for (int cell=0; cell<workset.numCells; ++cell)
    {
      for (int node=0; node<numNodes; ++node)
      {
        Residual(cell,node,0)=basalRes(cell,node,0);
        Residual(cell,node,1)=basalRes(cell,node,1);
      }
    }
  }
  else
  {
    for (int cell=0; cell<workset.numCells; ++cell)
    {
      for (int node=0; node<numNodes; ++node)
      {
        Residual(cell,node,0)=0.;
        Residual(cell,node,1)=0.;
      }
    }
  }


//  Residual.deep_copy(ScalarT(0.0));
  if (numDims == 3) { //3D case
    if (eqn_type == FELIX) {
      for (std::size_t cell=0; cell < workset.numCells; ++cell) {

        if(useStereographicMap) {
          double R = stereographicMapList->get<double>("Earth Radius", 6371);
          double x_0 = stereographicMapList->get<double>("X_0", 0);//-136);
          double y_0 = stereographicMapList->get<double>("Y_0", 0);//-2040);
          double R2 = std::pow(R,2);
          for (std::size_t qp=0; qp < numQPs; ++qp) {
            //evaluate non-linear viscosity, given by Glen's law, at quadrature points
            ScalarT mu = muFELIX(cell,qp);
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

            for (std::size_t node=0; node < numNodes; ++node) {
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
        }
        else {
          for (std::size_t qp=0; qp < numQPs; ++qp) {
            ScalarT mu = muFELIX(cell,qp);
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
        }

        for (std::size_t qp=0; qp < numQPs; ++qp) {
          ScalarT frc0 = force(cell,qp,0);
          ScalarT frc1 = force(cell,qp,1);
          for (std::size_t node=0; node < numNodes; ++node) {
               Residual(cell,node,0) += frc0*wBF(cell,node,qp);
               Residual(cell,node,1) += frc1*wBF(cell,node,qp);
          }
        }
      }
    }
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
    else if (eqn_type == FELIX_XZ) {
    for (std::size_t cell=0; cell < workset.numCells; ++cell) {
      for (std::size_t node=0; node < numNodes; ++node) {
          for (std::size_t qp=0; qp < numQPs; ++qp) {
             //z dimension is treated as 2nd dimension
             //PDEs is: -d/dx(4*mu*du/dx) - d/dz(mu*du/dz) - f1 0
             Residual(cell,node,0) += 4.0*muFELIX(cell,qp)*Ugrad(cell,qp,0,0)*wGradBF(cell,node,qp,0)
                                   + muFELIX(cell,qp)*Ugrad(cell,qp,0,1)*wGradBF(cell,node,qp,1)+force(cell,qp,0)*wBF(cell,node,qp);
          }
       }
     }
    }
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
#else
  if (numDims == 3) { //3D case
    if (eqn_type == FELIX) {
     Kokkos::parallel_for(FELIX_3D_Policy(0,workset.numCells), *this);
   //  Kokkos::parallel_for ( workset.numCells, StokesFOResid_3D_FELIX < ScalarT, PHX::Device, PHX::MDField<ScalarT,Cell,Node,VecDim>, PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim>, PHX::MDField<ScalarT,Cell,QuadPoint,VecDim>, PHX::MDField<ScalarT,Cell,QuadPoint,VecDim,Dim>, PHX::MDField<ScalarT,Cell,QuadPoint>, PHX::MDField<MeshScalarT,Cell,Node,QuadPoint> > (Residual, wGradBF, force, Ugrad, muFELIX, wBF, numNodes, numQPs));
    }
    else if (eqn_type == POISSON) {
      Kokkos::parallel_for(POISSON_3D_Policy(0,workset.numCells), *this);
    }
  }
  else { //2D case
   if (eqn_type == FELIX) {
     Kokkos::parallel_for(FELIX_2D_Policy(0,workset.numCells), *this);
   }
   if (eqn_type == FELIX_XZ) {
     Kokkos::parallel_for(FELIX_XZ_2D_Policy(0,workset.numCells), *this);
   }
   else if (eqn_type == POISSON) {
    Kokkos::parallel_for(POISSON_2D_Policy(0,workset.numCells), *this);
   }
  }
#endif



}

//**********************************************************************
}

