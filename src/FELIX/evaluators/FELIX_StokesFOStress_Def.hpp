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
StokesFOStress<EvalT, Traits>::
StokesFOStress(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl) :
  Ugrad    (p.get<std::string> ("Velocity Gradient QP Variable Name"), dl->qp_vecgradient),
  surfaceHeight    (p.get<std::string> ("Surface Height QP Name"), dl->qp_scalar),
  muFELIX  (p.get<std::string> ("Viscosity QP Variable Name"), dl->qp_scalar),
  coordVec (p.get<std::string>("Coordinate Vector Name"),dl->qp_gradient),
  Stress (p.get<std::string> ("Stress Variable Name"), dl->qp_tensor)
{
#ifdef OUTPUT_TO_SCREEN
  Teuchos::RCP<Teuchos::FancyOStream> output(Teuchos::VerboseObjectBase::getDefaultOStream());

  int procRank = Teuchos::GlobalMPISession::getRank();
  int numProcs = Teuchos::GlobalMPISession::getNProc();
  output->setProcRankAndSize (procRank, numProcs);
  output->setOutputToRootOnly (0);
#endif

  this->addDependentField(Ugrad);
  this->addDependentField(surfaceHeight);
  this->addDependentField(muFELIX);
  this->addDependentField(coordVec);

  if(useStereographicMap)
  {
    U = decltype(U)(p.get<std::string>("Velocity QP Variable Name"), dl->qp_vector);
    this->addDependentField(U);
  }

  stereographicMapList = p.get<Teuchos::ParameterList*>("Stereographic Map");
  useStereographicMap = stereographicMapList->get("Use Stereographic Map", false);
  if(useStereographicMap)
  {
    coordVec = decltype(coordVec)(p.get<std::string>("Coordinate Vector Name"),dl->qp_gradient);
    this->addDependentField(coordVec);
  }

  this->addEvaluatedField(Stress);

  this->setName("StokesFOStress"+PHX::typeAsString<EvalT>());

  std::vector<PHX::Device::size_type> dims;
  Ugrad.fieldTag().dataLayout().dimensions(dims);
  numQPs   = dims[1];
  vecDimFO  = dims[2];
  numDims  = dims[3];

#ifdef OUTPUT_TO_SCREEN
  *out << " in FELIX Stokes FO Stress! " << std::endl;
  *out << " vecDimFO = " << vecDimFO << std::endl;
  *out << " numDims = " << numDims << std::endl;
  *out << " numQPs = " << numQPs << std::endl;
  *out << " numNodes = " << numNodes << std::endl;
#endif

  Teuchos::ParameterList* p_list =
    p.get<Teuchos::ParameterList*>("Physical Parameter List");
  rho_g = p_list->get<double>("Ice Density", 910.0)*p_list->get<double>("Gravity Acceleration", 9.8);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void StokesFOStress<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(surfaceHeight,fm);
  this->utils.setFieldData(Ugrad,fm);
  this->utils.setFieldData(muFELIX,fm);
  this->utils.setFieldData(coordVec, fm);
  this->utils.setFieldData(Stress,fm);
  if(useStereographicMap)
    this->utils.setFieldData(U,fm);
}
//**********************************************************************

template<typename EvalT, typename Traits>
void StokesFOStress<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
#ifdef OUTPUT_TO_SCREEN
  Teuchos::RCP<Teuchos::FancyOStream> out(Teuchos::VerboseObjectBase::getDefaultOStream());

  int procRank = Teuchos::GlobalMPISession::getRank();
  int numProcs = Teuchos::GlobalMPISession::getNProc();
  output->setProcRankAndSize (procRank, numProcs);
  output->setOutputToRootOnly (0);
#endif

  TEUCHOS_TEST_FOR_EXCEPTION (vecDimFO != 2, Teuchos::Exceptions::InvalidParameter,
                              std::endl << "Error in FELIX::StokesFOStress constructor:  " <<
                              "Invalid Parameter vecDim.  Problem implemented for 2 dofs per node (u and v). " << std::endl);

  TEUCHOS_TEST_FOR_EXCEPTION (numDims != 3, Teuchos::Exceptions::InvalidParameter,
                              std::endl << "Error in FELIX::StokesFOStress constructor:  " <<
                              "Invalid Parameter numDims.  FELIX::StokesFOStress is for 3D " << std::endl);

  // Initialize residual to 0.0

//  Kokkos::deep_copy(Residual.get_view(), ScalarT(0.0));



  for (std::size_t cell=0; cell < workset.numCells; ++cell) {

    if(useStereographicMap) {
      double R = stereographicMapList->get<double>("Earth Radius", 6371);
      double x_0 = stereographicMapList->get<double>("X_0", 0);//-136);
      double y_0 = stereographicMapList->get<double>("Y_0", 0);//-2040);
      double R2 = std::pow(R,2);
      for (std::size_t qp=0; qp < numQPs; ++qp) {
        //evaluate non-linear viscosity, given by Glen's law, at quadrature points
        ScalarT mu = muFELIX(cell,qp);
        ScalarT p = rho_g*(surfaceHeight(cell,qp) - coordVec(cell,qp,2));
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

        Stress(cell, qp, 0, 0) = strs00;
        Stress(cell, qp, 0, 1) = strs01;
        Stress(cell, qp, 0, 2) = strs02;
        Stress(cell, qp, 1, 0) = strs01;
        Stress(cell, qp, 1, 1) = strs11;
        Stress(cell, qp, 1, 2) = strs12;
        Stress(cell, qp, 2, 0) = strs02;
        Stress(cell, qp, 2, 1) = strs12;
        Stress(cell, qp, 2, 2) = -p;
      }
    }
    else {
      for (std::size_t qp=0; qp < numQPs; ++qp) {
        ScalarT mu = muFELIX(cell,qp);
        ScalarT p = rho_g*(surfaceHeight(cell,qp) - coordVec(cell,qp,2));
        ScalarT strs00 = 2.0*mu*(2.0*Ugrad(cell,qp,0,0) + Ugrad(cell,qp,1,1)) -p;
        ScalarT strs11 = 2.0*mu*(2.0*Ugrad(cell,qp,1,1) + Ugrad(cell,qp,0,0)) -p;
        ScalarT strs01 = mu*(Ugrad(cell,qp,1,0)+ Ugrad(cell,qp,0,1));
        ScalarT strs02 = mu*Ugrad(cell,qp,0,2);
        ScalarT strs12 = mu*Ugrad(cell,qp,1,2);

        Stress(cell, qp, 0, 0) = strs00;
        Stress(cell, qp, 0, 1) = strs01;
        Stress(cell, qp, 0, 2) = strs02;
        Stress(cell, qp, 1, 0) = strs01;
        Stress(cell, qp, 1, 1) = strs11;
        Stress(cell, qp, 1, 2) = strs12;
        Stress(cell, qp, 2, 0) = strs02;
        Stress(cell, qp, 2, 1) = strs12;
        Stress(cell, qp, 2, 2) = -p;
      }
    }

  }





}

//**********************************************************************
}

