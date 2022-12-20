//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN

namespace LandIce {

//**********************************************************************
template<typename EvalT, typename Traits, typename SurfHeightST>
StokesFOStress<EvalT, Traits, SurfHeightST>::
StokesFOStress(const Teuchos::ParameterList& p,
               const Teuchos::RCP<Albany::Layouts>& dl) :
  surfaceHeight (p.get<std::string> ("Surface Height QP Name"), dl->qp_scalar),
  Ugrad         (p.get<std::string> ("Velocity Gradient QP Variable Name"), dl->qp_vecgradient),
  muLandIce     (p.get<std::string> ("Viscosity QP Variable Name"), dl->qp_scalar),
  coordVec      (p.get<std::string>("Coordinate Vector Name"),dl->qp_gradient),
  Stress        (p.get<std::string> ("Stress Variable Name"), dl->qp_tensor)
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
  this->addDependentField(muLandIce);
  this->addDependentField(coordVec);

  stereographicMapList = p.get<Teuchos::ParameterList*>("Stereographic Map");
  useStereographicMap = stereographicMapList->get("Use Stereographic Map", false);

  if(useStereographicMap)
  {
    U = decltype(U)(p.get<std::string>("Velocity QP Variable Name"), dl->qp_vector);
    this->addDependentField(U);
    coordVec = decltype(coordVec)(p.get<std::string>("Coordinate Vector Name"),dl->qp_gradient);
    this->addDependentField(coordVec);
  }

  this->addEvaluatedField(Stress);

  this->setName("StokesFOStress"+PHX::print<EvalT>());

  std::vector<PHX::Device::size_type> dims;
  Ugrad.fieldTag().dataLayout().dimensions(dims);
  numQPs   = dims[1];
  vecDimFO  = dims[2];
  numDims  = dims[3];

#ifdef OUTPUT_TO_SCREEN
  *out << " in LandIce Stokes FO Stress! " << std::endl;
  *out << " vecDimFO = " << vecDimFO << std::endl;
  *out << " numDims = " << numDims << std::endl;
  *out << " numQPs = " << numQPs << std::endl;
  *out << " numNodes = " << numNodes << std::endl;
#endif

  Teuchos::ParameterList* p_list = p.get<Teuchos::ParameterList*>("Physical Parameter List");
  rho_g = p_list->get<double>("Ice Density", 910.0) * p_list->get<double>("Gravity Acceleration", 9.8);
}

//**********************************************************************

template<typename EvalT, typename Traits, typename SurfHeightST>
void StokesFOStress<EvalT, Traits, SurfHeightST>::
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
      "\nError in LandIce::StokesFOStress constructor: Invalid Parameter vecDim.\n"
      "  Problem implemented for 2 dofs per node (u and v).\n");

  TEUCHOS_TEST_FOR_EXCEPTION (numDims != 3, Teuchos::Exceptions::InvalidParameter,
      "\nError in LandIce::StokesFOStress constructor: Invalid Parameter vecDim.\n"
      "  LandIce::StokesFOStress is for 3D.\n");

  for (std::size_t cell=0; cell < workset.numCells; ++cell) {

    if(useStereographicMap) {
      double R = stereographicMapList->get<double>("Earth Radius", 6371);
      double x_0 = stereographicMapList->get<double>("X_0", 0);//-136);
      double y_0 = stereographicMapList->get<double>("Y_0", 0);//-2040);
      double R2 = std::pow(R,2);
      for (std::size_t qp=0; qp < numQPs; ++qp) {
        //evaluate non-linear viscosity, given by Glen's law, at quadrature points
        ScalarT mu = muLandIce(cell,qp);
        ScalarT p = rho_g*(surfaceHeight(cell,qp) - coordVec(cell,qp,2));
        MeshScalarT x = coordVec(cell,qp,0)-x_0;
        MeshScalarT y = coordVec(cell,qp,1)-y_0;
        MeshScalarT h = 4.0*R2/(4.0*R2 + x*x + y*y);
        //MeshScalarT h2 = h*h;
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
        ScalarT mu = muLandIce(cell,qp);
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

} // namespace LandIce
