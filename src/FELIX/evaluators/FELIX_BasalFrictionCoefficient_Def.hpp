//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Albany_Layouts.hpp"

//uncomment the following line if you want debug output to be printed to screen
#define OUTPUT_TO_SCREEN

namespace FELIX
{

//**********************************************************************
// PARTIAL SPECIALIZATION: Stokes Present ******************************
//**********************************************************************
template<typename EvalT, typename Traits, bool IsHydrology>
BasalFrictionCoefficient<EvalT, Traits, IsHydrology, true>::
BasalFrictionCoefficient (const Teuchos::ParameterList& p,
                          const Teuchos::RCP<Albany::Layouts>& dl) :
  beta        (p.get<std::string> ("Basal Friction Coefficient Variable Name"), dl->qp_scalar)
{
#ifdef OUTPUT_TO_SCREEN
  Teuchos::RCP<Teuchos::FancyOStream> output(Teuchos::VerboseObjectBase::getDefaultOStream());

  int procRank = Teuchos::GlobalMPISession::getRank();
  int numProcs = Teuchos::GlobalMPISession::getNProc();
  output->setProcRankAndSize (procRank, numProcs);
  output->setOutputToRootOnly (0);
#endif

  Teuchos::ParameterList& beta_list = *p.get<Teuchos::ParameterList*>("Parameter List");

  std::string betaType = (beta_list.isParameter("Type") ? beta_list.get<std::string>("Type") : "Given Field");

  TEUCHOS_TEST_FOR_EXCEPTION (!dl->isSideLayouts, Teuchos::Exceptions::InvalidParameter,
                              "Error! The layout structure does not appear to be that of a side set.\n");

  basalSideName = p.get<std::string>("Side Set Name");
  numQPs        = dl->qp_scalar->dimension(2);
  numNodes      = dl->node_scalar->dimension(2);

  this->addEvaluatedField(beta);

  if (betaType == "Given Constant")
  {
#ifdef OUTPUT_TO_SCREEN
    *output << "Given constant and uniform beta, value loaded from xml input file.\n";
#endif
    beta_type = GIVEN_CONSTANT;
    beta_given_val = beta_list.get<double>("Constant Given Beta Value");
  }
  else if ((betaType == "Given Field")|| (betaType == "Exponent Of Given Field") || (betaType == "Galerkin Projection Of Exponent Of Given Field"))
  {
#ifdef OUTPUT_TO_SCREEN
    *output << "Given constant beta field, loaded from mesh or file.\n";
#endif
    if (betaType == "Given Field")
      beta_type = GIVEN_FIELD;
    else if (betaType == "Galerkin Projection Of Exponent Of Given Field")
      beta_type = GAL_PROJ_EXP_GIVEN_FIELD;
    else
      beta_type = EXP_GIVEN_FIELD;

    if(beta_type == GAL_PROJ_EXP_GIVEN_FIELD) {
      beta_given_field = PHX::MDField<ParamScalarT>(p.get<std::string> ("Basal Friction Coefficient Variable Name") + " Given", dl->node_scalar);
      this->addDependentField (beta_given_field);
      bF = PHX::MDField<RealType,Cell,Node,Side,QuadPoint>(p.get<std::string> ("BF Variable Name"), dl->node_qp_scalar);
      this->addDependentField (bF);
    }
    else {
      beta_given_field = PHX::MDField<ParamScalarT>(p.get<std::string> ("Basal Friction Coefficient Variable Name") + " Given", dl->qp_scalar);
      this->addDependentField (beta_given_field);
    }
  }
  else if (betaType == "Power Law")
  {
    beta_type = POWER_LAW;


#ifdef OUTPUT_TO_SCREEN
    *output << "Velocity-dependent beta (power law):\n\n"
            << "      beta = mu * N * |u|^p \n\n"
            << "  with N being the effective pressure, |u| the sliding velocity\n";
#endif

    N              = PHX::MDField<HydroScalarT,Cell,Side,QuadPoint>(p.get<std::string> ("Effective Pressure QP Variable Name"), dl->qp_scalar);
    u_norm         = PHX::MDField<ScalarT,Cell,Side,QuadPoint>(p.get<std::string> ("Sliding Velocity QP Variable Name"), dl->qp_scalar);
    muParam        = PHX::MDField<ScalarT,Dim>("Coulomb Friction Coefficient", dl->shared_param);
    lambdaParam    = PHX::MDField<ScalarT,Dim>("Bed Roughness", dl->shared_param);
    powerParam     = PHX::MDField<ScalarT,Dim>("Power Exponent", dl->shared_param);

    this->addDependentField (muParam);
    this->addDependentField (lambdaParam);
    this->addDependentField (powerParam);
    this->addDependentField (u_norm);
    this->addDependentField (N);
  }
  else if (betaType == "Regularized Coulomb")
  {
    beta_type = REGULARIZED_COULOMB;

    printedMu      = -9999.999;
    printedLambda  = -9999.999;
    printedQ       = -9999.999;
    if (beta_list.isParameter("Constant Flow Factor A"))
    {
      A = beta_list.get<double>("Constant Flow Factor A");

      // A*N^{1/q} is dimensionally correct only for q=1/3. To fix this, we modify A
      // so that the formula becomes (A_mod*N)^{1/q}. This means that A_mod = A^{1/3}
      A = std::cbrt(A);
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! The case with variable flow factor has not been implemented yet.\n");
    }
#ifdef OUTPUT_TO_SCREEN
    *output << "Velocity-dependent beta (regularized coulomb law):\n\n"
            << "      beta = mu * N * |u|^{p-1} / [|u| + lambda*A*N^(1/p)]^p\n\n"
            << "  with N being the effective pressure, |u| the sliding velocity\n";
#endif

    N              = PHX::MDField<HydroScalarT,Cell,Side,QuadPoint>(p.get<std::string> ("Effective Pressure QP Variable Name"), dl->qp_scalar);
    u_norm         = PHX::MDField<ScalarT,Cell,Side,QuadPoint>(p.get<std::string> ("Sliding Velocity QP Variable Name"), dl->qp_scalar);
    muParam        = PHX::MDField<ScalarT,Dim>("Coulomb Friction Coefficient", dl->shared_param);
    lambdaParam    = PHX::MDField<ScalarT,Dim>("Bed Roughness", dl->shared_param);
    powerParam     = PHX::MDField<ScalarT,Dim>("Power Exponent", dl->shared_param);

    this->addDependentField (muParam);
    this->addDependentField (lambdaParam);
    this->addDependentField (powerParam);
    this->addDependentField (N);
    this->addDependentField (u_norm);
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
        std::endl << "Error in FELIX::BasalFrictionCoefficient:  \"" << betaType << "\" is not a valid parameter for Beta Type\n");
  }

  auto& stereographicMapList = p.get<Teuchos::ParameterList*>("Stereographic Map");
  use_stereographic_map = stereographicMapList->get("Use Stereographic Map", false);
  if(use_stereographic_map)
  {
    coordVec = PHX::MDField<MeshScalarT,Cell,Side,QuadPoint,Dim>(p.get<std::string>("Coordinate Vector Variable Name"), dl->qp_coords);

    double R = stereographicMapList->get<double>("Earth Radius", 6371);
    x_0 = stereographicMapList->get<double>("X_0", 0);//-136);
    y_0 = stereographicMapList->get<double>("Y_0", 0);//-2040);
    R2 = std::pow(R,2);

    this->addDependentField(coordVec);
  }

  this->setName("BasalFrictionCoefficient"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits, bool IsHydrology>
void BasalFrictionCoefficient<EvalT, Traits, IsHydrology, true>::
postRegistrationSetup (typename Traits::SetupData d,
                       PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(beta,fm);

  switch (beta_type)
  {
    case GIVEN_CONSTANT:
      beta.deep_copy(ScalarT(beta_given_val));
      break;
    case GIVEN_FIELD:
    case EXP_GIVEN_FIELD:
      this->utils.setFieldData(beta_given_field,fm);
      break;
    case GAL_PROJ_EXP_GIVEN_FIELD:
      this->utils.setFieldData(bF,fm);
      this->utils.setFieldData(beta_given_field,fm);
      break;
    case POWER_LAW:
    case REGULARIZED_COULOMB:
      this->utils.setFieldData(muParam,fm);
      this->utils.setFieldData(lambdaParam,fm);
      this->utils.setFieldData(powerParam,fm);
      this->utils.setFieldData(N,fm);
      this->utils.setFieldData(u_norm,fm);
  }

  if (use_stereographic_map)
    this->utils.setFieldData(coordVec,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits, bool IsHydrology>
void BasalFrictionCoefficient<EvalT, Traits, IsHydrology, true>::
evaluateFields (typename Traits::EvalData workset)
{
  if (workset.sideSets->find(basalSideName)==workset.sideSets->end())
    return;

  ScalarT mu, lambda, power;

  if (beta_type==POWER_LAW || beta_type==REGULARIZED_COULOMB)
  {
    mu = muParam(0);
    lambda = lambdaParam(0);
    power = powerParam(0);

    TEUCHOS_TEST_FOR_EXCEPTION (power<-1.0, Teuchos::Exceptions::InvalidParameter,
                                "\nError in FELIX::BasalFrictionCoefficient: \"Power Exponent\" must be greater than (or equal to) -1.\n");
#ifdef OUTPUT_TO_SCREEN
    Teuchos::RCP<Teuchos::FancyOStream> output(Teuchos::VerboseObjectBase::getDefaultOStream());
    int procRank = Teuchos::GlobalMPISession::getRank();
    int numProcs = Teuchos::GlobalMPISession::getNProc();
    output->setProcRankAndSize (procRank, numProcs);
    output->setOutputToRootOnly (0);

    if (printedMu!=mu)
    {
      *output << "[Basal Friction Coefficient<" << PHX::typeAsString<EvalT>() << ">] mu = " << mu << "\n";
      printedMu = mu;
    }
    if (printedLambda!=lambda)
    {
      *output << "[Basal Friction Coefficient<" << PHX::typeAsString<EvalT>() << ">] lambda = " << lambda << "\n";
      printedLambda = lambda;
    }
    if (printedQ!=power)
    {
      *output << "[Basal Friction Coefficient<" << PHX::typeAsString<EvalT>() << ">]] power = " << power << "\n";
      printedQ = power;
    }
#endif
  }

  const std::vector<Albany::SideStruct>& sideSet = workset.sideSets->at(basalSideName);
  for (auto const& it_side : sideSet)
  {
    // Get the local data of side and cell
    const int cell = it_side.elem_LID;
    const int side = it_side.side_local_id;

    switch (beta_type)
    {
      case GIVEN_CONSTANT:
        return;   // We can save ourself some useless iterations

      case GIVEN_FIELD:
        for (int qp=0; qp<numQPs; ++qp)
        {
          beta(cell,side,qp) = beta_given_field(cell,side,qp);
        }
        break;

      case POWER_LAW:
        for (int qp=0; qp<numQPs; ++qp)
        {
          beta(cell,side,qp) = mu * N(cell,side,qp) * std::pow (u_norm(cell,side,qp), power);
        }
        break;

      case REGULARIZED_COULOMB:
        for (int qp=0; qp<numQPs; ++qp)
        {
          ScalarT q = u_norm(cell,side,qp) / ( u_norm(cell,side,qp) + lambda*std::pow(A*N(cell,side,qp),1./power) );
          beta(cell,side,qp) = mu * N(cell,side,qp) * std::pow( q, power) / u_norm(cell,side,qp);
        }
        break;

      case EXP_GIVEN_FIELD:
        for (int qp=0; qp<numQPs; ++qp)
        {
          beta(cell,side,qp) = std::exp(beta_given_field(cell,side,qp));
        }
        break;

      case GAL_PROJ_EXP_GIVEN_FIELD:
        for (int qp=0; qp<numQPs; ++qp)
        {
          beta(cell,side,qp) = 0;
          for (int node=0; node<numNodes; ++node) 
            beta(cell,side,qp) += std::exp(beta_given_field(cell,side,node))*bF(cell,side,node,qp);
        }
      break;
    }

    // Correct the value if we are using a stereographic map
    if (use_stereographic_map)
    {
      for (int qp=0; qp<numQPs; ++qp)
      {
        MeshScalarT x = coordVec(cell,side,qp,0) - x_0;
        MeshScalarT y = coordVec(cell,side,qp,1) - y_0;
        MeshScalarT h = 4.0*R2/(4.0*R2 + x*x + y*y);
        beta(cell,side,qp) *= h*h;
      }
    }
  }
}

//**********************************************************************
// PARTIAL SPECIALIZATION: Hydrology ***********************************
//**********************************************************************
template<typename EvalT, typename Traits>
BasalFrictionCoefficient<EvalT, Traits, true, false>::
BasalFrictionCoefficient (const Teuchos::ParameterList& p,
                          const Teuchos::RCP<Albany::Layouts>& dl) :
  beta        (p.get<std::string> ("Basal Friction Coefficient Variable Name"), dl->qp_scalar)
{
#ifdef OUTPUT_TO_SCREEN
  Teuchos::RCP<Teuchos::FancyOStream> output(Teuchos::VerboseObjectBase::getDefaultOStream());
#endif

  Teuchos::ParameterList& beta_list = *p.get<Teuchos::ParameterList*>("Parameter List");

  std::string betaType = (beta_list.isParameter("Type") ? beta_list.get<std::string>("Type") : "Given Field");

  numQPs   = dl->qp_scalar->dimension(1);
  numNodes = dl->node_scalar->dimension(1);

  this->addEvaluatedField(beta);

  if (betaType == "Given Constant")
  {
#ifdef OUTPUT_TO_SCREEN
    *output << "Given constant and uniform beta, value loaded from xml input file.\n";
#endif
    beta_type = GIVEN_CONSTANT;
    beta_given_val = beta_list.get<double>("Constant Given Beta Value");
  }
  else if ((betaType == "Given Field")|| (betaType == "Exponent Of Given Field")|| (betaType == "Galerkin Projection Of Exponent Of Given Field"))
  {
#ifdef OUTPUT_TO_SCREEN
    *output << "Given constant beta field, loaded from mesh or file.\n";
#endif
    if (betaType == "Given Field")
      beta_type = GIVEN_FIELD;
    else if (betaType == "Exponent Of Given Field")
      beta_type = EXP_GIVEN_FIELD;
    else
      beta_type = GAL_PROJ_EXP_GIVEN_FIELD;

    if(beta_type == GAL_PROJ_EXP_GIVEN_FIELD) {
      beta_given_field = PHX::MDField<ParamScalarT>(p.get<std::string> ("Basal Friction Coefficient Variable Name") + " Given", dl->node_scalar);
      this->addDependentField (beta_given_field);
      bF = PHX::MDField<RealType,Cell,Node,QuadPoint>(p.get<std::string> ("BF Variable Name"), dl->node_qp_scalar);
      this->addDependentField (bF);
    }
    else {
      beta_given_field = PHX::MDField<ParamScalarT>(p.get<std::string> ("Basal Friction Coefficient Variable Name") + " Given", dl->qp_scalar);
      this->addDependentField (beta_given_field);
    }



  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
        std::endl << "Error in FELIX::BasalFrictionCoefficient:  \"" << betaType << "\" is not a valid parameter for Beta Type\n");
  }

  auto& stereographicMapList = p.get<Teuchos::ParameterList*>("Stereographic Map");
  use_stereographic_map = stereographicMapList->get("Use Stereographic Map", false);
  if(use_stereographic_map)
  {
    coordVec = PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim>(p.get<std::string>("Coordinate Vector Variable Name"), dl->qp_coords);

    double R = stereographicMapList->get<double>("Earth Radius", 6371);
    x_0 = stereographicMapList->get<double>("X_0", 0);//-136);
    y_0 = stereographicMapList->get<double>("Y_0", 0);//-2040);
    R2 = std::pow(R,2);

    this->addDependentField(coordVec);
  }

  this->setName("BasalFrictionCoefficient"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void BasalFrictionCoefficient<EvalT, Traits, true, false>::
postRegistrationSetup (typename Traits::SetupData d,
                       PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(beta,fm);

  switch (beta_type)
  {
    case GIVEN_CONSTANT:
      beta.deep_copy(ParamScalarT(beta_given_val));
      break;
    case GIVEN_FIELD:
    case EXP_GIVEN_FIELD:
      this->utils.setFieldData(beta_given_field,fm);
      break;
    case GAL_PROJ_EXP_GIVEN_FIELD:
      this->utils.setFieldData(bF,fm);
      this->utils.setFieldData(beta_given_field,fm);
      break;
  }

  if (use_stereographic_map)
    this->utils.setFieldData(coordVec,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void BasalFrictionCoefficient<EvalT, Traits, true, false>::
evaluateFields (typename Traits::EvalData workset)
{
  switch (beta_type)
  {
    case GIVEN_CONSTANT:
      break;   // We don't have anything to do

    case GIVEN_FIELD:
      for (int cell=0; cell<workset.numCells; ++cell)
        for (int qp=0; qp<numQPs; ++qp)
          {
            beta(cell,qp) = beta_given_field(cell,qp);
          }
      break;

    case GAL_PROJ_EXP_GIVEN_FIELD:
      for (int cell=0; cell<workset.numCells; ++cell)
        for (int qp=0; qp<numQPs; ++qp)
        {
          beta(cell,qp) = 0;
          for (int node=0; node<numNodes; ++node) 
            beta(cell,qp) += std::exp(beta_given_field(cell,node))*bF(cell,node,qp);
        }
      break;
  }

  // Correct the value if we are using a stereographic map
  if (use_stereographic_map)
  {
    for (int cell=0; cell<workset.numCells; ++cell)
    {
      for (int qp=0; qp<numQPs; ++qp)
      {
        MeshScalarT x = coordVec(cell,qp,0) - x_0;
        MeshScalarT y = coordVec(cell,qp,1) - y_0;
        MeshScalarT h = 4.0*R2/(4.0*R2 + x*x + y*y);
        beta(cell,qp) *= h*h;
      }
    }
  }
}

} // Namespace FELIX
