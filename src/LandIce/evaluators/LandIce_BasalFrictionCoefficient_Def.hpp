//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Albany_DiscretizationUtils.hpp"
#include "Albany_Layouts.hpp"

#include "LandIce_BasalFrictionCoefficient.hpp"

#include <string.hpp> // for 'upper_case' (comes from src/utility; not to be confused with <string>)

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN

namespace LandIce
{

template<typename EvalT, typename Traits, bool IsHydrology, bool IsStokes, bool ThermoCoupled>
BasalFrictionCoefficient<EvalT, Traits, IsHydrology, IsStokes, ThermoCoupled>::
BasalFrictionCoefficient (const Teuchos::ParameterList& p,
                          const Teuchos::RCP<Albany::Layouts>& dl)
{
#ifdef OUTPUT_TO_SCREEN
  Teuchos::RCP<Teuchos::FancyOStream> output(Teuchos::VerboseObjectBase::getDefaultOStream());

  int procRank = Teuchos::GlobalMPISession::getRank();
  int numProcs = Teuchos::GlobalMPISession::getNProc();
  output->setProcRankAndSize (procRank, numProcs);
  output->setOutputToRootOnly (0);
#endif

  n = -1; //dummy value
  Teuchos::ParameterList& beta_list = *p.get<Teuchos::ParameterList*>("Parameter List");
  zero_on_floating = beta_list.get<bool> ("Zero Beta On Floating Ice", false);

  std::string betaType = util::upper_case((beta_list.isParameter("Type") ? beta_list.get<std::string>("Type") : "Given Field"));

  if (IsStokes)
  {
    TEUCHOS_TEST_FOR_EXCEPTION (!dl->isSideLayouts, Teuchos::Exceptions::InvalidParameter,
                                "Error! The layout structure does not appear to be that of a side set.\n");

    basalSideName = p.get<std::string>("Side Set Name");
    numQPs        = dl->qp_scalar->dimension(2);
    numNodes      = dl->node_scalar->dimension(2);
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION (dl->isSideLayouts, Teuchos::Exceptions::InvalidParameter,
                                "Error! The layout structure appears to be that of a side set.\n");

    numQPs    = dl->qp_scalar->dimension(1);
    numNodes  = dl->node_scalar->dimension(1);
  }

  nodal = p.isParameter("Nodal") ? p.get<bool>("Nodal") : false;
  Teuchos::RCP<PHX::DataLayout> layout;
  if (nodal) {
    layout = dl->node_scalar;
  } else {
    layout = dl->qp_scalar;
  }

  beta = PHX::MDField<ScalarT>(p.get<std::string> ("Basal Friction Coefficient Variable Name"), layout);
  this->addEvaluatedField(beta);

  printedMu      = -9999.999;
  printedLambda  = -9999.999;
  printedQ       = -9999.999;

  if (betaType == "GIVEN CONSTANT")
  {
    beta_type = GIVEN_CONSTANT;
    given_val = beta_list.get<double>("Constant Given Beta Value");
#ifdef OUTPUT_TO_SCREEN
    *output << "Given constant and uniform beta, value = " << given_val << " (loaded from xml input file).\n";
#endif
  }
  else if ((betaType == "GIVEN FIELD")|| (betaType == "EXPONENT OF GIVEN FIELD") || (betaType == "GALERKIN PROJECTION OF EXPONENT OF GIVEN FIELD"))
  {
#ifdef OUTPUT_TO_SCREEN
    *output << "Given constant beta field, loaded from mesh or file.\n";
#endif
    if (betaType == "GIVEN FIELD") {
      beta_type = GIVEN_FIELD;
    } else if (betaType == "GALERKIN PROJECTION OF EXPONENT OF GIVEN FIELD") {
      beta_type = GAL_PROJ_EXP_GIVEN_FIELD;
      if (nodal) {
        // There's no Galerkin projection to do. It's just like EXP_GIVEN_FIELD
        beta_type = EXP_GIVEN_FIELD;
      }
    } else {
      beta_type = EXP_GIVEN_FIELD;
    }

    std::string given_field_name = beta_list.get<std::string> ("Given Field Variable Name");
    if (IsStokes) {
      given_field_name += "_" + basalSideName;
    }
    if(beta_type == GAL_PROJ_EXP_GIVEN_FIELD) {
      given_field = PHX::MDField<const ParamScalarT>(given_field_name, dl->node_scalar);
      this->addDependentField (given_field);
      BF = PHX::MDField<const RealType>(p.get<std::string> ("BF Variable Name"), dl->node_qp_scalar);
      this->addDependentField (BF);
    }
    else {
      given_field = PHX::MDField<const ParamScalarT>(given_field_name, layout);
      this->addDependentField (given_field);
    }
  }
  else if (betaType == "POWER LAW")
  {
    beta_type = POWER_LAW;


#ifdef OUTPUT_TO_SCREEN
    *output << "Velocity-dependent beta (power law):\n\n"
            << "      beta = mu * N * |u|^{q-1} \n\n"
            << "  with N being the effective pressure, |u| the sliding velocity\n";
#endif

    N              = PHX::MDField<const HydroScalarT>(p.get<std::string> ("Effective Pressure Variable Name"), layout);
    u_norm         = PHX::MDField<const IceScalarT>(p.get<std::string> ("Sliding Velocity Variable Name"), layout);
    muPowerLaw     = PHX::MDField<const ScalarT,Dim>("Power Law Coefficient", dl->shared_param);
    powerParam     = PHX::MDField<const ScalarT,Dim>("Power Exponent", dl->shared_param);

    this->addDependentField (muPowerLaw);
    this->addDependentField (powerParam);
    this->addDependentField (u_norm);
    this->addDependentField (N);
  }
  else if (betaType == "REGULARIZED COULOMB")
  {
    beta_type = REGULARIZED_COULOMB;

#ifdef OUTPUT_TO_SCREEN
    *output << "Velocity-dependent beta (regularized coulomb law):\n\n"
            << "      beta = mu * N * |u|^{q-1} / [|u| + lambda*A*N^n]^q\n\n"
            << "  with N being the effective pressure, |u| the sliding velocity\n";
#endif

    n            = p.get<Teuchos::ParameterList*>("Viscosity Parameter List")->get<double>("Glen's Law n");
    N            = PHX::MDField<const HydroScalarT>(p.get<std::string> ("Effective Pressure Variable Name"), layout);
    u_norm       = PHX::MDField<const IceScalarT>(p.get<std::string> ("Sliding Velocity Variable Name"), layout);
    muCoulomb    = PHX::MDField<const ScalarT,Dim>("Coulomb Friction Coefficient", dl->shared_param);
    powerParam   = PHX::MDField<const ScalarT,Dim>("Power Exponent", dl->shared_param);
    ice_softness = PHX::MDField<const TempScalarT>(p.get<std::string>("Ice Softness Variable Name"), dl->cell_scalar2);

    this->addDependentField (muCoulomb);
    this->addDependentField (powerParam);
    this->addDependentField (N);
    this->addDependentField (u_norm);
    this->addDependentField (ice_softness);

    distributedLambda = beta_list.get<bool>("Distributed Bed Roughness",false);
    if (distributedLambda)
    {
      lambdaField = PHX::MDField<const ParamScalarT>(p.get<std::string> ("Bed Roughness Variable Name"), layout);
      this->addDependentField (lambdaField);
    }
    else
    {
      lambdaParam    = PHX::MDField<ScalarT,Dim>("Bed Roughness", dl->shared_param);
      this->addDependentField (lambdaParam);
    }
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
        std::endl << "Error in LandIce::BasalFrictionCoefficient:  \"" << betaType << "\" is not a valid parameter for Beta Type\n");
  }

  if(zero_on_floating) {
    bed_topo_field = PHX::MDField<const ParamScalarT>(p.get<std::string> ("Bed Topography Variable Name"), layout);
    thickness_field = PHX::MDField<const ParamScalarT>(p.get<std::string> ("Ice Thickness Variable Name"), layout);
    Teuchos::ParameterList& phys_param_list = *p.get<Teuchos::ParameterList*>("Physical Parameter List");
    rho_i = phys_param_list.get<double> ("Ice Density");
    rho_w = phys_param_list.get<double> ("Water Density");
    this->addDependentField (bed_topo_field);
    this->addDependentField (thickness_field);
  }

  auto& stereographicMapList = p.get<Teuchos::ParameterList*>("Stereographic Map");
  use_stereographic_map = stereographicMapList->get("Use Stereographic Map", false);
  if(use_stereographic_map)
  {
    layout = nodal ? dl->node_vector : dl->qp_coords;
    coordVec = PHX::MDField<MeshScalarT>(p.get<std::string>("Coordinate Vector Variable Name"), layout);

    double R = stereographicMapList->get<double>("Earth Radius", 6371);
    x_0 = stereographicMapList->get<double>("X_0", 0);//-136);
    y_0 = stereographicMapList->get<double>("Y_0", 0);//-2040);
    R2 = std::pow(R,2);

    this->addDependentField(coordVec);
  }

  logParameters = beta_list.get<bool>("Use log scalar parameters",false);

  if (p.isType<bool>("Enable Memoizer") && p.get<bool>("Enable Memoizer"))
    memoizer.enable_memoizer();

  this->setName("BasalFrictionCoefficient"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits, bool IsHydrology, bool IsStokes, bool ThermoCoupled>
void BasalFrictionCoefficient<EvalT, Traits, IsHydrology, IsStokes, ThermoCoupled>::
postRegistrationSetup (typename Traits::SetupData d,
                       PHX::FieldManager<Traits>& fm)
{
  if (beta_type == GIVEN_CONSTANT)
    beta.deep_copy(ScalarT(given_val));
}

//**********************************************************************
template<typename EvalT, typename Traits, bool IsHydrology, bool IsStokes, bool ThermoCoupled>
void BasalFrictionCoefficient<EvalT, Traits, IsHydrology, IsStokes, ThermoCoupled>::
evaluateFields (typename Traits::EvalData workset)
{
  if (memoizer.have_stored_data(workset)) return;

  ParamScalarT mu, lambda, power;

  if (beta_type == POWER_LAW)
  {
    if (logParameters)
    {
      mu = std::exp(Albany::convertScalar<const ParamScalarT>(muPowerLaw(0)));
      power = std::exp(Albany::convertScalar<const ParamScalarT>(powerParam(0)));
    }
    else
    {
      mu = Albany::convertScalar<const ParamScalarT>(muPowerLaw(0));
      power = Albany::convertScalar<const ParamScalarT>(powerParam(0));
    }
#ifdef OUTPUT_TO_SCREEN
    Teuchos::RCP<Teuchos::FancyOStream> output(Teuchos::VerboseObjectBase::getDefaultOStream());
    int procRank = Teuchos::GlobalMPISession::getRank();
    int numProcs = Teuchos::GlobalMPISession::getNProc();
    output->setProcRankAndSize (procRank, numProcs);
    output->setOutputToRootOnly (0);

    if (printedMu!=mu)
    {
      *output << "[Basal Friction Coefficient" << PHX::typeAsString<EvalT>() << "] mu = " << mu << " [kPa yr^q m^{-q}]\n";
      printedMu = mu;
    }
    if (printedQ!=power)
    {
      *output << "[Basal Friction Coefficient" << PHX::typeAsString<EvalT>() << "] power = " << power << "\n";
      printedQ = power;
    }
#endif

    TEUCHOS_TEST_FOR_EXCEPTION (power<0, Teuchos::Exceptions::InvalidParameter,
                                "\nError in LandIce::BasalFrictionCoefficient: 'Power Exponent' must be >= 0.\n");
    TEUCHOS_TEST_FOR_EXCEPTION (mu<0, Teuchos::Exceptions::InvalidParameter,
                                "\nError in LandIce::BasalFrictionCoefficient: 'Coulomb Friction Coefficient' must be >= 0.\n");
  }

  if (beta_type==REGULARIZED_COULOMB)
  {
    if (logParameters)
    {
      mu = std::exp(Albany::convertScalar<const ParamScalarT>(muCoulomb(0)));
      power = std::exp(Albany::convertScalar<const ParamScalarT>(powerParam(0)));
      if (!distributedLambda)
        lambda = std::exp(Albany::convertScalar<const ParamScalarT>(lambdaParam(0)));
    }
    else
    {
      mu = Albany::convertScalar<const ParamScalarT>(muCoulomb(0));
      power = Albany::convertScalar<const ParamScalarT>(powerParam(0));
      if (!distributedLambda)
        lambda = Albany::convertScalar<const ParamScalarT>(lambdaParam(0));
    }
#ifdef OUTPUT_TO_SCREEN
    Teuchos::RCP<Teuchos::FancyOStream> output(Teuchos::VerboseObjectBase::getDefaultOStream());
    int procRank = Teuchos::GlobalMPISession::getRank();
    int numProcs = Teuchos::GlobalMPISession::getNProc();
    output->setProcRankAndSize (procRank, numProcs);
    output->setOutputToRootOnly (0);

    if (!distributedLambda && printedLambda!=lambda)
    {
      *output << "[Basal Friction Coefficient" << PHX::typeAsString<EvalT>() << "] lambda = " << lambda << "\n";
      printedLambda = lambda;
    }
    if (printedMu!=mu)
    {
      *output << "[Basal Friction Coefficient" << PHX::typeAsString<EvalT>() << "] mu = " << mu << "\n";
      printedMu = mu;
    }
    if (printedQ!=power)
    {
      *output << "[Basal Friction Coefficient" << PHX::typeAsString<EvalT>() << "] power = " << power << "\n";
      printedQ = power;
    }
#endif

    TEUCHOS_TEST_FOR_EXCEPTION (power<0, Teuchos::Exceptions::InvalidParameter,
                                "\nError in LandIce::BasalFrictionCoefficient: 'Power Exponent' must be >= 0.\n");
    TEUCHOS_TEST_FOR_EXCEPTION (mu<0, Teuchos::Exceptions::InvalidParameter,
                                "\nError in LandIce::BasalFrictionCoefficient: 'Coulomb Friction Coefficient' must be >= 0.\n");
    TEUCHOS_TEST_FOR_EXCEPTION (!distributedLambda && lambda<0, Teuchos::Exceptions::InvalidParameter,
                                "\nError in LandIce::BasalFrictionCoefficient: \"Bed Roughness\" must be >= 0.\n");
  }

  if (IsStokes)
    evaluateFieldsSide(workset,mu,lambda,power);
  else
    evaluateFieldsCell(workset,mu,lambda,power);
}

template<typename EvalT, typename Traits, bool IsHydrology, bool IsStokes, bool ThermoCoupled>
void BasalFrictionCoefficient<EvalT, Traits, IsHydrology, IsStokes, ThermoCoupled>::
evaluateFieldsSide (typename Traits::EvalData workset, ScalarT mu, ScalarT lambda, ScalarT power)
{
  if (workset.sideSets->find(basalSideName)==workset.sideSets->end())
    return;

  const int dim = nodal ? numNodes : numQPs;

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
        for (int ipt=0; ipt<dim; ++ipt)
        {
          beta(cell,side,ipt) = given_field(cell,side,ipt);
        }
        break;

      case POWER_LAW:
        for (int ipt=0; ipt<dim; ++ipt)
        {
          beta(cell,side,ipt) = mu * N(cell,side,ipt) * std::pow (u_norm(cell,side,ipt), power-1);
        }
        break;

      case REGULARIZED_COULOMB:
        for (int ipt=0; ipt<dim; ++ipt)
        {
          ScalarT q = u_norm(cell,side,ipt) / ( u_norm(cell,side,ipt) + lambda*ice_softness(cell,side)*std::pow(N(cell,side,ipt),n) );
          beta(cell,side,ipt) = mu * N(cell,side,ipt) * std::pow( q, power) / u_norm(cell,side,ipt);
        }
        break;

      case EXP_GIVEN_FIELD:
        for (int ipt=0; ipt<dim; ++ipt)
        {
          beta(cell,side,ipt) = std::exp(given_field(cell,side,ipt));
        }
        break;

      case GAL_PROJ_EXP_GIVEN_FIELD:
        for (int qp=0; qp<numQPs; ++qp)
        {
          beta(cell,side,qp) = 0;
          for (int node=0; node<numNodes; ++node)
            beta(cell,side,qp) += std::exp(given_field(cell,side,node))*BF(cell,side,node,qp);
        }
      break;
    }

    if(zero_on_floating)
    {
      for (int ipt=0; ipt<dim; ++ipt) {
        ParamScalarT isGrounded = rho_i*thickness_field(cell,side,ipt) > -rho_w*bed_topo_field(cell,side,ipt);
        beta(cell,side,ipt) *=  isGrounded;
      }
    }

    // Correct the value if we are using a stereographic map
    if (use_stereographic_map)
    {
      for (int ipt=0; ipt<dim; ++ipt)
      {
        MeshScalarT x = coordVec(cell,side,ipt,0) - x_0;
        MeshScalarT y = coordVec(cell,side,ipt,1) - y_0;
        MeshScalarT h = 4.0*R2/(4.0*R2 + x*x + y*y);
        beta(cell,side,ipt) *= h*h;
      }
    }
  }
}

template<typename EvalT, typename Traits, bool IsHydrology, bool IsStokes, bool ThermoCoupled>
void BasalFrictionCoefficient<EvalT, Traits, IsHydrology, IsStokes, ThermoCoupled>::
evaluateFieldsCell (typename Traits::EvalData workset, ScalarT mu, ScalarT lambda, ScalarT power)
{
  const int dim = nodal ? numNodes : numQPs;
  switch (beta_type)
  {
    case GIVEN_CONSTANT:
      break;   // We don't have anything to do

    case GIVEN_FIELD:
      for (int cell=0; cell<workset.numCells; ++cell)
        for (int ipt=0; ipt<dim; ++ipt)
            beta(cell,ipt) = given_field(cell,ipt);
      break;

    case POWER_LAW:
      for (int cell=0; cell<workset.numCells; ++cell)
        for (int ipt=0; ipt<dim; ++ipt)
          beta(cell,ipt) = mu * N(cell,ipt) * std::pow (u_norm(cell,ipt), power);
      break;

    case REGULARIZED_COULOMB:
      if (distributedLambda)
      {
        if (logParameters)
          for (int cell=0; cell<workset.numCells; ++cell)
            for (int ipt=0; ipt<dim; ++ipt)
            {
              ScalarT tmp = u_norm(cell,ipt) / ( u_norm(cell,ipt) + lambdaField(cell,ipt)*ice_softness(cell)*std::pow(std::exp(N(cell,ipt)),3) );
              beta(cell,ipt) = mu * std::exp(N(cell,ipt)) * std::pow( tmp, power) / u_norm(cell,ipt);
            }
        else
          for (int cell=0; cell<workset.numCells; ++cell)
            for (int ipt=0; ipt<dim; ++ipt)
            {
              ScalarT q = u_norm(cell,ipt) / ( u_norm(cell,ipt) + lambdaField(cell,ipt)*ice_softness(cell)*std::pow(std::max(N(cell,ipt),0.0),3) );
              beta(cell,ipt) = mu * std::max(N(cell,ipt),0.0) * std::pow( q, power) / u_norm(cell,ipt);
            }
      }
      else
      {
        if (logParameters)
          for (int cell=0; cell<workset.numCells; ++cell)
            for (int ipt=0; ipt<dim; ++ipt)
            {
              ScalarT q = u_norm(cell,ipt) / ( u_norm(cell,ipt) + lambda*ice_softness(cell)*std::pow(std::exp(N(cell,ipt)),3) );
              beta(cell,ipt) = mu * std::exp(N(cell,ipt)) * std::pow( q, power) / u_norm(cell,ipt);
            }
        else
          for (int cell=0; cell<workset.numCells; ++cell)
            for (int ipt=0; ipt<dim; ++ipt)
            {
              ScalarT q = u_norm(cell,ipt) / ( u_norm(cell,ipt) + lambda*ice_softness(cell)*std::pow(std::max(N(cell,ipt),0.0),3) );
              beta(cell,ipt) = mu * std::max(N(cell,ipt),0.0) * std::pow( q, power) / u_norm(cell,ipt);
            }
      }
      break;

    case EXP_GIVEN_FIELD:
      for (int cell=0; cell<workset.numCells; ++cell)
        for (int ipt=0; ipt<dim; ++ipt)
        {
          beta(cell,ipt) = std::exp(given_field(cell,ipt));
        }
      break;

    case GAL_PROJ_EXP_GIVEN_FIELD:
      for (int cell=0; cell<workset.numCells; ++cell)
        for (int ipt=0; ipt<dim; ++ipt)
        {
          beta(cell,ipt) = 0;
          for (int node=0; node<numNodes; ++node)
            beta(cell,ipt) += std::exp(given_field(cell,node))*BF(cell,node,ipt);
        }
  }

  // Correct the value if we are using a stereographic map
  if (use_stereographic_map)
  {
    for (int cell=0; cell<workset.numCells; ++cell)
    {
      for (int ipt=0; ipt<dim; ++ipt)
      {
        MeshScalarT x = coordVec(cell,ipt,0) - x_0;
        MeshScalarT y = coordVec(cell,ipt,1) - y_0;
        MeshScalarT h = 4.0*R2/(4.0*R2 + x*x + y*y);
        beta(cell,ipt) *= h*h;
      }
    }
  }
}

} // Namespace LandIce
