//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Intrepid2_FunctionSpaceTools.hpp"
#include "Albany_Layouts.hpp"

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN

namespace FELIX
{

template<typename EvalT, typename Traits, bool IsHydrology, bool IsStokes>
BasalFrictionCoefficientNode<EvalT, Traits, IsHydrology, IsStokes>::
BasalFrictionCoefficientNode (const Teuchos::ParameterList& p,
                          const Teuchos::RCP<Albany::Layouts>& dl) :
  beta        (p.get<std::string> ("Basal Friction Coefficient Variable Name"), dl->node_scalar)
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

  if (IsStokes)
  {
    TEUCHOS_TEST_FOR_EXCEPTION (!dl->isSideLayouts, Teuchos::Exceptions::InvalidParameter,
                                "Error! The layout structure does not appear to be that of a side set.\n");

    basalSideName = p.get<std::string>("Side Set Name");
    numNodes      = dl->node_scalar->dimension(2);
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION (dl->isSideLayouts, Teuchos::Exceptions::InvalidParameter,
                                "Error! The layout structure appears to be that of a side set.\n");

    numNodes  = dl->node_scalar->dimension(1);
  }

  this->addEvaluatedField(beta);

  if (betaType == "Given Constant")
  {
#ifdef OUTPUT_TO_SCREEN
    *output << "Given constant and uniform beta, value loaded from xml input file.\n";
#endif
    beta_type = GIVEN_CONSTANT;
    beta_given_val = beta_list.get<double>("Constant Given Beta Value");
  }
  else if ((betaType == "Given Field")|| (betaType == "Exponent of Given Field"))
  {
#ifdef OUTPUT_TO_SCREEN
    *output << "Given constant beta field, loaded from mesh or file.\n";
#endif
    if (betaType == "Given Field")
      beta_type = GIVEN_FIELD;
    else
      beta_type = EXP_GIVEN_FIELD;

    beta_given_field = PHX::MDField<ParamScalarT>(p.get<std::string> ("Basal Friction Coefficient Variable Name") + " Given", dl->node_scalar);

    this->addDependentField (beta_given_field.fieldTag());
  }
  else if (betaType == "Power Law")
  {
    beta_type = POWER_LAW;


#ifdef OUTPUT_TO_SCREEN
    *output << "Velocity-dependent beta (power law):\n\n"
            << "      beta = mu * N * |u|^p \n\n"
            << "  with N being the effective pressure, |u| the sliding velocity\n";
#endif

    N              = PHX::MDField<HydroScalarT>(p.get<std::string> ("Effective Pressure Variable Name"), dl->node_scalar);
    u_norm         = PHX::MDField<IceScalarT>(p.get<std::string> ("Sliding Velocity Variable Name"), dl->node_scalar);
    muParam        = PHX::MDField<ScalarT,Dim>("Coulomb Friction Coefficient", dl->shared_param);
    powerParam     = PHX::MDField<ScalarT,Dim>("Power Exponent", dl->shared_param);

    this->addDependentField (muParam.fieldTag());
    this->addDependentField (powerParam.fieldTag());
    this->addDependentField (u_norm.fieldTag());
    this->addDependentField (N.fieldTag());

    distributedLambda = beta_list.get<bool>("Distributed Bed Roughness",false);
    if (distributedLambda)
    {
      lambdaField = PHX::MDField<ParamScalarT>(p.get<std::string> ("Bed Roughness Variable Name"), dl->node_scalar);
      this->addDependentField (lambdaField.fieldTag());
    }
    else
    {
      lambdaParam    = PHX::MDField<ScalarT,Dim>("Bed Roughness", dl->shared_param);
      this->addDependentField (lambdaParam.fieldTag());
    }
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
      //A = std::cbrt(A);
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! The case with variable flow factor has not been implemented yet.\n");
    }
#ifdef OUTPUT_TO_SCREEN
    *output << "Velocity-dependent beta (regularized coulomb law):\n\n"
            << "      beta = mu * N * |u|^{p-1} / [|u| + lambda*A*N^(1/n)]^p\n\n"
            << "  with N being the effective pressure, |u| the sliding velocity\n";
#endif

    N              = PHX::MDField<HydroScalarT>(p.get<std::string> ("Effective Pressure Variable Name"), dl->node_scalar);
    u_norm         = PHX::MDField<IceScalarT>(p.get<std::string> ("Sliding Velocity Variable Name"), dl->node_scalar);
    muParam        = PHX::MDField<ScalarT,Dim>("Coulomb Friction Coefficient", dl->shared_param);
    powerParam     = PHX::MDField<ScalarT,Dim>("Power Exponent", dl->shared_param);

    this->addDependentField (muParam.fieldTag());
    this->addDependentField (powerParam.fieldTag());
    this->addDependentField (N.fieldTag());
    this->addDependentField (u_norm.fieldTag());

    distributedLambda = beta_list.get<bool>("Distributed Bed Roughness",false);
    if (distributedLambda)
    {
      lambdaField = PHX::MDField<ParamScalarT>(p.get<std::string> ("Bed Roughness Variable Name"), dl->node_scalar);
      this->addDependentField (lambdaField.fieldTag());
    }
    else
    {
      lambdaParam    = PHX::MDField<ScalarT,Dim>("Bed Roughness", dl->shared_param);
      this->addDependentField (lambdaParam.fieldTag());
    }
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
        std::endl << "Error in FELIX::BasalFrictionCoefficientNode:  \"" << betaType << "\" is not a valid parameter for Beta Type\n");
  }

  logParameters = beta_list.get<bool>("Use log scalar parameters",false);

  this->setName("BasalFrictionCoefficientNode"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits, bool IsHydrology, bool IsStokes>
void BasalFrictionCoefficientNode<EvalT, Traits, IsHydrology, IsStokes>::
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
    case POWER_LAW:
    case REGULARIZED_COULOMB:
      this->utils.setFieldData(muParam,fm);
      this->utils.setFieldData(powerParam,fm);
      this->utils.setFieldData(N,fm);
      this->utils.setFieldData(u_norm,fm);
      if (distributedLambda)
        this->utils.setFieldData(lambdaField,fm);
      else
        this->utils.setFieldData(lambdaParam,fm);
  }

  if (use_stereographic_map)
    this->utils.setFieldData(coordVec,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits, bool IsHydrology, bool IsStokes>
void BasalFrictionCoefficientNode<EvalT, Traits, IsHydrology, IsStokes>::
evaluateFields (typename Traits::EvalData workset)
{
  ScalarT mu, lambda, power;

  if (beta_type==POWER_LAW || beta_type==REGULARIZED_COULOMB)
  {
    if (logParameters)
    {
      mu = std::exp(muParam(0));
      power = std::exp(powerParam(0));

      if (!distributedLambda)
        lambda = std::exp(lambdaParam(0));
    }
    else
    {
      mu = muParam(0);
      power = powerParam(0);
      if (!distributedLambda)
        lambda = lambdaParam(0);
    }
#ifdef OUTPUT_TO_SCREEN
    Teuchos::RCP<Teuchos::FancyOStream> output(Teuchos::VerboseObjectBase::getDefaultOStream());
    int procRank = Teuchos::GlobalMPISession::getRank();
    int numProcs = Teuchos::GlobalMPISession::getNProc();
    output->setProcRankAndSize (procRank, numProcs);
    output->setOutputToRootOnly (0);

    if (!distributedLambda && printedLambda!=lambda)
    {
      *output << "[Basal Friction Coefficient Node " << PHX::typeAsString<EvalT>() << "] lambda = " << lambda << "\n";
      printedLambda = lambda;
    }
    if (printedMu!=mu)
    {
      *output << "[Basal Friction Coefficient Node " << PHX::typeAsString<EvalT>() << "] mu = " << mu << "\n";
      printedMu = mu;
    }
    if (printedQ!=power)
    {
      *output << "[Basal Friction Coefficient Node " << PHX::typeAsString<EvalT>() << "] power = " << power << "\n";
      printedQ = power;
    }
#endif

    TEUCHOS_TEST_FOR_EXCEPTION (power<0, Teuchos::Exceptions::InvalidParameter,
                                "\nError in FELIX::BasalFrictionCoefficientNode: 'Power Exponent' must be >= 0.\n");
    TEUCHOS_TEST_FOR_EXCEPTION (mu<0, Teuchos::Exceptions::InvalidParameter,
                                "\nError in FELIX::BasalFrictionCoefficientNode: 'Coulomb Friction Coefficient' must be >= 0.\n");
    TEUCHOS_TEST_FOR_EXCEPTION (!distributedLambda && lambda<0, Teuchos::Exceptions::InvalidParameter,
                                "\nError in FELIX::BasalFrictionCoefficientNode: \"Bed Roughness\" must be >= 0.\n");
  }

  if (IsStokes)
    evaluateFieldsSide(workset,mu,lambda,power);
  else
    evaluateFieldsCell(workset,mu,lambda,power);
}

template<typename EvalT, typename Traits, bool IsHydrology, bool IsStokes>
void BasalFrictionCoefficientNode<EvalT, Traits, IsHydrology, IsStokes>::
evaluateFieldsSide (typename Traits::EvalData workset, ScalarT mu, ScalarT lambda, ScalarT power)
{
  if (workset.sideSets->find(basalSideName)==workset.sideSets->end())
    return;

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
        for (int node=0; node<numNodes; ++node)
        {
          beta(cell,side,node) = beta_given_field(cell,side,node);
        }
        break;

      case POWER_LAW:
        for (int node=0; node<numNodes; ++node)
        {
          beta(cell,side,node) = mu * N(cell,side,node) * std::pow (u_norm(cell,side,node), power);
        }
        break;

      case REGULARIZED_COULOMB:
        if (distributedLambda)
          for (int node=0; node<numNodes; ++node)
          {
            ScalarT q = u_norm(cell,side,node) / ( u_norm(cell,side,node) + lambdaField(cell,side,node)*std::pow(A*N(cell,side,node),1./power) );
            beta(cell,side,node) = mu * N(cell,side,node) * std::pow( q, power) / u_norm(cell,side,node);
          }
        else
          for (int node=0; node<numNodes; ++node)
          {
            ScalarT q = u_norm(cell,side,node) / ( u_norm(cell,side,node) + lambda*std::pow(A*N(cell,side,node),1./power) );
            beta(cell,side,node) = mu * N(cell,side,node) * std::pow( q, power) / u_norm(cell,side,node);
          }
        break;

      case EXP_GIVEN_FIELD:
        for (int node=0; node<numNodes; ++node)
        {
          beta(cell,side,node) = std::exp(beta_given_field(cell,side,node));
        }
        break;
    }
  }
}

template<typename EvalT, typename Traits, bool IsHydrology, bool IsStokes>
void BasalFrictionCoefficientNode<EvalT, Traits, IsHydrology, IsStokes>::
evaluateFieldsCell (typename Traits::EvalData workset, ScalarT mu, ScalarT lambda, ScalarT power)
{
  switch (beta_type)
  {
    case GIVEN_CONSTANT:
      break;   // We don't have anything to do

    case GIVEN_FIELD:
      for (int cell=0; cell<workset.numCells; ++cell)
        for (int node=0; node<numNodes; ++node)
            beta(cell,node) = beta_given_field(cell,node);
      break;

    case POWER_LAW:
      for (int cell=0; cell<workset.numCells; ++cell)
        for (int node=0; node<numNodes; ++node)
          beta(cell,node) = mu * N(cell,node) * std::pow (u_norm(cell,node), power);
      break;

    case REGULARIZED_COULOMB:
      if (distributedLambda)
      {
        if (logParameters)
          for (int cell=0; cell<workset.numCells; ++cell)
            for (int node=0; node<numNodes; ++node)
            {
              ScalarT q = u_norm(cell,node) / ( u_norm(cell,node) + lambdaField(cell,node)*A*std::pow(std::exp(N(cell,node)),3) );
              beta(cell,node) = mu * std::exp(N(cell,node)) * std::pow( q, power) / u_norm(cell,node);
            }
        else
          for (int cell=0; cell<workset.numCells; ++cell)
            for (int node=0; node<numNodes; ++node)
            {
              ScalarT q = u_norm(cell,node) / ( u_norm(cell,node) + lambdaField(cell,node)*A*std::pow(std::max(N(cell,node),0.0),3) );
              beta(cell,node) = mu * std::max(N(cell,node),0.0) * std::pow( q, power) / u_norm(cell,node);
            }
      }
      else
      {
        if (logParameters)
          for (int cell=0; cell<workset.numCells; ++cell)
            for (int node=0; node<numNodes; ++node)
            {
              ScalarT q = u_norm(cell,node) / ( u_norm(cell,node) + lambda*A*std::pow(std::exp(N(cell,node)),3) );
              beta(cell,node) = mu * std::exp(N(cell,node)) * std::pow( q, power) / u_norm(cell,node);
            }
        else
          for (int cell=0; cell<workset.numCells; ++cell)
            for (int node=0; node<numNodes; ++node)
            {
              ScalarT q = u_norm(cell,node) / ( u_norm(cell,node) + lambda*A*std::pow(std::max(N(cell,node),0.0),3) );
              beta(cell,node) = mu * std::max(N(cell,node),0.0) * std::pow( q, power) / u_norm(cell,node);
            }
      }
      break;

    case EXP_GIVEN_FIELD:
      for (int cell=0; cell<workset.numCells; ++cell)
        for (int node=0; node<numNodes; ++node)
        {
          beta(cell,node) = std::exp(beta_given_field(cell,node));
        }
      break;
  }
}

} // Namespace FELIX
