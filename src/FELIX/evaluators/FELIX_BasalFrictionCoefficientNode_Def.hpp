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
#define OUTPUT_TO_SCREEN

namespace FELIX
{

template<typename EvalT, typename Traits>
BasalFrictionCoefficientNode<EvalT, Traits>::
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

  numNodes  = dl->node_scalar->dimension(1);

  this->addEvaluatedField(beta);

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
          << "      beta = mu * N * |u|^{p-1} / [|u| + lambda*A*N^(1/p)]^p\n\n"
          << "  with N being the effective pressure, |u| the sliding velocity\n";
#endif

  N              = PHX::MDField<ParamScalarT>(p.get<std::string> ("Effective Pressure Variable Name"), dl->node_scalar);
  u_norm         = PHX::MDField<ParamScalarT>(p.get<std::string> ("Sliding Velocity Variable Name"), dl->node_scalar);
  muParam        = PHX::MDField<ScalarT,Dim>("Coulomb Friction Coefficient", dl->shared_param);
  lambdaParam    = PHX::MDField<ScalarT,Dim>("Bed Roughness", dl->shared_param);
  powerParam     = PHX::MDField<ScalarT,Dim>("Power Exponent", dl->shared_param);

  this->addDependentField (muParam);
  this->addDependentField (lambdaParam);
  this->addDependentField (powerParam);
  this->addDependentField (N);
  this->addDependentField (u_norm);

  logParameters = beta_list.get<bool>("Use log scalar parameters",false);

  this->setName("BasalFrictionCoefficientNode"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void BasalFrictionCoefficientNode<EvalT, Traits>::
postRegistrationSetup (typename Traits::SetupData d,
                       PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(muParam,fm);
  this->utils.setFieldData(lambdaParam,fm);
  this->utils.setFieldData(powerParam,fm);
  this->utils.setFieldData(N,fm);
  this->utils.setFieldData(u_norm,fm);

  this->utils.setFieldData(beta,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void BasalFrictionCoefficientNode<EvalT, Traits>::
evaluateFields (typename Traits::EvalData workset)
{
  ScalarT mu, lambda, power;

  mu = muParam(0);
  lambda = lambdaParam(0);
  power = powerParam(0);

  if (logParameters)
  {
    mu = std::exp(muParam(0));
    lambda = std::exp(lambdaParam(0));
    power = std::exp(powerParam(0));
  }
  else
  {
    mu = muParam(0);
    lambda = lambdaParam(0);
    power = powerParam(0);
  }

#ifdef OUTPUT_TO_SCREEN
  Teuchos::RCP<Teuchos::FancyOStream> output(Teuchos::VerboseObjectBase::getDefaultOStream());
  int procRank = Teuchos::GlobalMPISession::getRank();
  int numProcs = Teuchos::GlobalMPISession::getNProc();
  output->setProcRankAndSize (procRank, numProcs);
  output->setOutputToRootOnly (0);

  if (printedLambda!=lambda)
  {
    *output << "[Basal Friction Coefficient<" << PHX::typeAsString<EvalT>() << ">] lambda = " << lambda << "\n";
    printedLambda = lambda;
  }
  if (printedMu!=mu)
  {
    *output << "[Basal Friction Coefficient<" << PHX::typeAsString<EvalT>() << ">] mu = " << mu << "\n";
    printedMu = mu;
  }
  if (printedQ!=power)
  {
    *output << "[Basal Friction Coefficient<" << PHX::typeAsString<EvalT>() << ">]] power = " << power << "\n";
    printedQ = power;
  }
#endif

  TEUCHOS_TEST_FOR_EXCEPTION (power<0, Teuchos::Exceptions::InvalidParameter,
                              "\nError in FELIX::BasalFrictionCoefficient: 'Power Exponent' must be >= 0.\n");
  TEUCHOS_TEST_FOR_EXCEPTION (mu<0, Teuchos::Exceptions::InvalidParameter,
                              "\nError in FELIX::BasalFrictionCoefficient: 'Coulomb Friction Coefficient' must be >= 0.\n");
  TEUCHOS_TEST_FOR_EXCEPTION (lambda<0, Teuchos::Exceptions::InvalidParameter,
                              "\nError in FELIX::BasalFrictionCoefficient: \"Bed Roughness\" must be >= 0.\n");

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

} // Namespace FELIX
