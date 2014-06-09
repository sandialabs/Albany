//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Albany_AbstractProblem.hpp"

// Generic implementations that can be used by derived problems

Albany::AbstractProblem::AbstractProblem(
  const Teuchos::RCP<Teuchos::ParameterList>& params_,
  const Teuchos::RCP<ParamLib>& paramLib_,
  //const Teuchos::RCP<DistParamLib>& distParamLib_,
  const int neq_) :
  out(Teuchos::VerboseObjectBase::getDefaultOStream()),
  neq(neq_),
  params(params_),
  paramLib(paramLib_),
  //distParamLib(distParamLib_),
  rigidBodyModes(Teuchos::rcp(new Piro::MLRigidBodyModes(neq_)))
{}

unsigned int
Albany::AbstractProblem::numEquations() const
{
  TEUCHOS_TEST_FOR_EXCEPTION( neq <= 0,
                    Teuchos::Exceptions::InvalidParameter,
                    "A Problem must have at least 1 equation: "<<neq);
  return neq;
}

void
Albany::AbstractProblem::setNumEquations(const int neq_)
{
  neq = neq_;
  rigidBodyModes->setNumPDEs(neq_);
}


Teuchos::ArrayRCP<Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> > >
Albany::AbstractProblem::getFieldManager()
{ return fm; }

Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> >
Albany::AbstractProblem::getDirichletFieldManager()
{ return dfm; }

Teuchos::ArrayRCP<Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> > >
Albany::AbstractProblem::getNeumannFieldManager()
{ return nfm; }

Teuchos::RCP<Teuchos::ParameterList>
Albany::AbstractProblem::getGenericProblemParams(std::string listname) const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
     Teuchos::rcp(new Teuchos::ParameterList(listname));;
  validPL->set<std::string>("Name", "", "String to designate Problem Class");
  validPL->set<int>("Number of Spatial Processors", -1, "Number of spatial processors in multi-level parallelism");
  validPL->set<bool>("Enable Cubit Shape Parameters", false, "Flag to enable shape change capability");
  validPL->set<std::string>("Cubit Base Filename", "", "Base name of three Cubit files");
  validPL->set<int>("Phalanx Graph Visualization Detail", 0,
                    "Flag to select outpuy of Phalanx Graph and level of detail");
  validPL->set<bool>("Use Physics-Based Preconditioner", false,
      "Flag to create signal that this problem will creat its own preconditioner");

  validPL->sublist("Initial Condition", false, "");
  validPL->sublist("Initial Condition Dot", false, "");
  validPL->sublist("Source Functions", false, "");
  validPL->sublist("Absorption", false, "");
  validPL->sublist("Response Functions", false, "");
  validPL->sublist("Parameters", false, "");
  validPL->sublist("Distributed Parameters", false, "");
  validPL->sublist("Teko", false, "");
  validPL->sublist("Dirichlet BCs", false, "");
  validPL->sublist("Neumann BCs", false, "");
  validPL->sublist("Adaptation", false, "");
  validPL->sublist("Catalyst", false, "");
  validPL->set<bool>("Solve Adjoint", false, "");

  validPL->set<bool>("Ignore Residual In Jacobian", false,
                     "Ignore residual calculations while computing the Jacobian (only generally appropriate for linear problems)");
  validPL->set<double>("Perturb Dirichlet", 0.0,
                     "Add this (small) perturbation to the diagonal to prevent Mass Matrices from being singular for Dirichlets)");

  validPL->sublist("Model Order Reduction", false, "Specify the options relative to model order reduction");

  // Candidates for deprecation. Pertain to the solution rather than the problem definition.
  validPL->set<std::string>("Solution Method", "Steady", "Flag for Steady, Transient, or Continuation");
  validPL->set<double>("Homotopy Restart Step", 1., "Flag for Felix Homotopy Restart Step");
  validPL->set<std::string>("Second Order", "No", "Flag to indicate that a transient problem has two time derivs");
  validPL->set<bool>("Print Response Expansion", true, "");

  // Deprecated parameters, kept solely for backward compatibility
  validPL->set<bool>("Compute Sensitivities", true, "Deprecated; Use parameter located under \"Piro\"/\"Analysis\"/\"Solve\" instead.");
  validPL->set<bool>("Stochastic", false, "Deprecated; Unused; Run using AlbanySG executable and specify SG parameters under \"Piro\"");
  validPL->sublist("Stochastic Galerkin", false, "Deprecated; Unused; Run using AlbanySG executable and specify SG parameters under \"Piro\"");

  return validPL;
}
