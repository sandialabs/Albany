/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/

#include "Albany_AbstractProblem.hpp"

// Generic implementations that can be used by derived problems

Albany::AbstractProblem::AbstractProblem(
         const Teuchos::RCP<Teuchos::ParameterList>& params_,
         const Teuchos::RCP<ParamLib>& paramLib_,
         const int neq_) :
  out(Teuchos::VerboseObjectBase::getDefaultOStream()),
  neq(neq_),
  params(params_),
  paramLib(paramLib_)
{}

unsigned int 
Albany::AbstractProblem::numEquations() const 
{
  TEST_FOR_EXCEPTION( neq <= 0,
                    Teuchos::Exceptions::InvalidParameter,
                    "A Problem must have at least 1 equation: "<<neq);
  return neq;
}

void
Albany::AbstractProblem::setNumEquations(const int neq_)
{ neq = neq_; }

Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> >
Albany::AbstractProblem::getFieldManager()
{ return fm; }

Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> >
Albany::AbstractProblem::getDirichletFieldManager()
{ return dfm; }

Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> >
Albany::AbstractProblem::getResponseFieldManager()
{ return rfm; }

Teuchos::RCP<Teuchos::ParameterList>
Albany::AbstractProblem::getGenericProblemParams(std::string listname) const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
     Teuchos::rcp(new Teuchos::ParameterList(listname));;
  validPL->set<std::string>("Name", "", "String to designate Problem Class");
  validPL->set<int>("Number of Spatial Processors", -1, "Number of spatial processors in multi-level parallelism");
  validPL->set<std::string>("Solution Method", "Steady", "Flag for Steady, Transient, or Continuation");
  validPL->set<std::string>("Second Order", "No", "Flag to indicate that a transient problem has two time derivs");
  validPL->set<bool>("Stochastic", false, "Flag to indicate a StochasticGalerkin problem");
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
  validPL->sublist("Stochastic Galerkin", false, "");
  validPL->sublist("Teko", false, "");
  validPL->sublist("Dirichlet BCs", false, "");
  validPL->set<bool>("Solve Adjoint", false, "");

  validPL->set<bool>("Ignore Residual In Jacobian", false, 
		     "Ignore residual calculations while computing the Jacobian (only generally appropriate for linear problems)");
  validPL->set<double>("Perturb Dirichlet", 0.0, 
		     "Add this (small) perturbation to the diagonal to prevent Mass Matrices from being singular for Dirichlets)");

  return validPL;
}
