//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_AbstractProblem.hpp"
#include "NOX_StatusTest_Generic.H"

namespace Albany
{
// Generic implementations that can be used by derived problems

AbstractProblem::
AbstractProblem(
  const Teuchos::RCP<Teuchos::ParameterList>& params_,
  const Teuchos::RCP<ParamLib>& paramLib_,
  //const Teuchos::RCP<DistributedParameterLibrary>& distParamLib_,
  const int neq_) :
  out(Teuchos::VerboseObjectBase::getDefaultOStream()),
  neq(neq_),
  number_of_time_deriv(-1),
  SolutionMethodName(Unknown),
  params(params_),
  paramLib(paramLib_),
  rigidBodyModes(Teuchos::rcp(new RigidBodyModes()))
{

 /* 
  * Set the number of time derivatives. Semantics are to set the number of time derivatives:
  * x = 0, xdot = 1, xdotdot = 2
  * using the Discretization parameter "Number Of Time Derivatives" if this is specified, or if not
  * set it to zero if the problem is steady, or to one if it is transient. This needs to be overridden
  * in each problem is this logic is not sufficient.
  */

  /* Override this logic by specifying the below in the Discretization PL with

  <Parameter name="Number Of Time Derivatives" type="int" value="2"/>
  */

  std::string solutionMethod = params->get("Solution Method", "Steady");
  if(solutionMethod == "Steady")
  {
    number_of_time_deriv = 0;
    SolutionMethodName = Steady;
  } else if(solutionMethod == "Continuation") {
    number_of_time_deriv = 0;
    SolutionMethodName = Continuation;
  } else if(solutionMethod == "Transient") {
    number_of_time_deriv = 1;
    SolutionMethodName = Transient;
  } else
    TEUCHOS_TEST_FOR_EXCEPTION(true,
            std::logic_error, "Solution Method must be Steady, Transient, "
            << "Continuation, not : " << solutionMethod);

  // Set the number in the Problem PL
  params->set<int>("Number Of Time Derivatives", number_of_time_deriv);
}

unsigned int
AbstractProblem::numEquations() const
{
  TEUCHOS_TEST_FOR_EXCEPTION( neq <= 0,
                    Teuchos::Exceptions::InvalidParameter,
                    "A Problem must have at least 1 equation: "<<neq);
  return neq;
}

const std::map<int,std::vector<std::string> >&
AbstractProblem::getSideSetEquations() const
{
  return sideSetEquations;
}

void
AbstractProblem::setNumEquations(const int neq_)
{
  neq = neq_;
}

// Get the solution method type name
SolutionMethodType 
AbstractProblem::getSolutionMethod()
{
    return SolutionMethodName;
}

Teuchos::ArrayRCP<Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> > >
AbstractProblem::getFieldManager()
{ return fm; }

Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> >
AbstractProblem::getDirichletFieldManager()
{ return dfm; }

Teuchos::ArrayRCP<Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> > >
AbstractProblem::getNeumannFieldManager()
{ return nfm; }

Teuchos::RCP<Teuchos::ParameterList>
AbstractProblem::getGenericProblemParams(std::string listname) const
{
  auto validPL = Teuchos::rcp(new Teuchos::ParameterList(listname));;
  validPL->set<std::string>("Name", "", "String to designate Problem Class");
  validPL->set<int>("Number of Spatial Processors", -1, "Number of spatial processors in multi-level parallelism");
  validPL->set<int>("Phalanx Graph Visualization Detail", 0,
                    "Flag to select outpuy of Phalanx Graph and level of detail");
  validPL->set<bool>("Use Physics-Based Preconditioner", false,
                     "Flag to create signal that this problem will creat its own preconditioner");
  validPL->set<std::string>("Physics-Based Preconditioner", "None",
                            "Type of preconditioner that problem will create");

  validPL->sublist("Initial Condition", false, "");
  validPL->sublist("Initial Condition Dot", false, "");
  validPL->sublist("Initial Condition DotDot", false, "");
  validPL->sublist("Source Functions", false, "");
  validPL->sublist("Absorption", false, "");
  validPL->sublist("Response Functions", false, "");
  validPL->sublist("Parameters", false, "");
  validPL->sublist("Random Parameters", false, "");
  validPL->sublist("Linear Combination Parameters", false, "");
  validPL->sublist("LogNormal Parameter", false, "");
  validPL->sublist("Teko", false, "");
  validPL->sublist("Hessian", false, "");
  validPL->sublist("XFEM", false, "");
  validPL->sublist("Dirichlet BCs", false, "");
  validPL->sublist("Neumann BCs", false, "");
  validPL->sublist("Adaptation", false, "");
  validPL->set<bool>("Overwrite Nominal Values With Final Point",false,
                     "Whether 'reportFinalPoint' should be allowed to overwrite nominal values");
  validPL->set<int>("Number Of Time Derivatives", 1, "Number of time derivatives in use in the problem");

  validPL->set<bool>("Use MDField Memoization", false, "Use memoization to avoid recomputing MDFields");
  validPL->set<bool>("Use MDField Memoization For Parameters", false, "Use memoization to avoid recomputing MDFields dependent on parameters");
  validPL->set<bool>("Ignore Residual In Jacobian", false,
                     "Ignore residual calculations while computing the Jacobian (only generally appropriate for linear problems)");
  validPL->set<double>("Perturb Dirichlet", 0.0,
                     "Add this (small) perturbation to the diagonal to prevent Mass Matrices from being singular for Dirichlets)");

  // Candidates for deprecation. Pertain to the solution rather than the problem definition.
  validPL->set<std::string>("Solution Method", "Steady", "Flag for Steady, Transient, or Continuation");
  validPL->set<double>("Homotopy Restart Step", 1., "Flag for LandIce Homotopy Restart Step");
  validPL->set<std::string>("Second Order", "No", "Flag to indicate that a transient problem has two time derivs");
  validPL->set<bool>("Print Response Expansion", true, "");

  // Deprecated parameters, kept solely for backward compatibility
  validPL->set<bool>("Compute Sensitivities", true, "Deprecated; Use parameter located under \"Piro\"/\"Analysis\"/\"Solve\" instead.");

  validPL->set<int>("Cubature Degree", 3, "Cubature Degree");

  // NOX status test that allows constutive models to cut the global time step
  // needed at the Problem scope when running Schwarz coupling
  validPL->set<Teuchos::RCP<NOX::StatusTest::Generic>>("Constitutive Model NOX Status Test", Teuchos::RCP<NOX::StatusTest::Generic>(), "NOX status test that facilitates communication between a ModelEvaluator and a NOX solver");

  return validPL;
}

} // namespace Albany
