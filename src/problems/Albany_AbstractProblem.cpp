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
#include "PHAL_FactoryTraits.hpp"

// Generic implementations that can be used by derived problems

Albany::AbstractProblem::AbstractProblem(
         const Teuchos::RCP<Teuchos::ParameterList>& params_,
         const Teuchos::RCP<ParamLib>& paramLib_,
         const int neq_) :
    params(params_),
    paramLib(paramLib_),
    DBCparams(params_->sublist("Dirichlet BCs")),
    neq(neq_),
    out(Teuchos::VerboseObjectBase::getDefaultOStream()) 
{}

unsigned int 
Albany::AbstractProblem::numEquations() const 
{return neq;}


Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> >
Albany::AbstractProblem::getFieldManager()
{ return fm; }


Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> >
Albany::AbstractProblem::getDirichletFieldManager()
{ return dfm; }

Teuchos::RCP<Teuchos::ParameterList>
Albany::AbstractProblem::getGenericProblemParams(std::string listname) const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
     Teuchos::rcp(new Teuchos::ParameterList(listname));;
  validPL->set<std::string>("Name", "", "String to designate Problem Class");
  validPL->set<bool>("Transient", false, "Flag to indicate time-dependent problem");
  validPL->set<bool>("Continuation", false, "Flag to indicate a continuation problem");
  validPL->set<bool>("Stochastic", false, "Flag to indicate a StochasticGalerkin problem");
  validPL->set<bool>("Enable Cubit Shape Parameters", false, "Flag to enable shape change capability");
  validPL->set<std::string>("Cubit Base Filename", "", "Base name of three Cubit files");
  validPL->set<int>("Phalanx Graph Visualization Detail", 0,
                    "Flag to select outpuy of Phalanx Graph and level of detail");
  validPL->set<bool>("Use Physics-Based Preconditioner", false, 
      "Flag to create signal that this problem will creat its own preconditioner");
  validPL->set<int>("Workset Size", 0,
                    "Choose size of elements to be processed together (0 for all at once)");

  validPL->sublist("Initial Condition", false, "");
  validPL->sublist("Source Functions", false, "");
  validPL->sublist("Response Functions", false, "");
  validPL->sublist("Parameters", false, "");
  validPL->sublist("Stochastic Galerkin", false, "");
  validPL->sublist("Teko", false, "");
  validPL->sublist("Dirichlet BCs", false, "");

  return validPL;
}

void
Albany::AbstractProblem::constructDirichletEvaluators(
  const std::vector<std::string>& nodeSetIDs)
{
   using Teuchos::RCP;
   using Teuchos::rcp;
   using Teuchos::ParameterList;
   using PHX::DataLayout;
   using PHX::MDALayout;
   using std::vector;
   using std::map;
   using std::string;
   using PHAL::FactoryTraits;
   using PHAL::AlbanyTraits;

   DBCparams.validateParameters(*(getValidDirichletBCParameters(nodeSetIDs)),0);

   map<string, RCP<ParameterList> > evaluators_to_build;
   RCP<DataLayout> dummy = rcp(new MDALayout<Dummy>(0));
   vector<string> dbcs;

   // Check for all possible BCs (every dof on every nodeset) to see which is set
   for (int i=0; i<nodeSetIDs.size(); i++) {
     for (int j=0; j<dofNames.size(); j++) {
       std::string ss = constructDBCName(nodeSetIDs[i],dofNames[j]);

       if (DBCparams.isParameter(ss)) {
         RCP<ParameterList> p = rcp(new ParameterList);
         int type = FactoryTraits<AlbanyTraits>::id_dirichlet;
         p->set<int>("Type", type);

         p->set< RCP<DataLayout> >("Data Layout", dummy);
         p->set< string >  ("Dirichlet Name", ss);
         p->set< RealType >("Dirichlet Value", DBCparams.get<double>(ss));
         p->set< string >  ("Node Set ID", nodeSetIDs[i]);
         p->set< int >     ("Number of Equations", dofNames.size());
         p->set< int >     ("Equation Offset", j);
         p->set<RCP<ParamLib> >("Parameter Library", paramLib);

         std::stringstream ess; ess << "Evaluator for " << ss;
         evaluators_to_build[ess.str()] = p;

         dbcs.push_back(ss);
       }
     }
   }
   string allDBC="Evaluator for all Dirichlet BCs";
   {
      RCP<ParameterList> p = rcp(new ParameterList);
      int type = FactoryTraits<AlbanyTraits>::id_dirichlet_aggregator;
      p->set<int>("Type", type);

      p->set<vector<string>* >("DBC Names", &dbcs);
      p->set< RCP<DataLayout> >("Data Layout", dummy);
      p->set<string>("DBC Aggregator Name", allDBC);
      evaluators_to_build[allDBC] = p;
   }

   // Build Field Evaluators for each evaluation type
   PHX::EvaluatorFactory<AlbanyTraits,FactoryTraits<AlbanyTraits> > factory;
   RCP< vector< RCP<PHX::Evaluator_TemplateManager<AlbanyTraits> > > > evaluators;
   evaluators = factory.buildEvaluators(evaluators_to_build);

   // Create a DirichletFieldManager
   dfm = Teuchos::rcp(new PHX::FieldManager<AlbanyTraits>);

   // Register all Evaluators
   PHX::registerEvaluators(evaluators, *dfm);

   PHX::Tag<AlbanyTraits::Residual::ScalarT> res_tag0(allDBC, dummy);
   dfm->requireField<AlbanyTraits::Residual>(res_tag0);

   PHX::Tag<AlbanyTraits::Jacobian::ScalarT> jac_tag0(allDBC, dummy);
   dfm->requireField<AlbanyTraits::Jacobian>(jac_tag0);

   PHX::Tag<AlbanyTraits::Tangent::ScalarT> tan_tag0(allDBC, dummy);
   dfm->requireField<AlbanyTraits::Tangent>(tan_tag0);

   PHX::Tag<AlbanyTraits::SGResidual::ScalarT> sgres_tag0(allDBC, dummy);
   dfm->requireField<AlbanyTraits::SGResidual>(sgres_tag0);

   PHX::Tag<AlbanyTraits::SGJacobian::ScalarT> sgjac_tag0(allDBC, dummy);
   dfm->requireField<AlbanyTraits::SGJacobian>(sgjac_tag0);
}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::AbstractProblem::getValidDirichletBCParameters(
  const std::vector<std::string>& nodeSetIDs) const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
     Teuchos::rcp(new Teuchos::ParameterList("Valid Dirichlet BC List"));;

  for (int i=0; i<nodeSetIDs.size(); i++) {
    for (int j=0; j<dofNames.size(); j++) {
      std::string ss = constructDBCName(nodeSetIDs[i],dofNames[j]);
      validPL->set<double>(ss, 0.0, "Value of BC corresponding to nodeSetID and dofName");
    }
  }

  return validPL;
}

std::string
Albany::AbstractProblem::constructDBCName(const std::string ns,
                                          const std::string dof) const
{
  std::stringstream ss; ss << "DBC on NS " << ns << " for DOF " << dof;
  return ss.str();
}
