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

// JTO note: I do not like this, I am open to suggestions on a better way
#ifndef ALBANY_LCM
#include "PHAL_FactoryTraits.hpp"
#else
#include "LCM/LCM_FactoryTraits.hpp"
#endif

// Generic implementations that can be used by derived problems

Albany::AbstractProblem::AbstractProblem(
         const Teuchos::RCP<Teuchos::ParameterList>& params_,
         const Teuchos::RCP<ParamLib>& paramLib_,
         const int neq_) :
  out(Teuchos::VerboseObjectBase::getDefaultOStream()),
  neq(neq_),
  nstates(0),
  params(params_),
  DBCparams(params_->sublist("Dirichlet BCs")),
  paramLib(paramLib_)
{}

unsigned int 
Albany::AbstractProblem::numEquations() const 
{return neq;}

unsigned int 
Albany::AbstractProblem::numStates() const 
{return nstates;}

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
  validPL->set<int>("Workset Size", 0,
                    "Choose size of elements to be processed together (0 for all at once)");

  validPL->sublist("Initial Condition", false, "");
  validPL->sublist("Initial Condition Dot", false, "");
  validPL->sublist("Source Functions", false, "");
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

   // JTO note: I do not like this, I am open to suggestions on a better way
#ifndef ALBANY_LCM
   using PHAL::FactoryTraits;
#else
   using LCM::FactoryTraits;
#endif
   using PHAL::AlbanyTraits;

   DBCparams.validateParameters(*(getValidDirichletBCParameters(nodeSetIDs)),0);

   map<string, RCP<ParameterList> > evaluators_to_build;
   RCP<DataLayout> dummy = rcp(new MDALayout<Dummy>(0));
   vector<string> dbcs;

   // Check for all possible standard BCs (every dof on every nodeset) to see which is set
   for (std::size_t i=0; i<nodeSetIDs.size(); i++) {
     for (std::size_t j=0; j<dofNames.size(); j++) {
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

   for (std::size_t i=0; i<nodeSetIDs.size(); i++) 
   {
     std::string ss = constructDBCName(nodeSetIDs[i],"K");
     
     if (DBCparams.isSublist(ss)) 
     {
       // grab the sublist
       ParameterList& sub_list = DBCparams.sublist(ss);

#ifdef ALBANY_LCM       
       if (sub_list.get<string>("BC Function") == "Kfield" )
       {
	 RCP<ParameterList> p = rcp(new ParameterList);
	 int type = FactoryTraits<AlbanyTraits>::id_kfield_bc;
	 p->set<int>("Type", type);

	 // This BC needs a shear modulus and poissons ratio defined
	 TEST_FOR_EXCEPTION(!params->isSublist("Shear Modulus"), 
			    Teuchos::Exceptions::InvalidParameter, 
			    "This BC needs a Shear Modulus");
	 ParameterList& shmd_list = params->sublist("Shear Modulus");
	 TEST_FOR_EXCEPTION(!(shmd_list.get("Shear Modulus Type","") == "Constant"), 
			    Teuchos::Exceptions::InvalidParameter,
			    "Invalid Shear Modulus type");
	 p->set< RealType >("Shear Modulus", shmd_list.get("Value", 1.0));

	 TEST_FOR_EXCEPTION(!params->isSublist("Poissons Ratio"), 
			    Teuchos::Exceptions::InvalidParameter, 
			    "This BC needs a Poissons Ratio");
	 ParameterList& pr_list = params->sublist("Poissons Ratio");
	 TEST_FOR_EXCEPTION(!(pr_list.get("Poissons Ratio Type","") == "Constant"), 
			    Teuchos::Exceptions::InvalidParameter,
			    "Invalid Poissons Ratio type");
	 p->set< RealType >("Poissons Ratio", pr_list.get("Value", 1.0));

	 // Extract BC parameters
	 p->set< string >("Kfield KI Name", "Kfield KI");
	 p->set< string >("Kfield KII Name", "Kfield KII");
	 p->set< RealType >("KI Value", sub_list.get<double>("Kfield KI"));
	 p->set< RealType >("KII Value", sub_list.get<double>("Kfield KII"));

	 // Fill up ParameterList with things DirichletBase wants
	 p->set< RCP<DataLayout> >("Data Layout", dummy);
	 p->set< string >  ("Dirichlet Name", ss);
         p->set< RealType >("Dirichlet Value", 0.0);
	 p->set< string >  ("Node Set ID", nodeSetIDs[i]);
         p->set< int >     ("Number of Equations", dofNames.size());
	 p->set< int >     ("Equation Offset", 0);
	 
	 p->set<RCP<ParamLib> >("Parameter Library", paramLib);
	 std::stringstream ess; ess << "Evaluator for " << ss;
	 evaluators_to_build[ess.str()] = p;

	 dbcs.push_back(ss);
       }
#endif
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

   PHX::Tag<AlbanyTraits::MPResidual::ScalarT> mpres_tag0(allDBC, dummy);
   dfm->requireField<AlbanyTraits::MPResidual>(mpres_tag0);

   PHX::Tag<AlbanyTraits::MPJacobian::ScalarT> mpjac_tag0(allDBC, dummy);
   dfm->requireField<AlbanyTraits::MPJacobian>(mpjac_tag0);
}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::AbstractProblem::getValidDirichletBCParameters(
  const std::vector<std::string>& nodeSetIDs) const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
     Teuchos::rcp(new Teuchos::ParameterList("Valid Dirichlet BC List"));;

  for (std::size_t i=0; i<nodeSetIDs.size(); i++) {
    for (std::size_t j=0; j<dofNames.size(); j++) {
      std::string ss = constructDBCName(nodeSetIDs[i],dofNames[j]);
      validPL->set<double>(ss, 0.0, "Value of BC corresponding to nodeSetID and dofName");
    }
  }
  
  for (std::size_t i=0; i<nodeSetIDs.size(); i++) 
  {
    std::string ss = constructDBCName(nodeSetIDs[i],"K");
    validPL->sublist(ss, false, "");
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

Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > >
Albany::AbstractProblem::getIntrepidBasis(const CellTopologyData& ctd, bool compositeTet)
{
   using Teuchos::rcp;
   using Intrepid::FieldContainer;
   Teuchos::RCP<Intrepid::Basis<RealType, FieldContainer<RealType> > > intrepidBasis;
   const int& numNodes = ctd.node_count;
   const int& numDim = ctd.dimension;
   std::string name = ctd.name;

   cout << "CellTopology is " << name << " with nodes " << numNodes << "  dim " << numDim << endl;

   if (name == "Line_2" )
       intrepidBasis = rcp(new Intrepid::Basis_HGRAD_LINE_C1_FEM<RealType, FieldContainer<RealType> >() );
// No HGRAD_LINE_C2 in Intrepid
   else if (name == "Line_3" )
       intrepidBasis = rcp(new Intrepid::Basis_HGRAD_LINE_Cn_FEM<RealType, FieldContainer<RealType> >(2, Intrepid::POINTTYPE_EQUISPACED) );
   else if (name == "Triangle_3" )
         intrepidBasis = rcp(new Intrepid::Basis_HGRAD_TRI_C1_FEM<RealType, FieldContainer<RealType> >() );
   else if (name == "Triangle_6" )
         intrepidBasis = rcp(new Intrepid::Basis_HGRAD_TRI_C2_FEM<RealType, FieldContainer<RealType> >() );
   else if (name == "Quadrilateral_4" )
         intrepidBasis = rcp(new Intrepid::Basis_HGRAD_QUAD_C1_FEM<RealType, FieldContainer<RealType> >() );
   else if (name == "Quadrilateral_9" )
         intrepidBasis = rcp(new Intrepid::Basis_HGRAD_QUAD_C2_FEM<RealType, FieldContainer<RealType> >() );
   else if (name == "Hexahedron_8" )
         intrepidBasis = rcp(new Intrepid::Basis_HGRAD_HEX_C1_FEM<RealType, FieldContainer<RealType> >() );
   else if (name == "Hexahedron_27" )
         intrepidBasis = rcp(new Intrepid::Basis_HGRAD_HEX_C2_FEM<RealType, FieldContainer<RealType> >() );
   else if (name == "Tetrahedron_4" )
           intrepidBasis = rcp(new Intrepid::Basis_HGRAD_TET_C1_FEM<RealType, FieldContainer<RealType> >() );
   else if (name == "Tetrahedron_10" && !compositeTet )
           intrepidBasis = rcp(new Intrepid::Basis_HGRAD_TET_C2_FEM<RealType, FieldContainer<RealType> >() );
   else if (name == "Tetrahedron_10" && compositeTet )
           intrepidBasis = rcp(new Intrepid::Basis_HGRAD_TET_COMP12_FEM<RealType, FieldContainer<RealType> >() );
   else
     TEST_FOR_EXCEPTION( //JTO compiler doesn't like this --> ctd.name != "Recognized Element Name", 
			true,
			Teuchos::Exceptions::InvalidParameter,
			"Albany::AbstractProblem::getIntrepidBasis did not recognize element name: "
			<< ctd.name);

   return intrepidBasis;
}
