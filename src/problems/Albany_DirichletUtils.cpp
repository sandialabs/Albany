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

#include "Albany_DirichletUtils.hpp"
#include "PHAL_FactoryTraits.hpp"

Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> > 
Albany::DirichletUtils::constructDirichletEvaluators(
      const std::vector<std::string>& nodeSetIDs,
      const std::vector<std::string>& dirichletNames,
      Teuchos::RCP<Teuchos::ParameterList> params,
      Teuchos::RCP<ParamLib> paramLib)
{
   using Teuchos::RCP;
   using Teuchos::rcp;
   using Teuchos::ParameterList;
   using PHX::DataLayout;
   using PHX::MDALayout;
   using std::vector;
   using std::string;

   using PHAL::AlbanyTraits;

   int type = PHAL::DirichletFactoryTraits<AlbanyTraits>::id_dirichlet;
   
   Teuchos::ParameterList DBCparams = params->sublist("Dirichlet BCs");
   DBCparams.validateParameters(*(getValidDirichletBCParameters(nodeSetIDs,dirichletNames)),0);

   std::map<string, RCP<ParameterList> > evaluators_to_build;
   RCP<DataLayout> dummy = rcp(new MDALayout<Dummy>(0));
   vector<string> dbcs;

   // Check for all possible standard BCs (every dof on every nodeset) to see which is set
   for (std::size_t i=0; i<nodeSetIDs.size(); i++) {
     for (std::size_t j=0; j<dirichletNames.size(); j++) {
       std::string ss = constructDBCName(nodeSetIDs[i],dirichletNames[j]);

       if (DBCparams.isParameter(ss)) {
         RCP<ParameterList> p = rcp(new ParameterList);
         p->set<int>("Type", type);

         p->set< RCP<DataLayout> >("Data Layout", dummy);
         p->set< string >  ("Dirichlet Name", ss);
         p->set< RealType >("Dirichlet Value", DBCparams.get<double>(ss));
         p->set< string >  ("Node Set ID", nodeSetIDs[i]);
        // p->set< int >     ("Number of Equations", dirichletNames.size());
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
       // Only for LCM problems, but no harm in leaving off ifdefs
       if (sub_list.get<string>("BC Function") == "Kfield" )
       {
	 RCP<ParameterList> p = rcp(new ParameterList);
	 type = PHAL::DirichletFactoryTraits<AlbanyTraits>::id_kfield_bc;
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
         //p->set< int >     ("Number of Equations", dirichletNames.size());
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
      type = PHAL::DirichletFactoryTraits<AlbanyTraits>::id_dirichlet_aggregator;
      p->set<int>("Type", type);

      p->set<vector<string>* >("DBC Names", &dbcs);
      p->set< RCP<DataLayout> >("Data Layout", dummy);
      p->set<string>("DBC Aggregator Name", allDBC);
      evaluators_to_build[allDBC] = p;
   }

   // Build Field Evaluators for each evaluation type
   PHX::EvaluatorFactory<AlbanyTraits,PHAL::DirichletFactoryTraits<AlbanyTraits> > factory;
   RCP< vector< RCP<PHX::Evaluator_TemplateManager<AlbanyTraits> > > > evaluators;
   evaluators = factory.buildEvaluators(evaluators_to_build);

   // Create a DirichletFieldManager
   Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> > dfm
     = Teuchos::rcp(new PHX::FieldManager<AlbanyTraits>);

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

   return dfm;
}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::DirichletUtils::getValidDirichletBCParameters(
  const std::vector<std::string>& nodeSetIDs,
  const std::vector<std::string>& dirichletNames) const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
     Teuchos::rcp(new Teuchos::ParameterList("Valid Dirichlet BC List"));;

  for (std::size_t i=0; i<nodeSetIDs.size(); i++) {
    for (std::size_t j=0; j<dirichletNames.size(); j++) {
      std::string ss = constructDBCName(nodeSetIDs[i],dirichletNames[j]);
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
Albany::DirichletUtils::constructDBCName(const std::string ns,
                                       const std::string dof) const
{
  std::stringstream ss; ss << "DBC on NS " << ns << " for DOF " << dof;
  return ss.str();
}
