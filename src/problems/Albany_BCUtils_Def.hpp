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

#include "Albany_BCUtils.hpp"

template<typename BCTraits>
Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> > 
Albany::BCUtils<BCTraits>::constructBCEvaluators(
      const std::vector<std::string>& nodeorsideSetIDs,
      const std::vector<std::string>& bcNames,
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

   if(!params->isSublist(traits_type::bcParamsPl)){ // If the BC sublist is not in the input file, this means that
      // node or side sets are in the Exodus file but not defined in the problem statement.This is OK, we
      // just don't build a bc field manager

      return Teuchos::null;

    }

   Teuchos::ParameterList BCparams = params->sublist(traits_type::bcParamsPl);
   BCparams.validateParameters(*(getValidBCParameters(nodeorsideSetIDs, bcNames)),0);

   std::map<string, RCP<ParameterList> > evaluators_to_build;
   RCP<DataLayout> dummy = rcp(new MDALayout<Dummy>(0));
   vector<string> bcs;

   // Check for all possible standard BCs (every dof on every nodeorsideset) to see which is set
   for (std::size_t i=0; i<nodeorsideSetIDs.size(); i++) {
     for (std::size_t j=0; j<bcNames.size(); j++) {
       std::string ss = constructBCName(nodeorsideSetIDs[i],bcNames[j]);

       if (BCparams.isParameter(ss)) {
         RCP<ParameterList> p = rcp(new ParameterList);
         p->set<int>("Type", traits_type::type);

         p->set< RCP<DataLayout> >("Data Layout", dummy);
         p->set< string >  ("Dirichlet Name", ss);
         p->set< RealType >("Dirichlet Value", BCparams.get<double>(ss));
         p->set< string >  ("Node Set ID", nodeorsideSetIDs[i]);
        // p->set< int >     ("Number of Equations", dirichletNames.size());
         p->set< int >     ("Equation Offset", j);

         p->set<RCP<ParamLib> >("Parameter Library", paramLib);

         std::stringstream ess; ess << "Evaluator for " << ss;
         evaluators_to_build[ess.str()] = p;

         bcs.push_back(ss);
       }
     }
   }

#ifdef ALBANY_LCM
   ///
   /// Time dependent BC specific
   ///
   for (std::size_t i=0; i<nodeorsideSetIDs.size(); i++) 
   {
     for (std::size_t j=0; j<bcNames.size(); j++) 
     {
       std::string ss = constructTimeDepBCName(nodeorsideSetIDs[i],bcNames[j]);
       
       if (BCparams.isSublist(ss)) 
       {
	 // grab the sublist 
         ParameterList& sub_list = BCparams.sublist(ss);
	 RCP<ParameterList> p = rcp(new ParameterList);
	 p->set<int>("Type", traits_type::typeTd);
	   
	 // Extract the time values into a vector
	 //vector<RealType> timeValues = sub_list.get<Teuchos::Array<RealType> >("Time Values").toVector();
	 //RCP< vector<RealType> > t_ptr = Teuchos::rcpFromRef(timeValues); 
	 //p->set< RCP< vector<RealType> > >("Time Values", t_ptr);
	 p->set< Teuchos::Array<RealType> >("Time Values", sub_list.get<Teuchos::Array<RealType> >("Time Values"));

	 //cout << "timeValues: " << timeValues[0] << " " << timeValues[1] << endl;

	 // Extract the BC values into a vector
	 //vector<RealType> BCValues = sub_list.get<Teuchos::Array<RealType> >("BC Values").toVector();
	 //RCP< vector<RealType> > b_ptr = Teuchos::rcpFromRef(BCValues); 
	 //p->set< RCP< vector<RealType> > >("BC Values", b_ptr);
	 //p->set< vector<RealType> >("BC Values", BCValues);
	 p->set< Teuchos::Array<RealType> >("BC Values", sub_list.get<Teuchos::Array<RealType> >("BC Values"));
         p->set< RCP<DataLayout> >("Data Layout", dummy);
	 p->set< string >  ("Dirichlet Name", ss);
         p->set< RealType >("Dirichlet Value", 0.0);
	 p->set< int >     ("Equation Offset", j);
	 p->set<RCP<ParamLib> >("Parameter Library", paramLib);
	 p->set< string >  ("Node Set ID", nodeorsideSetIDs[i]);

         std::stringstream ess; ess << "Evaluator for " << ss;
         evaluators_to_build[ess.str()] = p;

	 bcs.push_back(ss);
       }
     }
   }

   ///
   /// Torsion BC specific
   ///
   for (std::size_t i=0; i<nodeorsideSetIDs.size(); i++) 
   {
     std::string ss = constructBCName(nodeorsideSetIDs[i],"twist");
     
     if (BCparams.isSublist(ss)) 
     {
       // grab the sublist
       ParameterList& sub_list = BCparams.sublist(ss);

       if (sub_list.get<string>("BC Function") == "Torsion" )
       {
	 RCP<ParameterList> p = rcp(new ParameterList);
	 p->set<int>("Type", traits_type::typeTo);

         p->set< RealType >("Theta Dot", sub_list.get< RealType >("Theta Dot"));

	 // Fill up ParameterList with things DirichletBase wants
	 p->set< RCP<DataLayout> >("Data Layout", dummy);
	 p->set< string >  ("Dirichlet Name", ss);
         p->set< RealType >("Dirichlet Value", 0.0);
	 p->set< string >  ("Node Set ID", nodeorsideSetIDs[i]);
         //p->set< int >     ("Number of Equations", dirichletNames.size());
	 p->set< int >     ("Equation Offset", 0);
	 
	 p->set<RCP<ParamLib> >("Parameter Library", paramLib);
	 std::stringstream ess; ess << "Evaluator for " << ss;
	 evaluators_to_build[ess.str()] = p;

	 bcs.push_back(ss);
       }
     }
   }

   ///
   /// Kfield BC specific
   ///
   for (std::size_t i=0; i<nodeorsideSetIDs.size(); i++) 
   {
     std::string ss = constructBCName(nodeorsideSetIDs[i],"K");
     
     if (BCparams.isSublist(ss)) 
     {
       // grab the sublist
       ParameterList& sub_list = BCparams.sublist(ss);

       if (sub_list.get<string>("BC Function") == "Kfield" )
       {
	 RCP<ParameterList> p = rcp(new ParameterList);
	 p->set<int>("Type", traits_type::typeKf);

	 p->set< Teuchos::Array<RealType> >("Time Values", sub_list.get<Teuchos::Array<RealType> >("Time Values"));
	 p->set< Teuchos::Array<RealType> >("KI Values", sub_list.get<Teuchos::Array<RealType> >("KI Values"));
	 p->set< Teuchos::Array<RealType> >("KII Values", sub_list.get<Teuchos::Array<RealType> >("KII Values"));

	 // This BC needs a shear modulus and poissons ratio defined
	 TEUCHOS_TEST_FOR_EXCEPTION(!params->isSublist("Shear Modulus"), 
				    Teuchos::Exceptions::InvalidParameter, 
				    "This BC needs a Shear Modulus");
	 ParameterList& shmd_list = params->sublist("Shear Modulus");
	 TEUCHOS_TEST_FOR_EXCEPTION(!(shmd_list.get("Shear Modulus Type","") == "Constant"), 
				    Teuchos::Exceptions::InvalidParameter,
				    "Invalid Shear Modulus type");
	 p->set< RealType >("Shear Modulus", shmd_list.get("Value", 1.0));

	 TEUCHOS_TEST_FOR_EXCEPTION(!params->isSublist("Poissons Ratio"), 
				    Teuchos::Exceptions::InvalidParameter, 
				    "This BC needs a Poissons Ratio");
	 ParameterList& pr_list = params->sublist("Poissons Ratio");
	 TEUCHOS_TEST_FOR_EXCEPTION(!(pr_list.get("Poissons Ratio Type","") == "Constant"), 
				    Teuchos::Exceptions::InvalidParameter,
				    "Invalid Poissons Ratio type");
	 p->set< RealType >("Poissons Ratio", pr_list.get("Value", 1.0));


//	 p->set< Teuchos::Array<RealType> >("BC Values", sub_list.get<Teuchos::Array<RealType> >("BC Values"));
//	 p->set< RCP<DataLayout> >("Data Layout", dummy);

	 // Extract BC parameters
	 p->set< string >("Kfield KI Name", "Kfield KI");
	 p->set< string >("Kfield KII Name", "Kfield KII");
	 p->set< RealType >("KI Value", sub_list.get<double>("Kfield KI"));
	 p->set< RealType >("KII Value", sub_list.get<double>("Kfield KII"));

	 // Fill up ParameterList with things DirichletBase wants
	 p->set< RCP<DataLayout> >("Data Layout", dummy);
	 p->set< string >  ("Dirichlet Name", ss);
         p->set< RealType >("Dirichlet Value", 0.0);
	 p->set< string >  ("Node Set ID", nodeorsideSetIDs[i]);
         //p->set< int >     ("Number of Equations", dirichletNames.size());
	 p->set< int >     ("Equation Offset", 0);
	 
	 p->set<RCP<ParamLib> >("Parameter Library", paramLib);
	 std::stringstream ess; ess << "Evaluator for " << ss;
	 evaluators_to_build[ess.str()] = p;

	 bcs.push_back(ss);
       }
     }
   }
#endif

   string allBC="Evaluator for all Dirichlet BCs";
   {
      RCP<ParameterList> p = rcp(new ParameterList);
      p->set<int>("Type", traits_type::typeDa);

      p->set<vector<string>* >("DBC Names", &bcs);
      p->set< RCP<DataLayout> >("Data Layout", dummy);
      p->set<string>("DBC Aggregator Name", allBC);
      evaluators_to_build[allBC] = p;
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

   PHX::Tag<AlbanyTraits::Residual::ScalarT> res_tag0(allBC, dummy);
   dfm->requireField<AlbanyTraits::Residual>(res_tag0);

   PHX::Tag<AlbanyTraits::Jacobian::ScalarT> jac_tag0(allBC, dummy);
   dfm->requireField<AlbanyTraits::Jacobian>(jac_tag0);

   PHX::Tag<AlbanyTraits::Tangent::ScalarT> tan_tag0(allBC, dummy);
   dfm->requireField<AlbanyTraits::Tangent>(tan_tag0);

   PHX::Tag<AlbanyTraits::SGResidual::ScalarT> sgres_tag0(allBC, dummy);
   dfm->requireField<AlbanyTraits::SGResidual>(sgres_tag0);

   PHX::Tag<AlbanyTraits::SGJacobian::ScalarT> sgjac_tag0(allBC, dummy);
   dfm->requireField<AlbanyTraits::SGJacobian>(sgjac_tag0);

   PHX::Tag<AlbanyTraits::SGTangent::ScalarT> sgtan_tag0(allBC, dummy);
   dfm->requireField<AlbanyTraits::SGTangent>(sgtan_tag0);

   PHX::Tag<AlbanyTraits::MPResidual::ScalarT> mpres_tag0(allBC, dummy);
   dfm->requireField<AlbanyTraits::MPResidual>(mpres_tag0);

   PHX::Tag<AlbanyTraits::MPJacobian::ScalarT> mpjac_tag0(allBC, dummy);
   dfm->requireField<AlbanyTraits::MPJacobian>(mpjac_tag0);

   PHX::Tag<AlbanyTraits::MPTangent::ScalarT> mptan_tag0(allBC, dummy);
   dfm->requireField<AlbanyTraits::MPTangent>(mptan_tag0);

   return dfm;
}

template<typename BCTraits>
Teuchos::RCP<const Teuchos::ParameterList>
Albany::BCUtils<BCTraits>::getValidBCParameters(
  const std::vector<std::string>& nodeorsideSetIDs,
  const std::vector<std::string>& bcNames) const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
     Teuchos::rcp(new Teuchos::ParameterList("Valid Dirichlet BC List"));;

  for (std::size_t i=0; i<nodeorsideSetIDs.size(); i++) {
    for (std::size_t j=0; j<bcNames.size(); j++) {
      std::string ss = constructBCName(nodeorsideSetIDs[i],bcNames[j]);
      std::string tt = constructTimeDepBCName(nodeorsideSetIDs[i],bcNames[j]);
      validPL->set<double>(ss, 0.0, "Value of BC corresponding to nodeSetID and dofName");
      validPL->sublist(tt, false, "SubList of BC corresponding to nodeSetID and dofName");
    }
  }
  
  for (std::size_t i=0; i<nodeorsideSetIDs.size(); i++) 
  {
    std::string ss = constructBCName(nodeorsideSetIDs[i],"K");
    validPL->sublist(ss, false, "");
  }

  return validPL;
}

template<typename BCTraits>
std::string
Albany::BCUtils<BCTraits>::constructBCName(const std::string ns,
					 const std::string dof) const
{
  std::stringstream ss; ss << "DBC on NS " << ns << " for DOF " << dof;
  return ss.str();
}

template<typename BCTraits>
std::string
Albany::BCUtils<BCTraits>::constructTimeDepBCName(const std::string ns,
						const std::string dof) const
{
  std::stringstream ss; ss << "Time Dependent " << constructBCName(ns, dof);
  return ss.str();
}
