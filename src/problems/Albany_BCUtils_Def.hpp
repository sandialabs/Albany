//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#include "Albany_BCUtils.hpp"

// Dirichlet specialization

template<>
Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> >
Albany::BCUtils<Albany::DirichletTraits>::constructBCEvaluators(
  const std::vector<std::string>& nodeSetIDs,
  const std::vector<std::string>& bcNames,
  Teuchos::RCP<Teuchos::ParameterList> params,
  Teuchos::RCP<ParamLib> paramLib,
  int numEqn) {

  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::ParameterList;
  using PHX::DataLayout;
  using PHX::MDALayout;
  using std::vector;
  using std::string;

  using PHAL::AlbanyTraits;

  if(!haveBCSpecified(params)) { // If the BC sublist is not in the input file,
    // but we are inside this function, this means that
    // node sets are contained in the Exodus file but are not defined in the problem statement.This is OK, we
    // just don't do anything

    return Teuchos::null;

  }

  Teuchos::ParameterList BCparams = params->sublist(traits_type::bcParamsPl);
  BCparams.validateParameters(*(traits_type::getValidBCParameters(nodeSetIDs, bcNames)), 0);

  std::map<std::string, RCP<ParameterList> > evaluators_to_build;
  RCP<DataLayout> dummy = rcp(new MDALayout<Dummy>(0));
  vector<std::string> bcs;

  // Check for all possible standard BCs (every dof on every nodeset) to see which is set
  for(std::size_t i = 0; i < nodeSetIDs.size(); i++) {
    for(std::size_t j = 0; j < bcNames.size(); j++) {
      std::string ss = traits_type::constructBCName(nodeSetIDs[i], bcNames[j]);

      if(BCparams.isParameter(ss)) {
        RCP<ParameterList> p = rcp(new ParameterList);
        p->set<int>("Type", traits_type::type);

        p->set< RCP<DataLayout> >("Data Layout", dummy);
        p->set< std::string > ("Dirichlet Name", ss);
        p->set< RealType >("Dirichlet Value", BCparams.get<double>(ss));
        p->set< std::string > ("Node Set ID", nodeSetIDs[i]);
        // p->set< int >     ("Number of Equations", dirichletNames.size());
        p->set< int > ("Equation Offset", j);

        p->set<RCP<ParamLib> >("Parameter Library", paramLib);

        std::stringstream ess;
        ess << "Evaluator for " << ss;
        evaluators_to_build[ess.str()] = p;

        bcs.push_back(ss);
      }
    }
  }

  ///
  /// Apply a function based on a coordinate value to the boundary
  ////
  for(std::size_t i = 0; i < nodeSetIDs.size(); i++) {
    std::string ss = traits_type::constructBCName(nodeSetIDs[i], "CoordFunc");

    if(BCparams.isSublist(ss)) {
      // grab the sublist
      ParameterList& sub_list = BCparams.sublist(ss);

      // Directly apply the coordinate values at the boundary as a DBC (Laplace Beltrami mesh equations)
      if(sub_list.get<std::string>("BC Function") == "Identity") {

        RCP<ParameterList> p = rcp(new ParameterList);
        p->set<int>("Type", traits_type::typeFb);

        // Fill up ParameterList with things DirichletBase wants
        p->set< RCP<DataLayout> >("Data Layout", dummy);
        p->set< std::string > ("Dirichlet Name", ss);
        p->set< RealType > ("Dirichlet Value", 0.0);
        p->set< std::string > ("Node Set ID", nodeSetIDs[i]);
        p->set< int > ("Number of Equations", numEqn);
        p->set< int > ("Equation Offset", 0);

        p->set<RCP<ParamLib> > ("Parameter Library", paramLib);
        std::stringstream ess;
        ess << "Evaluator for " << ss;
        evaluators_to_build[ess.str()] = p;

        bcs.push_back(ss);
      }

      // Add other functional boundary conditions here. Note that Torsion could fit into this framework
    }
  }

#ifdef ALBANY_LCM

  ///
  /// Time dependent BC specific
  ///
  for(std::size_t i = 0; i < nodeSetIDs.size(); i++) {
    for(std::size_t j = 0; j < bcNames.size(); j++) {
      std::string ss = traits_type::constructTimeDepBCName(nodeSetIDs[i], bcNames[j]);

      if(BCparams.isSublist(ss)) {
        // grab the sublist
        ParameterList& sub_list = BCparams.sublist(ss);
        RCP<ParameterList> p = rcp(new ParameterList);
        p->set<int>("Type", traits_type::typeTd);

        // Extract the time values into a vector
        //vector<RealType> timeValues = sub_list.get<Teuchos::Array<RealType> >("Time Values").toVector();
        //RCP< vector<RealType> > t_ptr = Teuchos::rcpFromRef(timeValues);
        //p->set< RCP< vector<RealType> > >("Time Values", t_ptr);
        p->set< Teuchos::Array<RealType> >("Time Values", sub_list.get<Teuchos::Array<RealType> >("Time Values"));

        //cout << "timeValues: " << timeValues[0] << " " << timeValues[1] << std::endl;

        // Extract the BC values into a vector
        //vector<RealType> BCValues = sub_list.get<Teuchos::Array<RealType> >("BC Values").toVector();
        //RCP< vector<RealType> > b_ptr = Teuchos::rcpFromRef(BCValues);
        //p->set< RCP< vector<RealType> > >("BC Values", b_ptr);
        //p->set< vector<RealType> >("BC Values", BCValues);
        p->set< Teuchos::Array<RealType> >("BC Values", sub_list.get<Teuchos::Array<RealType> >("BC Values"));
        p->set< RCP<DataLayout> >("Data Layout", dummy);
        p->set< std::string > ("Dirichlet Name", ss);
        p->set< RealType >("Dirichlet Value", 0.0);
        p->set< int > ("Equation Offset", j);
        p->set<RCP<ParamLib> >("Parameter Library", paramLib);
        p->set< std::string > ("Node Set ID", nodeSetIDs[i]);
        p->set<int>("Cubature Degree", BCparams.get("Cubature Degree", 0)); //if set to zero, the cubature degree of the side will be set to that of the element

        std::stringstream ess;
        ess << "Evaluator for " << ss;
        evaluators_to_build[ess.str()] = p;

        bcs.push_back(ss);
      }
    }
  }

  ///
  /// Torsion BC specific
  ////
  for(std::size_t i = 0; i < nodeSetIDs.size(); i++) {
    std::string ss = traits_type::constructBCName(nodeSetIDs[i], "twist");

    if(BCparams.isSublist(ss)) {
      // grab the sublist
      ParameterList& sub_list = BCparams.sublist(ss);

      if(sub_list.get<std::string>("BC Function") == "Torsion") {
        RCP<ParameterList> p = rcp(new ParameterList);
        p->set<int>("Type", traits_type::typeTo);

        p->set< RealType >("Theta Dot", sub_list.get< RealType >("Theta Dot"));
        p->set< RealType >("X0", sub_list.get< RealType >("X0"));
        p->set< RealType >("Y0", sub_list.get< RealType >("Y0"));

        // Fill up ParameterList with things DirichletBase wants
        p->set< RCP<DataLayout> >("Data Layout", dummy);
        p->set< std::string > ("Dirichlet Name", ss);
        p->set< RealType >("Dirichlet Value", 0.0);
        p->set< std::string > ("Node Set ID", nodeSetIDs[i]);
        //p->set< int >     ("Number of Equations", dirichletNames.size());
        p->set< int > ("Equation Offset", 0);
        p->set<int>("Cubature Degree", BCparams.get("Cubature Degree", 0)); //if set to zero, the cubature degree of the side will be set to that of the element


        p->set<RCP<ParamLib> >("Parameter Library", paramLib);
        std::stringstream ess;
        ess << "Evaluator for " << ss;
        evaluators_to_build[ess.str()] = p;

        bcs.push_back(ss);
      }
    }
  }

  ///
  /// Schwarz BC specific
  ///
  for (std::size_t i = 0; i < nodeSetIDs.size(); i++) {

    std::string
    ss = traits_type::constructBCName(nodeSetIDs[i], "Schwarz");

    if (BCparams.isSublist(ss)) {
      // grab the sublist
      ParameterList &
      sub_list = BCparams.sublist(ss);

      if (sub_list.get<std::string>("BC Function") == "Schwarz") {

        RCP<ParameterList>
        p = rcp(new ParameterList);

        p->set<int>("Type", traits_type::typeSw);

        p->set<std::string>(
            "Coupled Block",
            sub_list.get<std::string>("Coupled Block")
        );

        // Fill up ParameterList with things DirichletBase wants
        p->set< RCP<DataLayout> >("Data Layout", dummy);
        p->set< std::string > ("Dirichlet Name", ss);
        p->set< RealType >("Dirichlet Value", 0.0);
        p->set< std::string > ("Node Set ID", nodeSetIDs[i]);
        p->set< int > ("Equation Offset", 0);
        // if set to zero, the cubature degree of the side
        // will be set to that of the element
        p->set<int>("Cubature Degree", BCparams.get("Cubature Degree", 0));
        p->set<RCP<ParamLib> >("Parameter Library", paramLib);

        std::stringstream
        ess;

        ess << "Evaluator for " << ss;
        evaluators_to_build[ess.str()] = p;

        bcs.push_back(ss);
      }
    }
  }

  ///
  /// Kfield BC specific
  ///
  for(std::size_t i = 0; i < nodeSetIDs.size(); i++) {
    std::string ss = traits_type::constructBCName(nodeSetIDs[i], "K");

    if(BCparams.isSublist(ss)) {
      // grab the sublist
      ParameterList& sub_list = BCparams.sublist(ss);

      if(sub_list.get<std::string>("BC Function") == "Kfield") {
        RCP<ParameterList> p = rcp(new ParameterList);
        p->set<int>("Type", traits_type::typeKf);

        p->set< Teuchos::Array<RealType> >("Time Values", sub_list.get<Teuchos::Array<RealType> >("Time Values"));
        p->set< Teuchos::Array<RealType> >("KI Values", sub_list.get<Teuchos::Array<RealType> >("KI Values"));
        p->set< Teuchos::Array<RealType> >("KII Values", sub_list.get<Teuchos::Array<RealType> >("KII Values"));

        // // This BC needs a shear modulus and poissons ratio defined
        // TEUCHOS_TEST_FOR_EXCEPTION(!params->isSublist("Shear Modulus"),
        //               Teuchos::Exceptions::InvalidParameter,
        //               "This BC needs a Shear Modulus");
        // ParameterList& shmd_list = params->sublist("Shear Modulus");
        // TEUCHOS_TEST_FOR_EXCEPTION(!(shmd_list.get("Shear Modulus Type","") == "Constant"),
        //               Teuchos::Exceptions::InvalidParameter,
        //               "Invalid Shear Modulus type");
        // p->set< RealType >("Shear Modulus", shmd_list.get("Value", 1.0));

        // TEUCHOS_TEST_FOR_EXCEPTION(!params->isSublist("Poissons Ratio"),
        //               Teuchos::Exceptions::InvalidParameter,
        //               "This BC needs a Poissons Ratio");
        // ParameterList& pr_list = params->sublist("Poissons Ratio");
        // TEUCHOS_TEST_FOR_EXCEPTION(!(pr_list.get("Poissons Ratio Type","") == "Constant"),
        //               Teuchos::Exceptions::InvalidParameter,
        //               "Invalid Poissons Ratio type");
        // p->set< RealType >("Poissons Ratio", pr_list.get("Value", 1.0));


        //   p->set< Teuchos::Array<RealType> >("BC Values", sub_list.get<Teuchos::Array<RealType> >("BC Values"));
        //   p->set< RCP<DataLayout> >("Data Layout", dummy);

        // Extract BC parameters
        p->set< std::string >("Kfield KI Name", "Kfield KI");
        p->set< std::string >("Kfield KII Name", "Kfield KII");
        p->set< RealType >("KI Value", sub_list.get<double>("Kfield KI"));
        p->set< RealType >("KII Value", sub_list.get<double>("Kfield KII"));
        p->set< RealType >("Shear Modulus", sub_list.get<double>("Shear Modulus"));
        p->set< RealType >("Poissons Ratio", sub_list.get<double>("Poissons Ratio"));
        p->set<int>("Cubature Degree", BCparams.get("Cubature Degree", 0)); //if set to zero, the cubature degree of the side will be set to that of the element


        // Fill up ParameterList with things DirichletBase wants
        p->set< RCP<DataLayout> >("Data Layout", dummy);
        p->set< std::string > ("Dirichlet Name", ss);
        p->set< RealType >("Dirichlet Value", 0.0);
        p->set< std::string > ("Node Set ID", nodeSetIDs[i]);
        //p->set< int >     ("Number of Equations", dirichletNames.size());
        p->set< int > ("Equation Offset", 0);

        p->set<RCP<ParamLib> >("Parameter Library", paramLib);
        std::stringstream ess;
        ess << "Evaluator for " << ss;
        evaluators_to_build[ess.str()] = p;

        bcs.push_back(ss);
      }
    }
  }

#endif

  std::string allBC = "Evaluator for all Dirichlet BCs";
  {
    RCP<ParameterList> p = rcp(new ParameterList);
    p->set<int>("Type", traits_type::typeDa);

    p->set<vector<std::string>* >("DBC Names", &bcs);
    p->set< RCP<DataLayout> >("Data Layout", dummy);
    p->set<std::string>("DBC Aggregator Name", allBC);
    evaluators_to_build[allBC] = p;
  }

  return buildFieldManager(evaluators_to_build, allBC, dummy);
}


template<>
Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> >
Albany::BCUtils<Albany::NeumannTraits>::constructBCEvaluators(
  const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs,
  const std::vector<std::string>& bcNames,
  const Teuchos::ArrayRCP<std::string>& dof_names,
  bool isVectorField,
  int offsetToFirstDOF,
  const std::vector<std::string>& conditions,
  const Teuchos::Array<Teuchos::Array<int> >& offsets,
  const Teuchos::RCP<Albany::Layouts>& dl,
  Teuchos::RCP<Teuchos::ParameterList> params,
  Teuchos::RCP<ParamLib> paramLib,
  const Teuchos::RCP<QCAD::MaterialDatabase>& materialDB) {
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::ParameterList;
  using PHX::DataLayout;
  using PHX::MDALayout;
  using std::vector;
  using std::string;

  using PHAL::AlbanyTraits;

  // Drop into the "Neumann BCs" sublist
  Teuchos::ParameterList BCparams = params->sublist(traits_type::bcParamsPl);
  BCparams.validateParameters(*(traits_type::getValidBCParameters(meshSpecs->ssNames, bcNames, conditions)), 0);


  std::map<std::string, RCP<ParameterList> > evaluators_to_build;
  vector<std::string> bcs;

  // Check for all possible standard BCs (every dof on every sideset) to see which is set
  for(std::size_t i = 0; i < meshSpecs->ssNames.size(); i++) {
    for(std::size_t j = 0; j < bcNames.size(); j++) {
      for(std::size_t k = 0; k < conditions.size(); k++) {

        // construct input.xml string like:
        // "NBC on SS sidelist_12 for DOF T set dudn"
        //  or
        // "NBC on SS sidelist_12 for DOF T set (dudx, dudy)"
        // or
        // "NBC on SS surface_1 for DOF all set P"

        std::string ss = traits_type::constructBCName(meshSpecs->ssNames[i], bcNames[j], conditions[k]);

        // Have a match of the line in input.xml

        if(BCparams.isParameter(ss)) {

          //           std::cout << "Constructing NBC: " << ss << std::endl;

          TEUCHOS_TEST_FOR_EXCEPTION(BCparams.isType<std::string>(ss), std::logic_error,
                                     "NBC array information in XML file must be of type Array(double)\n");

          // These are read in the Albany::Neumann constructor (PHAL_Neumann_Def.hpp)

          RCP<ParameterList> p = rcp(new ParameterList);

          p->set<int> ("Type", traits_type::type);

          p->set<RCP<ParamLib> > ("Parameter Library", paramLib);

          p->set<std::string> ("Side Set ID", meshSpecs->ssNames[i]);
          p->set<Teuchos::Array< int > > ("Equation Offset", offsets[j]);
          p->set< RCP<Albany::Layouts> > ("Layouts Struct", dl);
          p->set< RCP<MeshSpecsStruct> > ("Mesh Specs Struct", meshSpecs);

          p->set<std::string> ("Coordinate Vector Name", "Coord Vec");
          p->set<int>("Cubature Degree", BCparams.get("Cubature Degree", 0)); //if set to zero, the cubature degree of the side will be set to that of the element

          if(conditions[k] == "robin") {
            p->set<std::string> ("DOF Name", dof_names[j]);
            p->set<bool> ("Vector Field", isVectorField);

            if(isVectorField) {
              p->set< RCP<DataLayout> >("DOF Data Layout", dl->node_vector);
            }

            else               p->set< RCP<DataLayout> >("DOF Data Layout", dl->node_scalar);
          }
#ifdef ALBANY_FELIX
          else if(conditions[k] == "basal") {
            std::string betaName = BCparams.get("BetaXY", "Constant");
            double L = BCparams.get("L", 1.0);
            p->set<std::string> ("BetaXY", betaName);
            p->set<string>("Beta Field Name", "Basal Friction");
            p->set<double> ("L", L);
            p->set<std::string> ("DOF Name", dof_names[0]);
            p->set<bool> ("Vector Field", isVectorField);
            if (isVectorField) p->set< RCP<DataLayout> >("DOF Data Layout", dl->node_vector);
            else               p->set< RCP<DataLayout> >("DOF Data Layout", dl->node_scalar);
          }
          else if(conditions[k] == "basal_scalar_field") {
            p->set<string>("Beta Field Name", "Basal Friction");
            p->set<std::string> ("DOF Name", dof_names[0]);
            p->set<bool> ("Vector Field", isVectorField);
            if (isVectorField) p->set< RCP<DataLayout> >("DOF Data Layout", dl->node_vector);
            else               p->set< RCP<DataLayout> >("DOF Data Layout", dl->node_scalar);
          }
          else if(conditions[k] == "lateral") {
            std::string betaName = BCparams.get("BetaXY", "Constant");
            double L = BCparams.get("L", 1.0);
            p->set<std::string>("Thickness Field Name", "Thickness");
            p->set<std::string>("Elevation Field Name", "Surface Height");
            p->set<std::string>  ("DOF Name", dof_names[0]);
           	p->set<bool> ("Vector Field", isVectorField);
           	if (isVectorField) {p->set< RCP<DataLayout> >("DOF Data Layout", dl->node_vector);}
            else               p->set< RCP<DataLayout> >("DOF Data Layout", dl->node_scalar);
          }
#endif

          // Pass the input file line
          p->set< std::string > ("Neumann Input String", ss);
          p->set< Teuchos::Array<double> > ("Neumann Input Value", BCparams.get<Teuchos::Array<double> >(ss));
          p->set< std::string > ("Neumann Input Conditions", conditions[k]);

          // If we are doing a Neumann internal boundary with a "scaled jump" (includes "robin" too)
          // The material DB database needs to be passed to the BC object

          if(conditions[k] == "scaled jump" || conditions[k] == "robin") {

            TEUCHOS_TEST_FOR_EXCEPTION(materialDB == Teuchos::null,
                                       Teuchos::Exceptions::InvalidParameter,
                                       "This BC needs a material database specified");

            p->set< RCP<QCAD::MaterialDatabase> >("MaterialDB", materialDB);


          }


          // Inputs: X, Y at nodes, Cubature, and Basis
          //p->set<std::string>("Node Variable Name", "Neumann");

          std::stringstream ess;
          ess << "Evaluator for " << ss;
          evaluators_to_build[ess.str()] = p;


          bcs.push_back(ss);
        }
      }
    }
  }

#ifdef ALBANY_LCM

  ///
  /// Time dependent BC specific
  ///
  for(std::size_t i = 0; i < meshSpecs->ssNames.size(); i++) {
    for(std::size_t j = 0; j < bcNames.size(); j++) {
      for(std::size_t k = 0; k < conditions.size(); k++) {

        // construct input.xml string like:
        // "Time Dependent NBC on SS sidelist_12 for DOF T set dudn"
        //  or
        // "Time Dependent NBC on SS sidelist_12 for DOF T set (dudx, dudy)"
        // or
        // "Time Dependent NBC on SS surface_1 for DOF all set P"

        std::string ss = traits_type::constructTimeDepBCName(meshSpecs->ssNames[i], bcNames[j], conditions[k]);

        // Have a match of the line in input.xml

        if(BCparams.isSublist(ss)) {

          // grab the sublist
          ParameterList& sub_list = BCparams.sublist(ss);

          //           std::cout << "Constructing Time Dependent NBC: " << ss << std::endl;

          // These are read in the LCM::TimeTracBC constructor (LCM/evaluators/TimeTrac_Def.hpp)

          RCP<ParameterList> p = rcp(new ParameterList);

          p->set<int> ("Type", traits_type::typeTd);

          p->set< Teuchos::Array<RealType> >("Time Values",
                                             sub_list.get<Teuchos::Array<RealType> >("Time Values"));

          // Note, we use a TwoDArray here as we expect the user to specify multiple components of
          // the traction vector at each "time" step.

          p->set< Teuchos::TwoDArray<RealType> >("BC Values",
                                                 sub_list.get<Teuchos::TwoDArray<RealType> >("BC Values"));

          p->set<RCP<ParamLib> > ("Parameter Library", paramLib);

          p->set<std::string> ("Side Set ID", meshSpecs->ssNames[i]);
          p->set<Teuchos::Array< int > > ("Equation Offset", offsets[j]);
          p->set< RCP<Albany::Layouts> > ("Layouts Struct", dl);
          p->set< RCP<MeshSpecsStruct> > ("Mesh Specs Struct", meshSpecs);
          p->set<int>("Cubature Degree", BCparams.get("Cubature Degree", 0)); //if set to zero, the cubature degree of the side will be set to that of the element


          p->set<std::string> ("Coordinate Vector Name", "Coord Vec");

          if(conditions[k] == "robin") {
            p->set<std::string> ("DOF Name", dof_names[j]);
            p->set<bool> ("Vector Field", isVectorField);

            if(isVectorField) p->set< RCP<DataLayout> >("DOF Data Layout", dl->node_vector);

            else               p->set< RCP<DataLayout> >("DOF Data Layout", dl->node_scalar);
          }

          else if(conditions[k] == "basal") {
            p->set<std::string> ("DOF Name", dof_names[0]);
            p->set<bool> ("Vector Field", isVectorField);

            if(isVectorField) p->set< RCP<DataLayout> >("DOF Data Layout", dl->node_vector);

            else               p->set< RCP<DataLayout> >("DOF Data Layout", dl->node_scalar);
          }

          // Pass the input file line
          p->set< std::string > ("Neumann Input String", ss);
          p->set< Teuchos::Array<double> > ("Neumann Input Value", Teuchos::tuple<double>(0.0, 0.0, 0.0));
          p->set< std::string > ("Neumann Input Conditions", conditions[k]);

          // If we are doing a Neumann internal boundary with a "scaled jump" (includes "robin" too)
          // The material DB database needs to be passed to the BC object

          if(conditions[k] == "scaled jump" || conditions[k] == "robin") {

            TEUCHOS_TEST_FOR_EXCEPTION(materialDB == Teuchos::null,
                                       Teuchos::Exceptions::InvalidParameter,
                                       "This BC needs a material database specified");

            p->set< RCP<QCAD::MaterialDatabase> >("MaterialDB", materialDB);


          }


          std::stringstream ess;
          ess << "Evaluator for " << ss;
          evaluators_to_build[ess.str()] = p;


          bcs.push_back(ss);
        }
      }
    }
  }

#endif


  // Build evaluator for Gather Coordinate Vector

  std::string NeuGCV = "Evaluator for Gather Coordinate Vector";
  {
    RCP<ParameterList> p = rcp(new ParameterList);
    p->set<int>("Type", traits_type::typeGCV);

    // Input: Periodic BC flag
    p->set<bool>("Periodic BC", false);

    // Output:: Coordindate Vector at vertices
    p->set< RCP<DataLayout> > ("Coordinate Data Layout",  dl->vertices_vector);
    p->set< std::string >("Coordinate Vector Name", "Coord Vec");

    evaluators_to_build[NeuGCV] = p;
  }

// Build evaluator for Gather Basal Friction
#ifdef ALBANY_FELIX
   string NeuGBF="Evaluator for Gather Basal Friction";
   {
     RCP<ParameterList> p = rcp(new ParameterList());
     p->set<int>("Type", traits_type::typeGBF);

     // for new way
     p->set< RCP<DataLayout> >  ("Data Layout",  dl->node_scalar);
     p->set< string >("Basal Friction Name", "Basal Friction");

     evaluators_to_build[NeuGBF] = p;
   }

   string NeuGT="Evaluator for Gather Thickness";
  {
	RCP<ParameterList> p = rcp(new ParameterList());
	p->set<int>("Type", traits_type::typeGT);

	// for new way
	p->set< RCP<DataLayout> >  ("Data Layout",  dl->node_scalar);
	p->set< string >("Thickness Name", "Thickness");

	evaluators_to_build[NeuGT] = p;
  }

  string NeuGSH="Evaluator for Gather Surface Height";
    {
  	RCP<ParameterList> p = rcp(new ParameterList());
  	p->set<int>("Type", traits_type::typeGSH);

  	// for new way
  	p->set< RCP<DataLayout> >  ("Data Layout",  dl->node_scalar);
  	p->set< string >("Surface Height Name", "Surface Height");

  	evaluators_to_build[NeuGSH] = p;
    }
#endif

  // Build evaluator for Gather Solution

  std::string NeuGS = "Evaluator for Gather Solution";
  {
    RCP<ParameterList> p = rcp(new ParameterList());
    p->set<int>("Type", traits_type::typeGS);

    // for new way
    p->set< RCP<Albany::Layouts> >("Layouts Struct", dl);

    p->set< Teuchos::ArrayRCP<std::string> >("Solution Names", dof_names);

    p->set<bool>("Vector Field", isVectorField);

    if(isVectorField) p->set< RCP<DataLayout> >("Data Layout", dl->node_vector);

    else               p->set< RCP<DataLayout> >("Data Layout", dl->node_scalar);

    p->set<int>("Offset of First DOF", offsetToFirstDOF);
    p->set<bool>("Disable Transient", true);

    evaluators_to_build[NeuGS] = p;
  }


  // Build evaluator that causes the evaluation of all the NBCs

  std::string allBC = "Evaluator for all Neumann BCs";
  {
    RCP<ParameterList> p = rcp(new ParameterList);
    p->set<int>("Type", traits_type::typeNa);

    p->set<vector<std::string>* >("NBC Names", &bcs);
    p->set< RCP<DataLayout> >("Data Layout", dl->dummy);
    p->set<std::string>("NBC Aggregator Name", allBC);
    evaluators_to_build[allBC] = p;
  }

  return buildFieldManager(evaluators_to_build, allBC, dl->dummy);
}

template<typename BCTraits>
Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> >
Albany::BCUtils<BCTraits>::buildFieldManager(const std::map < std::string,
    Teuchos::RCP<Teuchos::ParameterList> > & evaluators_to_build,
    std::string& allBC, Teuchos::RCP<PHX::DataLayout>& dummy) {

  using PHAL::AlbanyTraits;

  // Build Field Evaluators for each evaluation type
  PHX::EvaluatorFactory<AlbanyTraits, typename Albany::BCUtils<BCTraits>::traits_type::factory_type > factory;

  Teuchos::RCP< std::vector< Teuchos::RCP<PHX::Evaluator_TemplateManager<AlbanyTraits> > > > evaluators;
  evaluators = factory.buildEvaluators(evaluators_to_build);

  // Create a DirichletFieldManager
  Teuchos::RCP<PHX::FieldManager<AlbanyTraits> > fm
    = Teuchos::rcp(new PHX::FieldManager<AlbanyTraits>);

  // Register all Evaluators
  PHX::registerEvaluators(evaluators, *fm);

  PHX::Tag<AlbanyTraits::Residual::ScalarT> res_tag0(allBC, dummy);
  fm->requireField<AlbanyTraits::Residual>(res_tag0);

  PHX::Tag<AlbanyTraits::Jacobian::ScalarT> jac_tag0(allBC, dummy);
  fm->requireField<AlbanyTraits::Jacobian>(jac_tag0);

  PHX::Tag<AlbanyTraits::Tangent::ScalarT> tan_tag0(allBC, dummy);
  fm->requireField<AlbanyTraits::Tangent>(tan_tag0);

#ifdef ALBANY_SG_MP
  PHX::Tag<AlbanyTraits::SGResidual::ScalarT> sgres_tag0(allBC, dummy);
  fm->requireField<AlbanyTraits::SGResidual>(sgres_tag0);

  PHX::Tag<AlbanyTraits::SGJacobian::ScalarT> sgjac_tag0(allBC, dummy);
  fm->requireField<AlbanyTraits::SGJacobian>(sgjac_tag0);

  PHX::Tag<AlbanyTraits::SGTangent::ScalarT> sgtan_tag0(allBC, dummy);
  fm->requireField<AlbanyTraits::SGTangent>(sgtan_tag0);

  PHX::Tag<AlbanyTraits::MPResidual::ScalarT> mpres_tag0(allBC, dummy);
  fm->requireField<AlbanyTraits::MPResidual>(mpres_tag0);

  PHX::Tag<AlbanyTraits::MPJacobian::ScalarT> mpjac_tag0(allBC, dummy);
  fm->requireField<AlbanyTraits::MPJacobian>(mpjac_tag0);

  PHX::Tag<AlbanyTraits::MPTangent::ScalarT> mptan_tag0(allBC, dummy);
  fm->requireField<AlbanyTraits::MPTangent>(mptan_tag0);
#endif //ALBANY_SG_MP

  return fm;
}

// Various specializations

Teuchos::RCP<const Teuchos::ParameterList>
Albany::DirichletTraits::getValidBCParameters(
  const std::vector<std::string>& nodeSetIDs,
  const std::vector<std::string>& bcNames) {

  Teuchos::RCP<Teuchos::ParameterList> validPL =
    Teuchos::rcp(new Teuchos::ParameterList("Valid Dirichlet BC List"));;

  for(std::size_t i = 0; i < nodeSetIDs.size(); i++) {
    for(std::size_t j = 0; j < bcNames.size(); j++) {
      std::string ss = Albany::DirichletTraits::constructBCName(nodeSetIDs[i], bcNames[j]);
      std::string tt = Albany::DirichletTraits::constructTimeDepBCName(nodeSetIDs[i], bcNames[j]);
      validPL->set<double>(ss, 0.0, "Value of BC corresponding to nodeSetID and dofName");
      validPL->sublist(tt, false, "SubList of BC corresponding to nodeSetID and dofName");
    }
  }

  for(std::size_t i = 0; i < nodeSetIDs.size(); i++) {
    std::string ss = Albany::DirichletTraits::constructBCName(nodeSetIDs[i], "K");
    std::string tt = Albany::DirichletTraits::constructBCName(nodeSetIDs[i], "twist");
    std::string ww = Albany::DirichletTraits::constructBCName(nodeSetIDs[i], "Schwarz");
    std::string uu = Albany::DirichletTraits::constructBCName(nodeSetIDs[i], "CoordFunc");
    validPL->sublist(ss, false, "");
    validPL->sublist(tt, false, "");
    validPL->sublist(ww, false, "");
    validPL->sublist(uu, false, "");
  }

  return validPL;

}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::NeumannTraits::getValidBCParameters(
  const std::vector<std::string>& sideSetIDs,
  const std::vector<std::string>& bcNames,
  const std::vector<std::string>& conditions) {

  Teuchos::RCP<Teuchos::ParameterList> validPL =
    Teuchos::rcp(new Teuchos::ParameterList("Valid Neumann BC List"));;

  for(std::size_t i = 0; i < sideSetIDs.size(); i++) { // loop over all side sets in the mesh
    for(std::size_t j = 0; j < bcNames.size(); j++) { // loop over all possible types of condition
      for(std::size_t k = 0; k < conditions.size(); k++) { // loop over all possible types of condition

        std::string ss = Albany::NeumannTraits::constructBCName(sideSetIDs[i], bcNames[j], conditions[k]);
        std::string tt = Albany::NeumannTraits::constructTimeDepBCName(sideSetIDs[i], bcNames[j], conditions[k]);

        /*
                if(numDim == 2)
                  validPL->set<Teuchos::Array<double> >(ss, Teuchos::tuple<double>(0.0, 0.0),
                    "Value of BC corresponding to sideSetID and boundary condition");
                else
                  validPL->set<Teuchos::Array<double> >(ss, Teuchos::tuple<double>(0.0, 0.0, 0.0),
                    "Value of BC corresponding to sideSetID and boundary condition");
        */
        Teuchos::Array<double> defaultData;
        validPL->set<Teuchos::Array<double> >(ss, defaultData,
                                              "Value of BC corresponding to sideSetID and boundary condition");


        validPL->sublist(tt, false, "SubList of BC corresponding to sideSetID and boundary condition");
      }
    }
  }

  validPL->set<std::string>("BetaXY", "Constant", "Function Type for Basal BC");
  validPL->set<int>("Cubature Degree", 3,"Cubature Degree for Neumann BC");
  validPL->set<double>("L", 1, "Length Scale for ISMIP-HOM Tests");
  return validPL;

}

std::string
Albany::DirichletTraits::constructBCName(const std::string ns, const std::string dof) {

  std::stringstream ss;
  ss << "DBC on NS " << ns << " for DOF " << dof;

  return ss.str();
}

std::string
Albany::NeumannTraits::constructBCName(const std::string ns, const std::string dof,
                                       const std::string condition) {
  std::stringstream ss;
  ss << "NBC on SS " << ns << " for DOF " << dof << " set " << condition;
  return ss.str();
}

std::string
Albany::DirichletTraits::constructTimeDepBCName(const std::string ns, const std::string dof) {
  std::stringstream ss;
  ss << "Time Dependent " << Albany::DirichletTraits::constructBCName(ns, dof);
  return ss.str();
}

std::string
Albany::NeumannTraits::constructTimeDepBCName(const std::string ns,
    const std::string dof, const std::string condition) {
  std::stringstream ss;
  ss << "Time Dependent " << Albany::NeumannTraits::constructBCName(ns, dof, condition);
  return ss.str();
}

