//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_BCUtils.hpp"
#include "Albany_Macros.hpp"

#include <Phalanx_Evaluator_Factory.hpp>
#include <boost/algorithm/string.hpp>

namespace {
const char decorator[] = "Evaluator for ";

// Name decorator.
inline std::string
evaluatorsToBuildName(const std::string& bc_name)
{
  std::stringstream ess;
  ess << decorator << bc_name;
  return ess.str();
}

// Either (1) the inverse of above or (2) identity, in case the decorator is not
// used.
inline std::string
plName(const std::string& name)
{
  const std::size_t pos = name.find(decorator);
  if (pos == std::string::npos) return name;
  return name.substr(pos + sizeof(decorator) - 1);
}

// DBCs do not depend on each other. However, BCs are not always compatible at
// corners, and so order of evaluation can matter. Establish an order here. The
// order is the order the BC is listed in the XML input file.
void
imposeOrder(
    const Teuchos::ParameterList& bc_pl,
    const std::map<std::string, Teuchos::RCP<Teuchos::ParameterList>>&
        evname2pl)
{
  using Teuchos::ParameterList;
  using Teuchos::RCP;
  typedef std::map<std::string, Teuchos::RCP<Teuchos::ParameterList>> S2PL;
  typedef std::map<std::string, int>                                  S2int;

  const std::string parm_name("BCOrder");
  const char*       parm_val = "BCOrder_";

  // Get the order of the BCs as they are written in the XML file.
  // ParameterList::ConstIterator preserves the text ordering.
  S2int order;
  int   ne = 0;
  for (ParameterList::ConstIterator it = bc_pl.begin(); it != bc_pl.end(); ++it)
    order[it->first] = ne++;

  std::vector<bool> found(ne, false);
  for (S2PL::const_iterator it = evname2pl.begin(); it != evname2pl.end();
       ++it) {
    const std::string     name     = plName(it->first);
    S2int::const_iterator order_it = order.find(name);
    if (order_it == order.end()) {
      // It is not an error to add an evaluator not directly mapped to an XML
      // entry.

      // LB: if the bc is of the form 'DBC off NS...', we ignore the miss,
      //     since only one evaluator will be build for all the 'DBC off NS...'
      //     bc specified for a given dof (and therefore, no match can be found
      //     here).

      if (name.find("DBC off NS") != std::string::npos) {
        bool found_off_ns = false;
        // We actually have an evaluator for this 'DBC off NS...', but it
        // probably have more than one NS in its name
        for (order_it = order.begin(); order_it != order.end(); ++order_it) {
          if (order_it->first.find("DBC off NS") != std::string::npos) {
            // We found a 'DBC off NS...' bc. Let's check the NS and DOF inside
            // name are inside this evaluator name

            // Get NS name
            size_t      ns_pos  = name.find("NS ");
            size_t      ns_size = name.find(" ", ns_pos + 3);
            std::string ns_name = name.substr(ns_pos + 3, ns_size);

            // Get DOF name
            size_t      dof_pos  = name.find("DOF ");
            size_t      dof_size = name.find(" ", dof_pos + 4);
            std::string dof_name = name.substr(dof_pos + 4, dof_size);

            // Check that NS and DOF name are inside this evaluator name
            if (order_it->first.find(ns_name) != std::string::npos &&
                order_it->first.find(dof_name) != std::string::npos) {
              found_off_ns = true;
              break;
            }
          }
        }

        // If we did not find the right DirichletOffNodeSet evaluator, it's
        // really a miss
        if (!found_off_ns) continue;
      } else {
        // It really is a miss
        continue;
      }
    }
    const int index = order_it->second;
    found[index]    = true;
    if (index > 0) {
      std::stringstream dependency;
      dependency << parm_val << index - 1;
      it->second->set<std::string>(parm_name + " Dependency", dependency.str());
    }
    if (index + 1 < ne) {
      std::stringstream evaluates;
      evaluates << parm_val << index;
      it->second->set<std::string>(parm_name + " Evaluates", evaluates.str());
    }
  }

  // Protect against not having all dependencies satisfied. Phalanx would detect
  // this, of course, but here I can provide more information.
  bool all_found = true;
  for (std::vector<bool>::const_iterator it = found.begin(); it != found.end();
       ++it)
    if (!*it) {
      all_found = false;
      break;
    }
  if (!all_found) {
    std::stringstream msg;
    msg << ne << " BCs were specified in " << bc_pl.name() << ", but not all "
        << " were detected and ordered. The parameter list gives:\n";
    for (S2int::const_iterator it = order.begin(); it != order.end(); ++it)
      msg << "  " << it->first << "\n";
    msg << "But BCUtils provided:\n";
    for (S2PL::const_iterator it = evname2pl.begin(); it != evname2pl.end();
         ++it)
      msg << "  " << plName(it->first) << "\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, msg.str());
  }
}
}  // namespace

// Dirichlet specialization

template <>
Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>
Albany::BCUtils<Albany::DirichletTraits>::constructBCEvaluators(
    const std::vector<std::string>&      nodeSetIDs,
    const std::vector<std::string>&      bcNames,
    Teuchos::RCP<Teuchos::ParameterList> params,
    Teuchos::RCP<ParamLib>               paramLib,
    int                                  numEqn)
{
  using PHAL::AlbanyTraits;
  using PHX::DataLayout;
  using PHX::MDALayout;
  using Teuchos::ParameterList;
  using Teuchos::RCP;
  using Teuchos::rcp;

  use_sdbcs_ = false;
  nodeSetIDs_.resize(nodeSetIDs.size());
  for (int i = 0; i < nodeSetIDs.size(); i++) nodeSetIDs_[i] = nodeSetIDs[i];

  if (!haveBCSpecified(
          params)) {  // If the BC sublist is not in the input file,
    // but we are inside this function, this means that
    // node sets are contained in the Exodus file but are not defined in the
    // problem statement.This is OK, we
    // just don't do anything

    return Teuchos::null;
  }

  // Build the list of evaluators (with their parameters) that have to be built
  std::map<std::string, RCP<Teuchos::ParameterList>> evaluators_to_build;
  buildEvaluatorsList(
      evaluators_to_build, nodeSetIDs, bcNames, params, paramLib, numEqn);

  imposeOrder(params->sublist(traits_type::bcParamsPl), evaluators_to_build);

  // Build Field Evaluators for each evaluation type
  PHX::EvaluatorFactory<AlbanyTraits, Albany::DirichletTraits::factory_type>
      factory;

  RCP<std::vector<RCP<PHX::Evaluator_TemplateManager<AlbanyTraits>>>>
      evaluators;
  evaluators = factory.buildEvaluators(evaluators_to_build);

  // Build the field manager
  std::string          allBC = "Evaluator for all Dirichlet BCs";
  RCP<PHX::DataLayout> dummy = rcp(new PHX::MDALayout<Dummy>(0));
  return buildFieldManager(evaluators, allBC, dummy);
}

template <>
Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>
Albany::BCUtils<Albany::NeumannTraits>::constructBCEvaluators(
    const Teuchos::RCP<Albany::MeshSpecsStruct>&  meshSpecs,
    const std::vector<std::string>&               bcNames,
    const Teuchos::ArrayRCP<std::string>&         dof_names,
    bool                                          isVectorField,
    int                                           offsetToFirstDOF,
    const std::vector<std::string>&               conditions,
    const Teuchos::Array<Teuchos::Array<int>>&    offsets,
    const Teuchos::RCP<Albany::Layouts>&          dl,
    Teuchos::RCP<Teuchos::ParameterList>          params,
    Teuchos::RCP<ParamLib>                        paramLib,
    const Teuchos::RCP<Albany::MaterialDatabase>& materialDB)
{
  using PHAL::AlbanyTraits;
  using PHX::DataLayout;
  using PHX::MDALayout;
  using Teuchos::ParameterList;
  using Teuchos::RCP;
  using Teuchos::rcp;

  if (!haveBCSpecified(
          params)) {  // If the BC sublist is not in the input file,
    // but we are inside this function, this means that
    // node sets are contained in the Exodus file but are not defined in the
    // problem statement.This is OK, we
    // just don't do anything

    return Teuchos::null;
  }

  // Build the list of evaluators to build, with all the needed parameters
  std::map<std::string, RCP<Teuchos::ParameterList>> evaluators_to_build;
  buildEvaluatorsList(
      evaluators_to_build,
      meshSpecs,
      bcNames,
      dof_names,
      isVectorField,
      offsetToFirstDOF,
      conditions,
      offsets,
      dl,
      params,
      paramLib,
      materialDB);

  // Build Field Evaluators for each evaluation type
  PHX::EvaluatorFactory<AlbanyTraits, Albany::NeumannTraits::factory_type>
      factory;

  Teuchos::RCP<
      std::vector<Teuchos::RCP<PHX::Evaluator_TemplateManager<AlbanyTraits>>>>
      evaluators;
  evaluators = factory.buildEvaluators(evaluators_to_build);

  std::string allBC = "Evaluator for all Neumann BCs";

  return buildFieldManager(evaluators, allBC, dl->dummy);
}

template <>
Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>
Albany::BCUtils<Albany::NeumannTraits>::constructBCEvaluators(
    const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs,
    const std::vector<std::string>&              bcNames,
    const Teuchos::ArrayRCP<std::string>&        dof_names,
    bool                                         isVectorField,
    int                                          offsetToFirstDOF,
    const std::vector<std::string>&              conditions,
    const Teuchos::Array<Teuchos::Array<int>>&   offsets,
    const Teuchos::RCP<Albany::Layouts>&         dl,
    Teuchos::RCP<Teuchos::ParameterList>         params,
    Teuchos::RCP<ParamLib>                       paramLib,
    const std::vector<Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits>>>&
                                                  extra_evaluators,
    const Teuchos::RCP<Albany::MaterialDatabase>& materialDB)
{
  using PHAL::AlbanyTraits;
  using PHX::DataLayout;
  using PHX::MDALayout;
  using Teuchos::ParameterList;
  using Teuchos::RCP;
  using Teuchos::rcp;

  if (!haveBCSpecified(
          params)) {  // If the BC sublist is not in the input file,
    // but we are inside this function, this means that
    // node sets are contained in the Exodus file but are not defined in the
    // problem statement.This is OK, we
    // just don't do anything

    return Teuchos::null;
  }

  // Build the list of evaluators to build, with all the needed parameters
  std::map<std::string, RCP<Teuchos::ParameterList>> evaluators_to_build;
  buildEvaluatorsList(
      evaluators_to_build,
      meshSpecs,
      bcNames,
      dof_names,
      isVectorField,
      offsetToFirstDOF,
      conditions,
      offsets,
      dl,
      params,
      paramLib,
      materialDB);

  // Build Field Evaluators for each evaluation type
  PHX::EvaluatorFactory<AlbanyTraits, Albany::NeumannTraits::factory_type>
      factory;

  RCP<std::vector<RCP<PHX::Evaluator_TemplateManager<AlbanyTraits>>>>
      evaluators;
  evaluators = factory.buildEvaluators(evaluators_to_build);

  std::string                          allBC = "Evaluator for all Neumann BCs";
  RCP<PHX::FieldManager<AlbanyTraits>> fm =
      buildFieldManager(evaluators, allBC, dl->dummy);

  std::vector<RCP<PHX::Evaluator<AlbanyTraits>>>::const_iterator it;
  for (it = extra_evaluators.begin(); it != extra_evaluators.end(); ++it) {
    fm->registerEvaluatorForAllEvaluationTypes(*it);
  }

  return fm;
}

template <>
void
Albany::BCUtils<Albany::DirichletTraits>::buildEvaluatorsList(
    std::map<std::string, Teuchos::RCP<Teuchos::ParameterList>>&
                                         evaluators_to_build,
    const std::vector<std::string>&      nodeSetIDs,
    const std::vector<std::string>&      bcNames,
    Teuchos::RCP<Teuchos::ParameterList> params,
    Teuchos::RCP<ParamLib>               paramLib,
    int                                  numEqn)
{
  using PHAL::AlbanyTraits;
  using PHX::DataLayout;
  using PHX::MDALayout;
  using std::string;
  using Teuchos::ParameterList;
  using Teuchos::RCP;
  using Teuchos::rcp;

  use_sdbcs_ = false;

  ParameterList BCparams = params->sublist(traits_type::bcParamsPl);
  BCparams.validateParameters(
      *(traits_type::getValidBCParameters(nodeSetIDs, bcNames)), 0);

  RCP<DataLayout>          dummy = rcp(new PHX::MDALayout<Dummy>(0));
  RCP<std::vector<string>> bcs   = rcp(new std::vector<string>());

  offsets_.resize(nodeSetIDs.size());
  // Check for all possible standard BCs (every dof on every nodeset) to see
  // which is set

  for (std::size_t i = 0; i < nodeSetIDs.size(); i++) {
    for (std::size_t j = 0; j < bcNames.size(); j++) {
      string ss = traits_type::constructBCName(nodeSetIDs[i], bcNames[j]);
      if (BCparams.isParameter(ss)) {
        RCP<ParameterList> p = rcp(new ParameterList);

        p->set<int>("Type", traits_type::type);
        p->set<RCP<DataLayout>>("Data Layout", dummy);
        p->set<string>("Dirichlet Name", ss);
        p->set<RealType>("Dirichlet Value", BCparams.get<double>(ss));
        p->set<string>("Node Set ID", nodeSetIDs[i]);
        // p->set< int >     ("Number of Equations", dirichletNames.size());
        p->set<int>("Equation Offset", j);
        offsets_[i].push_back(j);
        p->set<RCP<ParamLib>>("Parameter Library", paramLib);

        evaluators_to_build[evaluatorsToBuildName(ss)] = p;

        bcs->push_back(ss);
        use_dbcs_ = true;
      }
    }
  }

  ///
  /// Apply a function based on a coordinate value to the boundary
  ///
  for (std::size_t i = 0; i < nodeSetIDs.size(); i++) {
    string ss = traits_type::constructBCName(nodeSetIDs[i], "CoordFunc");

    if (BCparams.isSublist(ss)) {
      // grab the sublist
      ParameterList& sub_list = BCparams.sublist(ss);

      // Directly apply the coordinate values at the boundary as a DBC (Laplace
      // Beltrami mesh equations)
      if (sub_list.get<string>("BC Function") == "Identity") {
        RCP<ParameterList> p = rcp(new ParameterList);
        p->set<int>("Type", traits_type::typeFb);

        // Fill up ParameterList with things DirichletBase wants
        p->set<RCP<DataLayout>>("Data Layout", dummy);
        p->set<string>("Dirichlet Name", ss);
        p->set<RealType>("Dirichlet Value", 0.0);
        p->set<string>("Node Set ID", nodeSetIDs[i]);
        p->set<int>("Number of Equations", numEqn);
        p->set<int>("Equation Offset", 0);
        for (std::size_t j = 0; j < bcNames.size(); j++) {
          offsets_[i].push_back(j);
        }
        p->set<RCP<ParamLib>>("Parameter Library", paramLib);

        evaluators_to_build[evaluatorsToBuildName(ss)] = p;

        bcs->push_back(ss);
        use_dbcs_ = true;
      }

      // Add other functional boundary conditions here. Note that Torsion could
      // fit into this framework
    }
  }

  for (std::size_t i = 0; i < nodeSetIDs.size(); i++) {
    for (std::size_t j = 0; j < bcNames.size(); j++) {
      string ss = traits_type::constructBCNameField(nodeSetIDs[i], bcNames[j]);
      if (BCparams.isParameter(ss)) {
        RCP<ParameterList> p = rcp(new ParameterList);
        p->set<int>("Type", traits_type::typeF);
        p->set<RCP<DataLayout>>("Data Layout", dummy);
        p->set<string>("Dirichlet Name", ss);
        p->set<RealType>("Dirichlet Value", 0.0);
        p->set<string>("Field Name", BCparams.get<string>(ss));
        p->set<string>("Node Set ID", nodeSetIDs[i]);
        p->set<int>("Equation Offset", j);
        offsets_[i].push_back(j);
        p->set<RCP<ParamLib>>("Parameter Library", paramLib);
        evaluators_to_build[evaluatorsToBuildName(ss)] = p;
        bcs->push_back(ss);
        use_dbcs_ = true;
      }
    }
  }

  for (std::size_t i = 0; i < nodeSetIDs.size(); i++) {
    for (std::size_t j = 0; j < bcNames.size(); j++) {
      string ss = traits_type::constructSDBCNameField(nodeSetIDs[i], bcNames[j]);
      if (BCparams.isParameter(ss)) {
        RCP<ParameterList> p = rcp(new ParameterList);
        p->set<int>("Type", traits_type::typeSF);
        p->set<RCP<DataLayout>>("Data Layout", dummy);
        p->set<string>("Dirichlet Name", ss);
        p->set<RealType>("Dirichlet Value", 0.0);
        p->set<string>("Field Name", BCparams.get<string>(ss));
        p->set<string>("Node Set ID", nodeSetIDs[i]);
        p->set<int>("Equation Offset", j);
        offsets_[i].push_back(j);
        p->set<RCP<ParamLib>>("Parameter Library", paramLib);
        evaluators_to_build[evaluatorsToBuildName(ss)] = p;
        bcs->push_back(ss);
        use_sdbcs_ = true;
      }
    }
  }

  ///
  /// Time dependent BC specific
  ///
  for (std::size_t i = 0; i < nodeSetIDs.size(); i++) {
    for (std::size_t j = 0; j < bcNames.size(); j++) {
      string ss =
          traits_type::constructTimeDepBCName(nodeSetIDs[i], bcNames[j]);

      if (BCparams.isSublist(ss)) {
        // grab the sublist
        ParameterList&     sub_list = BCparams.sublist(ss);
        RCP<ParameterList> p        = rcp(new ParameterList);
        p->set<int>("Type", traits_type::typeTd);

        // Extract the time values into a vector
        if (sub_list.isParameter("Time File") == true) {
          std::string const filename = sub_list.get<std::string>("Time File");
          std::ifstream     file(filename);
          ALBANY_ASSERT(file.good() == true, "Error opening Time File");
          std::stringstream buffer;
          buffer << file.rdbuf();
          file.close();
          std::istringstream       iss(buffer.str());
          Teuchos::Array<RealType> time_values;
          iss >> time_values;
          p->set<Teuchos::Array<RealType>>("Time Values", time_values);
        } else {
          p->set<Teuchos::Array<RealType>>(
              "Time Values",
              sub_list.get<Teuchos::Array<RealType>>("Time Values"));
        }

        // Extract the BC values into a vector
        if (sub_list.isParameter("BC File") == true) {
          std::string const filename = sub_list.get<std::string>("BC File");
          std::ifstream     file(filename);
          ALBANY_ASSERT(file.good() == true, "Error opening BC File");
          std::stringstream buffer;
          buffer << file.rdbuf();
          file.close();
          std::istringstream       iss(buffer.str());
          Teuchos::Array<RealType> bc_values;
          iss >> bc_values;
          p->set<Teuchos::Array<RealType>>("BC Values", bc_values);
        } else {
          p->set<Teuchos::Array<RealType>>(
              "BC Values", sub_list.get<Teuchos::Array<RealType>>("BC Values"));
        }

        p->set<bool>("Mesh Deforms", sub_list.get<bool>("Mesh Deforms", false));
        p->set<RCP<DataLayout>>("Data Layout", dummy);
        p->set<string>("Dirichlet Name", ss);
        p->set<RealType>("Dirichlet Value", 0.0);
        p->set<int>("Equation Offset", j);
        offsets_[i].push_back(j);
        p->set<RCP<ParamLib>>("Parameter Library", paramLib);
        p->set<string>("Node Set ID", nodeSetIDs[i]);
        p->set<int>(
            "Cubature Degree",
            BCparams.get("Cubature Degree", 0));  // if set to zero, the
                                                  // cubature degree of the side
                                                  // will be set to that of the
                                                  // element

        evaluators_to_build[evaluatorsToBuildName(ss)] = p;

        bcs->push_back(ss);
        use_dbcs_ = true;
      }
    }
  }

  ///
  /// Time dependent SDBC specific
  ///
  for (std::size_t i = 0; i < nodeSetIDs.size(); ++i) {
    for (std::size_t j = 0; j < bcNames.size(); ++j) {
      string ss =
          traits_type::constructTimeDepSDBCName(nodeSetIDs[i], bcNames[j]);

      if (BCparams.isSublist(ss)) {
        use_sdbcs_ = true;

        // grab the sublist
        ParameterList& sub_list = BCparams.sublist(ss);

        RCP<ParameterList> p = rcp(new ParameterList);

        p->set<int>("Type", traits_type::typeTs);

        // Extract the time values into a vector
        if (sub_list.isParameter("Time File") == true) {
          std::string const filename = sub_list.get<std::string>("Time File");
          std::ifstream     file(filename);
          ALBANY_ASSERT(file.good() == true, "Error opening Time File");
          std::stringstream buffer;
          buffer << file.rdbuf();
          file.close();
          std::istringstream       iss(buffer.str());
          Teuchos::Array<RealType> time_values;
          iss >> time_values;
          p->set<Teuchos::Array<RealType>>("Time Values", time_values);
        } else {
          p->set<Teuchos::Array<RealType>>(
              "Time Values",
              sub_list.get<Teuchos::Array<RealType>>("Time Values"));
        }

        // Extract the BC values into a vector
        if (sub_list.isParameter("BC File") == true) {
          std::string const filename = sub_list.get<std::string>("BC File");
          std::ifstream     file(filename);
          ALBANY_ASSERT(file.good() == true, "Error opening BC File");
          std::stringstream buffer;
          buffer << file.rdbuf();
          file.close();
          std::istringstream       iss(buffer.str());
          Teuchos::Array<RealType> bc_values;
          iss >> bc_values;
          p->set<Teuchos::Array<RealType>>("BC Values", bc_values);
        } else {
          p->set<Teuchos::Array<RealType>>(
              "BC Values", sub_list.get<Teuchos::Array<RealType>>("BC Values"));
        }

        p->set<bool>("Mesh Deforms", sub_list.get<bool>("Mesh Deforms", false));
        p->set<RCP<DataLayout>>("Data Layout", dummy);
        p->set<string>("Dirichlet Name", ss);
        p->set<RealType>("Dirichlet Value", 0.0);
        p->set<int>("Equation Offset", j);
        offsets_[i].push_back(j);
        p->set<RCP<ParamLib>>("Parameter Library", paramLib);
        p->set<string>("Node Set ID", nodeSetIDs[i]);

        // if set to zero, the cubature degree of the side
        // will be set to that of the element
        p->set<int>("Cubature Degree", BCparams.get("Cubature Degree", 0));

        evaluators_to_build[evaluatorsToBuildName(ss)] = p;

        bcs->push_back(ss);
      }
    }
  }

#ifdef ALBANY_LCM
  ///
  /// Torsion BC specific
  ////
  for (std::size_t i = 0; i < nodeSetIDs.size(); i++) {
    string ss = traits_type::constructBCName(nodeSetIDs[i], "twist");

    if (BCparams.isSublist(ss)) {
      // grab the sublist
      ParameterList& sub_list = BCparams.sublist(ss);

      if (sub_list.get<string>("BC Function") == "Torsion") {
        RCP<ParameterList> p = rcp(new ParameterList);
        p->set<int>("Type", traits_type::typeTo);

        p->set<RealType>("Theta Dot", sub_list.get<RealType>("Theta Dot"));
        p->set<RealType>("X0", sub_list.get<RealType>("X0"));
        p->set<RealType>("Y0", sub_list.get<RealType>("Y0"));

        // Fill up ParameterList with things DirichletBase wants
        p->set<RCP<DataLayout>>("Data Layout", dummy);
        p->set<string>("Dirichlet Name", ss);
        p->set<RealType>("Dirichlet Value", 0.0);
        p->set<string>("Node Set ID", nodeSetIDs[i]);
        // p->set< int >     ("Number of Equations", dirichletNames.size());
        p->set<int>("Equation Offset", 0);
        for (std::size_t j = 0; j < bcNames.size(); j++) {
          offsets_[i].push_back(j);
        }
        p->set<int>(
            "Cubature Degree",
            BCparams.get("Cubature Degree", 0));  // if set to zero, the
                                                  // cubature degree of the side
                                                  // will be set to that of the
                                                  // element
        p->set<RCP<ParamLib>>("Parameter Library", paramLib);
        evaluators_to_build[evaluatorsToBuildName(ss)] = p;

        bcs->push_back(ss);
        use_dbcs_ = true;
      }
    }
  }

  ///
  /// Equilibrium Concentration BC specific
  ///
  for (std::size_t i = 0; i < nodeSetIDs.size(); i++) {
    for (std::size_t j = 0; j < bcNames.size(); j++) {
      string ss =
          traits_type::constructPressureDepBCName(nodeSetIDs[i], bcNames[j]);
      if (BCparams.isSublist(ss)) {
        // grab the sublist
        ParameterList& sub_list = BCparams.sublist(ss);

        // get the pressure offset
        int pressure_offset;
        for (std::size_t k = 0; k < bcNames.size(); k++) {
          if (bcNames[k] == "TAU") {
            pressure_offset = k;
            continue;
          }
        }

        if (sub_list.get<string>("BC Function") ==
            "Equilibrium Concentration") {
          RCP<ParameterList> p = rcp(new ParameterList);
          p->set<int>("Type", traits_type::typeEq);

          p->set<RealType>(
              "Applied Concentration",
              sub_list.get<RealType>("Applied Concentration"));
          p->set<RealType>(
              "Pressure Factor", sub_list.get<RealType>("Pressure Factor"));

          // Fill up ParameterList with things DirichletBase wants
          p->set<RCP<DataLayout>>("Data Layout", dummy);
          p->set<string>("Dirichlet Name", ss);
          p->set<RealType>("Dirichlet Value", 0.0);
          p->set<string>("Node Set ID", nodeSetIDs[i]);
          p->set<int>("Equation Offset", j);
          offsets_[i].push_back(j);
          p->set<int>("Pressure Offset", pressure_offset);

          evaluators_to_build[evaluatorsToBuildName(ss)] = p;

          bcs->push_back(ss);
          use_dbcs_ = true;
        }
      }
    }
  }

#endif  // ALBANY_LCM

  ///
  /// SDBC (S = "Symmetric", f.k.a. "Strong")
  ///
  for (std::size_t i = 0; i < nodeSetIDs.size(); i++) {
    for (std::size_t j = 0; j < bcNames.size(); j++) {
      string ss = traits_type::constructSDBCName(nodeSetIDs[i], bcNames[j]);
      if (BCparams.isParameter(ss)) {
        RCP<ParameterList> p = rcp(new ParameterList);
        use_sdbcs_           = true;
        p->set<int>("Type", traits_type::typeSt);
        p->set<RCP<DataLayout>>("Data Layout", dummy);
        p->set<string>("Dirichlet Name", ss);
        p->set<RealType>("Dirichlet Value", BCparams.get<double>(ss));
        p->set<string>("Node Set ID", nodeSetIDs[i]);
        p->set<int>("Equation Offset", j);
        offsets_[i].push_back(j);
        p->set<RCP<ParamLib>>("Parameter Library", paramLib);

        evaluators_to_build[evaluatorsToBuildName(ss)] = p;

        bcs->push_back(ss);
      }
    }
  }
#ifdef ALBANY_STK_EXPR_EVAL
  ///
  /// Expression Evaluated SDBC (S = "Symmetric", f.k.a. "Strong")
  ///
  for (std::size_t i = 0; i < nodeSetIDs.size(); i++) {
    for (std::size_t j = 0; j < bcNames.size(); j++) {
      string ss =
          traits_type::constructExprEvalSDBCName(nodeSetIDs[i], bcNames[j]);
      if (BCparams.isParameter(ss)) {
        RCP<ParameterList> p = rcp(new ParameterList);
        use_sdbcs_           = true;
        p->set<int>("Type", traits_type::typeEe);
        p->set<RCP<DataLayout>>("Data Layout", dummy);
        p->set<string>("Dirichlet Name", ss);
        p->set<std::string>(
            "Dirichlet Expression", BCparams.get<std::string>(ss));
        p->set<RealType>("Dirichlet Value", 0.0);
        p->set<string>("Node Set ID", nodeSetIDs[i]);
        p->set<int>("Equation Offset", j);
        offsets_[i].push_back(j);
        p->set<RCP<ParamLib>>("Parameter Library", paramLib);

        evaluators_to_build[evaluatorsToBuildName(ss)] = p;

        bcs->push_back(ss);
      }
    }
  }
#endif
  ///
  ///
  /// Scaled SDBC (S = "Symmetric", f.k.a. "Strong")
  ///
  for (std::size_t i = 0; i < nodeSetIDs.size(); i++) {
    for (std::size_t j = 0; j < bcNames.size(); j++) {
      string ss =
          traits_type::constructScaledSDBCName(nodeSetIDs[i], bcNames[j]);
      if (BCparams.isParameter(ss)) {
        RCP<ParameterList> p = rcp(new ParameterList);
        use_sdbcs_           = true;
        p->set<int>("Type", traits_type::typeSt);
        p->set<RCP<DataLayout>>("Data Layout", dummy);
        p->set<string>("Dirichlet Name", ss);
        Teuchos::Array<RealType> array =
            BCparams.get<Teuchos::Array<RealType>>(ss);
        p->set<RealType>("Dirichlet Value", array[0]);
        p->set<string>("Node Set ID", nodeSetIDs[i]);
        p->set<int>("Equation Offset", j);
        offsets_[i].push_back(j);
        p->set<RCP<ParamLib>>("Parameter Library", paramLib);

        evaluators_to_build[evaluatorsToBuildName(ss)] = p;

        bcs->push_back(ss);
      }
    }
  }

#if defined(ALBANY_LCM)
  ///
  /// Schwarz BC specific
  ///
  for (auto i = 0; i < nodeSetIDs.size(); ++i) {
    string ss = traits_type::constructBCName(nodeSetIDs[i], "Schwarz");

    if (BCparams.isSublist(ss)) {
      // grab the sublist
      ParameterList& sub_list = BCparams.sublist(ss);

      if (sub_list.get<string>("BC Function") == "Schwarz") {
        RCP<ParameterList> p = rcp(new ParameterList);

        p->set<int>("Type", traits_type::typeSw);

        p->set<string>(
            "Coupled Application", sub_list.get<string>("Coupled Application"));

        p->set<string>(
            "Coupled Block", sub_list.get<string>("Coupled Block", "NONE"));

        // Get the application from the main parameters list above
        // and pass it to the Schwarz BC evaluator.
        Teuchos::RCP<Albany::Application> const& application =
            params->get<Teuchos::RCP<Albany::Application>>("Application");

        p->set<Teuchos::RCP<Albany::Application>>("Application", application);

        // Fill up ParameterList with things DirichletBase wants
        p->set<RCP<DataLayout>>("Data Layout", dummy);
        p->set<string>("Dirichlet Name", ss);
        p->set<RealType>("Dirichlet Value", 0.0);
        p->set<string>("Node Set ID", nodeSetIDs[i]);
        p->set<int>("Equation Offset", 0);
        for (std::size_t j = 0; j < bcNames.size(); j++) {
          offsets_[i].push_back(j);
        }
        // if set to zero, the cubature degree of the side
        // will be set to that of the element
        p->set<int>("Cubature Degree", BCparams.get("Cubature Degree", 0));
        p->set<RCP<ParamLib>>("Parameter Library", paramLib);

        evaluators_to_build[evaluatorsToBuildName(ss)] = p;

        bcs->push_back(ss);
        use_dbcs_ = true;
      }
    }
  }

  ///
  /// Strong Schwarz BC specific
  ///
  for (auto i = 0; i < nodeSetIDs.size(); ++i) {
    string ss = traits_type::constructSDBCName(nodeSetIDs[i], "StrongSchwarz");

    if (BCparams.isSublist(ss)) {
      use_sdbcs_ = true;

      // grab the sublist
      ParameterList& sub_list = BCparams.sublist(ss);

      if (sub_list.get<string>("BC Function") == "StrongSchwarz") {
        RCP<ParameterList> p = rcp(new ParameterList);

        p->set<int>("Type", traits_type::typeSsw);

        p->set<string>(
            "Coupled Application", sub_list.get<string>("Coupled Application"));

        p->set<string>(
            "Coupled Block", sub_list.get<string>("Coupled Block", "NONE"));

        // Get the application from the main parameters list above
        // and pass it to the Schwarz BC evaluator.
        Teuchos::RCP<Albany::Application> const& application =
            params->get<Teuchos::RCP<Albany::Application>>("Application");

        p->set<Teuchos::RCP<Albany::Application>>("Application", application);

        // Fill up ParameterList with things DirichletBase wants
        p->set<RCP<DataLayout>>("Data Layout", dummy);
        p->set<string>("Dirichlet Name", ss);
        p->set<RealType>("Dirichlet Value", 0.0);
        p->set<string>("Node Set ID", nodeSetIDs[i]);
        p->set<int>("Equation Offset", 0);
        for (std::size_t j = 0; j < bcNames.size(); j++) {
          offsets_[i].push_back(j);
        }
        // if set to zero, the cubature degree of the side
        // will be set to that of the element
        p->set<int>("Cubature Degree", BCparams.get("Cubature Degree", 0));
        p->set<RCP<ParamLib>>("Parameter Library", paramLib);

        evaluators_to_build[evaluatorsToBuildName(ss)] = p;

        bcs->push_back(ss);
      }
    }
  }

  ///
  /// Kfield BC specific
  ///
  for (std::size_t i = 0; i < nodeSetIDs.size(); i++) {
    string ss = traits_type::constructBCName(nodeSetIDs[i], "K");

    if (BCparams.isSublist(ss)) {
      // grab the sublist
      ParameterList& sub_list = BCparams.sublist(ss);

      if (sub_list.get<string>("BC Function") == "Kfield") {
        RCP<ParameterList> p = rcp(new ParameterList);
        p->set<int>("Type", traits_type::typeKf);

        p->set<Teuchos::Array<RealType>>(
            "Time Values",
            sub_list.get<Teuchos::Array<RealType>>("Time Values"));
        p->set<Teuchos::Array<RealType>>(
            "KI Values", sub_list.get<Teuchos::Array<RealType>>("KI Values"));
        p->set<Teuchos::Array<RealType>>(
            "KII Values", sub_list.get<Teuchos::Array<RealType>>("KII Values"));

        // Extract BC parameters
        p->set<string>("Kfield KI Name", "Kfield KI");
        p->set<string>("Kfield KII Name", "Kfield KII");
        p->set<RealType>("KI Value", sub_list.get<double>("Kfield KI"));
        p->set<RealType>("KII Value", sub_list.get<double>("Kfield KII"));
        p->set<RealType>(
            "Shear Modulus", sub_list.get<double>("Shear Modulus"));
        p->set<RealType>(
            "Poissons Ratio", sub_list.get<double>("Poissons Ratio"));
        // p->set<int>("Cubature Degree", BCparams.get("Cubature Degree", 0));
        // //if set to zero, the cubature degree of the side will be set to that
        // of the element

        // Fill up ParameterList with things DirichletBase wants
        p->set<RCP<DataLayout>>("Data Layout", dummy);
        p->set<string>("Dirichlet Name", ss);
        p->set<RealType>("Dirichlet Value", 0.0);
        p->set<string>("Node Set ID", nodeSetIDs[i]);
        // p->set< int >     ("Number of Equations", dirichletNames.size());
        p->set<int>("Equation Offset", 0);
        for (std::size_t j = 0; j < bcNames.size(); j++) {
          offsets_[i].push_back(j);
        }
        p->set<RCP<ParamLib>>("Parameter Library", paramLib);
        evaluators_to_build[evaluatorsToBuildName(ss)] = p;

        bcs->push_back(ss);
        use_dbcs_ = true;
      }
    }
  }

#endif
  ///
  /// SideSet equations case: DBC to handle nodes not on the side set
  ///
  for (std::size_t j = 0; j < bcNames.size(); ++j) {
    RCP<std::vector<string>> nodeSets(new std::vector<string>(0));
    string                   dir_name = "DBC off NS";
    double*                  value    = NULL;
    for (auto i = 0; i < nodeSetIDs.size(); ++i) {
      string ss =
          traits_type::constructBCNameOffNodeSet(nodeSetIDs[i], bcNames[j]);

      if (BCparams.isParameter(ss)) {
        nodeSets->push_back(nodeSetIDs[i]);
        dir_name += " " + nodeSetIDs[i];
        if (value == NULL) {
          value  = new double();
          *value = BCparams.get<double>(ss);
        } else {
          // The solution is prescribed a unique value off the given side sets.
          TEUCHOS_TEST_FOR_EXCEPTION(
              *value != BCparams.get<double>(ss),
              std::logic_error,
              "Error! For a given DOF, all Off-Node-Set BC MUST have the same "
              "value.\n");
        }
      }
    }
    dir_name += " for DOF " + bcNames[j];

    if (nodeSets->size() > 0) {
      RCP<ParameterList> p = rcp(new ParameterList());

      p->set<int>("Type", traits_type::typeON);
      p->set<int>("Equation Offset", j);
      p->set<RCP<ParamLib>>("Parameter Library", paramLib);

      // Fill up ParameterList with things DirichletBase wants
      p->set<RCP<DataLayout>>("Data Layout", dummy);
      p->set<string>("Dirichlet Name", dir_name);
      p->set<RealType>("Dirichlet Value", *value);
      p->set<RCP<std::vector<string>>>("Node Sets", nodeSets);
      p->set<RCP<ParamLib>>("Parameter Library", paramLib);
      p->set<string>("Node Set ID", "");

      std::stringstream ess;
      ess << "Evaluator for " << dir_name;
      evaluators_to_build[ess.str()] = p;

      bcs->push_back(dir_name);
      use_dbcs_ = true;
    }
    delete value;
  }

/*  if ((use_dbcs_ == true) && (use_sdbcs_ == true)) {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        std::logic_error,
        "You are attempting to prescribe a mix of SDBCs and DBCs, which is not "
        "allowed!\n");
  }
*/
  string allBC = "Evaluator for all Dirichlet BCs";
  {
    RCP<ParameterList> p = rcp(new ParameterList);
    p->set<int>("Type", traits_type::typeDa);

    p->set<RCP<std::vector<string>>>("DBC Names", bcs);
    p->set<RCP<DataLayout>>("Data Layout", dummy);
    p->set<string>("DBC Aggregator Name", allBC);

    evaluators_to_build[allBC] = p;
  }
}

template <>
void
Albany::BCUtils<Albany::NeumannTraits>::buildEvaluatorsList(
    std::map<std::string, Teuchos::RCP<Teuchos::ParameterList>>&
                                                  evaluators_to_build,
    const Teuchos::RCP<Albany::MeshSpecsStruct>&  meshSpecs,
    const std::vector<std::string>&               bcNames,
    const Teuchos::ArrayRCP<std::string>&         dof_names,
    bool                                          isVectorField,
    int                                           offsetToFirstDOF,
    const std::vector<std::string>&               conditions,
    const Teuchos::Array<Teuchos::Array<int>>&    offsets,
    const Teuchos::RCP<Albany::Layouts>&          dl,
    Teuchos::RCP<Teuchos::ParameterList>          params,
    Teuchos::RCP<ParamLib>                        paramLib,
    const Teuchos::RCP<Albany::MaterialDatabase>& materialDB)
{
  using PHAL::AlbanyTraits;
  using PHX::DataLayout;
  using PHX::MDALayout;
  using std::string;
  using Teuchos::ParameterList;
  using Teuchos::RCP;
  using Teuchos::rcp;

  // Drop into the "Neumann BCs" sublist
  ParameterList BCparams = params->sublist(traits_type::bcParamsPl);
  BCparams.validateParameters(
      *(traits_type::getValidBCParameters(
          meshSpecs->ssNames, bcNames, conditions)),
      0);

  RCP<std::vector<string>> bcs = rcp(new std::vector<string>);

  // Check for all possible standard BCs (every dof on every sideset) to see
  // which is set
  for (std::size_t i = 0; i < meshSpecs->ssNames.size(); i++) {
    for (std::size_t j = 0; j < bcNames.size(); j++) {
      for (std::size_t k = 0; k < conditions.size(); k++) {
        // construct input.xml string like:
        // "NBC on SS sidelist_12 for DOF T set dudn"
        //  or
        // "NBC on SS sidelist_12 for DOF T set (dudx, dudy)"
        // or
        // "NBC on SS surface_1 for DOF all set P"
       
        // Set logic for certain NBCs which allow array inputs  
        bool allowArrayNBC = false;   
        if ((conditions[k] == "robin") || (conditions[k] == "radiate")
              || (conditions[k].find("(") < conditions[k].length())) {
          allowArrayNBC = true; 
        }

        string ss = traits_type::constructBCName(
            meshSpecs->ssNames[i], bcNames[j], conditions[k]);

        // Have a match of the line in input.xml

        if (BCparams.isParameter(ss)) {
          //           std::cout << "Constructing NBC: " << ss << std::endl;

          TEUCHOS_TEST_FOR_EXCEPTION(
              BCparams.isType<string>(ss),
              std::logic_error,
              "NBC array information in XML/YAML file must be of type "
              "Array(double)\n");

          // These are read in the Albany::Neumann constructor
          // (PHAL_Neumann_Def.hpp)

          RCP<ParameterList> p = rcp(new ParameterList);

          p->set<int>("Type", traits_type::type);

          p->set<RCP<ParamLib>>("Parameter Library", paramLib);

          p->set<string>("Side Set ID", meshSpecs->ssNames[i]);
          p->set<Teuchos::Array<int>>("Equation Offset", offsets[j]);
          p->set<RCP<Albany::Layouts>>("Layouts Struct", dl);
          p->set<RCP<MeshSpecsStruct>>("Mesh Specs Struct", meshSpecs);

          p->set<string>("Coordinate Vector Name", "Coord Vec");
          p->set<int>(
              "Cubature Degree",
              BCparams.get("Cubature Degree", 0));  // if set to zero, the
                                                    // cubature degree of the
                                                    // side will be set to that
                                                    // of the element

          if (conditions[k] == "robin" || conditions[k] == "radiate") {
            p->set<string>("DOF Name", dof_names[j]);
            p->set<bool>("Vector Field", isVectorField);

            if (isVectorField)
              p->set<RCP<DataLayout>>("DOF Data Layout", dl->node_vector);
            else
              p->set<RCP<DataLayout>>("DOF Data Layout", dl->node_scalar);
          }

          // Pass the input file line
          p->set<string>("Neumann Input String", ss);
          
          Teuchos::Array<double> niv = BCparams.get<Teuchos::Array<double>>(ss); 
          // Note, we use a Teuchos::Array  here to allow the user to specify
          // multiple components of the traction vector.
          // This is only allowed for certain BCs (see how allowArrayNBC) is set.
          if (!allowArrayNBC) {
            if (niv.size() != 1) {
              ALBANY_ASSERT(false, "NBC takes a scalar value.  You attempted to provide an array!");  
            }
          }
          else {
            if ((conditions[k] == "robin") || (conditions[k] == "radiate")) {
              if (niv.size() != 2) {
                ALBANY_ASSERT(false, "Robin NBC takes a 2-array!");  
              }
            }
            else {
              if (niv.size() != meshSpecs->numDim) {
                ALBANY_ASSERT(false, "Traction NBC takes an array of size numDim!");  
              }
            }
          }
          
          p->set<Teuchos::Array<double>>("Neumann Input Value", niv); 
          p->set<string>("Neumann Input Conditions", conditions[k]);

          // If we are doing a Neumann internal boundary with a "scaled jump",
          // the material DB database needs to be passed to the BC object
          // Note: 'robin' is a very generic name. It is ok to allow some
          // 'complex'
          //       robin conditions (with a scaled jump), but we should allow
          //       one to use 'robin' bc for the classic du/dn + alpha*u = g,
          //       which means the user should not have to specify a material DB

          if (conditions[k] == "scaled jump" || conditions[k] == "radiate") {
            TEUCHOS_TEST_FOR_EXCEPTION(
                materialDB == Teuchos::null,
                Teuchos::Exceptions::InvalidParameter,
                "This BC needs a material database specified");
          }
          p->set<RCP<Albany::MaterialDatabase>>("MaterialDB", materialDB);

          // Inputs: X, Y at nodes, Cubature, and Basis
          // p->set<string>("Node Variable Name", "Neumann");

          evaluators_to_build[evaluatorsToBuildName(ss)] = p;

          bcs->push_back(ss);
        }
      }
    }
  }

#if defined(ALBANY_LCM)

  ///
  /// Time dependent BC specific
  ///
  for (std::size_t i = 0; i < meshSpecs->ssNames.size(); i++) {
    for (std::size_t j = 0; j < bcNames.size(); j++) {
      for (std::size_t k = 0; k < conditions.size(); k++) {
        // construct input.xml string like:
        // "Time Dependent NBC on SS sidelist_12 for DOF T set dudn"
        //  or
        // "Time Dependent NBC on SS sidelist_12 for DOF T set (dudx, dudy)"
        // or
        // "Time Dependent NBC on SS surface_1 for DOF all set P"

        // Set logic for certain NBCs which allow array inputs  
        bool allowArrayNBC = false;   
        if ((conditions[k] == "robin") || (conditions[k] == "radiate")
              || (conditions[k].find("(") < conditions[k].length())) {
          allowArrayNBC = true; 
        }

        string ss = traits_type::constructTimeDepBCName(
            meshSpecs->ssNames[i], bcNames[j], conditions[k]);

        // Have a match of the line in input.xml

        if (BCparams.isSublist(ss)) {
          // grab the sublist
          ParameterList& sub_list = BCparams.sublist(ss);

          //           std::cout << "Constructing Time Dependent NBC: " << ss <<
          //           std::endl;

          // These are read in the LCM::TimeTracBC constructor
          // (LCM/evaluators/TimeTrac_Def.hpp)

          RCP<ParameterList> p = rcp(new ParameterList);

          p->set<int>("Type", traits_type::typeTd);

          Teuchos::Array<RealType> timevals = sub_list.get<Teuchos::Array<RealType>>("Time Values");
          
          // Note, we use a TwoDArray here to allow the user to specify
          // multiple components of the traction vector at each "time" step.
          // This is only allowed for certain BCs (see how allowArrayNBC) is set. 
          Teuchos::TwoDArray<RealType> bcvals = sub_list.get<Teuchos::TwoDArray<RealType>>("BC Values");
         
          // Check that bcvals and timevals have the same size.  If they do not, throw an error.
          if (timevals.size() != bcvals.getNumRows()) {
            ALBANY_ASSERT(false, "'Time Values' array must have same length as 'BC Values' array!");
          } 

          // IKT, 2/15/2020: Currently, the code downstream of this
          // assumes bcvals is a scalar for all but a few NBCs (see comment above).  
          // Throw an error if user attempts to specify array for NBCs where this 
          // is not allowed.
          if (!allowArrayNBC) {
            if (bcvals.getNumCols() != 1) {
              ALBANY_ASSERT(false, "Time Dependent NBC takes 1D array for 'BC Values'.  You attempted to provide a multi-D array!");  
            }
          }
          else {
            if ((conditions[k] == "robin") || (conditions[k] == "radiate")) {
              if (bcvals.getNumCols() != 2) {
                ALBANY_ASSERT(false, "Time Dependent robin NBC takes a 2-array for 'BC Values' at each time!");  
              }
            }
            else {
              if (bcvals.getNumCols() != meshSpecs->numDim) {
                ALBANY_ASSERT(false, "Time Dependent traction NBC takes an array of size numDim for 'BC Values' at each time!");  
              }
            }
          }

          p->set<Teuchos::Array<RealType>>("Time Values", timevals); 

          p->set<Teuchos::TwoDArray<RealType>>("BC Values", bcvals); 

          p->set<RCP<ParamLib>>("Parameter Library", paramLib);

          p->set<string>("Side Set ID", meshSpecs->ssNames[i]);
          p->set<Teuchos::Array<int>>("Equation Offset", offsets[j]);
          p->set<RCP<Albany::Layouts>>("Layouts Struct", dl);
          p->set<RCP<MeshSpecsStruct>>("Mesh Specs Struct", meshSpecs);
          p->set<int>(
              "Cubature Degree",
              BCparams.get("Cubature Degree", 0));  // if set to zero, the
                                                    // cubature degree of the
                                                    // side will be set to that
                                                    // of the element

          p->set<string>("Coordinate Vector Name", "Coord Vec");

          if (conditions[k] == "robin") {
            p->set<string>("DOF Name", dof_names[j]);
            p->set<bool>("Vector Field", isVectorField);

            if (isVectorField)
              p->set<RCP<DataLayout>>("DOF Data Layout", dl->node_vector);
            else
              p->set<RCP<DataLayout>>("DOF Data Layout", dl->node_scalar);
          }

          // Pass the input file line
          p->set<string>("Neumann Input String", ss);
          p->set<Teuchos::Array<double>>(
              "Neumann Input Value", Teuchos::tuple<double>(0.0, 0.0, 0.0));
          p->set<string>("Neumann Input Conditions", conditions[k]);

          // If we are doing a Neumann internal boundary with a "scaled jump"
          // (includes "robin" too)
          // The material DB database needs to be passed to the BC object

          if (conditions[k] == "scaled jump" || conditions[k] == "robin") {
            TEUCHOS_TEST_FOR_EXCEPTION(
                materialDB == Teuchos::null,
                Teuchos::Exceptions::InvalidParameter,
                "This BC needs a material database specified");

            p->set<RCP<Albany::MaterialDatabase>>("MaterialDB", materialDB);
          }

          evaluators_to_build[evaluatorsToBuildName(ss)] = p;

          bcs->push_back(ss);
        }
      }
    }
  }

#endif

  // Build evaluator for Gather Coordinate Vector
  string NeuGCV = "Evaluator for Gather Coordinate Vector";
  {
    RCP<ParameterList> p = rcp(new ParameterList);
    p->set<int>("Type", traits_type::typeGCV);

    // Input: Periodic BC flag
    p->set<bool>("Periodic BC", false);

    // Output:: Coordindate Vector at vertices
    p->set<RCP<DataLayout>>("Coordinate Data Layout", dl->vertices_vector);
    p->set<string>("Coordinate Vector Name", "Coord Vec");

    evaluators_to_build[NeuGCV] = p;
  }

  // Build evaluator for Gather Solution
  string NeuGS = "Evaluator for Gather Solution";
  {
    RCP<ParameterList> p = rcp(new ParameterList());
    p->set<int>("Type", traits_type::typeGS);

    // for new way
    p->set<RCP<Albany::Layouts>>("Layouts Struct", dl);

    p->set<Teuchos::ArrayRCP<string>>("Solution Names", dof_names);

    p->set<bool>("Vector Field", isVectorField);

    if (isVectorField)
      p->set<RCP<DataLayout>>("Data Layout", dl->node_vector);

    else
      p->set<RCP<DataLayout>>("Data Layout", dl->node_scalar);

    p->set<int>("Offset of First DOF", offsetToFirstDOF);
    p->set<bool>("Disable Transient", true);

    evaluators_to_build[NeuGS] = p;
  }

  // Build evaluator that causes the evaluation of all the NBCs
  string allBC = "Evaluator for all Neumann BCs";
  {
    RCP<ParameterList> p = rcp(new ParameterList);
    p->set<int>("Type", traits_type::typeNa);

    p->set<RCP<std::vector<string>>>("NBC Names", bcs);
    p->set<RCP<DataLayout>>("Data Layout", dl->dummy);
    p->set<string>("NBC Aggregator Name", allBC);
    evaluators_to_build[allBC] = p;
  }
}

template <typename BCTraits>
Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>>
Albany::BCUtils<BCTraits>::buildFieldManager(
    const Teuchos::RCP<std::vector<Teuchos::RCP<
        PHX::Evaluator_TemplateManager<PHAL::AlbanyTraits>>>> evaluators,
    std::string&                                              allBC,
    Teuchos::RCP<PHX::DataLayout>&                            dummy)
{
  using PHAL::AlbanyTraits;

  // Create a DirichletFieldManager
  Teuchos::RCP<PHX::FieldManager<AlbanyTraits>> fm =
      Teuchos::rcp(new PHX::FieldManager<AlbanyTraits>);

  // Register all Evaluators
  PHX::registerEvaluators(evaluators, *fm);

  PHX::Tag<AlbanyTraits::Residual::ScalarT> res_tag0(allBC, dummy);
  fm->requireField<AlbanyTraits::Residual>(res_tag0);

  PHX::Tag<AlbanyTraits::Jacobian::ScalarT> jac_tag0(allBC, dummy);
  fm->requireField<AlbanyTraits::Jacobian>(jac_tag0);

  PHX::Tag<AlbanyTraits::Tangent::ScalarT> tan_tag0(allBC, dummy);
  fm->requireField<AlbanyTraits::Tangent>(tan_tag0);

  PHX::Tag<AlbanyTraits::DistParamDeriv::ScalarT> dpd_tag0(allBC, dummy);
  fm->requireField<AlbanyTraits::DistParamDeriv>(dpd_tag0);

  return fm;
}

// Various specializations

Teuchos::RCP<const Teuchos::ParameterList>
Albany::DirichletTraits::getValidBCParameters(
    const std::vector<std::string>& nodeSetIDs,
    const std::vector<std::string>& bcNames)
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
      Teuchos::rcp(new Teuchos::ParameterList("Valid Dirichlet BC List"));

  for (std::size_t i = 0; i < nodeSetIDs.size(); i++) {
    for (std::size_t j = 0; j < bcNames.size(); j++) {
      std::string ss =
          Albany::DirichletTraits::constructBCName(nodeSetIDs[i], bcNames[j]);
      std::string tt = Albany::DirichletTraits::constructTimeDepBCName(
          nodeSetIDs[i], bcNames[j]);
      std::string ts = Albany::DirichletTraits::constructTimeDepSDBCName(
          nodeSetIDs[i], bcNames[j]);
      std::string pp = Albany::DirichletTraits::constructPressureDepBCName(
          nodeSetIDs[i], bcNames[j]);
      std::string ee = Albany::DirichletTraits::constructExprEvalSDBCName(
          nodeSetIDs[i], bcNames[j]);
      std::string st =
          Albany::DirichletTraits::constructSDBCName(nodeSetIDs[i], bcNames[j]);
      std::string sst = Albany::DirichletTraits::constructScaledSDBCName(
          nodeSetIDs[i], bcNames[j]);
      validPL->set<double>(
          ss, 0.0, "Value of BC corresponding to nodeSetID and dofName");
      validPL->set<double>(
          st, 0.0, "Value of SDBC corresponding to nodeSetID and dofName");
      validPL->set<std::string>(
          ee,
          "0.0",
          "Expression of SDBC corresponding to nodeSetID and dofName");
      Teuchos::Array<double> array(1);
      array[0] = 0.0;
      validPL->set<Teuchos::Array<double>>(
          sst,
          array,
          "Value of Scaled SDBC corresponding to nodeSetID and dofName");
      validPL->sublist(
          tt, false, "SubList of BC corresponding to nodeSetID and dofName");
      validPL->sublist(
          ts, false, "SubList of SDBC corresponding to nodeSetID and dofName");
      validPL->sublist(
          pp, false, "SubList of BC corresponding to nodeSetID and dofName");
      ss = Albany::DirichletTraits::constructBCNameField(
          nodeSetIDs[i], bcNames[j]);
      st = Albany::DirichletTraits::constructSDBCNameField(
          nodeSetIDs[i], bcNames[j]);
      ee = Albany::DirichletTraits::constructExprEvalSDBCNameField(
          nodeSetIDs[i], bcNames[j]);
      sst = Albany::DirichletTraits::constructScaledSDBCNameField(
          nodeSetIDs[i], bcNames[j]);
      validPL->set<std::string>(
          ss, "dirichlet field", "Field used to prescribe Dirichlet BCs");
      validPL->set<std::string>(
          st, "dirichlet field", "Field used to prescribe SDBCs");
      validPL->set<std::string>(
          ee, "dirichlet field", "Field used to prescribe Expression SDBCs");
      std::string onsbc = Albany::DirichletTraits::constructBCNameOffNodeSet(
          nodeSetIDs[i], bcNames[j]);
      validPL->set<double>(
          onsbc,
          0.0,
          "Value of BC to prescribe off the given nodeset (use multiple "
          "entries for multiple nodesets)");
    }
  }

  for (std::size_t i = 0; i < nodeSetIDs.size(); i++) {
    std::string ss =
        Albany::DirichletTraits::constructBCName(nodeSetIDs[i], "K");
    std::string tt =
        Albany::DirichletTraits::constructBCName(nodeSetIDs[i], "twist");
    std::string ww =
        Albany::DirichletTraits::constructBCName(nodeSetIDs[i], "Schwarz");
    std::string sw = Albany::DirichletTraits::constructSDBCName(
        nodeSetIDs[i], "StrongSchwarz");
    std::string uu =
        Albany::DirichletTraits::constructBCName(nodeSetIDs[i], "CoordFunc");
    std::string pd =
        Albany::DirichletTraits::constructBCName(nodeSetIDs[i], "lsfit");
    validPL->sublist(ss, false, "");
    validPL->sublist(tt, false, "");
    validPL->sublist(ww, false, "");
    validPL->sublist(sw, false, "");
    validPL->sublist(uu, false, "");
    validPL->sublist(pd, false, "");
  }

  return validPL;
}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::NeumannTraits::getValidBCParameters(
    const std::vector<std::string>& sideSetIDs,
    const std::vector<std::string>& bcNames,
    const std::vector<std::string>& conditions)
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
      Teuchos::rcp(new Teuchos::ParameterList("Valid Neumann BC List"));
  ;

  for (std::size_t i = 0; i < sideSetIDs.size();
       i++) {  // loop over all side sets in the mesh
    for (std::size_t j = 0; j < bcNames.size();
         j++) {  // loop over all possible types of condition
      for (std::size_t k = 0; k < conditions.size();
           k++) {  // loop over all possible types of condition

        std::string ss = Albany::NeumannTraits::constructBCName(
            sideSetIDs[i], bcNames[j], conditions[k]);
        std::string tt = Albany::NeumannTraits::constructTimeDepBCName(
            sideSetIDs[i], bcNames[j], conditions[k]);

        /*
         if(numDim == 2)
         validPL->set<Teuchos::Array<double>>(ss, Teuchos::tuple<double>(0.0,
         0.0),
         "Value of BC corresponding to sideSetID and boundary condition");
         else
         validPL->set<Teuchos::Array<double>>(ss, Teuchos::tuple<double>(0.0,
         0.0, 0.0),
         "Value of BC corresponding to sideSetID and boundary condition");
         */
        Teuchos::Array<double> defaultData;
        validPL->set<Teuchos::Array<double>>(
            ss,
            defaultData,
            "Value of BC corresponding to sideSetID and boundary condition");

        validPL->sublist(
            tt,
            false,
            "SubList of BC corresponding to sideSetID and boundary condition");
      }
    }
  }

  validPL->set<int>("Cubature Degree", 3, "Cubature Degree for Neumann BC");
  return validPL;
}

std::string
Albany::DirichletTraits::constructBCName(
    const std::string& ns,
    const std::string& dof)
{
  std::stringstream ss;
  ss << "DBC on NS " << ns << " for DOF " << dof;

  return ss.str();
}

std::string
Albany::DirichletTraits::constructSDBCName(
    const std::string& ns,
    const std::string& dof)
{
  std::stringstream ss;
  ss << "SDBC on NS " << ns << " for DOF " << dof;

  return ss.str();
}

std::string
Albany::DirichletTraits::constructExprEvalSDBCName(
    std::string const& ns,
    std::string const& dof)
{
  std::stringstream ss;
  ss << "ExpressionEvaluated SDBC on NS " << ns << " for DOF " << dof;

  return ss.str();
}

std::string
Albany::DirichletTraits::constructScaledSDBCName(
    const std::string& ns,
    const std::string& dof)
{
  std::stringstream ss;
  ss << "Scaled SDBC on NS " << ns << " for DOF " << dof;

  return ss.str();
}

std::string
Albany::DirichletTraits::constructBCNameField(
    const std::string& ns,
    const std::string& dof)
{
  std::stringstream ss;
  ss << "DBC on NS " << ns << " for DOF " << dof << " prescribe Field";

  return ss.str();
}

std::string
Albany::DirichletTraits::constructSDBCNameField(
    const std::string& ns,
    const std::string& dof)
{
  std::stringstream ss;
  ss << "SDBC on NS " << ns << " for DOF " << dof << " prescribe Field";

  return ss.str();
}

std::string
Albany::DirichletTraits::constructExprEvalSDBCNameField(
    std::string const& ns,
    std::string const& dof)
{
  std::stringstream ss;
  ss << "ExpressionEvaluated SDBC on NS " << ns << " for DOF " << dof
     << " prescribe Field";

  return ss.str();
}


std::string
Albany::DirichletTraits::constructScaledSDBCNameField(
    const std::string& ns,
    const std::string& dof)
{
  std::stringstream ss;
  ss << "Scaled SDBC on NS " << ns << " for DOF " << dof << " prescribe Field";

  return ss.str();
}

std::string
Albany::NeumannTraits::constructBCName(
    const std::string& ns,
    const std::string& dof,
    const std::string& condition)
{
  std::stringstream ss;
  ss << "NBC on SS " << ns << " for DOF " << dof << " set " << condition;
  return ss.str();
}

std::string
Albany::DirichletTraits::constructTimeDepBCName(
    const std::string& ns,
    const std::string& dof)
{
  std::stringstream ss;
  ss << "Time Dependent " << Albany::DirichletTraits::constructBCName(ns, dof);
  return ss.str();
}

std::string
Albany::DirichletTraits::constructTimeDepSDBCName(
    const std::string& ns,
    const std::string& dof)
{
  std::stringstream ss;
  ss << "Time Dependent "
     << Albany::DirichletTraits::constructSDBCName(ns, dof);
  return ss.str();
}

std::string
Albany::DirichletTraits::constructBCNameOffNodeSet(
    const std::string& ns,
    const std::string& dof)
{
  std::stringstream ss;
  ss << "DBC off NS " << ns << " for DOF " << dof;

  return ss.str();
}

std::string
Albany::DirichletTraits::constructPressureDepBCName(
    const std::string& ns,
    const std::string& dof)
{
  std::stringstream ss;
  ss << "Pressure Dependent "
     << Albany::DirichletTraits::constructBCName(ns, dof);
  return ss.str();
}

std::string
Albany::NeumannTraits::constructTimeDepBCName(
    const std::string& ns,
    const std::string& dof,
    const std::string& condition)
{
  std::stringstream ss;
  ss << "Time Dependent "
     << Albany::NeumannTraits::constructBCName(ns, dof, condition);
  return ss.str();
}
