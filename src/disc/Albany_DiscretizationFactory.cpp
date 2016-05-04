//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include "Teuchos_TestForException.hpp"
#include "Albany_DiscretizationFactory.hpp"
#if defined(HAVE_STK)
#include "Albany_STKDiscretization.hpp"
#ifdef ALBANY_AERAS
#include "Aeras_SpectralDiscretization.hpp"
#endif
#include "Albany_TmplSTKMeshStruct.hpp"
#include "Albany_STK3DPointStruct.hpp"
#include "Albany_GenericSTKMeshStruct.hpp"
#include "Albany_SideSetSTKMeshStruct.hpp"

#ifdef ALBANY_SEACAS
#include "Albany_IossSTKMeshStruct.hpp"
#endif
#if defined(ALBANY_EPETRA)
#include "Albany_AsciiSTKMeshStruct.hpp"
#include "Albany_AsciiSTKMesh2D.hpp"
#include "Albany_GmshSTKMeshStruct.hpp"
#ifdef ALBANY_FELIX
#include "Albany_ExtrudedSTKMeshStruct.hpp"
#endif
#endif
#ifdef ALBANY_FELIX
#include "Albany_STKDiscretizationStokesH.hpp"
#endif
#ifdef ALBANY_CUTR
#include "Albany_FromCubitSTKMeshStruct.hpp"
#endif
#endif
#ifdef ALBANY_SCOREC
#include "Albany_PUMIDiscretization.hpp"
#include "Albany_PUMIMeshStruct.hpp"
#endif
#ifdef ALBANY_GOAL
#include "Albany_GOALDiscretization.hpp"
#include "Albany_GOALMeshStruct.hpp"
#endif
#ifdef ALBANY_AMP
#include "Albany_SimDiscretization.hpp"
#include "Albany_SimMeshStruct.hpp"
#endif
#ifdef ALBANY_CATALYST
#include "Albany_Catalyst_Decorator.hpp"
#endif

#if defined(ALBANY_LCM) && defined(HAVE_STK) && defined(ALBANY_BGL)
#include "Topology_Utils.h"
#endif // ALBANY_LCM

Albany::DiscretizationFactory::DiscretizationFactory(
  const Teuchos::RCP<Teuchos::ParameterList>& topLevelParams,
  const Teuchos::RCP<const Teuchos_Comm>& commT_,
  const bool explicit_scheme_) :
  commT(commT_),
  explicit_scheme(explicit_scheme_) {

  discParams = Teuchos::sublist(topLevelParams, "Discretization", true);

  if(topLevelParams->isSublist("Piro"))

    piroParams = Teuchos::sublist(topLevelParams, "Piro", true);

  if(topLevelParams->isSublist("Problem")) {

    Teuchos::RCP<Teuchos::ParameterList> problemParams = Teuchos::sublist(topLevelParams, "Problem", true);

    if(problemParams->isSublist("Adaptation"))

      adaptParams = Teuchos::sublist(problemParams, "Adaptation", true);

    if(problemParams->isSublist("Catalyst"))

      catalystParams = Teuchos::sublist(problemParams, "Catalyst", true);

#ifdef ALBANY_AERAS
    Teuchos::RCP<Teuchos::ParameterList> hsParams;
    Teuchos::ArrayRCP<std::string> dof_names_tracers;
    if (problemParams->isSublist("Hydrostatic Problem")) {
      hsParams = Teuchos::sublist(problemParams, "Hydrostatic Problem", true);
      numLevels = hsParams->get("Number of Vertical Levels", 0);
      dof_names_tracers = arcpFromArray(hsParams->get<Teuchos::Array<std::string> >("Tracers",
            Teuchos::Array<std::string>()));
      numTracers = dof_names_tracers.size();

    }

    if (problemParams->isSublist("XZHydrostatic Problem")) {
      hsParams = Teuchos::sublist(problemParams, "XZHydrostatic Problem", true);
      numLevels = hsParams->get("Number of Vertical Levels", 0);
      dof_names_tracers = arcpFromArray(hsParams->get<Teuchos::Array<std::string> >("Tracers",
            Teuchos::Array<std::string>()));
      numTracers = dof_names_tracers.size();
    }
    if (problemParams->isSublist("Shallow Water Problem")) {
      numLevels = 0;
    }
#endif

  }

}

#ifdef ALBANY_CUTR
void
Albany::DiscretizationFactory::setMeshMover(const Teuchos::RCP<CUTR::CubitMeshMover>& meshMover_) {
  meshMover = meshMover_;
}
#endif

#if defined(ALBANY_LCM)

namespace {

void createInterfaceParts(
    Teuchos::RCP<Teuchos::ParameterList> const & adapt_params,
    Teuchos::RCP<Albany::AbstractMeshStruct> & mesh_struct
    )
{
#if defined(HAVE_STK) && defined(ALBANY_BGL) // LCM only uses STK for adaptation here
                                             // Top mod uses BGL
  bool const
  do_adaptation = adapt_params.is_null() == false;

  if (do_adaptation == false) return;

  std::string const &
  adaptation_method_name = adapt_params->get<std::string>("Method");

  bool const
  is_topology_modification = adaptation_method_name == "Topmod";

  if (is_topology_modification == false) return;

  std::string const &
  bulk_part_name = adapt_params->get<std::string>("Bulk Block Name");

  Albany::AbstractSTKMeshStruct &
  stk_mesh_struct = dynamic_cast<Albany::AbstractSTKMeshStruct &>(*mesh_struct);

  stk::mesh::MetaData &
  meta_data = *(stk_mesh_struct.metaData);

  stk::mesh::Part &
  bulk_part = *(meta_data.get_part(bulk_part_name));

  shards::CellTopology const &
  bulk_cell_topology = meta_data.get_cell_topology(bulk_part);

  std::string const &
  interface_part_name(adapt_params->get<std::string>("Interface Block Name"));

  shards::CellTopology const
  interface_cell_topology =
      LCM::interfaceCellTopogyFromBulkCellTopogy(bulk_cell_topology);

  stk::mesh::EntityRank const
  interface_dimension = static_cast<stk::mesh::EntityRank>(
      interface_cell_topology.getDimension());

  stk::mesh::Part &
  interface_part =
      meta_data.declare_part(interface_part_name, interface_dimension);

  stk::mesh::set_cell_topology(interface_part, interface_cell_topology);

#ifdef ALBANY_SEACAS
  stk::io::put_io_part_attribute(interface_part);
#endif // ALBANY_SEACAS

  // Augment the MeshSpecsStruct array with one additional entry for
  // the interface block. Essentially copy the last entry from the array
  // and modify some of its fields as needed.
  Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> > &
  mesh_specs_struct = stk_mesh_struct.getMeshSpecs();

  Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >::size_type
  number_blocks = mesh_specs_struct.size();

  Albany::MeshSpecsStruct &
  last_mesh_specs_struct = *(mesh_specs_struct[number_blocks - 1]);

  CellTopologyData const &
  interface_cell_topology_data =
      *(interface_cell_topology.getCellTopologyData());

  int const
  dimension = interface_cell_topology.getDimension();

  int const
  cubature_degree = last_mesh_specs_struct.cubatureDegree;

  std::vector<std::string>
  node_sets, side_sets;

  int const
  workset_size = last_mesh_specs_struct.worksetSize;

  std::string const &
  element_block_name = interface_part_name;

  std::map<std::string, int> &
  eb_name_to_index_map = last_mesh_specs_struct.ebNameToIndex;

  // Add entry to the map for this block
  eb_name_to_index_map.insert(
      std::make_pair(element_block_name, number_blocks));

  bool const
  is_interleaved = last_mesh_specs_struct.interleavedOrdering;

  Intrepid2::EIntrepidPLPoly const
  cubature_rule = last_mesh_specs_struct.cubatureRule;

  mesh_specs_struct.resize(number_blocks + 1);

  mesh_specs_struct[number_blocks] =
      Teuchos::rcp(
          new Albany::MeshSpecsStruct(
              interface_cell_topology_data,
              dimension,
              cubature_degree,
              node_sets,
              side_sets,
              workset_size,
              element_block_name,
              eb_name_to_index_map,
              is_interleaved,
              number_blocks > 1,
              cubature_rule));
#endif
  return;
}

} // anonymous namespace

#endif //ALBANY_LCM


Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >
Albany::DiscretizationFactory::createMeshSpecs()
{
  // First, create the mesh struct
#ifdef ALBANY_CUTR
  // Luca: WARNING, this does not compile. Frankly, I don't even know how it worked before in master,
  //       since neq was never available...
  int neq = 1;  // Hard coded neq=1. I have no idea where this number was supposed to be fetched from otherwise...
  meshStruct = createMeshStruct (discParams, adaptParams, commT, meshMover, neq);
#else
  meshStruct = createMeshStruct (discParams, adaptParams, commT);
#endif

#if defined(ALBANY_LCM) && defined(HAVE_STK)
  // Add an interface block. For now relies on STK, so we force a cast that
  // will fail if the underlying meshStruct is not based on STK.
  createInterfaceParts(adaptParams, meshStruct);
#endif // ALBANY_LCM

#if defined(ALBANY_AERAS) && defined(HAVE_STK)
  //IK, 2/9/15: if the method is Ioss Aeras or Exodus Aeras (corresponding to Aeras::SpectralDiscretization,
  //overwrite the meshSpecs of the meshStruct with an enriched one.
  std::string& method = discParams->get("Method", "STK1D");
  if (method == "Ioss Aeras" || method == "Exodus Aeras" || method == "STK1D Aeras") {
    //get "Element Degree" from parameter list.  Default value is 1.
    int points_per_edge = discParams->get("Element Degree", 1) + 1;
    Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> > &mesh_specs_struct = meshStruct->getMeshSpecs();
    Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >::size_type number_blocks = mesh_specs_struct.size();
    Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> > enriched_mesh_specs_struct;
    enriched_mesh_specs_struct.resize(number_blocks);
    for (Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >::size_type i=0; i< number_blocks; i++) {
      Teuchos::RCP<Albany::MeshSpecsStruct> orig_mesh_specs_struct = mesh_specs_struct[i];
      Aeras::AerasMeshSpectStruct aeras_mesh_specs_struct;
      enriched_mesh_specs_struct[i] = aeras_mesh_specs_struct.createAerasMeshSpecs(orig_mesh_specs_struct,
                                                                                   points_per_edge, discParams);
    }
    return enriched_mesh_specs_struct;
  }
  else
#endif
    return meshStruct->getMeshSpecs();
}

Teuchos::RCP<Albany::AbstractMeshStruct>
#ifdef ALBANY_CUTR
Albany::DiscretizationFactory::createMeshStruct (Teuchos::RCP<Teuchos::ParameterList> disc_params,
                                                 Teuchos::RCP<Teuchos::ParameterList> adapt_params,
                                                 Teuchos::RCP<const Teuchos_Comm> comm,
                                                 Teuchos::RCP<CUTR::CubitMeshMover> mesh_mover,
                                                 int num_eq)
#else
Albany::DiscretizationFactory::createMeshStruct (Teuchos::RCP<Teuchos::ParameterList> disc_params,
                                                 Teuchos::RCP<Teuchos::ParameterList> adapt_params,
                                                 Teuchos::RCP<const Teuchos_Comm> comm)
#endif
{
  std::string& method = disc_params->get("Method", "STK1D");
#if defined(HAVE_STK)
  if(method == "STK1D" || method == "STK1D Aeras") {
    return Teuchos::rcp(new Albany::TmplSTKMeshStruct<1>(disc_params, adapt_params, comm));
  }

  else if(method == "STK0D") {
    return Teuchos::rcp(new Albany::TmplSTKMeshStruct<0>(disc_params, adapt_params, comm));
  }

  else if(method == "STK2D") {
    return Teuchos::rcp(new Albany::TmplSTKMeshStruct<2>(disc_params, adapt_params, comm));
  }

  else if(method == "STK3D") {
    return Teuchos::rcp(new Albany::TmplSTKMeshStruct<3>(disc_params, adapt_params, comm));
  }

  else if(method == "STK3DPoint") {
    return Teuchos::rcp(new Albany::STK3DPointStruct(disc_params, comm));
  }

  else if(method == "Ioss" || method == "Exodus" ||  method == "Pamgen" || method == "Ioss Aeras" || method == "Exodus Aeras") {

#ifdef ALBANY_SEACAS
    return Teuchos::rcp(new Albany::IossSTKMeshStruct(disc_params, adapt_params, comm));
#else
    TEUCHOS_TEST_FOR_EXCEPTION(method == "Ioss" || method == "Exodus" ||  method == "Pamgen" || method == "Ioss Aeras" ||
                               method == "Exodus Aeras",
                               Teuchos::Exceptions::InvalidParameter,
                               "Error: Discretization method " << method
                               << " requested, but not compiled in" << std::endl);
#endif // ALBANY_SEACAS
  }
#if defined(ALBANY_EPETRA)
  else if(method == "Ascii") {
    return Teuchos::rcp(new Albany::AsciiSTKMeshStruct(disc_params, comm));
  }
  else if(method == "Ascii2D") {
    return Teuchos::rcp(new Albany::AsciiSTKMesh2D(disc_params, comm));
  }
  else if(method == "Gmsh") {
    return Teuchos::rcp(new Albany::GmshSTKMeshStruct(disc_params, comm));
  }
#ifdef ALBANY_FELIX
  else if(method == "Extruded")
  {
    Teuchos::RCP<Albany::AbstractMeshStruct> basalMesh;
    Teuchos::RCP<Teuchos::ParameterList> basal_params;
    if (disc_params->isSublist("Side Set Discretizations") && disc_params->sublist("Side Set Discretizations").isSublist("basalside"))
    {
      basal_params = Teuchos::rcp(new Teuchos::ParameterList(disc_params->sublist("Side Set Discretizations").sublist("basalside")));
    }
    else
    {
      // Backward compatibility: Ioss, with parameters mixed with the extruded mesh ones
      basal_params->set("Method","Ioss");
      basal_params->set("Use Serial Mesh", disc_params->get("Use Serial Mesh", false));
      basal_params->set("Exodus Input File Name", disc_params->get("Exodus Input File Name", "basalmesh.exo"));
    }

    basalMesh = createMeshStruct(basal_params, Teuchos::null, comm);
    return Teuchos::rcp(new Albany::ExtrudedSTKMeshStruct(disc_params, comm, basalMesh));
  }
#endif // ALBANY_FELIX
#endif // ALBANY_EPETRA
  else if(method == "Cubit") {
#ifdef ALBANY_CUTR
    // AGS"need to inherit from Generic"
    return Teuchos::rcp(new Albany::FromCubitSTKMeshStruct(mesh_mover, disc_params, num_eq));
#else
    TEUCHOS_TEST_FOR_EXCEPTION(method == "Cubit",
                               Teuchos::Exceptions::InvalidParameter,
                               "Error: Discretization method " << method
                               << " requested, but not compiled in" << std::endl);
#endif // ALBANY_CUTR
  }
  else
#endif // HAVE_STK
  if(method == "PUMI") {
#ifdef ALBANY_SCOREC
    return Teuchos::rcp(new Albany::PUMIMeshStruct(disc_params, comm));
#else
    TEUCHOS_TEST_FOR_EXCEPTION(method == "PUMI",
                               Teuchos::Exceptions::InvalidParameter,
                               "Error: Discretization method " << method
                               << " requested, but not compiled in" << std::endl);
#endif
  }
  else if(method == "PUMI Hierarchic") {
#ifdef ALBANY_GOAL
    return Teuchos::rcp(new Albany::GOALMeshStruct(disc_params, comm));
#else
    TEUCHOS_TEST_FOR_EXCEPTION(method == "PUMI Hierarchic",
                               Teuchos::Exceptions::InvalidParameter,
                               "Error: Discretization method " << method
                               << " requested, but not compiled in" << std::endl);
#endif
  }
  else if (method == "Sim") {
#ifdef ALBANY_AMP
    return Teuchos::rcp(new Albany::SimMeshStruct(disc_params, comm));
#else
    TEUCHOS_TEST_FOR_EXCEPTION(method == "Sim",
                               Teuchos::Exceptions::InvalidParameter,
                               "Error: Discretization method " << method
                               << " requested, but not compiled in" << std::endl);
#endif
  }
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter, std::endl <<
                               "Error!  Unknown discretization method in DiscretizationFactory: " << method <<
                               "!" << std::endl << "Supplied parameter list is " << std::endl << *disc_params <<
                               "\nValid Methods are: STK1D, STK2D, STK3D, STK3DPoint, Ioss, Ioss Aeras," <<
                               " Exodus, Exodus Aeras, Cubit, PUMI, PUMI Hierarchic, Sim, Ascii," <<
                               " Ascii2D, Extruded" << std::endl);
  }
}


Teuchos::RCP<Albany::AbstractDiscretization>
Albany::DiscretizationFactory::createDiscretization(unsigned int neq,
    const Teuchos::RCP<Albany::StateInfoStruct>& sis,
    const AbstractFieldContainer::FieldContainerRequirements& req,
    const Teuchos::RCP<Albany::RigidBodyModes>& rigidBodyModes) {
  createDiscretization(neq, empty_side_set_equations, sis, empty_side_set_sis, req, empty_side_set_req, rigidBodyModes);
}

Teuchos::RCP<Albany::AbstractDiscretization>
Albany::DiscretizationFactory::createDiscretization(
    unsigned int neq, const std::map<int,std::vector<std::string> >& sideSetEquations,
    const Teuchos::RCP<Albany::StateInfoStruct>& sis,
    const std::map<std::string,Teuchos::RCP<Albany::StateInfoStruct> >& side_set_sis,
    const AbstractFieldContainer::FieldContainerRequirements& req,
    const std::map<std::string,AbstractFieldContainer::FieldContainerRequirements>& side_set_req,
    const Teuchos::RCP<Albany::RigidBodyModes>& rigidBodyModes) {
  TEUCHOS_TEST_FOR_EXCEPTION(meshStruct == Teuchos::null,
                             std::logic_error,
                             "meshStruct accessed, but it has not been constructed" << std::endl);

  setupInternalMeshStruct(neq, sis, side_set_sis, req, side_set_req);
  Teuchos::RCP<Albany::AbstractDiscretization> result =
      createDiscretizationFromInternalMeshStruct(sideSetEquations,rigidBodyModes);

  // Wrap the discretization in the catalyst decorator if needed.
#ifdef ALBANY_CATALYST

  if(Teuchos::nonnull(catalystParams) && catalystParams->get<bool>("Interface Activated", false))
    result = Teuchos::rcp(static_cast<Albany::AbstractDiscretization*>(
                          new Catalyst::Decorator(result, catalystParams)));

#endif

  return result;
}

Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >
Albany::DiscretizationFactory::createMeshSpecs(Teuchos::RCP<Albany::AbstractMeshStruct> mesh) {
  meshStruct = mesh;
  return meshStruct->getMeshSpecs();
}

void
Albany::DiscretizationFactory::setupInternalMeshStruct(
  unsigned int neq,
  const Teuchos::RCP<Albany::StateInfoStruct>& sis,
  const std::map<std::string,Teuchos::RCP<Albany::StateInfoStruct> >& side_set_sis,
  const AbstractFieldContainer::FieldContainerRequirements& req,
  const std::map<std::string,AbstractFieldContainer::FieldContainerRequirements>& side_set_req) {
  meshStruct->setFieldAndBulkData(commT, discParams, neq, req, sis,
                                  meshStruct->getMeshSpecs()[0]->worksetSize, side_set_sis, side_set_req);
}

Teuchos::RCP<Albany::AbstractDiscretization>
Albany::DiscretizationFactory::createDiscretizationFromInternalMeshStruct(
  const std::map<int,std::vector<std::string> >& sideSetEquations,
  const Teuchos::RCP<Albany::RigidBodyModes>& rigidBodyModes) {

  if(!piroParams.is_null() && !rigidBodyModes.is_null())

    rigidBodyModes->setPiroPL(piroParams);

  std::string& method = discParams->get("Method", "STK1D");

  //IK, 1/8/15: added a method called "Ioss Aeras" and "Exodus Aeras" (which are equivalent)
  //which would tell the code to read in an Ioss (Exodus) linear mesh and then
  //refine it.  Changed the logic here so that the switch statement on meshSpecsType() is only hit if the method is not Ioss Aeras
  //or Exodus Aeras.
  //If it is Ioss Aeras or Exodus Aeras we use the Aeras::SpectralDiscretization class (right now just a dummy class that's a copy of
  //Albany::STKDiscretization).  The class will impelement the enrichment of a linear mesh to get higher order meshes.
  //
  //NOTE: one may want to create STK Aeras methods too if for example the Aeras::SpectralDiscretization class can refine
  //meshes created internally to Albany, if this is of interest.

  if(method != "Ioss Aeras" && method != "Exodus Aeras" && method != "STK1D Aeras") {
    switch(meshStruct->meshSpecsType()) {
#if defined(HAVE_STK)
      case Albany::AbstractMeshStruct::STK_MS: {
        Teuchos::RCP<Albany::AbstractSTKMeshStruct> ms = Teuchos::rcp_dynamic_cast<Albany::AbstractSTKMeshStruct>(meshStruct);
        Teuchos::RCP<Albany::STKDiscretization> disc;
#ifdef ALBANY_FELIX
        if (method == "Extruded")
          disc = Teuchos::rcp(new Albany::STKDiscretizationStokesH(ms, commT, rigidBodyModes));
        else
#endif
          disc = Teuchos::rcp(new Albany::STKDiscretization(ms, commT, rigidBodyModes, sideSetEquations));
        disc->updateMesh();
        return disc;
      }
      break;
#endif
#ifdef ALBANY_SCOREC
      case Albany::AbstractMeshStruct::PUMI_MS: {
        Teuchos::RCP<Albany::PUMIMeshStruct> ms = Teuchos::rcp_dynamic_cast<Albany::PUMIMeshStruct>(meshStruct);
        return Teuchos::rcp(new Albany::PUMIDiscretization(ms, commT, rigidBodyModes));
      }
      break;
#endif
#ifdef ALBANY_GOAL
      case Albany::AbstractMeshStruct::GOAL_MS: {
        Teuchos::RCP<Albany::GOALMeshStruct> ms = Teuchos::rcp_dynamic_cast<Albany::GOALMeshStruct>(meshStruct);
        return Teuchos::rcp(new Albany::GOALDiscretization(ms, commT, rigidBodyModes));
      }
      break;
#endif

#ifdef ALBANY_AMP
      case Albany::AbstractMeshStruct::SIM_MS: {
        Teuchos::RCP<Albany::SimMeshStruct> ms = Teuchos::rcp_dynamic_cast<Albany::SimMeshStruct>(meshStruct);
        return Teuchos::rcp(new Albany::SimDiscretization(ms, commT, rigidBodyModes));
      }
      break;
#endif
    }
  }

#if defined(ALBANY_AERAS) && defined(HAVE_STK)
  else if (method == "Ioss Aeras" || method == "Exodus Aeras" || method == "STK1D Aeras") {
    //IK, 1/8/15: Added construction of Aeras::SpectralDiscretization object.
    //WARNING: meshSpecsType() right now is set to STK_MS even for an Aeras::SpectralDiscretization, b/c that's how
    //the code is structured.  That should be OK since meshSpecsType() is not used anywhere except this function.
    //But one may want to change it to, e.g., AERAS_MS, to prevent confusion.
      Teuchos::RCP<Albany::AbstractSTKMeshStruct> ms = Teuchos::rcp_dynamic_cast<Albany::AbstractSTKMeshStruct>(meshStruct);
      return Teuchos::rcp(new Aeras::SpectralDiscretization(discParams, ms, numLevels, numTracers, commT, explicit_scheme, rigidBodyModes));
    }
#endif
  return Teuchos::null;
}
