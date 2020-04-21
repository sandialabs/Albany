//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include "Teuchos_TestForException.hpp"
#include "Albany_DiscretizationFactory.hpp"
#if defined(ALBANY_STK)
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
#include "Albany_AsciiSTKMeshStruct.hpp"
#include "Albany_AsciiSTKMesh2D.hpp"
#include "Albany_GmshSTKMeshStruct.hpp"
#ifdef ALBANY_LANDICE
#include "Albany_STKDiscretizationStokesH.hpp"
#include "Albany_ExtrudedSTKMeshStruct.hpp"
#endif
#endif
#ifdef ALBANY_SCOREC
#include "Albany_PUMIDiscretization.hpp"
#include "Albany_PUMIMeshStruct.hpp"
#endif
#ifdef ALBANY_CATALYST
#include "Albany_Catalyst_Decorator.hpp"
#endif

Albany::DiscretizationFactory::DiscretizationFactory(
        const Teuchos::RCP<Teuchos::ParameterList>& topLevelParams,
        const Teuchos::RCP<const Teuchos_Comm>& commT_,
        const bool explicit_scheme_) :
commT(commT_),
explicit_scheme(explicit_scheme_) {

    discParams = Teuchos::sublist(topLevelParams, "Discretization", true);

    if (topLevelParams->isSublist("Piro"))

        piroParams = Teuchos::sublist(topLevelParams, "Piro", true);

    if (topLevelParams->isSublist("Problem")) {

        Teuchos::RCP<Teuchos::ParameterList> problemParams = Teuchos::sublist(topLevelParams, "Problem", true);

        if (problemParams->isSublist("Adaptation"))

            adaptParams = Teuchos::sublist(problemParams, "Adaptation", true);

        if (problemParams->isSublist("Catalyst"))

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


Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >
Albany::DiscretizationFactory::createMeshSpecs() {
    // First, create the mesh struct
    meshStruct = createMeshStruct(discParams, adaptParams, commT);

#if defined(ALBANY_AERAS) && defined(ALBANY_STK)
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
        for (Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >::size_type i = 0; i < number_blocks; i++) {
            Teuchos::RCP<Albany::MeshSpecsStruct> orig_mesh_specs_struct = mesh_specs_struct[i];
            Aeras::AerasMeshSpectStruct aeras_mesh_specs_struct;
            enriched_mesh_specs_struct[i] = aeras_mesh_specs_struct.createAerasMeshSpecs(orig_mesh_specs_struct,
                    points_per_edge, discParams);
        }
        return enriched_mesh_specs_struct;
    } else
#endif
        return meshStruct->getMeshSpecs();
}

Teuchos::RCP<Albany::AbstractMeshStruct>
Albany::DiscretizationFactory::createMeshStruct(Teuchos::RCP<Teuchos::ParameterList> disc_params,
        Teuchos::RCP<Teuchos::ParameterList> adapt_params,
        Teuchos::RCP<const Teuchos_Comm> comm)
{
    std::string& method = disc_params->get("Method", "STK1D");
#if defined(ALBANY_STK)
    if (method == "STK1D" || method == "STK1D Aeras") {
        return Teuchos::rcp(new Albany::TmplSTKMeshStruct<1>(disc_params, adapt_params, comm));
    } else if (method == "STK0D") {
        return Teuchos::rcp(new Albany::TmplSTKMeshStruct<0>(disc_params, adapt_params, comm));
    } else if (method == "STK2D") {
        return Teuchos::rcp(new Albany::TmplSTKMeshStruct<2>(disc_params, adapt_params, comm));
    } else if (method == "STK3D") {
        return Teuchos::rcp(new Albany::TmplSTKMeshStruct<3>(disc_params, adapt_params, comm));
    } else if (method == "STK3DPoint") {
        return Teuchos::rcp(new Albany::STK3DPointStruct(disc_params, comm));
    } else if (method == "Ioss" || method == "Exodus" || method == "Pamgen" || method == "Ioss Aeras" || method == "Exodus Aeras") {

#ifdef ALBANY_SEACAS
        return Teuchos::rcp(new Albany::IossSTKMeshStruct(disc_params, adapt_params, comm));
#else
        TEUCHOS_TEST_FOR_EXCEPTION(method == "Ioss" || method == "Exodus" || method == "Pamgen" || method == "Ioss Aeras" ||
                method == "Exodus Aeras",
                Teuchos::Exceptions::InvalidParameter,
                "Error: Discretization method " << method
                << " requested, but not compiled in" << std::endl);
#endif // ALBANY_SEACAS
    }
    else if (method == "Ascii") {
        return Teuchos::rcp(new Albany::AsciiSTKMeshStruct(disc_params, comm));
    } else if (method == "Ascii2D") {
        return Teuchos::rcp(new Albany::AsciiSTKMesh2D(disc_params, comm));
#ifdef ALBANY_SEACAS  // Fails to compile without SEACAS
    } else if (method == "Hacky Ascii2D") {
        //FixME very hacky! needed for printing 2d mesh
        Teuchos::RCP<Albany::GenericSTKMeshStruct> meshStruct2D;
        meshStruct2D = Teuchos::rcp(new Albany::AsciiSTKMesh2D(disc_params, comm));
        Teuchos::RCP<Albany::StateInfoStruct> sis = Teuchos::rcp(new Albany::StateInfoStruct);
        Albany::AbstractFieldContainer::FieldContainerRequirements req;
        int neq = 2;
        meshStruct2D->setFieldAndBulkData(comm, disc_params, neq, req,
                sis, meshStruct2D->getMeshSpecs()[0]->worksetSize);
        Ioss::Init::Initializer io;
        Teuchos::RCP<stk::io::StkMeshIoBroker> mesh_data = Teuchos::rcp(new stk::io::StkMeshIoBroker(MPI_COMM_WORLD));
        mesh_data->set_bulk_data(*meshStruct2D->bulkData);
        const std::string& output_filename = disc_params->get("Exodus Output File Name", "ice_mesh.2d.exo");
        size_t idx = mesh_data->create_output_mesh(output_filename, stk::io::WRITE_RESULTS);
        mesh_data->process_output_request(idx, 0.0);
#endif // ALBANY_SEACAS
    } else if (method == "Gmsh") {
        return Teuchos::rcp(new Albany::GmshSTKMeshStruct(disc_params, comm));
    }
#ifdef ALBANY_LANDICE
    else if (method == "Extruded") {
        Teuchos::RCP<Albany::AbstractMeshStruct> basalMesh;
        Teuchos::RCP<Teuchos::ParameterList> basal_params;
        //compute basal Workset size starting from Discretization
        int extruded_ws_size = disc_params->get("Workset Size", 50);
        int basal_ws_size = -1;
        if(extruded_ws_size != -1) {
          basal_ws_size =  extruded_ws_size/ (disc_params->get<int>("NumLayers") * ((disc_params->get<std::string>("Element Shape") == "Tetrahedron") ? 3 : 1));
          basal_ws_size = std::max(basal_ws_size,1); //makes sure is at least 1.
        }
        if (disc_params->isSublist("Side Set Discretizations") && disc_params->sublist("Side Set Discretizations").isSublist("basalside")) {
            basal_params = Teuchos::rcp(new Teuchos::ParameterList(disc_params->sublist("Side Set Discretizations").sublist("basalside")));
            if(!disc_params->sublist("Side Set Discretizations").isParameter("Workset Size"))
              basal_params->set("Workset Size", basal_ws_size);
        } else {
            // Backward compatibility: Ioss, with parameters mixed with the extruded mesh ones
            basal_params->set("Method", "Ioss");
            basal_params->set("Use Serial Mesh", disc_params->get("Use Serial Mesh", false));
            basal_params->set("Exodus Input File Name", disc_params->get("Exodus Input File Name", "basalmesh.exo"));
            basal_params->set("Workset Size", basal_ws_size);
        }
        basalMesh = createMeshStruct(basal_params, Teuchos::null, comm);
        return Teuchos::rcp(new Albany::ExtrudedSTKMeshStruct(disc_params, comm, basalMesh));
    }
#endif // ALBANY_LANDICE
    else if (method == "Cubit") {
        TEUCHOS_TEST_FOR_EXCEPTION(method == "Cubit",
                Teuchos::Exceptions::InvalidParameter,
                "Error: Discretization method " << method
                << " requested, but no longe supported as of 10/2017" << std::endl);
    } else
#endif // ALBANY_STK
        if (method == "PUMI") {
#ifdef ALBANY_SCOREC
        return Teuchos::rcp(new Albany::PUMIMeshStruct(disc_params, comm));
#else
        TEUCHOS_TEST_FOR_EXCEPTION(method == "PUMI",
                Teuchos::Exceptions::InvalidParameter,
                "Error: Discretization method " << method
                << " requested, but not compiled in" << std::endl);
#endif
    } 

    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter, std::endl <<
                "Error!  Unknown discretization method in DiscretizationFactory: " << method <<
                "!" << std::endl << "Supplied parameter list is " << std::endl << *disc_params <<
                "\nValid Methods are: STK1D, STK2D, STK3D, STK3DPoint, Ioss, Ioss Aeras," <<
                " Exodus, Exodus Aeras, PUMI, PUMI Hierarchic, Sim, Ascii," <<
                " Ascii2D, Extruded" << std::endl);

  return Teuchos::null;
}

Teuchos::RCP<Albany::AbstractDiscretization>
Albany::DiscretizationFactory::createDiscretization(unsigned int neq,
        const Teuchos::RCP<Albany::StateInfoStruct>& sis,
        const AbstractFieldContainer::FieldContainerRequirements& req,
        const Teuchos::RCP<Albany::RigidBodyModes>& rigidBodyModes) {

    return createDiscretization(neq, empty_side_set_equations, sis, empty_side_set_sis, req, empty_side_set_req, rigidBodyModes);
}

Teuchos::RCP<Albany::AbstractDiscretization>
Albany::DiscretizationFactory::createDiscretization(
        unsigned int neq, const std::map<int, std::vector<std::string> >& sideSetEquations,
        const Teuchos::RCP<Albany::StateInfoStruct>& sis,
        const std::map<std::string, Teuchos::RCP<Albany::StateInfoStruct> >& side_set_sis,
        const AbstractFieldContainer::FieldContainerRequirements& req,
        const std::map<std::string, AbstractFieldContainer::FieldContainerRequirements>& side_set_req,
        const Teuchos::RCP<Albany::RigidBodyModes>& rigidBodyModes) {
    TEUCHOS_TEST_FOR_EXCEPTION(meshStruct == Teuchos::null,
            std::logic_error,
            "meshStruct accessed, but it has not been constructed" << std::endl);

    setupInternalMeshStruct(neq, sis, side_set_sis, req, side_set_req);
    Teuchos::RCP<Albany::AbstractDiscretization> result =
            createDiscretizationFromInternalMeshStruct(sideSetEquations, rigidBodyModes);

    // Wrap the discretization in the catalyst decorator if needed.
#ifdef ALBANY_CATALYST

    if (Teuchos::nonnull(catalystParams) && catalystParams->get<bool>("Interface Activated", false))
        result = Teuchos::rcp(static_cast<Albany::AbstractDiscretization*> (
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
        const AbstractFieldContainer::FieldContainerRequirements& req) {
    setupInternalMeshStruct(neq, sis, empty_side_set_sis, req, empty_side_set_req);
}

void
Albany::DiscretizationFactory::setupInternalMeshStruct(
        unsigned int neq,
        const Teuchos::RCP<Albany::StateInfoStruct>& sis,
        const std::map<std::string, Teuchos::RCP<Albany::StateInfoStruct> >& side_set_sis,
        const AbstractFieldContainer::FieldContainerRequirements& req,
        const std::map<std::string, AbstractFieldContainer::FieldContainerRequirements>& side_set_req) {
    meshStruct->setFieldAndBulkData(commT, discParams, neq, req, sis,
            meshStruct->getMeshSpecs()[0]->worksetSize, side_set_sis, side_set_req);
}

Teuchos::RCP<Albany::AbstractDiscretization>
Albany::DiscretizationFactory::createDiscretizationFromInternalMeshStruct(
        const Teuchos::RCP<Albany::RigidBodyModes>& rigidBodyModes) {
    return createDiscretizationFromInternalMeshStruct(empty_side_set_equations, rigidBodyModes);
}

Teuchos::RCP<Albany::AbstractDiscretization>
Albany::DiscretizationFactory::createDiscretizationFromInternalMeshStruct(
        const std::map<int, std::vector<std::string> >& sideSetEquations,
        const Teuchos::RCP<Albany::RigidBodyModes>& rigidBodyModes) {

    if (!piroParams.is_null() && !rigidBodyModes.is_null())

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

    if (method != "Ioss Aeras" && method != "Exodus Aeras" && method != "STK1D Aeras") {
        switch (meshStruct->meshSpecsType()) {
#if defined(ALBANY_STK)
            case Albany::AbstractMeshStruct::STK_MS:
            {
                Teuchos::RCP<Albany::AbstractSTKMeshStruct> ms = Teuchos::rcp_dynamic_cast<Albany::AbstractSTKMeshStruct>(meshStruct);
                Teuchos::RCP<Albany::STKDiscretization> disc;
#ifdef ALBANY_LANDICE
                if (method=="Extruded") {
                  disc = Teuchos::rcp(new Albany::STKDiscretizationStokesH(discParams, ms, commT, rigidBodyModes));
                } else
#endif
                {
                  disc = Teuchos::rcp(new Albany::STKDiscretization(discParams, ms, commT, rigidBodyModes, sideSetEquations));
                }
                disc->updateMesh();
                return disc;
                break;
            }
#endif
#ifdef ALBANY_SCOREC
            case Albany::AbstractMeshStruct::PUMI_MS:
            {
                Teuchos::RCP<Albany::PUMIMeshStruct> ms = Teuchos::rcp_dynamic_cast<Albany::PUMIMeshStruct>(meshStruct);
                return Teuchos::rcp(new Albany::PUMIDiscretization(ms, commT, rigidBodyModes));
                break;
            }
#endif
        }
    }
#if defined(ALBANY_AERAS) && defined(ALBANY_STK)
    else if (method == "Ioss Aeras" || method == "Exodus Aeras" || method == "STK1D Aeras") {
        //IK, 1/8/15: Added construction of Aeras::SpectralDiscretization object.
        //WARNING: meshSpecsType() right now is set to STK_MS even for an Aeras::SpectralDiscretization, b/c that's how
        //the code is structured.  That should be OK since meshSpecsType() is not used anywhere except this function.
        //But one may want to change it to, e.g., AERAS_MS, to prevent confusion.
        Teuchos::RCP<Albany::AbstractSTKMeshStruct> ms = Teuchos::rcp_dynamic_cast<Albany::AbstractSTKMeshStruct>(meshStruct);
        auto disc = Teuchos::rcp(new Aeras::SpectralDiscretization(discParams, ms, numLevels, numTracers, commT, explicit_scheme, rigidBodyModes));
        disc->updateMesh(); 
        return disc; 
    }
#endif
    return Teuchos::null;
}

/* This function overwrite previous discretization parameter list */
void
Albany::DiscretizationFactory::setDiscretizationParameters(Teuchos::RCP<Teuchos::ParameterList> disc_params) {
    discParams = disc_params;
}
