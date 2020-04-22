//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include "Teuchos_TestForException.hpp"
#include "Albany_DiscretizationFactory.hpp"
#if defined(ALBANY_STK)
#include "Albany_STKDiscretization.hpp"
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
    }
}


Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >
Albany::DiscretizationFactory::createMeshSpecs() {
    // First, create the mesh struct
    meshStruct = createMeshStruct(discParams, adaptParams, commT);
    return meshStruct->getMeshSpecs();
}

Teuchos::RCP<Albany::AbstractMeshStruct>
Albany::DiscretizationFactory::createMeshStruct(Teuchos::RCP<Teuchos::ParameterList> disc_params,
        Teuchos::RCP<Teuchos::ParameterList> adapt_params,
        Teuchos::RCP<const Teuchos_Comm> comm)
{
    std::string& method = disc_params->get("Method", "STK1D");
#if defined(ALBANY_STK)
    if (method == "STK1D") {
        return Teuchos::rcp(new Albany::TmplSTKMeshStruct<1>(disc_params, adapt_params, comm));
    } else if (method == "STK0D") {
        return Teuchos::rcp(new Albany::TmplSTKMeshStruct<0>(disc_params, adapt_params, comm));
    } else if (method == "STK2D") {
        return Teuchos::rcp(new Albany::TmplSTKMeshStruct<2>(disc_params, adapt_params, comm));
    } else if (method == "STK3D") {
        return Teuchos::rcp(new Albany::TmplSTKMeshStruct<3>(disc_params, adapt_params, comm));
    } else if (method == "STK3DPoint") {
        return Teuchos::rcp(new Albany::STK3DPointStruct(disc_params, comm));
    } else if (method == "Ioss" || method == "Exodus" || method == "Pamgen") {

#ifdef ALBANY_SEACAS
        return Teuchos::rcp(new Albany::IossSTKMeshStruct(disc_params, adapt_params, comm));
#else
        TEUCHOS_TEST_FOR_EXCEPTION(method == "Ioss" || method == "Exodus" || method == "Pamgen",
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
                "\nValid Methods are: STK1D, STK2D, STK3D, STK3DPoint, Ioss," <<
                " Exodus, PUMI, PUMI Hierarchic, Sim, Ascii," <<
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
    return Teuchos::null;
}

/* This function overwrite previous discretization parameter list */
void
Albany::DiscretizationFactory::setDiscretizationParameters(Teuchos::RCP<Teuchos::ParameterList> disc_params) {
    discParams = disc_params;
}
