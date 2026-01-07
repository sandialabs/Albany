//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include "Teuchos_TestForException.hpp"
#include "Albany_DiscretizationFactory.hpp"

#include "Albany_ExtrudedDiscretization.hpp"
#include "Albany_STKDiscretization.hpp"
// #include "Albany_BlockedSTKDiscretization.hpp"
#include "Albany_TmplSTKMeshStruct.hpp"
#include "Albany_STK3DPointStruct.hpp"
#include "Albany_GenericSTKMeshStruct.hpp"
#include "Albany_SideSetSTKMeshStruct.hpp"
#include "Albany_AsciiSTKMeshStruct.hpp"
#include "Albany_AsciiSTKMesh2D.hpp"
#include "Albany_GmshSTKMeshStruct.hpp"
#include "Albany_ExtrudedSTKMeshStruct.hpp"
#include "Albany_ExtrudedMesh.hpp"
#include "Albany_Utils.hpp" // For CalculateNumberParams

#ifdef ALBANY_OMEGAH
#include "Albany_OmegahGenericMesh.hpp"
#include "Albany_OmegahDiscretization.hpp"
#endif

#ifdef ALBANY_SEACAS
#include "Albany_IossSTKMeshStruct.hpp"
#endif

namespace Albany {

DiscretizationFactory::DiscretizationFactory(
        const Teuchos::RCP<Teuchos::ParameterList>& topLevelParams,
        const Teuchos::RCP<const Teuchos_Comm>& comm_,
        const bool explicit_scheme_) :
          comm(comm_),
          explicit_scheme(explicit_scheme_) 
{
  discParams = Teuchos::sublist(topLevelParams, "Discretization", true);
  if (topLevelParams->isSublist("Piro"))
    piroParams = Teuchos::sublist(topLevelParams, "Piro", true);
  if (topLevelParams->isSublist("Problem")) {
    Teuchos::RCP<Teuchos::ParameterList> problemParams = Teuchos::sublist(topLevelParams, "Problem", true);
    num_params = CalculateNumberParams(problemParams); 
  }
}


Teuchos::ArrayRCP<Teuchos::RCP<MeshSpecsStruct> >
DiscretizationFactory::createMeshSpecs() {
    // First, create the mesh struct
    meshStruct = createMeshStruct(discParams, comm, num_params);
    return meshStruct->meshSpecs;
}

Teuchos::RCP<AbstractMeshStruct>
DiscretizationFactory::createMeshStruct(Teuchos::RCP<Teuchos::ParameterList> disc_params,
        Teuchos::RCP<const Teuchos_Comm> comm, const int numParams)
{
    std::string& method = disc_params->get("Method", "STK1D");
    Teuchos::RCP<AbstractMeshStruct> mesh;
    if (method == "STK1D") {
        mesh = Teuchos::rcp(new TmplSTKMeshStruct<1>(disc_params, comm, numParams));
    } else if (method == "STK0D") {
        mesh = Teuchos::rcp(new TmplSTKMeshStruct<0>(disc_params, comm, numParams));
    } else if (method == "STK2D") {
        mesh = Teuchos::rcp(new TmplSTKMeshStruct<2>(disc_params, comm, numParams));
    } else if (method == "STK3D") {
        mesh = Teuchos::rcp(new TmplSTKMeshStruct<3>(disc_params, comm, numParams));
    } else if (method == "STK3D") {
        mesh = Teuchos::rcp(new TmplSTKMeshStruct<3>(disc_params, comm, numParams));
    } else if (method == "STK3DPoint") {
        mesh = Teuchos::rcp(new STK3DPointStruct(disc_params, comm, numParams));
    } else if (method == "Ioss" || method == "Exodus" || method == "Pamgen") {

#ifdef ALBANY_SEACAS
        mesh = Teuchos::rcp(new IossSTKMeshStruct(disc_params, comm, numParams));
#else
        TEUCHOS_TEST_FOR_EXCEPTION(method == "Ioss" || method == "Exodus" || method == "Pamgen",
                Teuchos::Exceptions::InvalidParameter,
                "Error: Discretization method " << method
                << " requested, but not compiled in" << std::endl);
#endif // ALBANY_SEACAS
    }
#ifdef ALBANY_OMEGAH
    else if (method == "Omegah") {
        mesh = Teuchos::rcp(new OmegahGenericMesh(disc_params));
    }
#endif
    else if (method == "Ascii") {
        mesh = Teuchos::rcp(new AsciiSTKMeshStruct(disc_params, comm, numParams));
    } else if (method == "Ascii2D") {
        mesh = Teuchos::rcp(new AsciiSTKMesh2D(disc_params, comm, numParams));
#ifdef ALBANY_SEACAS  // Fails to compile without SEACAS
    } else if (method == "Hacky Ascii2D") {
        //FixME very hacky! needed for printing 2d mesh
        Teuchos::RCP<GenericSTKMeshStruct> meshStruct2D;
        meshStruct2D = Teuchos::rcp(new AsciiSTKMesh2D(disc_params, comm, numParams));
        meshStruct2D->setFieldData(comm,Teuchos::null,{});
        meshStruct2D->setBulkData(comm);
        Ioss::Init::Initializer io;
        Teuchos::RCP<stk::io::StkMeshIoBroker> mesh_data = Teuchos::rcp(new stk::io::StkMeshIoBroker(MPI_COMM_WORLD));
        mesh_data->set_bulk_data(*meshStruct2D->bulkData);
        const std::string& output_filename = disc_params->get("Exodus Output File Name", "ice_mesh.2d.exo");
        size_t idx = mesh_data->create_output_mesh(output_filename, stk::io::WRITE_RESULTS);
        mesh_data->process_output_request(idx, 0.0);
#endif // ALBANY_SEACAS
    } else if (method == "Gmsh") {
        mesh = Teuchos::rcp(new GmshSTKMeshStruct(disc_params, comm, numParams));
    }
    else if (method == "Extruded") {
        Teuchos::RCP<AbstractMeshStruct> basalMesh;

        // Get basal_params
        auto ss_disc_params = Teuchos::sublist(disc_params,"Side Set Discretizations");
        auto basal_params = Teuchos::sublist(ss_disc_params,"basalside");
        if (!basal_params->isParameter("Number Of Time Derivatives")) {
          basal_params->set("Number Of Time Derivatives",disc_params->get<int>("Number Of Time Derivatives"));
        }

        // Set basal workset size
        int extruded_ws_size = disc_params->get("Workset Size", -1);
        if (extruded_ws_size == -1) {
          basal_params->set("Workset Size", -1);
        } else if (!basal_params->isParameter("Workset Size")) {
          // Compute basal workset size based on extruded workset size
          int basal_ws_size = extruded_ws_size / disc_params->get<int>("NumLayers");
          basal_ws_size = std::max(basal_ws_size,1); //makes sure is at least 1.
          basal_params->set("Workset Size", basal_ws_size);
        }

        basalMesh = createMeshStruct(basal_params, comm, numParams);
        mesh = Teuchos::rcp(new ExtrudedMesh(basalMesh, disc_params, comm));
    }
    else if (method == "STKExtruded") {
        Teuchos::RCP<AbstractMeshStruct> basalMesh;

        // Get basal_params
        Teuchos::RCP<Teuchos::ParameterList> basal_params;
        if (disc_params->isSublist("Side Set Discretizations") && disc_params->sublist("Side Set Discretizations").isSublist("basalside")) {
            basal_params = Teuchos::rcp(new Teuchos::ParameterList(disc_params->sublist("Side Set Discretizations").sublist("basalside")));
        } else {
            // Backward compatibility: Ioss, with parameters mixed with the extruded mesh ones
            basal_params->set("Method", "Ioss");
            basal_params->set("Use Serial Mesh", disc_params->get("Use Serial Mesh", false));
            basal_params->set("Exodus Input File Name", disc_params->get("Exodus Input File Name", "basalmesh.exo"));
        }

        // Set basal workset size
        int extruded_ws_size = disc_params->get("Workset Size", -1);
        if (extruded_ws_size == -1) {
          basal_params->set("Workset Size", -1);
        } else if (!basal_params->isParameter("Workset Size")) {
          // Compute basal workset size based on extruded workset size
          int basal_ws_size = extruded_ws_size / disc_params->get<int>("NumLayers");
          basal_ws_size = std::max(basal_ws_size,1); //makes sure is at least 1.
          basal_params->set("Workset Size", basal_ws_size);
        }

        basalMesh = createMeshStruct(basal_params, comm, numParams);
        mesh = Teuchos::rcp(new ExtrudedSTKMeshStruct(disc_params, comm, basalMesh, numParams));
    }
    else if (method == "Cubit") {
        TEUCHOS_TEST_FOR_EXCEPTION(method == "Cubit",
                Teuchos::Exceptions::InvalidParameter,
                "Error: Discretization method " << method
                << " requested, but no longer supported as of 10/2017" << std::endl);
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter, std::endl <<
                  "Error!  Unknown discretization method in DiscretizationFactory: " << method <<
                  "!" << std::endl << "Supplied parameter list is " << std::endl << *disc_params <<
                  "\nValid Methods are: STK1D, STK2D, STK3D, STK3DPoint, Ioss," <<
                  " Exodus, Ascii," <<
                  " Ascii2D, STKExtruded, Extruded, Omegah" << std::endl);
  }

  if (disc_params->isSublist ("Side Set Discretizations")) {
    TEUCHOS_TEST_FOR_EXCEPTION (mesh->meshSpecs.size()!=1, std::logic_error,
        "Error! So far, side set mesh is allowed only for meshes with 1 element block.\n");
    auto ms = mesh->meshSpecs[0];

    const Teuchos::ParameterList& ssd_list = disc_params->sublist("Side Set Discretizations");
    const Teuchos::Array<std::string>& sideSets = ssd_list.get<Teuchos::Array<std::string> >("Side Sets");

    Teuchos::RCP<Teuchos::ParameterList> params_ss;
    int sideDim = ms->numDim - 1;
    for (const auto& ss_name : sideSets) {
      auto& ss_mesh = mesh->sideSetMeshStructs[ss_name];

      // Extruded meshes can create some side meshes, so check if mesh is already there
      if (ss_mesh.is_null()) {
        params_ss = Teuchos::rcp(new Teuchos::ParameterList(ssd_list.sublist(ss_name)));

        if (!params_ss->isParameter("Number Of Time Derivatives"))
          params_ss->set<int>("Number Of Time Derivatives",disc_params->get<int>("Number Of Time Derivatives"));

        // Set sideset discretization workset size based on sideset mesh spec if a single workset is used
        const auto &sideSetMeshSpecs = ms->sideSetMeshSpecs;
        auto sideSetMeshSpecIter = sideSetMeshSpecs.find(ss_name);
        TEUCHOS_TEST_FOR_EXCEPTION(sideSetMeshSpecIter == sideSetMeshSpecs.end(), std::runtime_error,
            "Cannot find " << ss_name << " in sideSetMeshSpecs!\n");

        std::string ss_method = params_ss->get<std::string>("Method");
        if (ss_method=="SideSetSTK") {
          ss_mesh = Teuchos::rcp(new SideSetSTKMeshStruct(*ms, params_ss, comm, numParams));

          auto mesh_stk = Teuchos::rcp_dynamic_cast<AbstractSTKMeshStruct>(mesh,true);
          auto ss_mesh_stk = Teuchos::rcp_dynamic_cast<SideSetSTKMeshStruct>(ss_mesh,true);
          ss_mesh_stk->setParentMeshInfo(*mesh_stk, ss_name);

          // If requested, we ignore the side maps already stored in the imported side mesh (if any)
          // This can be useful for side mesh of an extruded mesh, in the case it was constructed
          // as side mesh of an extruded mesh with a different ordering and/or different number
          // of layers. Notice that if that's the case, it probably is impossible to build a new
          // set of maps, since there is no way to correctly map the side nodes to the cell nodes.
          ss_mesh_stk->ignore_side_maps = params_ss->get<bool>("Ignore Side Maps", false);
        } else {
          // This can be the case if we restart from existing volume and side meshes
          ss_mesh = createMeshStruct (params_ss,comm, numParams);
        }
      }

      auto ss_ms = ss_mesh->meshSpecs;

      // Checking that the side mesh has the correct dimension (in case they were loaded from file,
      // and the user mistakenly gave the wrong file name)
      TEUCHOS_TEST_FOR_EXCEPTION (sideDim!=ss_ms[0]->numDim, std::logic_error,
          "Error! Mesh on side " << ss_name << " has the wrong dimension.\n");
    }

    auto stk_mesh = Teuchos::rcp_dynamic_cast<GenericSTKMeshStruct>(mesh);
    if (stk_mesh)
      stk_mesh->createSideMeshMaps();
  }

  return mesh;
}

Teuchos::RCP<AbstractDiscretization>
DiscretizationFactory::createDiscretization(
        unsigned int neq, const std::map<int, std::vector<std::string> >& sideSetEquations,
        const Teuchos::RCP<StateInfoStruct>& sis,
        const std::map<std::string, Teuchos::RCP<StateInfoStruct> >& side_set_sis,
        const Teuchos::RCP<RigidBodyModes>& rigidBodyModes) 
{
    TEUCHOS_FUNC_TIME_MONITOR("Albany_DiscrFactory: createDiscretization");
    TEUCHOS_TEST_FOR_EXCEPTION(meshStruct == Teuchos::null,
            std::logic_error,
            "meshStruct accessed, but it has not been constructed" << std::endl);

    auto disc = createDiscretizationFromMeshStruct(discParams,meshStruct, neq, sideSetEquations, rigidBodyModes);

    // Set data and solution fields info
    meshStruct->setFieldData(comm,sis,side_set_sis);
    disc->setFieldData();

    // Set mesh info
    meshStruct->setBulkData(comm);
    for (auto& it : meshStruct->sideSetMeshStructs) {
      // Some meshes (e.g. extruded) may have already set the bulk data in some
      // side meshes during the call above, so check first
      if (not it.second->isBulkDataSet()) {
        it.second->setBulkData(comm);
      }
    }

    // Create internal disc structures
    disc->updateMesh();

    return disc;
}

Teuchos::ArrayRCP<Teuchos::RCP<MeshSpecsStruct> >
DiscretizationFactory::createMeshSpecs(Teuchos::RCP<AbstractMeshStruct> mesh) {
    meshStruct = mesh;
    return meshStruct->meshSpecs;
}

Teuchos::RCP<AbstractDiscretization>
DiscretizationFactory::
createDiscretizationFromMeshStruct (const Teuchos::RCP<Teuchos::ParameterList>& params,
                                    const Teuchos::RCP<AbstractMeshStruct>& mesh,
                                    const int neq,
                                    const std::map<int, std::vector<std::string> >& sideSetEquations,
                                    const Teuchos::RCP<RigidBodyModes>& rigidBodyModes)
{
  TEUCHOS_FUNC_TIME_MONITOR("Albany_DiscrFactory: createDiscretizationFromMeshStruct");

  if (!piroParams.is_null() && !rigidBodyModes.is_null())
      rigidBodyModes->setPiroPL(piroParams);

  Teuchos::RCP<AbstractDiscretization> disc;
  if (mesh->meshSpecs[0]->mesh_type==MeshType::Extruded)
  {
    auto ext_mesh = Teuchos::rcp_dynamic_cast<ExtrudedMesh>(mesh);
    auto basal_mesh = ext_mesh->basal_mesh();
    auto basal_params = Teuchos::sublist(Teuchos::sublist(discParams,"Side Set Discretizations"),"basalside");
    auto basal_disc = createDiscretizationFromMeshStruct(basal_params,basal_mesh,neq,{},rigidBodyModes);
    disc = Teuchos::rcp(new ExtrudedDiscretization (params,neq,ext_mesh,basal_disc,comm,rigidBodyModes, sideSetEquations));
  } else if (mesh->meshLibName()=="STK") {
    auto ms = Teuchos::rcp_dynamic_cast<AbstractSTKMeshStruct>(mesh);
    disc = Teuchos::rcp(new STKDiscretization(params, neq, ms, comm, rigidBodyModes, sideSetEquations));
#ifdef ALBANY_OMEGAH
  } else if (mesh->meshLibName()=="Omega_h") {
    auto ms = Teuchos::rcp_dynamic_cast<OmegahGenericMesh>(mesh);
    disc = Teuchos::rcp(new OmegahDiscretization(params, neq, ms, comm, rigidBodyModes, sideSetEquations));
#endif
  }
  return disc;
}

/* This function overwrite previous discretization parameter list */
void
DiscretizationFactory::setDiscretizationParameters(Teuchos::RCP<Teuchos::ParameterList> disc_params) {
    discParams = disc_params;
}

} // namespace Albany
