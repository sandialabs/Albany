//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include "Teuchos_TestForException.hpp"
#include "Albany_DiscretizationFactory.hpp"

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

#ifdef ALBANY_OMEGAH
#include "Albany_OmegahBoxMesh.hpp"
#include "Albany_OmegahDiscretization.hpp"
#endif

#ifdef ALBANY_SEACAS
#include "Albany_IossSTKMeshStruct.hpp"
#endif

namespace Albany {

DiscretizationFactory::DiscretizationFactory(
        const Teuchos::RCP<Teuchos::ParameterList>& topLevelParams,
        const Teuchos::RCP<const Teuchos_Comm>& commT_,
        const bool explicit_scheme_) :
          commT(commT_),
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
    meshStruct = createMeshStruct(discParams, commT, num_params);
    return meshStruct->getMeshSpecs();
}

Teuchos::RCP<AbstractMeshStruct>
DiscretizationFactory::createMeshStruct(Teuchos::RCP<Teuchos::ParameterList> disc_params,
        Teuchos::RCP<const Teuchos_Comm> comm, const int numParams)
{
    std::string& method = disc_params->get("Method", "STK1D");
    if (method == "STK1D") {
        return Teuchos::rcp(new TmplSTKMeshStruct<1>(disc_params, comm, numParams));
    } else if (method == "STK0D") {
        return Teuchos::rcp(new TmplSTKMeshStruct<0>(disc_params, comm, numParams));
    } else if (method == "STK2D") {
        return Teuchos::rcp(new TmplSTKMeshStruct<2>(disc_params, comm, numParams));
    } else if (method == "STK3D") {
        return Teuchos::rcp(new TmplSTKMeshStruct<3>(disc_params, comm, numParams));
    } else if (method == "STK3D") {
        return Teuchos::rcp(new TmplSTKMeshStruct<3>(disc_params, comm, numParams));
    } else if (method == "STK3DPoint") {
        return Teuchos::rcp(new STK3DPointStruct(disc_params, comm, numParams));
    } else if (method == "Ioss" || method == "Exodus" || method == "Pamgen") {

#ifdef ALBANY_SEACAS
        return Teuchos::rcp(new IossSTKMeshStruct(disc_params, comm, numParams));
#else
        TEUCHOS_TEST_FOR_EXCEPTION(method == "Ioss" || method == "Exodus" || method == "Pamgen",
                Teuchos::Exceptions::InvalidParameter,
                "Error: Discretization method " << method
                << " requested, but not compiled in" << std::endl);
#endif // ALBANY_SEACAS
    }
#ifdef ALBANY_OMEGAH
    else if (method == "Box1D") {
        return Teuchos::rcp(new OmegahBoxMesh<1>(disc_params, comm, numParams));
    } else if (method == "Box2D") {
        return Teuchos::rcp(new OmegahBoxMesh<2>(disc_params, comm, numParams));
    } else if (method == "Box3D") {
        return Teuchos::rcp(new OmegahBoxMesh<3>(disc_params, comm, numParams));
    }
#endif
    else if (method == "Ascii") {
        return Teuchos::rcp(new AsciiSTKMeshStruct(disc_params, comm, numParams));
    } else if (method == "Ascii2D") {
        return Teuchos::rcp(new AsciiSTKMesh2D(disc_params, comm, numParams));
#ifdef ALBANY_SEACAS  // Fails to compile without SEACAS
    } else if (method == "Hacky Ascii2D") {
        //FixME very hacky! needed for printing 2d mesh
        Teuchos::RCP<GenericSTKMeshStruct> meshStruct2D;
        meshStruct2D = Teuchos::rcp(new AsciiSTKMesh2D(disc_params, comm, numParams));
        Teuchos::RCP<StateInfoStruct> sis = Teuchos::rcp(new StateInfoStruct);
        meshStruct2D->setFieldAndBulkData(comm,
                sis, meshStruct2D->getMeshSpecs()[0]->worksetSize);
        Ioss::Init::Initializer io;
        Teuchos::RCP<stk::io::StkMeshIoBroker> mesh_data = Teuchos::rcp(new stk::io::StkMeshIoBroker(MPI_COMM_WORLD));
        mesh_data->set_bulk_data(*meshStruct2D->bulkData);
        const std::string& output_filename = disc_params->get("Exodus Output File Name", "ice_mesh.2d.exo");
        size_t idx = mesh_data->create_output_mesh(output_filename, stk::io::WRITE_RESULTS);
        mesh_data->process_output_request(idx, 0.0);
#endif // ALBANY_SEACAS
    } else if (method == "Gmsh") {
        return Teuchos::rcp(new GmshSTKMeshStruct(disc_params, comm, numParams));
    }
    else if (method == "Extruded") {
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
        return Teuchos::rcp(new ExtrudedSTKMeshStruct(disc_params, comm, basalMesh, numParams));
    }
    else if (method == "Cubit") {
        TEUCHOS_TEST_FOR_EXCEPTION(method == "Cubit",
                Teuchos::Exceptions::InvalidParameter,
                "Error: Discretization method " << method
                << " requested, but no longer supported as of 10/2017" << std::endl);
    } else
      TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter, std::endl <<
                  "Error!  Unknown discretization method in DiscretizationFactory: " << method <<
                  "!" << std::endl << "Supplied parameter list is " << std::endl << *disc_params <<
                  "\nValid Methods are: STK1D, STK2D, STK3D, STK3DPoint, Ioss," <<
                  " Exodus, Ascii," <<
                  " Ascii2D, Extruded" << std::endl);

  return Teuchos::null;
}

Teuchos::RCP<AbstractDiscretization>
DiscretizationFactory::createDiscretization(unsigned int neq,
        const Teuchos::RCP<StateInfoStruct>& sis,
        const Teuchos::RCP<RigidBodyModes>& rigidBodyModes) 
{
    return createDiscretization(neq, empty_side_set_equations, sis, empty_side_set_sis, rigidBodyModes);
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

    Teuchos::RCP<AbstractDiscretization> result =
            createDiscretizationFromInternalMeshStruct(neq, sideSetEquations, rigidBodyModes);

    setMeshStructFieldData(sis, side_set_sis);
    setFieldData(result, sis);
    Teuchos::RCP<StateInfoStruct> dummy_sis;
    for (auto it : result->getSideSetDiscretizations()) {
      if (side_set_sis.count(it.first)==1) {
        setFieldData(it.second,side_set_sis.at(it.first));
      } else {
        setFieldData(it.second,{});
      }
    }
    setMeshStructBulkData(sis, side_set_sis);
    completeDiscSetup(result);

    return result;
}

Teuchos::ArrayRCP<Teuchos::RCP<MeshSpecsStruct> >
DiscretizationFactory::createMeshSpecs(Teuchos::RCP<AbstractMeshStruct> mesh) {
    meshStruct = mesh;
    return meshStruct->getMeshSpecs();
}

void
DiscretizationFactory::setMeshStructFieldData(
        const Teuchos::RCP<StateInfoStruct>& sis) {
    setMeshStructFieldData(sis, empty_side_set_sis);
}

void
DiscretizationFactory::setMeshStructFieldData(
        const Teuchos::RCP<StateInfoStruct>& sis,
        const std::map<std::string, Teuchos::RCP<StateInfoStruct> >& side_set_sis)
{
    TEUCHOS_FUNC_TIME_MONITOR("Albany_DiscrFactory: setMeshStructFieldData");
    meshStruct->setFieldData(commT, sis,
            meshStruct->getMeshSpecs()[0]->worksetSize, side_set_sis); 
}

void
DiscretizationFactory::setMeshStructBulkData(
        const Teuchos::RCP<StateInfoStruct>& sis) {
    setMeshStructBulkData(sis, empty_side_set_sis);
}

void
DiscretizationFactory::setMeshStructBulkData(
        const Teuchos::RCP<StateInfoStruct>& sis,
        const std::map<std::string, Teuchos::RCP<StateInfoStruct> >& side_set_sis)
{
    TEUCHOS_FUNC_TIME_MONITOR("Albany_DiscrFactory: setMeshStructBulkData");
    meshStruct->setBulkData(commT, sis,
            meshStruct->getMeshSpecs()[0]->worksetSize, side_set_sis);
}

Teuchos::RCP<AbstractDiscretization>
DiscretizationFactory::createDiscretizationFromInternalMeshStruct(
        const int neq,
        const Teuchos::RCP<RigidBodyModes>& rigidBodyModes) {
    return createDiscretizationFromInternalMeshStruct(neq, empty_side_set_equations, rigidBodyModes);
}

Teuchos::RCP<AbstractDiscretization>
DiscretizationFactory::createDiscretizationFromInternalMeshStruct(
        const int neq,
        const std::map<int, std::vector<std::string> >& sideSetEquations,
        const Teuchos::RCP<RigidBodyModes>& rigidBodyModes) {
    TEUCHOS_FUNC_TIME_MONITOR("Albany_DiscrFactory: createDiscretizationFromInternalMeshStruct");

    if (!piroParams.is_null() && !rigidBodyModes.is_null())

        rigidBodyModes->setPiroPL(piroParams);


  Teuchos::RCP<AbstractDiscretization> disc;
  if (meshStruct->meshType()=="STK") {
    auto ms = Teuchos::rcp_dynamic_cast<AbstractSTKMeshStruct>(meshStruct);
    disc = Teuchos::rcp(new STKDiscretization(discParams, neq, ms, commT, rigidBodyModes, sideSetEquations));
  } else if (meshStruct->meshType()=="Omega_h") {
    auto ms = Teuchos::rcp_dynamic_cast<OmegahAbstractMesh>(meshStruct);
    disc = Teuchos::rcp(new OmegahDiscretization(discParams, neq, ms, commT, rigidBodyModes, sideSetEquations));
  }
  return disc;
}

void
DiscretizationFactory::setFieldData(Teuchos::RCP<AbstractDiscretization> disc,
                                    const Teuchos::RCP<Albany::StateInfoStruct>& sis) {

  if (meshStruct->meshType()=="STK") {
    auto stk_disc = Teuchos::rcp_dynamic_cast<STKDiscretization>(disc);
    stk_disc->setFieldData(sis);
  } else if (meshStruct->meshType()=="Omega_h") {
    auto omh_disc = Teuchos::rcp_dynamic_cast<OmegahDiscretization>(disc);
    omh_disc->setFieldData(sis);
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION (true,std::runtime_error, "Unrecognized mesh type!\n");
  }
}

void
DiscretizationFactory::completeDiscSetup(Teuchos::RCP<AbstractDiscretization> disc) {
  TEUCHOS_FUNC_TIME_MONITOR("Albany_DiscrFactory: completeDiscSetup");

  if (meshStruct->meshType()=="STK") {
    auto stk_disc = Teuchos::rcp_dynamic_cast<STKDiscretization>(disc);
    stk_disc->updateMesh();
  } else if (meshStruct->meshType()=="Omega_h") {
    auto omh_disc = Teuchos::rcp_dynamic_cast<OmegahDiscretization>(disc);
    omh_disc->updateMesh();
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION (true,std::runtime_error, "Unrecognized mesh type!\n");
  }
}

/* This function overwrite previous discretization parameter list */
void
DiscretizationFactory::setDiscretizationParameters(Teuchos::RCP<Teuchos::ParameterList> disc_params) {
    discParams = disc_params;
}

} // namespace Albany
