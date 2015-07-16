//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include "Albany_SimMeshStruct.hpp"

#include <MeshSim.h>
#include <SimPartitionedMesh.h>
#include <gmi_sim.h>
#include <SimUtil.h>
#include <apfSIM.h>
#include <PCU.h>

Albany::SimMeshStruct::SimMeshStruct(
    const Teuchos::RCP<Teuchos::ParameterList>& params,
		const Teuchos::RCP<const Teuchos_Comm>& commT)
{
  SimUtil_start();
  Sim_readLicenseFile(0);
  SimPartitionedMesh_start(NULL, NULL);
  gmi_sim_start();
  gmi_register_sim();
  PCU_Comm_Init();

  params->validateParameters(
      *(SimMeshStruct::getValidDiscretizationParameters()), 0);

  outputFileName = params->get<std::string>("Sim Output File Name", "");
  outputInterval = params->get<int>("Sim Write Interval", 1); // write every time step default

  std::string native_file;
  if (params->isParameter("Acis Model Input File Name"))
    native_file = params->get<std::string>("Parasolid Model Input File Name");
  if(params->isParameter("Parasolid Model Input File Name"))
    native_file = params->get<std::string>("Parasolid Model Input File Name");
  std::string smd_file;
  if(params->isParameter("Sim Model Input File Name"))
    smd_file = params->get<std::string>("Sim Model Input File Name");

  model = gmi_sim_load(native_file.empty() ? 0 : native_file.c_str(),
                       smd_file.empty()    ? 0 : smd_file.c_str());
  pGModel sim_model = gmi_export_sim(model);

  std::string mesh_file = params->get<std::string>("Sim Input File Name");
  pParMesh sim_mesh = PM_load(mesh_file.c_str(), sthreadNone, sim_model, NULL);
  mesh = apf::createMesh(sim_mesh);

  APFMeshStruct::init(params, commT);
}

Albany::SimMeshStruct::~SimMeshStruct()
{
  mesh->destroyNative();
  apf::destroyMesh(mesh);
  gmi_destroy(model);
  PCU_Comm_Free();
  gmi_sim_stop();
  SimPartitionedMesh_stop();
  Sim_unregisterAllKeys();
  SimUtil_stop();
}

Albany::AbstractMeshStruct::msType
Albany::SimMeshStruct::meshSpecsType()
{
  return SIM_MS;
}

apf::Field*
Albany::SimMeshStruct::createNodalField(char const* name, int valueType)
{
  return apf::createSIMFieldOn(this->mesh, name, valueType);
}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::SimMeshStruct::getValidDiscretizationParameters() const
{

  Teuchos::RCP<Teuchos::ParameterList> validPL
     = APFMeshStruct::getValidDiscretizationParameters();

  validPL->set<int>("Sim Write Interval", 3, "Step interval to write solution data to output file");

  validPL->set<std::string>("Sim Input File Name", "", "File Name For Sim Mesh Input");
  validPL->set<std::string>("Sim Output File Name", "", "File Name For Sim Mesh Output");
  validPL->set<std::string>("Sim Model Input File Name", "", "File Name For Sim Mesh Output");

  return validPL;
}


