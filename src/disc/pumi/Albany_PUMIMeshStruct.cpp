//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include "Albany_PUMIMeshStruct.hpp"

#include <gmi_mesh.h>
#ifdef SCOREC_SIMMODEL
#include <gmi_sim.h>
#include <SimUtil.h>
#endif
#include <apfMDS.h>
#include <apfShape.h>
#include <ma.h>
#include <PCU.h>

class SizeFunction : public ma::IsotropicFunction {
  public:
    SizeFunction(double s) {size = s;}
    double getValue(ma::Entity*) {return size;}
  private:
    double size;
};

Albany::PUMIMeshStruct::PUMIMeshStruct(
    const Teuchos::RCP<Teuchos::ParameterList>& params,
		const Teuchos::RCP<const Teuchos_Comm>& commT)
{
  PCU_Comm_Init();
  params->validateParameters(
      *(PUMIMeshStruct::getValidDiscretizationParameters()), 0);

  outputFileName = params->get<std::string>("PUMI Output File Name", "");
  outputInterval = params->get<int>("PUMI Write Interval", 1); // write every time step default

  gmi_register_mesh();

  std::string model_file;
  if(params->isParameter("Mesh Model Input File Name"))
    model_file = params->get<std::string>("Mesh Model Input File Name");

#ifdef SCOREC_SIMMODEL
  Sim_readLicenseFile(0);
  gmi_sim_start();
  gmi_register_sim();

  if (params->isParameter("Acis Model Input File Name"))
    model_file = params->get<std::string>("Parasolid Model Input File Name");

  if(params->isParameter("Parasolid Model Input File Name"))
    model_file = params->get<std::string>("Parasolid Model Input File Name");
#endif

  if (params->isParameter("PUMI Input File Name")) {
    std::string mesh_file = params->get<std::string>("PUMI Input File Name");
    mesh = apf::loadMdsMesh(model_file.c_str(), mesh_file.c_str());
  } else {
    int nex = params->get<int>("1D Elements", 0);
    int ney = params->get<int>("2D Elements", 0);
    int nez = params->get<int>("3D Elements", 0);
    double wx = params->get<double>("1D Scale", 1);
    double wy = params->get<double>("2D Scale", 1);
    double wz = params->get<double>("3D Scale", 1);
    bool is = ! params->get<bool>("Hexahedral", true);
    buildBoxMesh(nex, ney, nez, wx, wy, wz, is);
  }

  model = mesh->getModel();

  // Tell the mesh that we'll handle deleting the model.
  apf::disownMdsModel(mesh);

  bool isQuadMesh = params->get<bool>("2nd Order Mesh",false);
  if (isQuadMesh)
    apf::changeMeshShape(mesh, apf::getLagrange(2), false);

  // Resize mesh after input if indicated in the input file
  // User has indicated a desired element size in input file
  if(params->isParameter("Resize Input Mesh Element Size")){
    SizeFunction sizeFunction(params->get<double>(
          "Resize Input Mesh Element Size", 0.1));
    int num_iters = params->get<int>(
        "Max Number of Mesh Adapt Iterations", 1);
    ma::Input* input = ma::configure(mesh,&sizeFunction);
    input->maximumIterations = num_iters;
    input->shouldSnap = false;
    ma::adapt(input);
  }

  APFMeshStruct::init(params, commT);
}

Albany::PUMIMeshStruct::~PUMIMeshStruct()
{
  setMesh(0);
  if (model)
    gmi_destroy(model);
  PCU_Comm_Free();
#ifdef SCOREC_SIMMODEL
  gmi_sim_stop();
  Sim_unregisterAllKeys();
#endif
}

Albany::AbstractMeshStruct::msType
Albany::PUMIMeshStruct::meshSpecsType()
{
  return PUMI_MS;
}

void Albany::PUMIMeshStruct::setMesh(apf::Mesh2* new_mesh)
{
  if (mesh) {
    mesh->destroyNative();
    apf::destroyMesh(mesh);
  }
  mesh = new_mesh;
}

apf::Field*
Albany::SimMeshStruct::createNodalField(char const* name, int valueType)
{
  return apf::createSIMFieldOn(this->mesh, name, valueType);
}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::PUMIMeshStruct::getValidDiscretizationParameters() const
{

  Teuchos::RCP<Teuchos::ParameterList> validPL
     = APFMeshStruct::getValidDiscretizationParameters();

  validPL->set<int>("PUMI Write Interval", 3, "Step interval to write solution data to output file");
  validPL->set<bool>("2nd Order Mesh", false, "Flag to indicate 2nd order Lagrange shape functions");

  validPL->set<std::string>("PUMI Input File Name", "", "File Name For PUMI Mesh Input");
  validPL->set<std::string>("PUMI Output File Name", "", "File Name For PUMI Mesh Output");
  validPL->set<std::string>("Mesh Model Input File Name", "", "meshmodel geometry file");

  // Parameters to refine the mesh after input
  validPL->set<double>("Resize Input Mesh Element Size", 1.0, "Resize mesh element to this size at input");
  validPL->set<int>("Max Number of Mesh Adapt Iterations", 1);

  validPL->set<int>("1D Elements", 0, "Number of Elements in X discretization");
  validPL->set<int>("2D Elements", 0, "Number of Elements in Y discretization");
  validPL->set<int>("3D Elements", 0, "Number of Elements in Z discretization");
  validPL->set<double>("1D Scale", 1.0, "Width of X discretization");
  validPL->set<double>("2D Scale", 1.0, "Depth of Y discretization");
  validPL->set<double>("3D Scale", 1.0, "Height of Z discretization");
  validPL->set<bool>("Hexahedral", true, "Build hexahedral elements");

  return validPL;
}

