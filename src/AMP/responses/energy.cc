#include <iostream>
#include <apf.h>
#include <apfMesh2.h>
#include <apfShape.h>
#include <gmi_mesh.h>
#include <apfMDS.h>
#include <PCU.h>
#include <assert.h>

class EnergyIntegrator : public apf::Integrator
{
  public:

    // constructor for energy integrator
    // inherits from base class apf::Integrator
    // the argument 1 is the quadrature degree
    EnergyIntegrator() : apf::Integrator(1)
    {
      // initialize the total energy
      energy = 0.0;
    }

    // destructor
    ~EnergyIntegrator() {}

    // this is called every time the integrator
    // begins to operate on a new element
    void inElement(apf::MeshElement* me)
    {
      // an apf::Element* contains information
      // about the field that was used to build
      // it and can be used to interpolate values
      // of a field in an element
      e_ = apf::createElement(temp,me);
    }

    // this is called after the integrator is
    // done operating on an element
    void outElement()
    {
      // it is necessary to destroy elements
      // to avoid memory leaks
      apf::destroyElement(e_);
    }

    // the integrator calls this for every integration point
    void atPoint(apf::Vector3 const& p, double w, double dv)
    {
      // this is the interpolated value of the temp field
      // at the integration point p
      T_ = apf::getScalar(e_,p);

      // sum values into the total energy
      energy += rho * Cp * T_ * w * dv;
    }

    // this is used to add local on-part integrated values
    // to global integrated values, a parallel communication
    void parallelReduce()
    {
      PCU_Add_Doubles(&energy,1);
    }

    double Cp;
    double rho;
    double energy;
    apf::Field* temp;

  private:
    double T_;
    apf::Element* e_;
};

double computeEnergy(apf::Mesh* m, apf::Field* T)
{
  // create the energy integrator
  EnergyIntegrator integrator;

  // set some fake material properties
  // and the temperature field for the integrator
  integrator.rho = 1.0;
  integrator.Cp = 4.25e6;
  integrator.temp = T;

  // this loops over all elements in the mesh and
  // performs the integration. a different routine
  // will need to be written for multiple materials
  integrator.process(m);

  integrator.parallelReduce();

  return integrator.energy;
}

int main(int argc, char** argv)
{
  assert(argc==4);
  const char* modelFile = argv[1];
  const char* meshFile = argv[2];
  const char* outFile = argv[3];
  MPI_Init(&argc,&argv);
  PCU_Comm_Init();
  gmi_register_mesh();
  apf::Mesh2* mesh = apf::loadMdsMesh(modelFile, meshFile);
  apf::Field* T = 
    apf::createLagrangeField(mesh, "Solution", apf::SCALAR, 1);
  double e = computeEnergy(mesh,T);
  std::cout << "Energy: " << e << std::endl;
  writeVtkFiles(outFile,mesh);
  apf::destroyField(T);
  mesh->destroyNative();
  apf::destroyMesh(mesh);
  PCU_Comm_Free();
  MPI_Finalize();
}

