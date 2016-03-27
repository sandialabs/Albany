//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "EnergyIntegral.hpp"

#include <Albany_APFMeshStruct.hpp>
#include <PCU.h>

namespace Albany {

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

double computeAMPEnergyIntegral(apf::Mesh* m)
{
  apf::Field* T = m->findField(Albany::APFMeshStruct::solution_name[0]);
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

void debugAMPMesh(apf::Mesh* m, char const* prefix)
{
  std::cout << "Energy integral: " << computeAMPEnergyIntegral(m) << '\n';
  apf::writeVtkFiles(prefix, m);
}

}
