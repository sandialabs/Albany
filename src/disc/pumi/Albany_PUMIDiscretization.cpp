//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_APFDiscretization.hpp"
#include "Albany_PUMIDiscretization.hpp"

#include <PCU.h>
#include <apf.h>
#include <apfShape.h>

Albany::PUMIDiscretization::PUMIDiscretization(
    Teuchos::RCP<Albany::PUMIMeshStruct> meshStruct_,
    const Teuchos::RCP<const Teuchos_Comm>& commT_,
    const Teuchos::RCP<Albany::RigidBodyModes>& rigidBodyModes_):
  APFDiscretization(meshStruct_, commT_, rigidBodyModes_)
{
  pumiMeshStruct = meshStruct_;
  if (pumiMeshStruct->meshSpecsType() == Albany::AbstractMeshStruct::PUMI_MS)
    init();
}

Albany::PUMIDiscretization::~PUMIDiscretization()
{
}

void
Albany::PUMIDiscretization::setRestartData()
{
  // want to first call apf::createField to import the state data
  // from mesh tags from the .smb file
  int dim = getNumDim();
  apf::Mesh* m = meshStruct->getMesh();
  apf::FieldShape* fs = apf::getIPShape(dim, meshStruct->cubatureDegree);
  for (std::size_t i=0; i < meshStruct->qpscalar_states.size(); ++i) {
    PUMIQPData<double, 2>& state = *(meshStruct->qpscalar_states[i]);
    /* check for existing field due to restart */
    if (!m->findField(state.name.c_str()))
      apf::createField(m,state.name.c_str(),apf::SCALAR,fs);
  }
  for (std::size_t i=0; i < meshStruct->qpvector_states.size(); ++i) {
    PUMIQPData<double, 3>& state = *(meshStruct->qpvector_states[i]);
    if (!m->findField(state.name.c_str()))
      apf::createField(m,state.name.c_str(),apf::VECTOR,fs);
  }
  for (std::size_t i=0; i < meshStruct->qptensor_states.size(); ++i) {
    PUMIQPData<double, 4>& state = *(meshStruct->qptensor_states[i]);
    if (!m->findField(state.name.c_str()))
      apf::createField(m,state.name.c_str(),apf::MATRIX,fs);
  }

  // then we want to copy the qp data from apf to Albany's data structs
  this->copyQPStatesFromAPF();

  // also want the time information to be correct
  for (std::size_t b=0; b < buckets.size(); ++b) {
    Albany::MDArray& told = stateArrays.elemStateArrays[b]["Time_old"];
    told(0) = pumiMeshStruct->restartDataTime;
  }

  // get rid of this qp data from apf
  this->removeQPStatesFromAPF();

}

void
Albany::PUMIDiscretization::setFELIXData()
{
  assert(meshStruct->qpscalar_states.size() == 0);
  assert(meshStruct->qpvector_states.size() == 0);
  assert(meshStruct->qptensor_states.size() == 0);

  apf::Mesh2* m = meshStruct->getMesh();

  /* loop over the fields that the FELIX problem wants to exist */
  for (std::size_t i=0; i < meshStruct->elemnodescalar_states.size(); ++i) {

    /* try to find the field on the mesh */
    PUMIQPData<double, 2>& state = *(meshStruct->elemnodescalar_states[i]);
    apf::Field* f = m->findField(state.name.c_str());

    /* if the field does not exist on the mesh, we initialize it to zero */
    if (!f) {

      if(! PCU_Comm_Self())
        std::cout << "initializing " << state.name << " to zero" << std::endl;

      for (std::size_t b=0; b < buckets.size(); ++b) {
        std::vector<apf::MeshEntity*>& buck = buckets[b];
        Albany::MDArray& ar = stateArrays.elemStateArrays[b][state.name];
        for (std::size_t e=0; e < buck.size(); ++e)
          for (std::size_t n=0; n < state.dims[1]; ++n)
            ar(e,n) = 0.0;
      }

    }

    /* otherwise we pull in the data from the mesh */
    else {

      if(! PCU_Comm_Self())
        std::cout << "initializing " << state.name << " from mesh" << std::endl;

      apf::NewArray<double> values;
      int num_nodes = state.dims[1];

      for (std::size_t b=0; b < buckets.size(); ++b) {
        std::vector<apf::MeshEntity*>& buck = buckets[b];
        Albany::MDArray& ar = stateArrays.elemStateArrays[b][state.name];
        for (std::size_t e=0; e < buck.size(); ++e) {
          for (std::size_t n=0; n < num_nodes; ++n) {
            apf::MeshElement* mesh_elem = apf::createMeshElement(m, buck[e]);
            apf::Element* elem = apf::createElement(f, mesh_elem);
            apf::getScalarNodes(elem, values);
            assert(values.size() == num_nodes);
            ar(e,n) = values[n];
            apf::destroyElement(elem);
            apf::destroyMeshElement(mesh_elem);
          }
        }
      }

    }
  }
}
