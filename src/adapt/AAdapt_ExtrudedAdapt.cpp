//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "AAdapt_ExtrudedAdapt.hpp"
#include "Albany_PUMIMeshStruct.hpp"

namespace AAdapt {

ExtrudedAdapt::ExtrudedAdapt(const Teuchos::RCP<Albany::APFDiscretization>& disc):
  MeshAdaptMethod(disc),
  spr_helper(disc) {
  mesh = mesh_struct->getMesh();
  model_extrusions.push_back(ma::ModelExtrusion(
        mesh->findModelEntity(1, 2),
        mesh->findModelEntity(2, 3),
        mesh->findModelEntity(1, 1)));
  model_extrusions.push_back(ma::ModelExtrusion(
        mesh->findModelEntity(2, 2),
        mesh->findModelEntity(3, 1),
        mesh->findModelEntity(2, 1)));
}

void ExtrudedAdapt::setParams(const Teuchos::RCP<Teuchos::ParameterList>& p) {
  spr_helper.setParams(p);
}

void ExtrudedAdapt::preProcessOriginalMesh() {
  std::cerr << "pre-processing original...\n";
  ma::intrude(mesh, model_extrusions, &nlayers);
  /* we will use the top layer velocity for error estimation */
  std::string flat_name = ma::getFlatName(
        Albany::APFMeshStruct::solution_name[0], nlayers - 1);
  /* this field starts out as a Packed field of size 2,
   * but SPR can only handle a VECTOR Field of size 3,
   * so we copy it to match.
   * we can also call this new field "Solution", because
   * after flattening converted the old "Solution" into
   * a bunch of flat fields "L0_Solution" etc.
   */
  apf::Field* flat_field = mesh->findField(flat_name.c_str());
  assert(flat_field);
  apf::Field* new_field = apf::createFieldOn(mesh,
      Albany::APFMeshStruct::solution_name[0], apf::VECTOR);
  ma::Iterator* it = mesh->begin(0);
  ma::Entity* v;
  while ((v = mesh->iterate(it))) {
    ma::Vector x;
    apf::getComponents(flat_field, v, 0, &x[0]);
    x[2] = 0;
    apf::setVector(new_field, v, 0, x);
  }
  mesh->end(it);
  std::cerr << "pre-processing original done.\n";
  std::cerr << "mesh dim is now " << mesh->getDimension() << ", element count "
    << mesh->count(mesh->getDimension()) << '\n';
}

void ExtrudedAdapt::preProcessShrunkenMesh() {
  std::cerr << "pre-processing shrunken...\n";
  spr_helper.preProcessOriginalMesh();
  std::cerr << "spr pre-processed original\n";
  spr_helper.preProcessShrunkenMesh();
  std::cerr << "pre-processing shrunken done.\n";
  apf::Field* new_field = mesh->findField(
      Albany::APFMeshStruct::solution_name[0]);
  assert(new_field);
  apf::destroyField(new_field);
}

void ExtrudedAdapt::adaptMesh(const Teuchos::RCP<Teuchos::ParameterList>& adapt_params_) {
  std::cerr << "adapting...\n";
  spr_helper.adaptMesh(adapt_params_);
  std::cerr << "adapting done.\n";
}

void ExtrudedAdapt::postProcessShrunkenMesh() {
  std::cerr << "post-processing shrunken...\n";
  spr_helper.postProcessShrunkenMesh();
  spr_helper.postProcessFinalMesh();
  std::cerr << "post-processing shrunken done.\n";
}

void ExtrudedAdapt::postProcessFinalMesh() {
  std::cerr << "post-processing final...\n";
  ma::extrude(mesh, model_extrusions, nlayers);
  std::cerr << "post-processing final done.\n";
}

}
