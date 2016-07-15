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
  ma::intrude(mesh, model_extrusions, &nlayers);
  spr_helper.setSolName(ma::getFlatName("solution", nlayers - 1));
}

void ExtrudedAdapt::preProcessShrunkenMesh() {
  spr_helper.preProcessOriginalMesh();
  spr_helper.preProcessShrunkenMesh();
}

void ExtrudedAdapt::adaptMesh(const Teuchos::RCP<Teuchos::ParameterList>& adapt_params_) {
  spr_helper.adaptMesh(adapt_params_);
}

void ExtrudedAdapt::postProcessShrunkenMesh() {
  spr_helper.postProcessShrunkenMesh();
  spr_helper.postProcessFinalMesh();
}

void ExtrudedAdapt::postProcessFinalMesh() {
  ma::extrude(mesh, model_extrusions, nlayers);
}

}
