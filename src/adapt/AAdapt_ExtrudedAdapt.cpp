//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "AAdapt_ExtrudedAdapt.hpp"
#include "Albany_PUMIMeshStruct.hpp"
#include "AAdapt_SPRSizeField.hpp"

namespace AAdapt {

ExtrudedAdapt::ExtrudedAdapt(const Teuchos::RCP<Albany::APFDiscretization>& disc):
  MeshAdaptMethod(disc) {
  helper = new SPRSizeField(disc);
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

ExtrudedAdapt::~ExtrudedAdapt() {
  delete helper;
}

void ExtrudedAdapt::setParams(const Teuchos::RCP<Teuchos::ParameterList>& p) {
  helper->setParams(p);
}

void ExtrudedAdapt::preProcessOriginalMesh() {
  std::cerr << "pre-processing original (flattening)...\n";
  ma::intrude(mesh, model_extrusions, &nlayers);
  std::cerr << "flattening done.\n";
  std::cerr << "mesh dim is now " << mesh->getDimension() << ", "
    << mesh->count(0) << " vertices, "
    << mesh->count(mesh->getDimension()) << " elements.\n";
  SPRSizeField* spr_helper = dynamic_cast<SPRSizeField*>(helper);
  if (spr_helper) {
  /* we will use the top layer velocity for error estimation */
    std::string flat_name = ma::getFlatName(
          Albany::APFMeshStruct::solution_name[0], nlayers - 1);
    spr_helper->setSolName(flat_name);
  }
  std::cerr << "pre-processing original done.\n";
}

void ExtrudedAdapt::preProcessShrunkenMesh() {
  std::cerr << "pre-processing shrunken (error estimation by SPR)...\n";
  helper->preProcessOriginalMesh();
  helper->preProcessShrunkenMesh();
  std::cerr << "error estimation by SPR done.\n";
}

void ExtrudedAdapt::adaptMesh(const Teuchos::RCP<Teuchos::ParameterList>& adapt_params_) {
  std::cerr << "adapting...\n";
  helper->adaptMesh(adapt_params_);
  std::cerr << "adapting done.\n";
}

void ExtrudedAdapt::postProcessShrunkenMesh() {
  std::cerr << "post-processing shrunken...\n";
  helper->postProcessShrunkenMesh();
  helper->postProcessFinalMesh();
  std::cerr << "post-processing shrunken done.\n";
}

void ExtrudedAdapt::postProcessFinalMesh() {
  std::cerr << "post-processing final (extrude)...\n";
  ma::extrude(mesh, model_extrusions, nlayers);
  std::cerr << "extrusion done.\n";
  std::cerr << "mesh dim is now " << mesh->getDimension() << ", "
    << mesh->count(0) << " vertices, "
    << mesh->count(mesh->getDimension()) << " elements.\n";
}

}
