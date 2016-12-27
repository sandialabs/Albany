//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "AAdapt_Omega_h_Method.hpp"
#include "Albany_PUMIMeshStruct.hpp"
#include "AAdapt_SPRSizeField.hpp"
#include "AAdapt_ConstantSizeField.hpp"
#include <apfOmega_h.h>
#include <apfMDS.h>

namespace AAdapt {

Omega_h_Method::Omega_h_Method(const Teuchos::RCP<Albany::APFDiscretization>& disc):
  MeshAdaptMethod(disc),
  library_osh(nullptr, nullptr),
  mesh_osh(&library_osh) {
  mesh_apf = mesh_struct->getMesh();
}

Omega_h_Method::~Omega_h_Method() {
  delete helper;
}

void Omega_h_Method::setParams(const Teuchos::RCP<Teuchos::ParameterList>& p) {
  std::string size_method = p->get<std::string>(
      "Size Method", "SPR");
  if (size_method == "SPR")
    helper = new SPRSizeField(apf_disc);
  else if (size_method == "Constant")
    helper = new ConstantSizeField(apf_disc);
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
        "Unknown \"Omega_h->Size Method\" option " << size_method << '\n');
  helper->setParams(p);
}

void Omega_h_Method::preProcessOriginalMesh() {
  helper->preProcessOriginalMesh();
}

void Omega_h_Method::preProcessShrunkenMesh() {
  helper->preProcessShrunkenMesh();
}

void Omega_h_Method::adaptMesh(const Teuchos::RCP<Teuchos::ParameterList>& adapt_params_) {
  apf::to_omega_h(&mesh_osh, mesh_apf);
  auto model_gmi = mesh_apf->getModel();
  mesh_apf->destroyNative();
  apf::destroyMesh(mesh_apf);
  Omega_h::AdaptOpts opts(&mesh_osh);
  Omega_h::adapt(&mesh_osh, opts);
  mesh_apf = apf::makeEmptyMdsMesh(model_gmi, mesh_osh.dim(), false);
  apf::from_omega_h(mesh_apf, &mesh_osh);
  mesh_osh = Omega_h::Mesh(&library_osh);
}

void Omega_h_Method::postProcessShrunkenMesh() {
  helper->postProcessShrunkenMesh();
}

void Omega_h_Method::postProcessFinalMesh() {
  helper->postProcessFinalMesh();
}

}
