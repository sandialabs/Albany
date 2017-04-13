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

#include <Omega_h_teuchos.hpp>

namespace AAdapt {

Omega_h_Method::Omega_h_Method(const Teuchos::RCP<Albany::APFDiscretization>& disc):
  MeshAdaptMethod(disc),
  should_target_count(false),
  should_target_error(false),
  library_osh(nullptr, nullptr),
  mesh_osh(&library_osh) {
  mesh_apf = mesh_struct->getMesh();
}

Omega_h_Method::~Omega_h_Method() {
}

void Omega_h_Method::setParams(const Teuchos::RCP<Teuchos::ParameterList>& p) {
  auto& omega_h_pl = p.sublist("Omega_h");
  Omega_h::update_adapt_opts(&adapt_opts, omega_h_pl);
  auto& metric_pl = omega_h_pl.sublist("Metric");
  Omega_h::update_metric_input(&metric_opts, metric_pl);
}

void Omega_h_Method::preProcessOriginalMesh() {
}

void Omega_h_Method::preProcessShrunkenMesh() {
}

void Omega_h_Method::adaptMesh(const Teuchos::RCP<Teuchos::ParameterList>& adapt_params_) {
  apf::to_omega_h(&mesh_osh, mesh_apf);
  apf::clear(mesh_apf);
  mesh_osh.set_parting(OMEGA_H_GHOSTED);
  Omega_h::add_implied_metric_tag(&mesh_osh);
  Omega_h::generate_target_metric_tag(&mesh_osh, metric_opts);
  while (Omega_h::approach_metric(&mesh_osh, adapt_opts)) {
    Omega_h::adapt(&mesh_osh, adapt_opts);
  }
  apf::from_omega_h(mesh_apf, &mesh_osh);
  mesh_osh = Omega_h::Mesh(&library_osh);
}

void Omega_h_Method::postProcessShrunkenMesh() {
}

void Omega_h_Method::postProcessFinalMesh() {
}

}
