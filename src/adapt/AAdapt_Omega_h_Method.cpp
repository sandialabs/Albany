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
#include <Omega_h_math.hpp>

namespace AAdapt {

Omega_h_Method::Omega_h_Method(const Teuchos::RCP<Albany::APFDiscretization>& disc):
  MeshAdaptMethod(disc),
  should_target_count(false),
  should_target_error(false),
  helper(nullptr),
  library_osh(nullptr, nullptr),
  mesh_osh(&library_osh) {
  mesh_apf = mesh_struct->getMesh();
}

Omega_h_Method::~Omega_h_Method() {
  delete helper;
}

void Omega_h_Method::setParams(const Teuchos::RCP<Teuchos::ParameterList>& p) {
  size_method = p->get<std::string>(
      "Size Method", "SPR");
  if (size_method == "SPR")
    helper = new SPRSizeField(apf_disc);
  else if (size_method == "Constant")
    helper = new ConstantSizeField(apf_disc);
  else if (size_method == "Hessian") {
    TEUCHOS_TEST_FOR_EXCEPTION(!p->isType<double>("Maximum Size"),
        std::logic_error,
        "Must specify \"Maximum Size\" for \"Hessian\" size field\n");
    maximum_size = p->get<double>("Maximum Size");
    if (p->isParameter("Error Bound")) {
      should_target_error = true;
      target_error = p->get<double>("Error Bound");
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION(!p->isType<long int>("Target Element Count"),
          std::logic_error,
          "Must specify either \"Error Bound\" or \"Target Element Count\"");
      should_target_count = true;
      target_count = p->get<double>("Target Element Count");
    }
  } else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
        "Unknown Omega_h \"Size Method\" option " << size_method << '\n');
  should_smooth_metric = p->isType<int>("Metric Smooth Steps");
  if (should_smooth_metric) {
    metric_smooth_steps = p->get<int>("Metric Smooth Steps");
  }
  should_limit_gradation = p->isType<double>("Gradation Rate Limit");
  if (should_limit_gradation) {
    gradation_rate = p->get<double>("Gradation Rate Limit");
  }
  should_prevent_overshoot = p->isType<double>("Overshoot Allowance");
  if (should_prevent_overshoot) {
    overshoot_allowance = p->get<double>("Overshoot Allowance");
  }
  should_use_curvature = p->isType<double>("Max Edge Angle");
  if (should_use_curvature) {
    segment_angle = p->get<double>("Max Edge Angle");
    TEUCHOS_TEST_FOR_EXCEPTION(!p->isType<double>("Maximum Size"),
        std::logic_error,
        "Must specify \"Maximum Size\" with \"Max Edge Angle\"\n");
    maximum_size = p->get<double>("Maximum Size");
  }
  if (helper) helper->setParams(p);
}

void Omega_h_Method::preProcessOriginalMesh() {
  if (helper) helper->preProcessOriginalMesh();
}

void Omega_h_Method::preProcessShrunkenMesh() {
  if (helper) helper->preProcessShrunkenMesh();
}

void Omega_h_Method::adaptMesh(const Teuchos::RCP<Teuchos::ParameterList>& adapt_params_) {
  apf::to_omega_h(&mesh_osh, mesh_apf);
  apf::clear(mesh_apf);
  mesh_osh.set_parting(OMEGA_H_GHOSTED);
  if (mesh_osh.has_tag(0, "size")) {
    mesh_osh.add_tag(0, "target_size", 1, OMEGA_H_SIZE,
        OMEGA_H_DO_OUTPUT, mesh_osh.get_array<double>(0, "size"));
    mesh_osh.remove_tag(0, "size");
  }
  if (size_method == "Hessian") {
    auto sol_name = Albany::APFMeshStruct::solution_name[0];
    TEUCHOS_TEST_FOR_EXCEPTION(!mesh_osh.has_tag(0, sol_name), std::logic_error,
        "No tag \"" << sol_name << "\" on Omega_h mesh\n");
    auto sol_tag = mesh_osh.get_tag<double>(0, sol_name);
    TEUCHOS_TEST_FOR_EXCEPTION(sol_tag->ncomps() != 1, std::logic_error,
        "Hessians from multivariable solutions not yet supported\n");
    auto sol = sol_tag->array();
    auto hessians = Omega_h::recover_hessians(&mesh_osh, sol);
    Omega_h::Reals metrics;
    if (should_target_error) {
      metrics = metric_from_hessians(mesh_osh.dim(), hessians,
          target_error, maximum_size);
    }
    if (should_target_count) {
      metrics = metric_for_nelems_from_hessians(&mesh_osh, target_count,
          1e-3, hessians, maximum_size);
    }
    mesh_osh.add_tag(0, "target_metric", Omega_h::symm_dofs(mesh_osh.dim()),
        OMEGA_H_METRIC, OMEGA_H_DO_OUTPUT, metrics);
  }
  if (should_use_curvature) {
    auto old_isos =  mesh_osh.get_array<double>(0, "target_size");
    auto curv_isos = Omega_h::get_curvature_isos(&mesh_osh,
        segment_angle, maximum_size);
    auto new_isos = Omega_h::min_each(old_isos, curv_isos);
    mesh_osh.set_tag(0, "target_size", new_isos);
  }
  if (should_smooth_metric) {
    for (int i = 0; i < metric_smooth_steps; ++i) {
      if (mesh_osh.has_tag(0, "target_metric")) {
        auto metrics =  mesh_osh.get_array<double>(0, "target_metric");
        metrics = Omega_h::smooth_metric_once(&mesh_osh, metrics);
        mesh_osh.set_tag(0, "target_metric", metrics);
      }
      if (mesh_osh.has_tag(0, "target_size")) {
        auto isos =  mesh_osh.get_array<double>(0, "target_size");
        isos = Omega_h::smooth_isos_once(&mesh_osh, isos);
        mesh_osh.set_tag(0, "target_size", isos);
      }
    }
  }
  if (should_limit_gradation) {
    if (mesh_osh.has_tag(0, "target_metric")) {
      auto metrics =  mesh_osh.get_array<double>(0, "target_metric");
      metrics = Omega_h::limit_size_field_gradation(&mesh_osh, metrics,
          gradation_rate);
      mesh_osh.set_tag(0, "target_metric", metrics);
    }
    if (mesh_osh.has_tag(0, "target_size")) {
      auto isos =  mesh_osh.get_array<double>(0, "target_size");
      isos = Omega_h::limit_size_field_gradation(&mesh_osh, isos,
          gradation_rate);
      mesh_osh.set_tag(0, "target_size", isos);
    }
  }
  if (mesh_osh.has_tag(0, "target_metric")) {
    auto implied_metrics = Omega_h::find_implied_metric(&mesh_osh);
    mesh_osh.add_tag(0, "metric", Omega_h::symm_dofs(mesh_osh.dim()),
        OMEGA_H_METRIC, OMEGA_H_DO_OUTPUT, implied_metrics);
  }
  if (mesh_osh.has_tag(0, "target_size")) {
    auto implied_size = Omega_h::find_implied_size(&mesh_osh);
    mesh_osh.add_tag(0, "size", 1,
        OMEGA_H_SIZE, OMEGA_H_DO_OUTPUT, implied_size);
  }
  Omega_h::AdaptOpts opts(&mesh_osh);
  if (should_prevent_overshoot) {
    opts.max_length_allowed = overshoot_allowance;
  }
  while (Omega_h::approach_size_field(&mesh_osh, opts)) {
    Omega_h::adapt(&mesh_osh, opts);
  }
  apf::from_omega_h(mesh_apf, &mesh_osh);
  mesh_osh = Omega_h::Mesh(&library_osh);
}

void Omega_h_Method::postProcessShrunkenMesh() {
  if (helper) helper->postProcessShrunkenMesh();
}

void Omega_h_Method::postProcessFinalMesh() {
  if (helper) helper->postProcessFinalMesh();
}

}
