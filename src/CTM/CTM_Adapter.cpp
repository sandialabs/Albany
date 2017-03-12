#include "CTM_Adapter.hpp"

#include <set>
#include <spr.h>
#include <apfSIM.h>
#include <SimField.h>
#include <SimPartitionedMesh.h>
#include <MeshSimAdapt.h>
#include <Albany_StateManager.hpp>
#include <Albany_SimDiscretization.hpp>
#include <Teuchos_ParameterList.hpp>

namespace CTM {

using Teuchos::rcp_dynamic_cast;

static RCP<ParameterList> get_valid_params() {
  auto p = rcp(new ParameterList);
  p->set<double>("Error Bound", 0.1, "Max relative error for adaptivity");
  p->set<double>("Max Size", 1e10, "Maximum allowed edge length (size field)");
  p->set<double>("Min Size", 1e-2, "Mininum allowed edge length (size field)");
  p->set<double>("Gradation", 0.3, "Mesh size gradation parameter");
  p->set<bool>("Debug", false, "Print debug VTK files");
  p->set<double>("Uniform Temperature New Layer", 20.0, "Uniform layer temperature");
  p->set<std::string>("SPR Solution Field", "Temp", "Field name for SPR to operate on");
  p->set<long int>("Target Element Count", 1000, "Desired # of elements for spr adaptivity");
  return p;
}

Adapter::Adapter(
    RCP<ParameterList> p,
    RCP<Albany::StateManager> tsm,
    RCP<Albany::StateManager> msm) {
  params = p;
  params->validateParameters(*get_valid_params(), 0);
  t_state_mgr = tsm;
  m_state_mgr = msm;
  t_disc = t_state_mgr->getDiscretization();
  m_disc = m_state_mgr->getDiscretization();
  out = Teuchos::VerboseObjectBase::getDefaultOStream();

  auto sim_disc = rcp_dynamic_cast<Albany::SimDiscretization>(m_disc);
  auto apf_ms = sim_disc->getAPFMeshStruct();
  auto apf_mesh = apf_ms->getMesh();
  auto apf_sim_mesh = dynamic_cast<apf::MeshSIM*>(apf_mesh);
  auto sim_mesh = apf_sim_mesh->getMesh();
  sim_model = M_model(sim_mesh);
  compute_layer_times();

  *out << std::endl;
  *out << "*********************" << std::endl;
  *out << "LAYER ADDING ENABLED " << std::endl;
  *out << "*********************" << std::endl;
}

void Adapter::compute_layer_times() {
}

bool Adapter::should_adapt(const int step) {
  return true;
}

void Adapter::adapt() {
  *out << "I'M ADAPTING!!!" << std::endl;
}

} // namespace CTM
