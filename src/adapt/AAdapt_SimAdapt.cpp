#include "AAdapt_SimAdapt.hpp"
#include "Albany_SimDiscretization.hpp"
#include <MeshSimAdapt.h>
#include <apfSIM.h>
#include <spr.h>

namespace AAdapt {

SimAdapt::SimAdapt(const Teuchos::RCP<Teuchos::ParameterList>& params_,
                   const Teuchos::RCP<ParamLib>& paramLib_,
                   const Albany::StateManager& StateMgr_,
                   const Teuchos::RCP<const Teuchos_Comm>& commT_):
  AbstractAdapterT(params_, paramLib_, StateMgr_, commT_)
{
  errorBound = params_->get<double>("Error Bound", 0.1);
}

bool SimAdapt::queryAdaptationCriteria(int iteration)
{
  return true;
}

bool SimAdapt::adaptMesh(const Teuchos::RCP<const Tpetra_Vector>& solution,
                         const Teuchos::RCP<const Tpetra_Vector>& ovlp_solution)
{
  Teuchos::RCP<Albany::AbstractDiscretization> disc =
    state_mgr_.getDiscretization();
  Teuchos::RCP<Albany::SimDiscretization> sim_disc =
    Teuchos::rcp_dynamic_cast<Albany::SimDiscretization>(disc);
  Teuchos::RCP<Albany::APFMeshStruct> apf_ms =
    sim_disc->getAPFMeshStruct();
  apf::Mesh* apf_m = apf_ms->getMesh();
  apf::MeshSIM* apf_msim = dynamic_cast<apf::MeshSIM*>(apf_m);
  bool should_transfer_ip_data = adapt_params_->get<bool>("Transfer IP Data", false);
  if (should_transfer_ip_data)
    sim_disc->attachQPData();
  apf::Field* sol_fld = apf_m->findField(Albany::APFMeshStruct::solution_name);
  apf::Field* grad_ip_fld = spr::getGradIPField(sol_fld, "grad_sol",
      apf_ms->cubatureDegree);
  apf::Field* size_fld = spr::getSPRSizeField(grad_ip_fld, errorBound);
  apf::destroyField(grad_ip_fld);
  std::stringstream ss;
  static int i = 0;
  ss << "size_" << i++ << '_';
  std::string s = ss.str();
  apf::writeVtkFiles(s.c_str(), apf_m);
  pParMesh sim_pm = apf_msim->getMesh();
  pMSAdapt adapter = MSA_new(sim_pm, 1);
  apf::MeshEntity* v;
  apf::MeshIterator* it = apf_m->begin(0);
  while ((v = apf_m->iterate(it))) {
    double size = apf::getScalar(size_fld, v, 0);
    MSA_setVertexSize(adapter, (pVertex) v, size);
  }
  apf_m->end(it);
  apf::destroyField(size_fld);
  pProgress progress = Progress_new();
  MSA_adapt(adapter, progress);
  Progress_delete(progress);
  MSA_delete(adapter);
  apf_m->verify();
  apf::writeVtkFiles("adapted", apf_m);
  abort();
  sim_disc->updateMesh(should_transfer_ip_data);
  if (should_transfer_ip_data)
    sim_disc->detachQPData();
  return true;
}

Teuchos::RCP<const Teuchos::ParameterList> SimAdapt::getValidAdapterParameters()
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericAdapterParams("ValidSimAdaptParams");
  validPL->set<bool>("Transfer IP Data", false, "Turn on solution transfer of integration point data");
  validPL->set<double>("Error Bound", 0.1, "Max relative error for error-based adaptivity");
  return validPL;
}

}
