#include "AAdapt_SimAdapt.hpp"
#include "Albany_SimDiscretization.hpp"
#include <MeshSimAdapt.h>
#include <apfSIM.h>

namespace AAdapt {

SimAdapt::SimAdapt(const Teuchos::RCP<Teuchos::ParameterList>& params_,
                   const Teuchos::RCP<ParamLib>& paramLib_,
                   const Albany::StateManager& StateMgr_,
                   const Teuchos::RCP<const Teuchos_Comm>& commT_):
  AbstractAdapterT(params_, paramLib_, StateMgr_, commT_)
{
}

bool SimAdapt::queryAdaptationCriteria(int iteration)
{
  return true;
}

bool SimAdapt::adaptMesh(const Teuchos::RCP<const Tpetra_Vector>& solution,
                         const Teuchos::RCP<const Tpetra_Vector>& ovlp_solution)
{
  Teuchos::RCP<Albany::AbstractDiscretization> disc = state_mgr_.getDiscretization();
  Teuchos::RCP<Albany::SimDiscretization> sim_disc =
    Teuchos::rcp_dynamic_cast<Albany::SimDiscretization>(disc);
  Teuchos::RCP<Albany::APFMeshStruct> apf_ms =
    sim_disc->getAPFMeshStruct();
  apf::Mesh* apf_m = apf_ms->getMesh();
  apf::MeshSIM* apf_msim = dynamic_cast<apf::MeshSIM*>(apf_m);
  bool should_transfer_ip_data = adapt_params_->get<bool>("Transfer IP Data", false);
  if (should_transfer_ip_data)
    sim_disc->attachQPData();
  pParMesh sim_pm = apf_msim->getMesh();
  pMSAdapt adapter = MSA_new(sim_pm, 1);
  /* TODO: set size field here */
  MSA_adapt(adapter, NULL);
  MSA_delete(adapter);
  apf_m->verify();
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
  return validPL;
}

}
