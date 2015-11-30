#include "AAdapt_SimAdapt.hpp"
#include "Albany_SimDiscretization.hpp"
#include <MeshSimAdapt.h>
#include <SimPartitionedMesh.h>
#include <SimField.h>
#include <apfSIM.h>
#include <spr.h>
#include <EnergyIntegral.hpp>

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
  std::string strategy = adapt_params_->get<std::string>("Remesh Strategy", "Step Number");
  if (strategy == "None")
    return false;
  if (strategy == "Continuous")
    return iteration > 1;
  if (strategy == "Step Number") {
    TEUCHOS_TEST_FOR_EXCEPTION(!adapt_params_->isParameter("Remesh Step Number"),
        std::logic_error,
        "Remesh Strategy " << strategy << " but no Remesh Step Number" << '\n');
    Teuchos::Array<int> remesh_iter = adapt_params_->get<Teuchos::Array<int> >("Remesh Step Number");
    for(int i = 0; i < remesh_iter.size(); i++)
      if(iteration == remesh_iter[i])
        return true;
    return false;
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
      "Unknown Remesh Strategy " << strategy << '\n');
  return false;
}

bool SimAdapt::adaptMesh(const Teuchos::RCP<const Tpetra_Vector>& solution,
                         const Teuchos::RCP<const Tpetra_Vector>& ovlp_solution)
{
  /* dig through all the abstrations to obtain pointers
     to the various structures needed */
  static int callcount = 0;
  Teuchos::RCP<Albany::AbstractDiscretization> disc =
    state_mgr_.getDiscretization();
  Teuchos::RCP<Albany::SimDiscretization> sim_disc =
    Teuchos::rcp_dynamic_cast<Albany::SimDiscretization>(disc);
  Teuchos::RCP<Albany::APFMeshStruct> apf_ms =
    sim_disc->getAPFMeshStruct();
  apf::Mesh* apf_m = apf_ms->getMesh();
  apf::MeshSIM* apf_msim = dynamic_cast<apf::MeshSIM*>(apf_m);
  pParMesh sim_pm = apf_msim->getMesh();
  /* ensure that users don't expect Simmetrix to transfer IP state */
  bool should_transfer_ip_data = adapt_params_->get<bool>("Transfer IP Data", false);
  /* remove this assert when Simmetrix support IP transfer */
  assert(!should_transfer_ip_data);
  /* compute the size field via SPR error estimation
     on the solution gradient */
  apf::Field* sol_fld = apf_m->findField(Albany::APFMeshStruct::solution_name);
  assert(apf::countComponents(sol_fld) == 1);
  apf::Field* grad_ip_fld = spr::getGradIPField(sol_fld, "grad_sol",
      apf_ms->cubatureDegree);
  apf::Field* size_fld = spr::getSPRSizeField(grad_ip_fld, errorBound);
//  Estimation meshFinal;

  apf::destroyField(grad_ip_fld);
  /* write the mesh with size field to file */
  std::stringstream ss;
  ss << "size_" << callcount << '_';
  std::string s = ss.str();
  apf::writeVtkFiles(s.c_str(), apf_m);
  /* create the Simmetrix adapter */
  pMSAdapt adapter = MSA_new(sim_pm, 1);
  /* copy the size field from APF to the Simmetrix adapter */
  apf::MeshEntity* v;
  apf::MeshIterator* it = apf_m->begin(0);
  double max_size = adapt_params_->get<double>("Max Size", 1e10);
  while ((v = apf_m->iterate(it))) {
    double size1 = apf::getScalar(size_fld, v, 0);
    double size = std::min(max_size, size1);
    MSA_setVertexSize(adapter, (pVertex) v, size);
  }
  apf_m->end(it);
  apf::destroyField(size_fld);
  /* tell the adapter to transfer the solution and residual fields */
  apf::Field* res_fld = apf_m->findField(Albany::APFMeshStruct::residual_name);
  pField sim_sol_fld = apf::getSIMField(sol_fld);
  pField sim_res_fld = apf::getSIMField(res_fld);
  pPList sim_fld_lst = PList_new();
  PList_append(sim_fld_lst, sim_sol_fld);
  PList_append(sim_fld_lst, sim_res_fld);
  if (apf_ms->useTemperatureHack) {
    /* transfer Temperature_old at the nodes */
    apf::Field* told_fld = apf_m->findField("temp_old");
    pField sim_told_fld = apf::getSIMField(told_fld);
    PList_append(sim_fld_lst, sim_told_fld);
  }
  MSA_setMapFields(adapter, sim_fld_lst);
  PList_delete(sim_fld_lst);

#ifdef SIMDEBUG
  char simname[80];
  sprintf(simname, "preadapt_%d.sms", callcount);
  PM_write(sim_pm, simname, sthreadDefault, 0);
  sprintf(simname, "preadapt_sol_%d.fld", callcount);
  Field_write(sim_sol_fld, simname, 0, 0, 0);
  sprintf(simname, "preadapt_res_%d.fld", callcount);
  Field_write(sim_res_fld, simname, 0, 0, 0);
#endif
  Albany::debugAMPMesh(apf_m, "before");
  /* run the adapter */
  pProgress progress = Progress_new();
  MSA_adapt(adapter, progress);
  Progress_delete(progress);
  MSA_delete(adapter);
#ifdef SIMDEBUG
  sprintf(simname, "adapted_%d.sms", callcount);
  PM_write(sim_pm, simname, sthreadDefault, 0);
  sprintf(simname, "adapted_sol_%d.fld", callcount);
  Field_write(sim_sol_fld, simname, 0, 0, 0);
  sprintf(simname, "adapted_res_%d.fld", callcount);
  Field_write(sim_res_fld, simname, 0, 0, 0);
#endif

  /* run APF verification on the resulting mesh */
  apf_m->verify();
  Albany::debugAMPMesh(apf_m, "after");
  /* update Albany structures to reflect the adapted mesh */
  sim_disc->updateMesh(should_transfer_ip_data);
  /* see the comment in Albany_APFDiscretization.cpp */
  sim_disc->initTemperatureHack();
  ++callcount;
  return true;
}

Teuchos::RCP<const Teuchos::ParameterList> SimAdapt::getValidAdapterParameters()
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericAdapterParams("ValidSimAdaptParams");
  validPL->set<bool>("Transfer IP Data", false, "Turn on solution transfer of integration point data");
  validPL->set<double>("Error Bound", 0.1, "Max relative error for error-based adaptivity");
  validPL->set<double>("Max Size", 1e10, "Maximum allowed edge length (size field)");
  return validPL;
}

}
