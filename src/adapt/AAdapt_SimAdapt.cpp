//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

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
  if (strategy == "Every N Step Number") {
            TEUCHOS_TEST_FOR_EXCEPTION(!adapt_params_->isParameter("Remesh Every N Step Number"),
                    std::logic_error,
                    "Remesh Strategy " << strategy << " but no Remesh Every N Step Number" << '\n');
            int remesh_iter = adapt_params_->get<int>("Remesh Every N Step Number", -1);
            // check user do not specify a zero or negative value
            TEUCHOS_TEST_FOR_EXCEPTION(remesh_iter <= 0, std::logic_error,
                    "Value must be positive" << '\n');
            if (iteration % remesh_iter == 0)
                return true;
            return false;
        }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
      "Unknown Remesh Strategy " << strategy << '\n');
  return false;
}

bool SimAdapt::adaptMesh()
{
    TEUCHOS_FUNC_TIME_MONITOR("SimAdapt: Adapt Mesh");
    
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

  /* grab the solution fields from the discretization.
     here we assume that the apf_ms->num_time_deriv = 0
     I think this is currently valid for all Sim* problems. */
  Albany::SolutionLayout soln_layout = sim_disc->getSolutionLayout();
  std::vector<apf::Field*> soln_fields;
  Teuchos::Array<std::string> soln_field_names = soln_layout.getDerivNames(0);
  int num_soln_fields = soln_layout.getNumSolFields();
  assert( soln_field_names.size() == num_soln_fields );
  for (int i=0; i < num_soln_fields; ++i) {
    soln_fields.push_back(apf_m->findField(soln_field_names[i].c_str()));
    assert( soln_fields[i] );
  }

  /* grab the residual fields from the discretization */
  std::vector<apf::Field*> res_fields;
  Teuchos::Array<std::string> res_names = sim_disc->getResNames();
  assert( res_names.size() == num_soln_fields );
  for (int i=0; i < num_soln_fields; ++i) {
    res_fields.push_back(apf_m->findField(res_names[i].c_str()));
    assert( res_fields[i] );
  }

  /* compute the size field via SPR error estimation on the gradient
     of the chosen solution field. */
  int spr_idx = adapt_params_->get<int>("SPR Solution Index", 0);
  apf::Field* grad_ip_fld = spr::getGradIPField(
      soln_fields[spr_idx], "grad_sol", apf_ms->cubatureDegree);
  apf::Field* size_fld;
  if (adapt_params_->isType<long int>("Target Element Count")) {
    long N = adapt_params_->get<long int>("Target Element Count");
    size_fld = spr::getTargetSPRSizeField(grad_ip_fld, N);
  }
  else if (adapt_params_->isType<double>("Error Bound")) {
    double error_bound = adapt_params_->get<double>("Error Bound", 0.1);
    size_fld = spr::getSPRSizeField(grad_ip_fld, error_bound);
  }
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
        "invalid SimAdapt SPR inputs\n");
  apf::destroyField(grad_ip_fld);

#ifdef SIMDEBUG
  /* write the mesh with size field to file */
  std::stringstream ss;
  ss << "size_" << callcount << '_';
  std::string s = ss.str();
  apf::writeVtkFiles(s.c_str(), apf_m);
#endif

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

  /* tell the Simmetrix adapter to transfer the soln/residual fields */
  std::vector<pField> sim_soln_fields;
  std::vector<pField> sim_res_fields;
  pPList sim_fld_lst = PList_new();
  for (int i=0; i < num_soln_fields; ++i) {
    sim_soln_fields.push_back(apf::getSIMField(soln_fields[i]));
    sim_res_fields.push_back(apf::getSIMField(res_fields[i]));
    PList_append(sim_fld_lst, sim_soln_fields[i]);
    PList_append(sim_fld_lst, sim_res_fields[i]);
  }

  /* Append the old temperature state if specified to transfer
     temperature at the nodes. */
  if (apf_ms->useTemperatureHack) {
    apf::Field* told_fld = apf_m->findField("temp_old");
    assert(told_fld);
    pField sim_told_fld = apf::getSIMField(told_fld);
    PList_append(sim_fld_lst, sim_told_fld);
  }

  MSA_setMapFields(adapter, sim_fld_lst);
  PList_delete(sim_fld_lst);

#ifdef SIMDEBUG
  apf::writeVtkFiles("before", apf_m);
  char simname[80];
  sprintf(simname, "preadapt_%d.sms", callcount);
  PM_write(sim_pm, simname, sthreadDefault, 0);
  for (int i = 0; i <= num_soln_fields; ++i) {
    sprintf(simname, "preadapt_sol_%d_%d.fld", i, callcount);
    Field_write(sim_soln_fields[i], simname, 0, 0, 0);
    sprintf(simname, "preadpt_res_%d_%d.fld", i, callcount);
    Field_write(sim_res_fields[i], simname, 0, 0, 0);
  }
#endif

  /* run the adapter */
  pProgress progress = Progress_new();
  MSA_adapt(adapter, progress);
  Progress_delete(progress);
  MSA_delete(adapter);

#ifdef SIMDEBUG
  sprintf(simname, "adapted_%d.sms", callcount);
  PM_write(sim_pm, simname, sthreadDefault, 0);
  for (int i = 0; i <= apf_ms->num_time_deriv; ++i) {
    sprintf(simname, "adapted_sol%d_%d.fld", i, callcount);
    Field_write(sim_sol_flds[i], simname, 0, 0, 0);
  }
  sprintf(simname, "adapted_res_%d.fld", callcount);
  Field_write(sim_res_fld, simname, 0, 0, 0);
  Albany::debugAMPMesh(apf_m, "after");
#endif
  
  /* run APF verification on the resulting mesh */
  apf_m->verify();

  /* update Albany structures to reflect the adapted mesh */
  sim_disc->updateMesh(should_transfer_ip_data, param_lib_);

  /* see the comment in Albany_APFDiscretization.cpp */
  sim_disc->initTemperatureHack();

  ++callcount;
  return true;
}

Teuchos::RCP<const Teuchos::ParameterList>
AAdapt::SimAdapt::getValidAdapterParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericAdapterParams("ValidSimAdaptParams");
  validPL->set<bool>("Transfer IP Data", false, "Turn on solution transfer of integration point data");
  validPL->set<bool>("Equilibrate", false, "Perform an equilibration step after adaptivity");
  validPL->set<double>("Error Bound", 0.1, "Max relative error for error-based adaptivity");
  validPL->set<double>("Max Size", 1e10, "Maximum allowed edge length (size field)");
  validPL->set<bool>("Add Layer", false, "Turn on/off adding layer");
  validPL->set<std::string>("Remesh Strategy", "", "Strategy for when to adapt");
  validPL->set<int>("Remesh Every N Step Number", 1, "Remesh every Nth load/time step");
  validPL->set<int>("SPR Solution Index", 0, "Solution field index for SPR to operate on");
  validPL->set<long int>("Target Element Count", 1000, "Desired number of elements for error-based adaptivity");
  return validPL;
}

}
