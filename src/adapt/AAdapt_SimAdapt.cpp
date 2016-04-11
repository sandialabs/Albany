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
#include "PHAL_Utilities.hpp"

#if defined ALBANY_AMP_ADD_LAYER
extern void DM_undoSlicing(pPList regions, int layerNum, pMesh mesh);
#endif

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

#if defined ALBANY_AMP_ADD_LAYER
void meshCurrentLayerOnly(pGModel model,pParMesh mesh,int currentLayer,bool skipSmesh)
{
  pACase mcase = MS_newMeshCase(model);
  MS_setMeshSize(mcase,GM_domain(model),1,0.003,0);  // 0.3 for clevis, 0.003 for bracket & cube
  MS_setMeshCurv(mcase,GM_domain(model),2,0.025);
  MS_setMinCurvSize(mcase,GM_domain(model),2,0.0025);

  // Top layer mesh sizes
  // Bracket (& cube) = 0.0003
  // Clevis  = 0.01
  GRIter regions = GM_regionIter(model);
  pGRegion gr;
  int layer;
  while (gr=GRIter_next(regions)) {
    if (GEN_numNativeIntAttribute(gr,"SimLayer")==1) {
      GEN_nativeIntAttribute(gr,"SimLayer",&layer);
      if (layer==currentLayer)
        MS_setMeshSize(mcase,gr,1,0.0003,0);
      else if (layer > currentLayer)
        MS_setNoMesh(mcase,gr,1);
    }
  }
  GRIter_delete(regions);
  
  if (!skipSmesh) {
    pSurfaceMesher sm = SurfaceMesher_new(mcase,mesh);
    // SurfaceMesher_setParamForDiscrete(sm, 1);
    SurfaceMesher_execute(sm,0);
    SurfaceMesher_delete(sm);
  }
  PM_setTotalNumParts(mesh,PMU_size());
  pVolumeMesher vm  = VolumeMesher_new(mcase,mesh);
  VolumeMesher_execute(vm,0);
  VolumeMesher_delete(vm);

  MS_deleteMeshCase(mcase);
}

void adaptMesh2(pGModel model,pParMesh mesh,int currentLayer,double layerThickness,pPList flds)
{
  pACase mcase = MS_newMeshCase(model);
  MS_setMeshSize(mcase,GM_domain(model),1,0.3,0);
  MS_setMeshCurv(mcase,GM_domain(model),2,0.025);
  MS_setMinCurvSize(mcase,GM_domain(model),2,0.0025);
  //MS_setGlobalSizeGradationRate(mcase,0.1);
  
  //M_write(mesh,"beforeSizes.sms",0,0);
  pMSAdapt adaptM = MSA_createFromCase(mcase,mesh);
  MSA_setSizeGradation(adaptM,1,0.1);
  pVertex v;
  for(int np=0;np<PM_numParts(mesh);np++) {
    VIter vertices = M_vertexIter(PM_mesh(mesh,np));
    while (v = VIter_next(vertices) ) {
      double xyz[3];
      V_coord(v,xyz);
      //  Commented out condition as it skips some points we want - SST
      // if (xyz[2] >= 0.0) {  // ignore everything below the plate
      // AND only set sizes for those mesh vertices that are in the 
      // closure of a SimLayer region and below the current layer.
      bool adaptMesh = false;
      pPList mfaces = V_faces(v);
      for(int i=0;i<PList_size(mfaces) && !adaptMesh;i++) {
        pFace mf = static_cast<pFace>(PList_item(mfaces,i));
        if (F_whatInType(mf)==Gface) {
          pGFace gf = static_cast<pGFace>(F_whatIn(mf));
          for(int j=0;j<2;j++) {
            pGRegion gr = GF_region(gf,j);
            if (gr && (GEN_numNativeIntAttribute(gr,"SimLayer")==1))
              adaptMesh = true;
          }
        } else if (F_whatInType(mf)==Gregion) {
          pGRegion gr = static_cast<pGRegion>(F_whatIn(mf));
          if (GEN_numNativeIntAttribute(gr,"SimLayer")==1)
            adaptMesh = true;
        }
      }
      PList_delete(mfaces);
      if (adaptMesh) {
        // make sure the mesh vertex is below the current layer
        int layer = xyz[2]/layerThickness;
        if (layer < currentLayer)
          MSA_setVertexSize(adaptM,v,0.3);
      }
      // }
    }
    VIter_delete(vertices);
  }
  //M_write(mesh,"afterSizes.sms",0,0);
  if (flds)
    MSA_setMapFields(adaptM,flds);
  MSA_adapt(adaptM,0);
  MSA_delete(adaptM);
  MS_deleteMeshCase(mcase);
}

void addNextLayer(pParMesh sim_pm,int nextLayer,pPList flds) {
  double layerThickness;
  pGModel model = M_model(sim_pm);
  GIP_nativeDoubleAttribute(GM_part(model),"SimLayerThickness",&layerThickness);
  
  // Collect the layer 0 regions
  GRIter regions = GM_regionIter(model);
  pGRegion gr1;
  int layer, maxLayer = -1;
  pPList combinedRegions = PList_new();
  while (gr1=GRIter_next(regions)) {
    if (GEN_numNativeIntAttribute(gr1,"SimLayer")==1) {
      GEN_nativeIntAttribute(gr1,"SimLayer",&layer);
      if (layer==0)
        PList_appUnique(combinedRegions,gr1);
      if (layer > maxLayer)
        maxLayer = layer;
    }
  }
  GRIter_delete(regions);
  if ( nextLayer > maxLayer )
    return;

  if (nextLayer>1) {
    pPartitionOpts popts = PM_newPartitionOpts();
    PartitionOpts_setTotalNumParts(popts, 1);
    PM_partition(sim_pm, popts, sthreadNone, 0);
    PartitionOpts_delete(popts);
    pMesh oneMesh = PM_mesh(sim_pm,0);
    std::cout << "Combine layer " << nextLayer-1 << "\n";
    DM_undoSlicing(combinedRegions,nextLayer-1,oneMesh);
  }
  PList_clear(combinedRegions);
  std::cout << "Mesh top layer\n";
  meshCurrentLayerOnly(model,sim_pm,nextLayer,false);
  /*
  if (nextLayer>1) {
    adaptMesh2(model,sim_pm,nextLayer,layerThickness,0);
  }
  */

  if (flds) {
    // Add temperature and residual fields to top layer
    // Add temperature HACK fields to top layer
    // NOTE: this assumes a SINGLE partition
    assert(PM_numParts(sim_pm)==1);
    pField sim_sol_fld = static_cast<pField>(PList_item(flds,0));
    pField sim_res_fld = static_cast<pField>(PList_item(flds,1));
    pField sim_hak_fld = 0;
    if (PList_size(flds)==3)
      sim_hak_fld = static_cast<pField>(PList_item(flds,2));
    pMesh localMesh = PM_mesh(sim_pm,0);
    pMEntitySet topLayerVerts = MEntitySet_new(localMesh);
    regions = GM_regionIter(model);
    pVertex mv;
    while (gr1=GRIter_next(regions)) {
      if (GEN_numNativeIntAttribute(gr1,"SimLayer")==1) {
        GEN_nativeIntAttribute(gr1,"SimLayer",&layer);
        if (layer == nextLayer) {
          for(int np=0;np<PM_numParts(sim_pm);np++) {
            VIter allVerts = M_classifiedVertexIter(PM_mesh(sim_pm,np),gr1,1);
            while ( mv = VIter_next(allVerts) )
              MEntitySet_add(topLayerVerts,mv);
            VIter_delete(allVerts);
          }
        }
      }
    }
    pPList unmapped = PList_new();
    pDofGroup dg;
    MESIter viter = MESIter_iter(topLayerVerts);
    while ( mv = reinterpret_cast<pVertex>(MESIter_next(viter)) ) {
      dg = Field_entDof(sim_sol_fld,mv,0);
      if (!dg) {
        PList_append(unmapped,mv);
        Field_applyEnt(sim_sol_fld,mv);
        Field_applyEnt(sim_res_fld,mv);
        if (sim_hak_fld)
          Field_applyEnt(sim_hak_fld,mv);
      } 
    }
    MESIter_delete(viter);
    pEntity ent;
    void *vptr;
    int c, nc = Field_numComp(sim_sol_fld);
    int nc2 = (sim_res_fld ? Field_numComp(sim_res_fld) : 0);
    int nc3 = (sim_hak_fld ? Field_numComp(sim_hak_fld) : 0);
    void *iter = 0;
    while (vptr = PList_next(unmapped,&iter)) {
      ent = reinterpret_cast<pEntity>(vptr);
      dg = Field_entDof(sim_sol_fld,ent,0);
      for (c=0; c < nc; c++)
        DofGroup_setValue(dg,c,0,21.0);
      if (sim_res_fld) {
        dg = Field_entDof(sim_res_fld,ent,0);
        for (c=0; c < nc2; c++)
          DofGroup_setValue(dg,c,0,0.0);
      }
      if (sim_hak_fld) {
        dg = Field_entDof(sim_hak_fld,ent,0);
        for (c=0; c < nc3; c++)
          DofGroup_setValue(dg,c,0,21.0);
      }
    }
    PList_delete(unmapped);
    MEntitySet_delete(localMesh,topLayerVerts);
    GRIter_delete(regions);
  }
  return;
}
#endif

bool SimAdapt::adaptMesh()
{
    TEUCHOS_FUNC_TIME_MONITOR("SimAdapt: Adapt Mesh");
    
# if !defined ALBANY_AMP_ADD_LAYER
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
  apf::Field* sol_flds[3];
  for (int i = 0; i <= apf_ms->num_time_deriv; ++i)
    sol_flds[i] = apf_m->findField(Albany::APFMeshStruct::solution_name[i]);
  apf::Field* grad_ip_fld = spr::getGradIPField(sol_flds[0], "grad_sol",
      apf_ms->cubatureDegree);
  apf::Field* size_fld = spr::getSPRSizeField(grad_ip_fld, errorBound);
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
  /* tell the adapter to transfer the solution and residual fields */
  apf::Field* res_fld = apf_m->findField(Albany::APFMeshStruct::residual_name);
  pField sim_sol_flds[3];
  for (int i = 0; i <= apf_ms->num_time_deriv; ++i)
    sim_sol_flds[i] = apf::getSIMField(sol_flds[i]);
  pField sim_res_fld = apf::getSIMField(res_fld);
  pPList sim_fld_lst = PList_new();
  for (int i = 0; i <= apf_ms->num_time_deriv; ++i)
    PList_append(sim_fld_lst, sim_sol_flds[i]);
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
  for (int i = 0; i <= apf_ms->num_time_deriv; ++i) {
    sprintf(simname, "preadapt_sol%d_%d.fld", i, callcount);
    Field_write(sim_sol_flds[i], simname, 0, 0, 0);
  }
  sprintf(simname, "preadapt_res_%d.fld", callcount);
  Field_write(sim_res_fld, simname, 0, 0, 0);
  Albany::debugAMPMesh(apf_m, "before");
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
  sim_disc->updateMesh(should_transfer_ip_data);
  /* see the comment in Albany_APFDiscretization.cpp */
  sim_disc->initTemperatureHack();
  ++callcount;
  return true;
  
#else
  
  /* dig through all the abstrations to obtain pointers
     to the various structures needed */
  static int callcount = 0;
  static int Simmetrix_currentLayer = 1;
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
  apf::Field* sol_flds[3];
  for (int i = 0; i <= apf_ms->num_time_deriv; ++i)
    sol_flds[i] = apf_m->findField(Albany::APFMeshStruct::solution_name[i]);
//  assert(apf::countComponents(sol_fld) == 1);
  apf::Field* grad_ip_fld = spr::getGradIPField(sol_flds[0], "grad_sol",
      apf_ms->cubatureDegree);
  apf::Field* size_fld = spr::getSPRSizeField(grad_ip_fld, errorBound);
//  Estimation meshFinal;

  apf::destroyField(grad_ip_fld);
  /* write the mesh with size field to file */
//  std::stringstream ss;
//  ss << "size_" << callcount << '_';
//  std::string s = ss.str();
//  apf::writeVtkFiles(s.c_str(), apf_m);
  /* create the Simmetrix adapter */
  pMSAdapt adapter = MSA_new(sim_pm, 1);
  MSA_setSizeGradation(adapter,1,0.6);
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
  pField sim_sol_flds[3];
  for (int i = 0; i <= apf_ms->num_time_deriv; ++i)
    sim_sol_flds[i] = apf::getSIMField(sol_flds[i]);
//  /* tell the adapter to transfer the solution and residual fields */
//  apf::Field* res_fld = apf_m->findField(Albany::APFMeshStruct::residual_name);
//  pField sim_sol_fld = apf::getSIMField(sol_fld);
  pField sim_res_fld = apf::getSIMField(res_fld);
  pPList sim_fld_lst = PList_new();
  for (int i = 0; i <= apf_ms->num_time_deriv; ++i)
    PList_append(sim_fld_lst, sim_sol_flds[i]);
//  PList_append(sim_fld_lst, sim_sol_fld);
  PList_append(sim_fld_lst, sim_res_fld);
  if (apf_ms->useTemperatureHack) {
    //std::cout << "Royal Mess HACK!!!!!!!!! \n";
    // transfer Temperature_old at the nodes
    apf::Field* told_fld = apf_m->findField("temp_old");
    pField sim_told_fld = apf::getSIMField(told_fld);
    PList_append(sim_fld_lst, sim_told_fld);
  }
  MSA_setMapFields(adapter, sim_fld_lst);
  //PList_delete(sim_fld_lst);

  // Constrain the top face & reset sizes
  pGModel model = M_model(sim_pm);
  GRIter regions = GM_regionIter(model);
  pGRegion gr1;
  int layer;
  while (gr1=GRIter_next(regions)) {
    if (GEN_numNativeIntAttribute(gr1,"SimLayer")==1) {
      GEN_nativeIntAttribute(gr1,"SimLayer",&layer);
      if (layer==Simmetrix_currentLayer) {
        pPList faceList = GR_faces(gr1);
        void *ent, *iter = 0;
        while(ent = PList_next(faceList,&iter)) {
          pGFace gf = static_cast<pGFace>(ent);
          if (GEN_numNativeIntAttribute(gf,"SimLayer")==1) {
            GEN_nativeIntAttribute(gf,"SimLayer",&layer);
            if (layer==Simmetrix_currentLayer) {
              MSA_setNoModification(adapter,gf);
              for(int np=0;np<PM_numParts(sim_pm);np++) {
                pVertex mv;
                VIter allVerts = M_classifiedVertexIter(PM_mesh(sim_pm,np),gf,1);
                while ( mv = VIter_next(allVerts) )
                  MSA_setVertexSize(adapter,mv,0.0003);
                VIter_delete(allVerts);
              }
              break;
            }
          }
        }
        PList_delete(faceList);
      }
    }
  }
  GRIter_delete(regions);

#ifdef SIMDEBUG
  char simname[80];
  sprintf(simname, "preadapt_%d.sms", callcount);
  PM_write(sim_pm, simname, sthreadDefault, 0);
  sprintf(simname, "preadapt_sol_%d.fld", callcount);
  Field_write(sim_sol_flds[0], simname, 0, 0, 0);
  sprintf(simname, "preadapt_res_%d.fld", callcount);
  Field_write(sim_res_fld, simname, 0, 0, 0);
#endif
//  char simname[80];
//  sprintf(simname, "before%d", callcount);
//  //Albany::debugAMPMesh(apf_m, "before");
//  Albany::debugAMPMesh(apf_m, simname);
  /* run the adapter */
  pProgress progress = Progress_new();
  //MSA_adapt(adapter, progress);
  Progress_delete(progress);
  MSA_delete(adapter);
#ifdef SIMDEBUG
  sprintf(simname, "adapted_%d.sms", callcount);
  PM_write(sim_pm, simname, sthreadDefault, 0);
  sprintf(simname, "adapted_sol_%d.fld", callcount);
  Field_write(sim_sol_flds[0], simname, 0, 0, 0);
  sprintf(simname, "adapted_res_%d.fld", callcount);
  Field_write(sim_res_fld, simname, 0, 0, 0);
#endif

  double currentTime = param_lib_->getRealValue<PHAL::AlbanyTraits::Residual>("Time");
  std::cout << "Current time = " << currentTime << "\n";
  if (currentTime >= Simmetrix_currentLayer*0.00003) {
    char meshFile[80];
    std::cout << "Adding layer " << Simmetrix_currentLayer << "\n";
    addNextLayer(sim_pm,Simmetrix_currentLayer,sim_fld_lst);
//    sprintf(meshFile, "layerMesh%d.sms", Simmetrix_currentLayer);
//    PM_write(sim_pm, meshFile, sthreadDefault, 0);
//  Albany::debugAMPMesh(apf_m, "afterLayerAdded");


    Simmetrix_currentLayer++;
  }
  PList_delete(sim_fld_lst);

  /* run APF verification on the resulting mesh */
  apf_m->verify();
  //  Albany::debugAMPMesh(apf_m, "after");
  /* update Albany structures to reflect the adapted mesh */
  sim_disc->updateMesh(should_transfer_ip_data);
  /* see the comment in Albany_APFDiscretization.cpp */
  sim_disc->initTemperatureHack();
//  sprintf(simname, "after%d", callcount);
//  Albany::debugAMPMesh(apf_m, simname);
  ++callcount;

  return true;
  
#endif
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
