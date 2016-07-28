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
/* BRD */
#include <SimModel.h>
/* BRD */
#include <apfSIM.h>
#include <spr.h>
#include <EnergyIntegral.hpp>

/* BRD */
#include "PHAL_Utilities.hpp"
extern void DM_undoSlicing(pPList regions,int layerNum, pMesh mesh);
extern void PM_localizePartiallyConnected(pParMesh);
extern void MSA_setPrebalance(pMSAdapt,int);
/* BRD */

namespace AAdapt {

SimAdapt::SimAdapt(const Teuchos::RCP<Teuchos::ParameterList>& params_,
                   const Teuchos::RCP<ParamLib>& paramLib_,
                   const Albany::StateManager& StateMgr_,
                   const Teuchos::RCP<const Teuchos_Comm>& commT_):
  AbstractAdapterT(params_, paramLib_, StateMgr_, commT_)
{
  errorBound = params_->get<double>("Error Bound", 0.1);
  /* BRD */
  Simmetrix_numLayers = -1;
  std::cout << "Pid = " << getpid() << "\n";
  sleep(30.0);
  /* BRD */
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

/* BRD */
void meshCurrentLayerOnly(pGModel model,pParMesh mesh,int currentLayer,double sliceThickness)
{
  //Sim_logOn("meshLog.lua");
  pACase mcase = MS_newMeshCase(model);
  MS_setMeshSize(mcase,GM_domain(model),1,10.0*sliceThickness,0);  // at least an order of magnitude more than slice thickness
  MS_setMeshCurv(mcase,GM_domain(model),2,0.025);
  MS_setMinCurvSize(mcase,GM_domain(model),2,0.0025);

  // Slice thickness
  // Bracket = 0.0003/0.0001 - real part but way too slow
  // sliced_cube.smd = 0.003/0.001 - best model/settings for testing
  // sliced_cube300microns.smd = 0.0003/0.0001 - realistic slices but way too slow
  // Clevis  = 0.03/0.01 
  // Use a mesh size for the current layer that is 1/3 the slice thickness
  GRIter regions = GM_regionIter(model);
  pGRegion gr;
  int layer;
  while (gr=GRIter_next(regions)) {
    if (GEN_numNativeIntAttribute(gr,"SimLayer")==1) {
      GEN_nativeIntAttribute(gr,"SimLayer",&layer);
      if (layer==currentLayer) {
        MS_setMeshSize(mcase,gr,1,sliceThickness/3.0,0);
        pPList regFaces = GR_faces(gr);
        void *fiter = 0;
        pGFace gf;
        while (gf = static_cast<pGFace>(PList_next(regFaces,&fiter))) {
          if (GEN_numNativeIntAttribute(gf,"SimLayer")==1) {
            GEN_nativeIntAttribute(gf,"SimLayer",&layer);
            if (layer==currentLayer+1) {
              MS_limitSurfaceMeshModification(mcase,gf,1);
              MS_useDiscreteGeometryMesh(mcase,gf,1);
            }
          }
        }
        PList_delete(regFaces);
      }
      else if (layer > currentLayer)
        MS_setNoMesh(mcase,gr,1);
    }
  }
  GRIter_delete(regions);
  //PM_write(mesh, "beforeSurfaceMesh.sms", sthreadDefault, 0);

  pSurfaceMesher sm = SurfaceMesher_new(mcase,mesh);
  // SurfaceMesher_setParamForDiscrete(sm, 1);
  SurfaceMesher_execute(sm,0);
  SurfaceMesher_delete(sm);
  if (currentLayer==1)
    PM_setTotalNumParts(mesh,PMU_size());
  pVolumeMesher vm  = VolumeMesher_new(mcase,mesh);
  VolumeMesher_execute(vm,0);
  VolumeMesher_delete(vm);

  MS_deleteMeshCase(mcase);
  //Sim_logOff();
}

void adaptMesh2(pGModel model,pParMesh mesh,int currentLayer,double sliceThickness,pPList flds)
{
  pACase mcase = MS_newMeshCase(model);
  MS_setMeshSize(mcase,GM_domain(model),1,100*sliceThickness,0);
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
        int layer = xyz[2]/sliceThickness;
        if (layer < currentLayer)
          MSA_setVertexSize(adaptM,v,100*sliceThickness);
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

void addNextLayer(pParMesh sim_pm,int nextLayer,int nSolFlds,pPList flds) {
  double sliceThickness;
  pGModel model = M_model(sim_pm);
  GIP_nativeDoubleAttribute(GM_part(model),"SimLayerThickness",&sliceThickness);
  
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

  PM_localizePartiallyConnected(sim_pm);
  //PM_merge(sim_pm);
  if (nextLayer>1) {
    std::cout << "Combine layer " << nextLayer-1 << "\n";
    pMesh oneMesh = PM_numParts(sim_pm) == 1 ? PM_mesh(sim_pm, 0) : 0;
    DM_undoSlicing(combinedRegions,nextLayer-1,oneMesh);
  }
  PList_clear(combinedRegions);
  std::cout << "Mesh top layer\n";
  meshCurrentLayerOnly(model,sim_pm,nextLayer,sliceThickness);
  /*
  if (nextLayer>1) {
    adaptMesh2(model,sim_pm,nextLayer,sliceThickness,0);
  }
  */

  if (flds) {
    // Add temperature and residual fields to top layer
    // Add temperature HACK fields to top layer
    std::cout << "Add field to top layer\n";
    pField sim_sol_flds[3] = {0,0,0};  // at most 3 - see calling routine
    int i;
    for (i=0;i<nSolFlds;i++) {
      sim_sol_flds[i] = static_cast<pField>(PList_item(flds,i));
    }
    pField sim_res_fld  = static_cast<pField>(PList_item(flds,i++));
    pField sim_hak_fld = 0;
    if (PList_size(flds)==i+1)
      sim_hak_fld = static_cast<pField>(PList_item(flds,i));
    pMEntitySet topLayerVerts = MEntitySet_new(PM_mesh(sim_pm,0));
    regions = GM_regionIter(model);
    pVertex mv;
    std::cout << "Collect new mesh verts\n";
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
    std::cout << "Create fields\n";
    while ( mv = reinterpret_cast<pVertex>(MESIter_next(viter)) ) {
      dg = Field_entDof(sim_sol_flds[0],mv,0);
      if (!dg) {
        PList_append(unmapped,mv);
        for(i=0;i<nSolFlds;i++)
          Field_applyEnt(sim_sol_flds[i],mv);
        Field_applyEnt(sim_res_fld,mv);
        if (sim_hak_fld)
          Field_applyEnt(sim_hak_fld,mv);
      } 
    }
    MESIter_delete(viter);
    pEntity ent;
    void *vptr;
    int c, ncs;
    int nc2 = (sim_res_fld ? Field_numComp(sim_res_fld) : 0);
    int nc3 = (sim_hak_fld ? Field_numComp(sim_hak_fld) : 0);
    void *iter = 0;
    std::cout << "Set field values\n";
    while (vptr = PList_next(unmapped,&iter)) {
      ent = reinterpret_cast<pEntity>(vptr);
      for(i=0;i<nSolFlds;i++) {
        dg = Field_entDof(sim_sol_flds[i],ent,0);
        ncs = Field_numComp(sim_sol_flds[i]);
        for (c=0; c < ncs; c++)
          DofGroup_setValue(dg,c,0,20.0);
      }
      if (sim_res_fld) {
        dg = Field_entDof(sim_res_fld,ent,0);
        for (c=0; c < nc2; c++)
          DofGroup_setValue(dg,c,0,0.0);
      }
      if (sim_hak_fld) {
        dg = Field_entDof(sim_hak_fld,ent,0);
        for (c=0; c < nc3; c++)
          DofGroup_setValue(dg,c,0,20.0);
      }
    }
    PList_delete(unmapped);
    MEntitySet_delete(PM_mesh(sim_pm,0),topLayerVerts);
    GRIter_delete(regions);
  }
  return;
}

void SimAdapt::computeLayerTimes(SGModel *model) {

  GRIter regions = GM_regionIter(model);
  pGRegion gr;
  int i, layer, maxLayer = -1;
  while (gr=GRIter_next(regions)) {
    if (GEN_numNativeIntAttribute(gr,"SimLayer")==1) {
      GEN_nativeIntAttribute(gr,"SimLayer",&layer);
      if (layer > maxLayer)
        maxLayer = layer;
    }
  }
  Simmetrix_numLayers = maxLayer+1;

  Simmetrix_layerTimes = new double[Simmetrix_numLayers];
  for(i=0;i<Simmetrix_numLayers;i++)
    Simmetrix_layerTimes[i] = 0.0;

  double ls = 85.;  // laser speed
  double tw = 0.013;   // track width
  if (GIP_numNativeDoubleAttribute(GM_part(model),"speed")==1)
    GIP_nativeDoubleAttribute(GM_part(model),"speed",&ls);
  if (GIP_numNativeDoubleAttribute(GM_part(model),"width")==1)
    GIP_nativeDoubleAttribute(GM_part(model),"width",&tw);
  std::cout << "Laser speed " << ls << "\n";
  std::cout << "Track width " << tw << "\n";

  GRIter_reset(regions);
  while (gr=GRIter_next(regions)) {
    if (GEN_numNativeIntAttribute(gr,"SimLayer")==1) {
      GEN_nativeIntAttribute(gr,"SimLayer",&layer);
      double area = 0.0;
      pPList faces = GR_faces(gr);
      pGFace gf;
      for(i=0;i<PList_size(faces);i++) {
        gf = static_cast<pGFace>(PList_item(faces,i));
        if(layer == maxLayer) {
          // top face of last layer is not tagged so
          // count faces that aren't on the boundary of the
          // previous layer.  Good enough for RoyalMess.
          pPList fregs = GF_regions(gf);
          if (PList_size(fregs)==1) {
            area += GF_area(gf,0);
          }
          PList_delete(fregs);
        }
        else {
          if (GEN_numNativeIntAttribute(gf,"SimLayer")==1) {
            int faceLayer;
            GEN_nativeIntAttribute(gf,"SimLayer",&faceLayer);
            if (faceLayer == layer+1) {
              area += GF_area(gf,0);
            }
          }
        }
      }
      PList_delete(faces);
      Simmetrix_layerTimes[layer] += area/(ls*tw);
    }
  }
  GRIter_delete(regions);

  double totalTime = Simmetrix_layerTimes[0];
  for(i=1;i < Simmetrix_numLayers; i++) {
    totalTime += Simmetrix_layerTimes[i];
    Simmetrix_layerTimes[i] = totalTime;
  }
}
/* BRD */

bool SimAdapt::adaptMesh()
{
  /* dig through all the abstrations to obtain pointers
     to the various structures needed */
  static int callcount = 0;
  /* BRD */
  static int Simmetrix_currentLayer = 0;
 /* BRD */
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
  /* BRD */
  MSA_setSizeGradation(adapter,1,0.6);  // no broomsticks allowed
  /* BRD */
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
  /* BRD */
  //PList_delete(sim_fld_lst);
  /* BRD */

  /* BRD */
  pGModel model = M_model(sim_pm);
  GRIter regions = GM_regionIter(model);
  pGRegion gr1;

  // Calculate layer times
  if (Simmetrix_numLayers < 0)
    computeLayerTimes(model);

  double sliceThickness;
  GIP_nativeDoubleAttribute(GM_part(model),"SimLayerThickness",&sliceThickness);

  // Constrain the top face & reset sizes
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
            if (layer==Simmetrix_currentLayer+1) {
              MSA_setNoModification(adapter,gf);
              for(int np=0;np<PM_numParts(sim_pm);np++) {
                pVertex mv;
                VIter allVerts = M_classifiedVertexIter(PM_mesh(sim_pm,np),gf,1);
                while ( mv = VIter_next(allVerts) )
                  MSA_setVertexSize(adapter,mv,sliceThickness/3.0);  // should be same as top layer size in meshModel
                VIter_delete(allVerts);
              }
            }
          }
        }
        PList_delete(faceList);
      }
    }
  }
  GRIter_delete(regions);
  /* BRD */

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
  /* BRD */
  MSA_setPrebalance(adapter, 0);
  /* BRD */
  //PM_write(sim_pm,"debugMesh.sms",sthreadDefault,0);
  MSA_adapt(adapter, progress);
  pPartitionOpts popts = PM_newPartitionOpts();
  PartitionOpts_setAdaptive(popts,1);
  PM_partition(sim_pm,popts,sthreadDefault,0);
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

  /* BRD */
  double currentTime = param_lib_->getRealValue<PHAL::AlbanyTraits::Residual>("Time");
  std::cout << "Current time = " << currentTime << "\n";
  if (currentTime >= Simmetrix_layerTimes[Simmetrix_currentLayer]) {
    char meshFile[80];
    std::cout << "Adding layer " << Simmetrix_currentLayer+1 << "\n";
    addNextLayer(sim_pm,Simmetrix_currentLayer+1,apf_ms->num_time_deriv+1,sim_fld_lst);
    sprintf(meshFile, "layerMesh%d.sms", Simmetrix_currentLayer+1);
    //PM_write(sim_pm, meshFile, sthreadDefault, 0);
    Simmetrix_currentLayer++;
  }
  PList_delete(sim_fld_lst);
  /* BRD */

  /* run APF verification on the resulting mesh */
  apf_m->verify();
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
