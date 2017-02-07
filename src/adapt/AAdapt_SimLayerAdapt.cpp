//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "AAdapt_SimLayerAdapt.hpp"
#include "Albany_SimDiscretization.hpp"
#include <MeshSimAdapt.h>
#include <SimPartitionedMesh.h>
#include <SimField.h>
#include <SimModel.h>
#include <apfSIM.h>
#include <spr.h>
#include <EnergyIntegral.hpp>
#include <set>
#include <PCU.h>

/* BRD */
#include "PHAL_AlbanyTraits.hpp"
extern VIter M_classificationVertexIter(pMesh, int);
extern void DM_undoSlicing(pPList regions,int layerNum, pUnstructuredMesh mesh);
/* BRD */

namespace AAdapt {

enum { ABSOLUTE = 1, RELATIVE = 2 };
enum { DONT_GRADE = 0, DO_GRADE = 1 };
enum { ONLY_CURV_TYPE = 2 };

SimLayerAdapt::SimLayerAdapt(const Teuchos::RCP<Teuchos::ParameterList>& params_,
                   const Teuchos::RCP<ParamLib>& paramLib_,
                   const Albany::StateManager& StateMgr_,
                   const Teuchos::RCP<const Teuchos_Comm>& commT_):
  AbstractAdapterT(params_, paramLib_, StateMgr_, commT_),
  out(Teuchos::VerboseObjectBase::getDefaultOStream())
{
  // get inititial temperature for new added layer
  initTempNewLayer = params_->get<double>("Uniform Temperature New Layer", 20.0);

  // Tell user that Uniform temperature is in effect
  *out << "***********************" << std::endl;
  *out << "Uniform Temperature New Layer = " << initTempNewLayer << std::endl;
  *out << "***********************" << std::endl;

  /* BRD */
  Simmetrix_numLayers = -1;
  Simmetrix_currentLayer = 0;
  Simmetrix_model = 0;
  /* BRD */
}

bool SimLayerAdapt::queryAdaptationCriteria(int iteration)
{
  /* BRD */
  if (!Simmetrix_model) {
    Teuchos::RCP<Albany::AbstractDiscretization> disc =
      state_mgr_.getDiscretization();
    Teuchos::RCP<Albany::SimDiscretization> sim_disc =
      Teuchos::rcp_dynamic_cast<Albany::SimDiscretization>(disc);
    Teuchos::RCP<Albany::APFMeshStruct> apf_ms =
      sim_disc->getAPFMeshStruct();
    apf::Mesh* apf_m = apf_ms->getMesh();
    apf::MeshSIM* apf_msim = dynamic_cast<apf::MeshSIM*>(apf_m);
    pParMesh sim_pm = apf_msim->getMesh();
    Simmetrix_model =  M_model(sim_pm);
    computeLayerTimes();
  }
  double currentTime = param_lib_->getRealValue<PHAL::AlbanyTraits::Residual>("Time");
  if (currentTime >= Simmetrix_layerTimes[Simmetrix_currentLayer]) {
    *out << "Need to remesh and add next layer\n";
    return true;
  }
  /* BRD */
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
void meshCurrentLayerOnly(pGModel model,pParMesh mesh,int currentLayer,double layerSize)
{
  pACase mcase = MS_newMeshCase(model);
  MS_setMeshSize(mcase,GM_domain(model), RELATIVE, 1.0, NULL);
  MS_setMeshCurv(mcase,GM_domain(model), ONLY_CURV_TYPE, 0.025);
  MS_setMinCurvSize(mcase,GM_domain(model), ONLY_CURV_TYPE, 0.0025);
  MS_setSurfaceShapeMetric(mcase, GM_domain(model),ShapeMetricType_AspectRatio, 25);
  MS_setVolumeShapeMetric(mcase, GM_domain(model), ShapeMetricType_AspectRatio, 25);

  GRIter regions = GM_regionIter(model);
  pGRegion gr;
  int layer;
  while (gr=GRIter_next(regions)) {
    if (GEN_numNativeIntAttribute(gr,"SimLayer")==1) {
      GEN_nativeIntAttribute(gr,"SimLayer",&layer);
      if (layer==currentLayer) {
        MS_setMeshSize(mcase,gr, ABSOLUTE, layerSize, NULL);
        pPList regFaces = GR_faces(gr);
        void *fiter = 0;
        pGFace gf;
        while (gf = static_cast<pGFace>(PList_next(regFaces,&fiter))) {
          if (GEN_numNativeIntAttribute(gf,"SimLayer")==1) {
            GEN_nativeIntAttribute(gf,"SimLayer",&layer);
            /*
            if (layer==currentLayer+1) {
              MS_limitSurfaceMeshModification(mcase,gf,1);
              MS_useDiscreteGeometryMesh(mcase,gf,1);
            }
            */
          }
        }
        PList_delete(regFaces);
      }
      else if (layer > currentLayer)
        MS_setNoMesh(mcase,gr,1);
    }
  }
  GRIter_delete(regions);
  
  double t1 = PCU_Time();
  pSurfaceMesher sm = SurfaceMesher_new(mcase,mesh);
  SurfaceMesher_execute(sm,0);
  SurfaceMesher_delete(sm);
  double t2 = PCU_Time();
  if (!PCU_Comm_Self())
    fprintf(stderr,"meshCurrentLayerOnly surface mesh %f seconds\n",t2-t1);
  double t3 = PCU_Time();
  pVolumeMesher vm  = VolumeMesher_new(mcase,mesh);
  VolumeMesher_setEnforceSize(vm, 1);
  VolumeMesher_execute(vm,0);
  VolumeMesher_delete(vm);
  double t4 = PCU_Time();
  if (!PCU_Comm_Self())
    fprintf(stderr,"meshCurrentLayerOnly volume mesh %f seconds\n",t4-t3);

  MS_deleteMeshCase(mcase);
  printf("after mesh CurrentLayerOnly: %d tets on cpu %d\n",M_numRegions(PM_mesh(mesh,0)),PMU_rank());
}

void addNextLayer2(
    pParMesh sim_pm,
    double layerSize,
    int nextLayer,
    double initTemp,
    double initDisp,
    std::vector<pField> const& soln_flds,
    std::vector<pField> const& res_flds,
    pField old_temp_fld) {

  /* ostream only on proc 0 */
  Teuchos::RCP<Teuchos::FancyOStream> out = Teuchos::VerboseObjectBase::getDefaultOStream();

  /* split up the solution / residual fields so that we know exactly what
     fields we are working with */
  pField temp_fld = 0;
  pField disp_fld = 0;
  pField temp_res_fld = 0;
  pField disp_res_fld = 0;

  int num_soln_flds = soln_flds.size();
  assert((num_soln_flds == 1) || (num_soln_flds == 2));

  if (soln_flds.size() == 1) {
    temp_fld = soln_flds[0];
    temp_res_fld = soln_flds[0];
    assert(Field_numComp(temp_fld) == 1);
    assert(Field_numComp(temp_res_fld) == 1);
  }

  else if (soln_flds.size() == 2) {
    disp_fld = soln_flds[0];
    disp_res_fld = res_flds[0];
    temp_fld = soln_flds[1];
    temp_res_fld = res_flds[1];
    assert(Field_numComp(disp_fld) == 3);
    assert(Field_numComp(disp_res_fld) == 3);
    assert(Field_numComp(temp_fld) == 1);
    assert(Field_numComp(temp_res_fld) == 1);
  }

  /* Collect the layer 0 regions */
  pGModel model = M_model(sim_pm);
  GRIter regions = GM_regionIter(model);
  pGRegion gr1;
  int layer, maxLayer = -1;
  pPList combinedRegions = PList_new();
  std::set<pGRegion> layGrs;
  while (gr1=GRIter_next(regions)) {
    if (GEN_numNativeIntAttribute(gr1,"SimLayer")==1) {
      GEN_nativeIntAttribute(gr1,"SimLayer",&layer);
      if (layer==0)
        PList_appUnique(combinedRegions,gr1);
      if (layer > maxLayer)
        maxLayer = layer;
      if (layer == nextLayer)
        layGrs.insert(gr1);
    }
  }
  GRIter_delete(regions);
  if ( nextLayer > maxLayer )
    return;

  /* perform migration */
  double t1 = PCU_Time();
  pMesh mesh = PM_mesh(sim_pm, 0);
  pMigrator mig = Migrator_new(sim_pm, 0);
  Migrator_reset(mig, 3);
  std::set<pRegion> doneRs;
  int rank = PMU_rank();
  if (rank != 0) {
    VIter vi = M_classificationVertexIter(mesh, 3);
    while (pVertex v = VIter_next(vi)) {
      pGEntity gent = EN_whatIn(v);
      for (std::set<pGRegion>::const_iterator layGrIt = layGrs.begin();
           layGrIt != layGrs.end(); ++layGrIt) {
        pGRegion layGr = *layGrIt;
        if (GEN_inClosure(layGr, gent)) {
          pPList vrs = V_regions(v);
          int i, nvrs = PList_size(vrs);
          for (i=0; i < nvrs; i++) {
            pRegion r = static_cast<pRegion>(PList_item(vrs, i));
            if (doneRs.find(r) == doneRs.end()) {
              Migrator_add(mig, r, 0, rank);
              doneRs.insert(r);
            }
          }
          PList_delete(vrs);
          break;
        }
      }
    }
    VIter_delete(vi);
  }
  Migrator_run(mig, 0);
  Migrator_delete(mig);
  double t2 = PCU_Time();
  if (!PCU_Comm_Self())
    fprintf(stderr, "addNextLayer(): migration %f seconds\n", t2 - t1);

  /* undo slicing if needed */
  if (nextLayer>1) {
    *out << "Combine layer " << nextLayer-1 << "\n";
    double t3 = PCU_Time();
    DM_undoSlicing(combinedRegions,nextLayer-1,sim_pm);
    double t4 = PCU_Time();
    if (!PCU_Comm_Self())
      fprintf(stderr, "addNextLayer(): DM_undoSlicing %f seconds\n", t4 - t3);
  }
  PList_clear(combinedRegions);

  /* mesh the top layer only */
  *out << "Mesh top layer\n";
  double t5 = PCU_Time();
  meshCurrentLayerOnly(model,sim_pm,nextLayer,layerSize);
  double t6 = PCU_Time();
  if (!PCU_Comm_Self())
    fprintf(stderr,"addNextLayer(): meshCurrentLayerOnly %f seconds\n",t6-t5);

  /* initialize field values on the top layer */
  double t7 = PCU_Time();
  {
    *out << "Add field to top layer\n";

    /* collect the top layer vertices */
    pMEntitySet topLayerVerts = MEntitySet_new(PM_mesh(sim_pm,0));
    regions = GM_regionIter(model);
    pVertex mv;
    *out << "Collect new mesh verts\n";
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

    /* make sure mesh entities know about the fields ? */
    pPList unmapped = PList_new();
    pDofGroup dg;
    MESIter viter = MESIter_iter(topLayerVerts);
    while ( mv = reinterpret_cast<pVertex>(MESIter_next(viter)) ) {
      dg = Field_entDof(temp_fld, mv, 0);
      if (!dg) {
        PList_append(unmapped, mv);
        if (temp_fld) Field_applyEnt(temp_fld, mv);
        if (disp_fld) Field_applyEnt(disp_fld, mv);
        if (temp_res_fld) Field_applyEnt(temp_res_fld, mv);
        if (disp_res_fld) Field_applyEnt(disp_res_fld, mv);
        if (old_temp_fld) Field_applyEnt(old_temp_fld, mv);
      }
    }
    MESIter_delete(viter);

    /* set the field values to the user-specified initial value */
    pEntity ent;
    void *vptr;
    void *iter = 0;
    while (vptr = PList_next(unmapped,&iter)) {
      ent = reinterpret_cast<pEntity>(vptr);
      if (temp_fld) {
        dg = Field_entDof(temp_fld, ent, 0);
        DofGroup_setValue(dg, 0, 0, initTemp);
      }
      if (old_temp_fld) {
        dg = Field_entDof(old_temp_fld, ent, 0);
        DofGroup_setValue(dg, 0, 0, initTemp);
      }
      if (temp_res_fld) {
        dg = Field_entDof(temp_res_fld, ent, 0);
        DofGroup_setValue(dg, 0, 0, 0.0);
      }
      if (disp_fld) {
        dg = Field_entDof(disp_fld, ent, 0);
        for (int c=0; c < 3; ++c)
          DofGroup_setValue(dg, c, 0, initDisp);
      }
      if (disp_res_fld) {
        dg = Field_entDof(disp_res_fld, ent, 0);
        for (int c=0; c < 3; ++c)
          DofGroup_setValue(dg, c, 0, 0.0);
      }
    }
    PList_delete(unmapped);
    MEntitySet_delete(PM_mesh(sim_pm,0),topLayerVerts);
    GRIter_delete(regions);
  }
  double t8 = PCU_Time();
  if (!PCU_Comm_Self())
    fprintf(stderr,"addNextLayer(): dealing with field for new layer %f seconds\n",t8-t7);

  return;
}

void SimLayerAdapt::computeLayerTimes() {

  GRIter regions = GM_regionIter(Simmetrix_model);
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

  double ls = 85.0; // 0.070995;  // laser speed
  double tw = 0.013; // 0.00013;   // track width
  if (GIP_numNativeDoubleAttribute(GM_part(Simmetrix_model),"speed")==1)
    GIP_nativeDoubleAttribute(GM_part(Simmetrix_model),"speed",&ls);
  if (GIP_numNativeDoubleAttribute(GM_part(Simmetrix_model),"width")==1)
    GIP_nativeDoubleAttribute(GM_part(Simmetrix_model),"width",&tw);
  *out << "Laser speed " << ls << "\n";
  *out << "Track width " << tw << "\n";

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
  if (adapt_params_->isParameter("First Layer Time")) {
    Simmetrix_layerTimes[0] = adapt_params_->get<double>("First Layer Time", 0.0);
  }
}
/* BRD */

bool SimLayerAdapt::adaptMesh()
{
  if (PMU_rank() == 0)
    fprintf(stderr,"adaptMesh(): coming in %f, cpu = %d\n", PCU_Time(),PMU_rank());

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
  printf("coming in: %d tets on cpu %d\n",M_numRegions(PM_mesh(sim_pm,0)),PMU_rank());

  /* ensure that users don't expect Simmetrix to transfer IP state */
  bool should_transfer_ip_data = adapt_params_->get<bool>("Transfer IP Data", false);
  assert(!should_transfer_ip_data);

  /* grab the solution / residual fields from the discretization.
     here we assume that the apf_ms->num_time_deriv = 0. */
  Albany::SolutionLayout soln_layout = sim_disc->getSolutionLayout();
  std::vector<apf::Field*> soln_fields;
  std::vector<apf::Field*> res_fields;
  Teuchos::Array<std::string> soln_field_names = soln_layout.getDerivNames(0);
  Teuchos::Array<std::string> res_names = sim_disc->getResNames();
  int num_soln_fields = soln_layout.getNumSolFields();
  assert( soln_field_names.size() == num_soln_fields );
  assert( res_names.size() == num_soln_fields );
  assert( num_soln_fields > 0 );
  for (int i=0; i < num_soln_fields; ++i) {
    soln_fields.push_back(apf_m->findField(soln_field_names[i].c_str()));
    res_fields.push_back(apf_m->findField(res_names[i].c_str()));
    assert( soln_fields[i] );
    assert( res_fields[i] );
  }

  /* compute the size field via SPR error estimation on the gradient
     of the chosen solution field */
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

  /* get some layer addition information */
  double t4 = PCU_Time();
  double sliceThickness;
  GIP_nativeDoubleAttribute(GM_part(Simmetrix_model),"SimLayerThickness",&sliceThickness);

  // Slice thickness
  // Bracket = 0.0003/0.0001 - real part but way too slow
  // sliced_cube.smd = 0.003/0.001 - best model/settings for testing
  // sliced_cube300microns.smd = 0.0003/0.0001 - realistic slices but way too slow
  // Clevis  = 0.03/0.01
  // Use a mesh size for the current layer that is 1/3 the slice thickness
  double layerSize = adapt_params_->get<double>("Layer Mesh Size", sliceThickness / 3.0);

  double max_size = adapt_params_->get<double>("Max Size", 1e10);
  double min_size = adapt_params_->get<double>("Min Size", 1e-2);
  double gradation = adapt_params_->get<double>("Gradation", 0.3);
  assert(min_size <= max_size);

  bool should_debug = adapt_params_->get<bool>("Debug", false);

  /* create the Simmetrix adapter */
  pACase mcase = MS_newMeshCase(Simmetrix_model);
  pModelItem domain = GM_domain(Simmetrix_model);
  MS_setMeshCurv(mcase,domain, ONLY_CURV_TYPE, 0.025);
  MS_setMinCurvSize(mcase,domain, ONLY_CURV_TYPE, 0.0025);
  MS_setMeshSize(mcase,domain, RELATIVE, 1.0, NULL);
  pMSAdapt adapter = MSA_createFromCase(mcase,sim_pm);
  MSA_setSizeGradation(adapter, DO_GRADE, gradation);  // no broomsticks allowed

  /* BRD */
  /* copy the size field from APF to the Simmetrix adapter */
  apf::MeshEntity* v;
  apf::MeshIterator* it = apf_m->begin(0);
  while ((v = apf_m->iterate(it))) {
    double size = apf::getScalar(size_fld, v, 0);
    size = std::min(max_size, size);
    size = std::max(min_size, size);
    MSA_setVertexSize(adapter, (pVertex) v, size);
    apf::setScalar(size_fld, v, 0, size);
  }
  apf_m->end(it);

  /* tell the Simmetrix adapter to transfer the soln/residual fields &
     old temperature field (if specified) */
  std::vector<pField> sim_soln_fields;
  std::vector<pField> sim_res_fields;
  pField sim_told_field = 0;
  pPList sim_fld_lst = PList_new();
  for (int i=0; i < num_soln_fields; ++i) {
    sim_soln_fields.push_back(apf::getSIMField(soln_fields[i]));
    sim_res_fields.push_back(apf::getSIMField(res_fields[i]));
    PList_append(sim_fld_lst, sim_soln_fields[i]);
    PList_append(sim_fld_lst, sim_res_fields[i]);
  }
  if (apf_ms->useTemperatureHack) {
    apf::Field* told_field = apf_m->findField("temp_old");
    sim_told_field = apf::getSIMField(told_field);
    PList_append(sim_fld_lst, sim_told_field);
  }
  MSA_setMapFields(adapter, sim_fld_lst);

  /* BRD */
  GRIter regions = GM_regionIter(Simmetrix_model);
  pGRegion gr1;

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
                while ( mv = VIter_next(allVerts) ) {
                  MSA_setVertexSize(adapter,mv,layerSize);  // should be same as top layer size in meshModel
                  apf::setScalar(size_fld, reinterpret_cast<apf::MeshEntity*>(mv), 0, layerSize);
                }
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

  if (should_debug) {
    std::stringstream ss;
    ss << "preadapt_" << callcount;
    std::string s = ss.str();
    apf::writeVtkFiles(s.c_str(), apf_m);
  }

  apf::destroyField(size_fld);
  double t5 = PCU_Time();
  if (!PCU_Comm_Self())
    fprintf(stderr,"adaptMesh(): preparing mesh adapt in %f seconds\n",t5-t4);

  double t5b = PCU_Time();

  /* run the adapter */
  pProgress progress = Progress_new();
  MSA_adapt(adapter, progress);
  Progress_delete(progress);
  MSA_delete(adapter);
  MS_deleteMeshCase(mcase);

  double t5bb = PCU_Time();
  if (!PCU_Comm_Self())
    fprintf(stderr,"adaptMesh(): mesh adapt in %f seconds\n",t5bb-t5b);
    
  if (should_debug) {
    std::stringstream ss;
    ss << "postadapt_" << callcount;
    std::string s = ss.str();
    apf::writeVtkFiles(s.c_str(), apf_m);
  }

  /* BRD */
  double currentTime = param_lib_->getRealValue<PHAL::AlbanyTraits::Residual>("Time");
  if (currentTime >= Simmetrix_layerTimes[Simmetrix_currentLayer]) {
    *out << "Adding layer " << Simmetrix_currentLayer+1 << "\n";
    addNextLayer2(sim_pm, layerSize, Simmetrix_currentLayer+1,  initTempNewLayer, 0,
        sim_soln_fields, sim_res_fields, sim_told_field);
    if (should_debug) {
      char meshFile[80];
      sprintf(meshFile, "layerMesh%d.sms", Simmetrix_currentLayer + 1);
      PM_write(sim_pm, meshFile, 0);
    }
    Simmetrix_currentLayer++;
  }
  PList_delete(sim_fld_lst);
  /* BRD */

  if (should_debug) {
    std::stringstream ss;
    ss << "postlayer_" << callcount;
    std::string s = ss.str();
    apf::writeVtkFiles(s.c_str(), apf_m);
  }

  /* rebuild the ALbany data structures needed for the analysis */
  double t8 = PCU_Time();
  apf_m->verify();
  sim_disc->updateMesh(should_transfer_ip_data, param_lib_);
  sim_disc->initTemperatureHack();
  double t9 = PCU_Time();
  if (!PCU_Comm_Self())
    fprintf(stderr,"adaptMesh(): finalize %f seconds\n",t9-t8);

  /* print out some finalization details */
  if (PMU_rank() == 0)
    fprintf(stderr,"adaptMesh(): going out in %f on cpu: %d\n", PCU_Time(),PMU_rank());
  printf("leaving: %d tets on cpu %d\n",M_numRegions(PM_mesh(sim_pm,0)),PMU_rank());

  ++callcount;
  return true;
}


Teuchos::RCP<const Teuchos::ParameterList> SimLayerAdapt::getValidAdapterParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericAdapterParams("ValidSimLayerAdaptParams");
  validPL->set<bool>("Transfer IP Data", false, "Turn on solution transfer of integration point data");
  validPL->set<double>("Error Bound", 0.1, "Max relative error for error-based adaptivity");
  validPL->set<double>("Max Size", 1e10, "Maximum allowed edge length (size field)");
  validPL->set<double>("Min Size", 1e-2, "Minimum allowed edge length (size field)");
  validPL->set<double>("Layer Mesh Size", 1e-2, "Mesh size to use for top layer (default thickness/3)");
  validPL->set<double>("Gradation", 0.3, "Mesh size gradation parameter");
  validPL->set<bool>("Debug", false, "Print debug VTK files");
  validPL->set<bool>("Add Layer", true, "Turn on/off adding layer");
  validPL->set<double>("Uniform Temperature New Layer", 20.0, "Uniform Layer Temperature");
  validPL->set<double>("First Layer Time", 0.0, "Overrides time to place first layer");
  validPL->set<bool>("Equilibrate", false, "Should equilibration be turned on after adaptation");
  validPL->set<std::string>("Remesh Strategy", "", "Strategy for when to adapt");
  validPL->set<int>("Remesh Every N Step Number", 1, "Remesh every Nth load/time step");
  validPL->set<int>("SPR Solution Index", 0, "Solution field index for SPR to operate on");
  validPL->set<long int>("Target Element Count", 1000, "Desired number of elements for error-based adaptivity");
  return validPL;
}

}
