#include "CTM_Adapter.hpp"

#include <set>
#include <PCU.h>
#include <spr.h>
#include <apfSIM.h>
#include <parma.h>
#include <SimField.h>
#include <MeshSimAdapt.h>
#include <Albany_Utils.hpp>
#include <Albany_StateManager.hpp>
#include <Albany_SimDiscretization.hpp>
#include <Teuchos_ParameterList.hpp>

extern VIter M_classificationVertexIter(pMesh, int);
extern void DM_undoSlicing(pPList regions,int layerNum, pUnstructuredMesh mesh);

namespace CTM {

using Teuchos::rcp_dynamic_cast;

static RCP<ParameterList> get_valid_params() {
  auto p = rcp(new ParameterList);
  p->set<double>("Error Bound", 0.1, "Max relative error for adaptivity");
  p->set<double>("Layer Mesh Size", 1e-2, "Mesh size to use for top layer");
  p->set<double>("Max Size", 1e10, "Maximum allowed edge length (size field)");
  p->set<double>("Min Size", 1e-2, "Mininum allowed edge length (size field)");
  p->set<double>("Gradation", 0.3, "Mesh size gradation parameter");
  p->set<bool>("Debug", false, "Print debug VTK files");
  p->set<double>("Uniform Temperature New Layer", 20.0, "Uniform layer temperature");
  p->set<std::string>("SPR Solution Field", "Temp", "Field name for SPR to operate on");
  p->set<long int>("Target Element Count", 1000, "Desired # of elements for spr adaptivity");
  return p;
}

static void validate_params(RCP<const ParameterList> p) {
  ALBANY_ALWAYS_ASSERT(p->isType<double>("Uniform Temperature New Layer"));
  p->validateParameters(*get_valid_params(), 0);
}

Adapter::Adapter(
    RCP<ParameterList> p,
    RCP<ParamLib> lib,
    RCP<Albany::StateManager> tsm,
    RCP<Albany::StateManager> msm) {

  // initializations
  params = p;
  validate_params(params);
  param_lib = lib;
  t_state_mgr = tsm;
  m_state_mgr = msm;
  t_disc = t_state_mgr->getDiscretization();
  m_disc = m_state_mgr->getDiscretization();
  out = Teuchos::VerboseObjectBase::getDefaultOStream();

  // model initialization
  auto sim_disc = rcp_dynamic_cast<Albany::SimDiscretization>(m_disc);
  auto apf_ms = sim_disc->getAPFMeshStruct();
  apf_mesh = apf_ms->getMesh();
  auto apf_sim_mesh = dynamic_cast<apf::MeshSIM*>(apf_mesh);
  sim_mesh = apf_sim_mesh->getMesh();
  sim_model = M_model(sim_mesh);

  // setup input parameter info
  setup_params();

  // compute layer information + print it
  *out << std::endl;
  *out << "*********************" << std::endl;
  *out << "LAYER ADDING ENABLED " << std::endl;
  *out << "*********************" << std::endl;
  compute_layer_info();
}

void Adapter::setup_params() {

  // initialize spr size field params
  use_error = false;
  use_target_elems = false;
  error_bound = 0.0;
  target_elems = 0;

  // grab params from parameter list as needed
  double slice_thickness;
  GIP_nativeDoubleAttribute(
      GM_part(sim_model), "SimLayerThickness", &slice_thickness);
  layer_size = params->get<double>("Layer Mesh Size", (slice_thickness/3.0));
  min_size = params->get<double>("Min Size", 1e-2);
  max_size = params->get<double>("Max Size", 1e10);
  gradation = params->get<double>("Gradation", 0.3);
  debug = params->get<bool>("Debug", false);
  if (params->isType<long int>("Target Element Count")) {
    use_target_elems = true;
    target_elems = params->get<long int>("Target Element Count", 1000);
  }
  else if (params->isType<double>("Error Bound")) {
    use_error = true;
    error_bound = params->get<double>("Error Bound", 0.1);
  }
  else
    ALBANY_ALWAYS_ASSERT_VERBOSE(false, "invalid spr logic\n");

}

static int compute_num_layers(SGModel* model) {
  GRIter regions = GM_regionIter(model);
  pGRegion gr;
  int layer;
  int max_layer = -1;
  while (gr = GRIter_next(regions)) {
    if (GEN_numNativeIntAttribute(gr, "SimLayer") == 1) {
      GEN_nativeIntAttribute(gr, "SimLayer", &layer);
      if (layer > max_layer)
        max_layer = layer;
    }
  }
  return max_layer + 1;
}

static void compute_laser_info(SGModel* model, double& ls, double& tw) {
  auto part = GM_part(model);
  if (GIP_numNativeDoubleAttribute(part, "speed") == 1)
    GIP_nativeDoubleAttribute(part, "speed", &ls);
  if (GIP_numNativeDoubleAttribute(part, "width") == 1)
    GIP_nativeDoubleAttribute(part, "width", &tw);
}

static void compute_layer_times(
    SGModel* model, std::vector<double>& times, double ls, double tw) {

  int max_layer = times.size() - 1;
  int layer;
  pGRegion gr;
  GRIter regions = GM_regionIter(model);
  while (gr = GRIter_next(regions)) {
    if (GEN_numNativeIntAttribute(gr, "SimLayer") == 1) {
      GEN_nativeIntAttribute(gr, "SimLayer", &layer);
      double area = 0.0;
      pPList faces = GR_faces(gr);
      pGFace gf;
      for (int i = 0; i < PList_size(faces); ++i) {
        gf = static_cast<pGFace>(PList_item(faces, i));
        if (layer == max_layer) {
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
          if (GEN_numNativeIntAttribute(gf, "SimLayer")==1) {
            int faceLayer;
            GEN_nativeIntAttribute(gf, "SimLayer", &faceLayer);
            if (faceLayer == layer+1) {
              area += GF_area(gf, 0);
            }
          }
        }
      }
      PList_delete(faces);
      times[layer] += area / (ls * tw);
    }
  }
  GRIter_delete(regions);

  double total_time = times[0];
  for (int i = 1; i < times.size(); ++i) {
    total_time += times[i];
    times[i] = total_time;
  }
}

void Adapter::compute_layer_info() {

  // compute some basic laser + layer info
  new_layer_temp = params->get<double>("Uniform Temperature New Layer", 20.0);
  double laser_speed = 85.0;
  double track_width = 0.013;
  compute_laser_info(sim_model, laser_speed, track_width);
  num_layers = compute_num_layers(sim_model);
  layer_times.resize(num_layers);
  compute_layer_times(sim_model, layer_times, laser_speed, track_width);
  current_layer = 0;

  // print out some useful information for users
  *out << " > new layer uniform temperature: " << new_layer_temp << std::endl;
  *out << " > number of layers: " << num_layers << std::endl;
  *out << " > laser speed: " << laser_speed << std::endl;
  *out << " > track width: " << track_width << std::endl;
}

bool Adapter::should_adapt(const double t_current) {
  if (t_current >= layer_times[current_layer])
    return true;
  else
    return false;
}

static apf::Field* get_temp_field(
    RCP<Albany::AbstractDiscretization> disc, apf::Mesh* mesh) {
  auto sim_disc = rcp_dynamic_cast<Albany::SimDiscretization>(disc);
  auto t_names = sim_disc->getSolutionLayout().getDerivNames(0);
  ALBANY_ALWAYS_ASSERT(t_names.size() == 1);
  auto t_name = t_names[0];
  apf::Field* f = mesh->findField(t_name.c_str());
  ALBANY_ALWAYS_ASSERT(f);
  return f;
}

static apf::Field* get_mech_field(
    RCP<Albany::AbstractDiscretization> disc, apf::Mesh* mesh) {
  auto sim_disc = rcp_dynamic_cast<Albany::SimDiscretization>(disc);
  auto m_names = sim_disc->getSolutionLayout().getDerivNames(0);
  ALBANY_ALWAYS_ASSERT(m_names.size() == 1);
  auto m_name = m_names[0];
  apf::Field* f = mesh->findField(m_name.c_str());
  ALBANY_ALWAYS_ASSERT(f);
  return f;
}

static apf::Field* compute_error_size(
    std::string const& fname, apf::Field* tf, apf::Field* mf,
    bool use_error, bool use_target, double eb, long te) {

  // get the field to operate on
  apf::Field* f;
  if (fname == apf::getName(tf)) f = tf;
  else if (fname == apf::getName(mf)) f = mf;
  else ALBANY_ALWAYS_ASSERT_VERBOSE(false, "unknown 'SPR Solution Field'");

  // get the gradient of the field
  // note we assume q_degree == 1, might want to check that somewhere
  auto gf = spr::getGradIPField(f, "grad_sol", 1);

  // get the requested spr size field
  apf::Field* sz;
  if (use_target)
    sz = spr::getTargetSPRSizeField(gf, te);
  else if (use_error)
    sz = spr::getSPRSizeField(gf, eb);

  // clean up
  apf::destroyField(gf);

  return sz;
}

static void copy_size_field(
    apf::Mesh* m, pMSAdapt adapter, apf::Field* szf,
    double min_size, double max_size) {
  apf::MeshEntity* v;
  auto vertices = m->begin(0);
  while ((v = m->iterate(vertices))) {
    double size = apf::getScalar(szf, v, 0);
    size = std::min(max_size, size);
    size = std::max(min_size, size);
    MSA_setVertexSize(adapter, (pVertex) v, size);
    apf::setScalar(szf, v, 0, size);
  }
  m->end(vertices);
}

static pPList set_transfer_fields(
    pMSAdapt adapter,  pField sim_tf, pField sim_mf) {
  pPList sim_list = PList_new();
  PList_append(sim_list, sim_tf);
  PList_append(sim_list, sim_mf);
  MSA_setMapFields(adapter, sim_list);
  return sim_list;
}

static void constrain_top(
    pMSAdapt adapter, SGModel* model, pParMesh mesh,
    apf::Field* szf, int current_layer, double layer_sz) {
  int layer;
  pGRegion gr1;
  GRIter regions = GM_regionIter(model);
  while (gr1 = GRIter_next(regions)) {
    if (GEN_numNativeIntAttribute(gr1, "SimLayer") == 1) {
      GEN_nativeIntAttribute(gr1, "SimLayer", &layer);
      if (layer == current_layer) {
        pPList face_list = GR_faces(gr1);
        void* ent;
        void* iter = 0;
        while (ent = PList_next(face_list, &iter)) {
          pGFace gf = static_cast<pGFace>(ent);
          if (GEN_numNativeIntAttribute(gf, "SimLayer") == 1) {
            GEN_nativeIntAttribute(gf, "SimLayer", &layer);
            if (layer == current_layer + 1) {
              MSA_setNoModification(adapter, gf);
              for (int np = 0; np < PM_numParts(mesh); ++np) {
                pVertex mv;
                VIter all_verts = M_classifiedVertexIter(PM_mesh(mesh, np), gf, 1);
                while (mv = VIter_next(all_verts) ) {
                  MSA_setVertexSize(adapter, mv, layer_sz);
                  apf::setScalar(szf, reinterpret_cast<apf::MeshEntity*>(mv), 0, layer_sz);
                }
                VIter_delete(all_verts);
              }
            }
          }
        }
        PList_delete(face_list);
      }
    }
  }
  GRIter_delete(regions);
}

enum { ABSOLUTE = 1, RELATIVE = 2 };
enum { DONT_GRADE = 0, DO_GRADE = 1 };
enum { ONLY_CURV_TYPE = 2 };

static void mesh_current_layer(
    pGModel model, pParMesh mesh, int current_layer, double size,
    RCP<Teuchos::FancyOStream> out) {

  // setup the mesher
  pACase mcase = MS_newMeshCase(model);
  MS_setMeshSize(mcase,GM_domain(model), RELATIVE, 1.0, NULL);
  MS_setMeshCurv(mcase,GM_domain(model), ONLY_CURV_TYPE, 0.025);
  MS_setMinCurvSize(mcase,GM_domain(model), ONLY_CURV_TYPE, 0.0025);
  MS_setSurfaceShapeMetric(mcase, GM_domain(model),ShapeMetricType_AspectRatio, 25);
  MS_setVolumeShapeMetric(mcase, GM_domain(model), ShapeMetricType_AspectRatio, 25);

  // set region to be meshed based on attributes
  GRIter regions = GM_regionIter(model);
  pGRegion gr;
  int layer;
  while (gr=GRIter_next(regions)) {
    if (GEN_numNativeIntAttribute(gr, "SimLayer") == 1) {
      GEN_nativeIntAttribute(gr, "SimLayer", &layer);
      if (layer == current_layer) {
        MS_setMeshSize(mcase, gr, ABSOLUTE, size, NULL);
        pPList regFaces = GR_faces(gr);
        void *fiter = 0;
        pGFace gf;
        while (gf = static_cast<pGFace>(PList_next(regFaces, &fiter))) {
          if (GEN_numNativeIntAttribute(gf,"SimLayer") == 1)
            GEN_nativeIntAttribute(gf,"SimLayer", &layer);
        }
        PList_delete(regFaces);
      }
      else if (layer > current_layer)
        MS_setNoMesh(mcase,gr,1);
    }
  }
  GRIter_delete(regions);

  // mesh the surface
  double t0 = PCU_Time();
  pSurfaceMesher sm = SurfaceMesher_new(mcase,mesh);
  SurfaceMesher_execute(sm,0);
  SurfaceMesher_delete(sm);
  double t1 = PCU_Time();
  *out << "adapt(): current layer surface meshed in " << t1-t0 << " seconds\n";

  // mesh the volume
  double t2 = PCU_Time();
  pVolumeMesher vm  = VolumeMesher_new(mcase,mesh);
  VolumeMesher_setEnforceSize(vm, 1);
  VolumeMesher_execute(vm,0);
  VolumeMesher_delete(vm);
  MS_deleteMeshCase(mcase);
  double t3 = PCU_Time();
  *out << "adapt(): current layer volume meshed in " << t3-t2 << " seconds\n";

  // print a final statement
  printf("adapt(): after meshing current layer only: %d tets on cpu %d\n",
      M_numRegions(PM_mesh(mesh,0)),PMU_rank());
}

static void add_next_layer(
    pParMesh sim_pm, double size, int next_layer,
    double init_temp, double init_disp,
    pField temp, pField disp, RCP<Teuchos::FancyOStream> out) {

 // collect the layer 0 regions
  auto model = M_model(sim_pm);
  auto regions = GM_regionIter(model);
  pGRegion gr1;
  int layer;
  int max_layer = -1;
  auto combined_regions = PList_new();
  std::set<pGRegion> lay_grs;
  while (gr1 = GRIter_next(regions)) {
    if (GEN_numNativeIntAttribute(gr1, "SimLayer") == 1) {
      GEN_nativeIntAttribute(gr1, "SimLayer", &layer);
      if (layer == 0)
        PList_appUnique(combined_regions, gr1);
      if (layer > max_layer)
        max_layer = layer;
      if (layer == next_layer)
        lay_grs.insert(gr1);
    }
  }
  GRIter_delete(regions);

  // bail out if no more layers to add
  if (next_layer > max_layer)
    return;

  // perform migration
  double t0 = PCU_Time();
  pMesh mesh = PM_mesh(sim_pm, 0);
  pMigrator mig = Migrator_new(sim_pm, 0);
  Migrator_reset(mig, 3);
  std::set<pRegion> doneRs;
  int rank = PMU_rank();
  if (rank != 0) {
    VIter vi = M_classificationVertexIter(mesh, 3);
    while (pVertex v = VIter_next(vi)) {
      pGEntity gent = EN_whatIn(v);
      for (std::set<pGRegion>::const_iterator layGrIt = lay_grs.begin();
           layGrIt != lay_grs.end(); ++layGrIt) {
        pGRegion layGr = *layGrIt;
        if (GEN_inClosure(layGr, gent)) {
          pPList vrs = V_regions(v);
          int nvrs = PList_size(vrs);
          for (int i=0; i < nvrs; i++) {
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
  double t1 = PCU_Time();
  *out << "adapt(): next layer migrated in " << t1-t0 << " seconds\n";

  // undo slicing if needed
  if (next_layer > 1) {
    *out << "adapt(): undoing slicing\n";
    double t2 = PCU_Time();
    DM_undoSlicing(combined_regions, next_layer-1, sim_pm);
    double t3 = PCU_Time();
    * out << "adapt(): undo slicing in " << t3-t2 << " seconds\n";
  }

  // mesh the top layer only
  *out << "adapt(): mesh top layer\n";
  mesh_current_layer(model, sim_pm, next_layer, size, out);

  // initialize field values on the top layer
  *out << "adapt(): add fields to top layer\n";
  {
    double t2 = PCU_Time();

    // collect the top layer vertices
    *out << "adapt(): collect top layer vertices\n";
    pMEntitySet topLayerVerts = MEntitySet_new(PM_mesh(sim_pm,0));
    regions = GM_regionIter(model);
    pVertex mv;
    while (gr1=GRIter_next(regions)) {
      if (GEN_numNativeIntAttribute(gr1, "SimLayer")==1) {
        GEN_nativeIntAttribute(gr1, "SimLayer", &layer);
        if (layer == next_layer) {
          for(int np=0;np<PM_numParts(sim_pm);np++) {
            VIter allVerts = M_classifiedVertexIter(PM_mesh(sim_pm, np), gr1, 1);
            while ( mv = VIter_next(allVerts) )
              MEntitySet_add(topLayerVerts,mv);
            VIter_delete(allVerts);
          }
        }
      }
    }

    // make sure mesh entities know about the fields
    pPList unmapped = PList_new();
    pDofGroup dg;
    MESIter viter = MESIter_iter(topLayerVerts);
    while ( mv = reinterpret_cast<pVertex>(MESIter_next(viter)) ) {
      dg = Field_entDof(temp, mv, 0);
      if (!dg) {
        PList_append(unmapped, mv);
        Field_applyEnt(temp, mv);
        Field_applyEnt(disp, mv);
      }
    }
    MESIter_delete(viter);

    // set the field values to the user-specified initial value
    pEntity ent;
    void *vptr;
    void *iter = 0;
    while (vptr = PList_next(unmapped,&iter)) {
      ent = reinterpret_cast<pEntity>(vptr);
      dg = Field_entDof(temp, ent, 0);
      DofGroup_setValue(dg, 0, 0, init_temp);
      dg = Field_entDof(disp, ent, 0);
      for (int c=0; c < 3; ++c)
        DofGroup_setValue(dg, c, 0, init_disp);
    }
    PList_delete(unmapped);
    MEntitySet_delete(PM_mesh(sim_pm,0),topLayerVerts);
    GRIter_delete(regions);

    double t3 = PCU_Time();
    *out << "adapt(): new layer fields set in " << t3-t2 << " seconds\n";
  }

}

static void write_debug(bool debug, apf::Mesh* m, const char* n, int c) {
  if (! debug) return;
  std::stringstream ss;
  ss << n << c;
  std::string s = ss.str();
  apf::writeVtkFiles(s.c_str(), m);
}

void Adapter::adapt(const double t_current) {

  static int call_count = 0;

  // print an initial statement
  *out << "adapt(): coming in " << PCU_Time()
       << ", cpu = " << PMU_rank() << std::endl;

  // print mesh stats before adaptation
  apf::printStats(apf_mesh);

  // get the solution fields
  auto t_apf_field = get_temp_field(t_disc, apf_mesh);
  auto m_apf_field = get_mech_field(m_disc, apf_mesh);
  auto t_sim_field = apf::getSIMField(t_apf_field);
  auto m_sim_field = apf::getSIMField(m_apf_field);

  // compute chosen spr error estimate on the chosen field
  auto spr_field_name = params->get<std::string>("SPR Solution Field", "");
  auto spr_size_field = compute_error_size(
      spr_field_name, t_apf_field, m_apf_field,
      use_error, use_target_elems, error_bound, target_elems);

  // print before adapt results
  printf("adapt(): before adapt - coming in: %d tets on cpu %d\n",
      M_numRegions(PM_mesh(sim_mesh, 0)), PMU_rank());

  double t0 = PCU_Time();

  // create the simmetrix adapter
  pACase mcase = MS_newMeshCase(sim_model);
  pModelItem domain = GM_domain(sim_model);
  MS_setMeshCurv(mcase, domain, ONLY_CURV_TYPE, 0.025);
  MS_setMinCurvSize(mcase, domain, ONLY_CURV_TYPE, 0.0025);
  MS_setMeshSize(mcase, domain, RELATIVE, 1.0, NULL);
  pMSAdapt adapter = MSA_createFromCase(mcase, sim_mesh);
  MSA_setSizeGradation(adapter, DO_GRADE, gradation);

  // copy the size field from APF to Simmetrix
  copy_size_field(apf_mesh, adapter, spr_size_field, min_size, max_size);

  // tell the simmetrix adapter which fields to transfer
  auto sim_field_list = set_transfer_fields(adapter, t_sim_field, m_sim_field);

  // constrain the top face
  constrain_top(adapter, sim_model, sim_mesh, spr_size_field,
      current_layer, layer_size);

  // write debug info + clean up
  write_debug(debug, apf_mesh, "preadapt_", call_count);
  apf::destroyField(spr_size_field);

  double t1 = PCU_Time();
  *out << "adapt(): preparing mesh adapt in " << t1-t0 << " seconds\n";

  // run the adapter
  double t2 = PCU_Time();
  pProgress progress = Progress_new();
  MSA_adapt(adapter, progress);
  Progress_delete(progress);
  MSA_delete(adapter);
  MS_deleteMeshCase(mcase);
  write_debug(debug, apf_mesh, "postadapt_", call_count);
  double t3 = PCU_Time();
  *out << "adapt(): mesh adapt in " << t3-t2 << " seconds\n";

  // add the layer
  *out << "adapt(): adding layer: " << current_layer+1 << std::endl;
  add_next_layer(sim_mesh, layer_size, current_layer+1, new_layer_temp, 0.0,
      t_sim_field, m_sim_field, out);
  write_debug(debug, apf_mesh, "postlayer_", call_count);
  current_layer++;

  // clean up
  PList_delete(sim_field_list);

  // partition the mesh
  double t4 = PCU_Time();
  Parma_PrintPtnStats(apf_mesh, "pre load balance:");
  PM_partition(sim_mesh, 0, 0);
  Parma_PrintPtnStats(apf_mesh, "post load balance:");
  double t5 = PCU_Time();
  *out << "adapt(): load balancing in " << t5-t4 << " seconds\n";

  // rebuild the data structures needed for analysis
  double t6 = PCU_Time();
  apf_mesh->verify();
  auto t_sim_disc = rcp_dynamic_cast<Albany::SimDiscretization>(t_disc);
  auto m_sim_disc = rcp_dynamic_cast<Albany::SimDiscretization>(m_disc);
  t_sim_disc->updateMesh(/* transfer ip = */ false, param_lib);
  m_sim_disc->updateMesh(/* transfer ip = */ false, param_lib);
  double t7 = PCU_Time();
  *out << "adapt(): update albany structures in " << t7-t6 << " seconds\n";

  // print stats after adaptation
  apf::printStats(apf_mesh);

  // print a final statement
  *out << "adapt(): going out in " << PCU_Time()
       << ", cpu = " << PMU_rank() << std::endl;
  printf("adapt(): leaving %d tets on cpu %d\n",
      M_numRegions(PM_mesh(sim_mesh,0)), PMU_rank());

  call_count++;
}

} // namespace CTM
