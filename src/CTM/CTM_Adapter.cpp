#include "CTM_Adapter.hpp"

#include <set>
#include <PCU.h>
#include <spr.h>
#include <apfSIM.h>
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
    pMSAdapt adapter, apf::Field* tf, apf::Field* mf) {
  pPList sim_list = PList_new();
  auto sim_tf = apf::getSIMField(tf);
  auto sim_mf = apf::getSIMField(mf);
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

static void write_debug(bool debug, apf::Mesh* m, const char* n, int c) {
  if (! debug) return;
  std::stringstream ss;
  ss << n << c;
  std::string s = ss.str();
  apf::writeVtkFiles(s.c_str(), m);
}

enum { ABSOLUTE = 1, RELATIVE = 2 };
enum { DONT_GRADE = 0, DO_GRADE = 1 };
enum { ONLY_CURV_TYPE = 2 };

void Adapter::adapt(const double t_current) {

  static int call_count = 0;

  // get the solution fields
  auto t_apf_field = get_temp_field(t_disc, apf_mesh);
  auto m_apf_field = get_mech_field(m_disc, apf_mesh);

  // compute chosen spr error estimate on the chosen field
  auto spr_field_name = params->get<std::string>("SPR Solution Field", "");
  auto spr_size_field = compute_error_size(
      spr_field_name, t_apf_field, m_apf_field,
      use_error, use_target_elems, error_bound, target_elems);

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
  auto sim_field_list = set_transfer_fields(adapter, t_apf_field, m_apf_field);

  // constrain the top face
  constrain_top(adapter, sim_model, sim_mesh, spr_size_field,
      current_layer, layer_size);

  // write debug info + clean up
  write_debug(debug, apf_mesh, "preadapt_", call_count);
  apf::destroyField(spr_size_field);

  double t1 = PCU_Time();
  *out << "adaptMesh(): preparing mesh adapt in " << t1-t0 << " seconds\n";
  double t2 = PCU_Time();

  // run the adapter
  pProgress progress = Progress_new();
  MSA_adapt(adapter, progress);
  Progress_delete(progress);
  MSA_delete(adapter);
  MS_deleteMeshCase(mcase);

  // write debug info
  write_debug(debug, apf_mesh, "postadapt_", call_count);

  double t3 = PCU_Time();
  *out << "adaptMesh(): mesh adapt in " << t3-t2 << " seconds\n";

  // add the layer if needed
  if (t_current >= layer_times[current_layer]) {
    *out << "adaptMesh(): adding layer: " << current_layer+1 << std::endl;
    write_debug(debug, apf_mesh, "postlayer_", call_count);
    current_layer++;
  }

  // clean up
  PList_delete(sim_field_list);

  double t4 = PCU_Time();

  // rebuild the data structures needed for analysis
  apf_mesh->verify();
  auto t_sim_disc = rcp_dynamic_cast<Albany::SimDiscretization>(t_disc);
  auto m_sim_disc = rcp_dynamic_cast<Albany::SimDiscretization>(m_disc);
  t_sim_disc->updateMesh(/* transfer ip = */ false, param_lib);
  m_sim_disc->updateMesh(/* transfer ip = */ false, param_lib);

  double t5 = PCU_Time();
  *out << "adaptMesh(): update albany structures in " << t5-t4 << " seconds\n";

  call_count++;

}

} // namespace CTM
