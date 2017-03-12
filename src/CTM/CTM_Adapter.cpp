#include "CTM_Adapter.hpp"

#include <set>
#include <spr.h>
#include <apfSIM.h>
#include <SimField.h>
#include <SimPartitionedMesh.h>
#include <MeshSimAdapt.h>
#include <Albany_Utils.hpp>
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

static void validate_params(RCP<const ParameterList> p) {
  ALBANY_ALWAYS_ASSERT(p->isType<double>("Uniform Temperature New Layer"));
  p->validateParameters(*get_valid_params(), 0);
}

Adapter::Adapter(
    RCP<ParameterList> p,
    RCP<Albany::StateManager> tsm,
    RCP<Albany::StateManager> msm) {
  params = p;
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

  *out << std::endl;
  *out << "*********************" << std::endl;
  *out << "LAYER ADDING ENABLED " << std::endl;
  *out << "*********************" << std::endl;
  compute_layer_info();
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

void Adapter::adapt() {
  *out << "I'M ADAPTING!!!" << std::endl;
}

} // namespace CTM
