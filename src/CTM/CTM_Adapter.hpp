#ifndef CTM_ADAPTER_HPP
#define CTM_ADAPTER_HPP

#include <Teuchos_FancyOStream.hpp>
#include <Albany_DataTypes.hpp>
#include <SimModel.h>
#include <SimPartitionedMesh.h>

namespace apf {
class Mesh;
} // namespace apf

namespace Teuchos {
class ParameterList;
} // namespace Teuchos

namespace Albany {
class StateManager;
class AbstractDiscretization;
} // namespace Albany

namespace CTM {

using Teuchos::RCP;
using Teuchos::ParameterList;

class Adapter {

  public:

    Adapter(
        RCP<ParameterList> params,
        RCP<ParamLib> param_lib,
        RCP<Albany::StateManager> t_state_mgr,
        RCP<Albany::StateManager> m_state_mgr);

    bool should_adapt(const double t_current);

    void adapt();

  private:

    RCP<ParameterList> params;
    RCP<ParamLib> param_lib;
    RCP<Albany::StateManager> t_state_mgr;
    RCP<Albany::StateManager> m_state_mgr;
    RCP<Albany::AbstractDiscretization> t_disc;
    RCP<Albany::AbstractDiscretization> m_disc;
    RCP<Teuchos::FancyOStream> out;

    SGModel* sim_model;
    pParMesh sim_mesh;
    apf::Mesh* apf_mesh;

    int num_layers;
    int current_layer;
    std::vector<double> layer_times;
    double new_layer_temp;

    bool use_error;
    bool use_target_elems;
    double error_bound;
    long target_elems;

    double layer_size;
    double min_size;
    double max_size;
    double gradation;

    bool debug;

    void setup_params();
    void compute_layer_info();
    void compute_spr_size();

};

} // namespace CTM

#endif
