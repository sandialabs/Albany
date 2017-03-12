#ifndef CTM_ADAPTER_HPP
#define CTM_ADAPTER_HPP

#include <Teuchos_FancyOStream.hpp>
#include <SimModel.h>

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
        RCP<Albany::StateManager> t_state_mgr,
        RCP<Albany::StateManager> m_state_mgr);

    bool should_adapt(const double t_current);

    void adapt();

  private:

    RCP<ParameterList> params;
    RCP<Albany::StateManager> t_state_mgr;
    RCP<Albany::StateManager> m_state_mgr;
    RCP<Albany::AbstractDiscretization> t_disc;
    RCP<Albany::AbstractDiscretization> m_disc;
    RCP<Teuchos::FancyOStream> out;

    SGModel* sim_model;

    int num_layers;
    int current_layer;
    std::vector<double> layer_times;
    double new_layer_temp;

    void compute_layer_info();

};

} // namespace CTM

#endif
