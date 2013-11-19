//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

namespace LCM
{

//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
ConstitutiveModel<EvalT, Traits>::
ConstitutiveModel(Teuchos::ParameterList* p,
    const Teuchos::RCP<Albany::Layouts>& dl) :
    num_state_variables_(0),
    compute_energy_(false),
    compute_tangent_(false),
    need_integration_pt_locations_(false),
    have_temperature_(false),
    have_damage_(false)
{
  // extract number of integration points and dimensions
  std::vector<PHX::DataLayout::size_type> dims;
  dl->qp_tensor->dimensions(dims);
  num_pts_  = dims[1];
  num_dims_ = dims[2];
  field_name_map_ =
      p->get<Teuchos::RCP<std::map<std::string, std::string> > >("Name Map");

  if (p->isType<bool>("Have Temperature")) {
    if (p->get<bool>("Have Temperature")) {
      have_temperature_ = true;
      expansion_coeff_ = p->get<RealType>("Thermal Expansion Coefficient", 0.0);
      ref_temperature_ = p->get<RealType>("Reference Temperature", 0.0);
      heat_capacity_ = p->get<RealType>("Heat Capacity", 1.0);
      density_ = p->get<RealType>("Density", 1.0);
    }
  }

  if (p->isType<bool>("Have Damage")) {
    if (p->get<bool>("Have Damage")) {
      have_damage_ = true;
    }
  }
}
//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
std::string ConstitutiveModel<EvalT, Traits>::
getStateVarName(int state_var)
{
  return state_var_names_[state_var];
}
//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
Teuchos::RCP<PHX::DataLayout> ConstitutiveModel<EvalT, Traits>::
getStateVarLayout(int state_var)
{
  return state_var_layouts_[state_var];
}
//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
std::string ConstitutiveModel<EvalT, Traits>::
getStateVarInitType(int state_var)
{
  return state_var_init_types_[state_var];
}
//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
double ConstitutiveModel<EvalT, Traits>::
getStateVarInitValue(int state_var)
{
  return state_var_init_values_[state_var];
}
//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
bool ConstitutiveModel<EvalT, Traits>::
getStateVarOldStateFlag(int state_var)
{
  return state_var_old_state_flags_[state_var];
}
//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
bool ConstitutiveModel<EvalT, Traits>::
getStateVarOutputFlag(int state_var)
{
  return state_var_output_flags_[state_var];
}
//------------------------------------------------------------------------------
}

