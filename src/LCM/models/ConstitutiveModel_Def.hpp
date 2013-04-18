//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

namespace LCM {

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  ConstitutiveModel<EvalT, Traits>::
  ConstitutiveModel(Teuchos::ParameterList* p,
                    const Teuchos::RCP<Albany::Layouts>& dl) :
    num_state_variables_(0),
    need_integration_pt_locations_(false)
  {
    // extract number of integration points and dimensions
    std::vector<PHX::DataLayout::size_type> dims;
    dl->qp_tensor->dimensions(dims);
    num_pts_  = dims[1];
    num_dims_ = dims[2];

    field_name_map_ = 
      p->get<Teuchos::RCP<std::map<std::string, std::string> > >("Name Map");
  }
  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  std::string ConstitutiveModel<EvalT, Traits>::
  getStateVarName(int state_var)
  {
    return state_var_names_[state_var];
  }
  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  Teuchos::RCP<PHX::DataLayout> ConstitutiveModel<EvalT, Traits>::
  getStateVarLayout(int state_var)
  {
    return state_var_layouts_[state_var];
  }

  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  std::string ConstitutiveModel<EvalT, Traits>::
  getStateVarInitType(int state_var)
  {
    return state_var_init_types_[state_var];
  }
  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  double ConstitutiveModel<EvalT, Traits>::
  getStateVarInitValue(int state_var)
  {
    return state_var_init_values_[state_var];
  }
  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  bool ConstitutiveModel<EvalT, Traits>::
  getStateVarOldStateFlag(int state_var)
  {
    return state_var_old_state_flags_[state_var];
  }
  //----------------------------------------------------------------------------
  template<typename EvalT, typename Traits>
  bool ConstitutiveModel<EvalT, Traits>::
  getStateVarOutputFlag(int state_var)
  {
    return state_var_output_flags_[state_var];
  }

} 

