//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "Albany_Utils.hpp"
#include "Teuchos_Array.hpp"

namespace LCM
{

//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
ConstitutiveModelParameters<EvalT, Traits>::
ConstitutiveModelParameters(Teuchos::ParameterList& p,
    const Teuchos::RCP<Albany::Layouts>& dl) :
    have_temperature_(false),
    dl_(dl)
{
  // get number of integration points and spatial dimensions
  std::vector<PHX::DataLayout::size_type> dims;
  dl_->qp_vector->dimensions(dims);
  num_pts_ = dims[1];
  num_dims_ = dims[2];

  // get the Parameter Library
  Teuchos::RCP<ParamLib> paramLib =
      p.get<Teuchos::RCP<ParamLib> >("Parameter Library", Teuchos::null);

  // get the material parameter list
  Teuchos::ParameterList* mat_params =
      p.get<Teuchos::ParameterList*>("Material Parameters");

  // Check for optional field: temperature
  if (p.isType<std::string>("Temperature Name")) {
    have_temperature_ = true;
    PHX::MDField<ScalarT, Cell, QuadPoint>
    tmp(p.get<std::string>("Temperature Name"), dl_->qp_scalar);
    temperature_ = tmp;
    this->addDependentField(temperature_);
  }

  // step through the possible parameters, registering as necessary
  //
  // elastic modulus
  std::string e_mod("Elastic Modulus");
  if (mat_params->isSublist(e_mod)) {
    PHX::MDField<ScalarT, Cell, QuadPoint> tmp(e_mod, dl_->qp_scalar);
    elastic_mod_ = tmp;
    field_map_.insert(std::make_pair(e_mod, elastic_mod_));
    parseParameters(e_mod, p, paramLib);
  }
  // Poisson's ratio
  std::string pr("Poissons Ratio");
  if (mat_params->isSublist(pr)) {
    PHX::MDField<ScalarT, Cell, QuadPoint> tmp(pr, dl_->qp_scalar);
    poissons_ratio_ = tmp;
    field_map_.insert(std::make_pair(pr, poissons_ratio_));
    parseParameters(pr, p, paramLib);
  }
  // bulk modulus
  std::string b_mod("Bulk Modulus");
  if (mat_params->isSublist(b_mod)) {
    PHX::MDField<ScalarT, Cell, QuadPoint> tmp(b_mod, dl_->qp_scalar);
    bulk_mod_ = tmp;
    field_map_.insert(std::make_pair(b_mod, bulk_mod_));
    parseParameters(b_mod, p, paramLib);
  }
  // shear modulus
  std::string s_mod("Shear Modulus");
  if (mat_params->isSublist(s_mod)) {
    PHX::MDField<ScalarT, Cell, QuadPoint> tmp(s_mod, dl_->qp_scalar);
    shear_mod_ = tmp;
    field_map_.insert(std::make_pair(s_mod, shear_mod_));
    parseParameters(s_mod, p, paramLib);
  }
  // yield strength
  std::string yield("Yield Strength");
  if (mat_params->isSublist(yield)) {
    PHX::MDField<ScalarT, Cell, QuadPoint> tmp(yield, dl_->qp_scalar);
    yield_strength_ = tmp;
    field_map_.insert(std::make_pair(yield, yield_strength_));
    parseParameters(yield, p, paramLib);
  }
  // hardening modulus
  std::string h_mod("Hardening Modulus");
  if (mat_params->isSublist(h_mod)) {
    PHX::MDField<ScalarT, Cell, QuadPoint> tmp(h_mod, dl_->qp_scalar);
    hardening_mod_ = tmp;
    field_map_.insert(std::make_pair(h_mod, hardening_mod_));
    parseParameters(h_mod, p, paramLib);
  }
  // recovery modulus
  std::string r_mod("Recovery Modulus");
  if (mat_params->isSublist(r_mod)) {
    PHX::MDField<ScalarT, Cell, QuadPoint> tmp(r_mod, dl_->qp_scalar);
    recovery_mod_ = tmp;
    field_map_.insert(std::make_pair(r_mod, recovery_mod_));
    parseParameters(r_mod, p, paramLib);
  }
  // concentration equilibrium parameter
  std::string c_eq("Concentration Equilibrium Parameter");
  if (mat_params->isSublist(c_eq)) {
    PHX::MDField<ScalarT, Cell, QuadPoint> tmp(c_eq, dl_->qp_scalar);
    conc_eq_param_ = tmp;
    field_map_.insert(std::make_pair(c_eq, conc_eq_param_));
    parseParameters(c_eq, p, paramLib);
  }
  // diffusion coefficient
  std::string d_coeff("Diffusion Coefficient");
  if (mat_params->isSublist(d_coeff)) {
    PHX::MDField<ScalarT, Cell, QuadPoint> tmp(d_coeff, dl_->qp_scalar);
    diff_coeff_ = tmp;
    field_map_.insert(std::make_pair(d_coeff, diff_coeff_));
    parseParameters(d_coeff, p, paramLib);
  }
  // thermal conductivity
  std::string th_cond("Thermal Conductivity");
  if (mat_params->isSublist(th_cond)) {
    PHX::MDField<ScalarT, Cell, QuadPoint> tmp(th_cond, dl_->qp_scalar);
    thermal_cond_ = tmp;
    field_map_.insert(std::make_pair(th_cond, thermal_cond_));
    parseParameters(th_cond, p, paramLib);
  }
  // flow rule coefficient
  std::string f_coeff("Flow Rule Coefficient");
  if (mat_params->isSublist(f_coeff)) {
    PHX::MDField<ScalarT, Cell, QuadPoint> tmp(f_coeff, dl_->qp_scalar);
    flow_coeff_ = tmp;
    field_map_.insert(std::make_pair(f_coeff, flow_coeff_));
    parseParameters(f_coeff, p, paramLib);
  }
  // flow rule exponent
  std::string f_exp("Flow Rule Exponent");
  if (mat_params->isSublist(f_exp)) {
    PHX::MDField<ScalarT, Cell, QuadPoint> tmp(f_exp, dl_->qp_scalar);
    flow_exp_ = tmp;
    field_map_.insert(std::make_pair(f_exp, flow_exp_));
    parseParameters(f_exp, p, paramLib);
  }

  // register evaluated fields
  typename
  std::map<std::string, PHX::MDField<ScalarT, Cell, QuadPoint> >::iterator it;
  for (it = field_map_.begin();
      it != field_map_.end();
      ++it) {
    this->addEvaluatedField(it->second);
  }
  this->setName(
      "Constitutive Model Parameters" + PHX::TypeString<EvalT>::value);
}

//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
void ConstitutiveModelParameters<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
    PHX::FieldManager<Traits>& fm)
{
  typename std::map<std::string, PHX::MDField<ScalarT, Cell, QuadPoint> >::iterator it;
  for (it = field_map_.begin();
      it != field_map_.end();
      ++it) {
    this->utils.setFieldData(it->second, fm);
    if (!is_constant_map_[it->first]) {
      this->utils.setFieldData(coord_vec_, fm);
    }
  }

  if (have_temperature_) this->utils.setFieldData(temperature_, fm);
}

//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
void ConstitutiveModelParameters<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  typename std::map<std::string, PHX::MDField<ScalarT, Cell, QuadPoint> >::iterator it;
  for (it = field_map_.begin();
      it != field_map_.end();
      ++it) {
    ScalarT constant_value = constant_value_map_[it->first];
    if (is_constant_map_[it->first]) {
      for (std::size_t cell(0); cell < workset.numCells; ++cell) {
        for (std::size_t pt(0); pt < num_pts_; ++pt) {
          it->second(cell, pt) = constant_value;
        }
      }
    } else {
      for (std::size_t cell(0); cell < workset.numCells; ++cell) {
        for (std::size_t pt(0); pt < num_pts_; ++pt) {
          Teuchos::Array<MeshScalarT> point(num_dims_);
          for (std::size_t i(0); i < num_dims_; ++i)
            point[i] = Sacado::ScalarValue<MeshScalarT>::eval(
                coord_vec_(cell, pt, i));
          it->second(cell, pt) =
              exp_rf_kl_map_[it->first]->evaluate(point, rv_map_[it->first]);
        }
      }
    }
    // FIXME deal with Arrhenius temperature dependence, too
    if (have_temperature_) {
      if (temp_type_map_[it->first] == "Linear" ) {
        RealType dPdT = dparam_dtemp_map_[it->first];
        RealType ref_temp = ref_temp_map_[it->first];
        for (std::size_t cell(0); cell < workset.numCells; ++cell) {
          for (std::size_t pt(0); pt < num_pts_; ++pt) {
            it->second(cell, pt) += dPdT * (temperature_(cell, pt) - ref_temp);
          }
        }
      } else if (temp_type_map_[it->first] == "Arrhenius") {
        RealType pre_exp_ = pre_exp_map_[it->first];
        RealType exp_param_ = exp_param_map_[it->first];
        for (std::size_t cell(0); cell < workset.numCells; ++cell) {
          for (std::size_t pt(0); pt < num_pts_; ++pt) {
            it->second(cell, pt) = pre_exp_ 
              * std::exp( -exp_param_ / temperature_(cell, pt) );
          }
        }
      }
    }
  }
}

//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
typename ConstitutiveModelParameters<EvalT, Traits>::ScalarT&
ConstitutiveModelParameters<EvalT, Traits>::getValue(const std::string &n)
{
  typename std::map<std::string, ScalarT>::iterator it;
  for (it = constant_value_map_.begin();
      it != constant_value_map_.end();
      ++it) {
    if (n == it->first) {
      return constant_value_map_[it->first];
    }
  }
  typename std::map<std::string, Teuchos::Array<ScalarT> >::iterator it2;
  for (int i(0); i < rv_map_[it2->first].size(); ++i) {
    if (n == Albany::strint(n + " KL Random Variable", i))
      return rv_map_[it2->first][i];
  }
}
//------------------------------------------------------------------------------
template<typename EvalT, typename Traits>
void ConstitutiveModelParameters<EvalT, Traits>::
parseParameters(const std::string &n,
    Teuchos::ParameterList &p,
    Teuchos::RCP<ParamLib> paramLib)
{
  Teuchos::ParameterList pl =
    p.get<Teuchos::ParameterList*>("Material Parameters")->sublist(n);
  std::string type_name(n + " Type");
  std::string type = pl.get(type_name, "Constant");
  if (type == "Constant") {
    is_constant_map_.insert(std::make_pair(n, true));
    constant_value_map_.insert(std::make_pair(n, pl.get("Value", 1.0)));
    new Sacado::ParameterRegistration<EvalT, SPL_Traits>(n, this, paramLib);
    if (have_temperature_) {
      if (pl.get<std::string>("Temperature Dependence Type", "Linear")
          == "Linear") {
        temp_type_map_.insert(std::make_pair(n,"Linear"));
        dparam_dtemp_map_.insert
        (std::make_pair(n,
          pl.get<RealType>("Linear Temperature Coefficient", 0.0)));
        ref_temp_map_.insert
        (std::make_pair(n, pl.get<RealType>("Reference Temperature", -1)));
      } else if (pl.get<std::string>("Temperature Dependence Type", "Linear")
          == "Arrhenius") {
        temp_type_map_.insert(std::make_pair(n,"Arrhenius"));
        pre_exp_map_.insert(
            std::make_pair(n, pl.get<RealType>("Pre Exponential", 0.0)));
        exp_param_map_.insert(
            std::make_pair(n, pl.get<RealType>("Exponential Parameter", 0.0)));
      }
    }
  } else if (type == "Truncated KL Expansion") {
    is_constant_map_.insert(std::make_pair(n, false));
    PHX::MDField<MeshScalarT, Cell, QuadPoint, Dim>
      fx(p.get<std::string>("QP Coordinate Vector Name"), dl_->qp_vector);
    coord_vec_ = fx;
    this->addDependentField(coord_vec_);

    exp_rf_kl_map_.
        insert(
        std::make_pair(n,
            Teuchos::rcp(
                new Stokhos::KL::ExponentialRandomField<MeshScalarT>(pl))));
    int num_KL = exp_rf_kl_map_[n]->stochasticDimension();

    // Add KL random variables as Sacado-ized parameters
    rv_map_.insert(std::make_pair(n, Teuchos::Array<ScalarT>(num_KL)));
    for (int i(0); i < num_KL; ++i) {
      std::string ss = Albany::strint(n + " KL Random Variable", i);
      new Sacado::ParameterRegistration<EvalT, SPL_Traits>(ss, this, paramLib);
      rv_map_[n][i] = pl.get(ss, 0.0);
    }
  }
}
//------------------------------------------------------------------------------
}

