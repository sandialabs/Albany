//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Albany_DiscretizationUtils.hpp"
#include "Albany_Layouts.hpp"
#include "Albany_KokkosUtils.hpp"

#include "LandIce_BasalFrictionCoefficient.hpp"

#include <string.hpp> // for 'upper_case' (comes from src/utility; not to be confused with <string>)

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN

namespace LandIce
{

template<typename EvalT, typename Traits, typename EffPressureST, typename VelocityST, typename TemperatureST>
BasalFrictionCoefficient<EvalT, Traits, EffPressureST, VelocityST, TemperatureST>::
BasalFrictionCoefficient (const Teuchos::ParameterList& p,
                          const Teuchos::RCP<Albany::Layouts>& dl)
{
#ifdef OUTPUT_TO_SCREEN
  Teuchos::RCP<Teuchos::FancyOStream> output(Teuchos::VerboseObjectBase::getDefaultOStream());

  int procRank = Teuchos::GlobalMPISession::getRank();
  int numProcs = Teuchos::GlobalMPISession::getNProc();
  output->setProcRankAndSize (procRank, numProcs);
  output->setOutputToRootOnly (0);
#endif

  n = -1; //dummy value
  Teuchos::ParameterList& beta_list = *p.get<Teuchos::ParameterList*>("Parameter List");
  zero_on_floating = beta_list.get<bool> ("Zero Beta On Floating Ice", false);

  //whether to first interpolate the given field and then exponetiate it (on quad points) or the other way around.
  interpolate_then_exponentiate = beta_list.get<bool> ("Interpolate Then Exponentiate Given Field", true); 

  std::string betaType = util::upper_case((beta_list.isParameter("Type") ? beta_list.get<std::string>("Type") : "Given Field"));

  is_side_equation = p.isParameter("Side Set Name");

  if (is_side_equation) {
    TEUCHOS_TEST_FOR_EXCEPTION (!dl->isSideLayouts, Teuchos::Exceptions::InvalidParameter,
                                "Error! The layout structure does not appear to be that of a side set.\n");

    basalSideName = p.get<std::string>("Side Set Name");
    numQPs        = dl->qp_scalar->extent(2);
    numNodes      = dl->node_scalar->extent(2);
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION (dl->isSideLayouts, Teuchos::Exceptions::InvalidParameter,
                                "Error! The layout structure appears to be that of a side set.\n");

    numQPs    = dl->qp_scalar->extent(1);
    numNodes  = dl->node_scalar->extent(1);
  }

  useCollapsedSidesets = dl->isSideLayouts && dl->useCollapsedSidesets;

  nodal = p.isParameter("Nodal") ? p.get<bool>("Nodal") : false;
  Teuchos::RCP<PHX::DataLayout> layout;
  if (useCollapsedSidesets) {
    layout = nodal ? dl->node_scalar_sideset : dl->qp_scalar_sideset;
  } else {
    layout = nodal ? dl->node_scalar : dl->qp_scalar;
  }

  beta = PHX::MDField<ScalarT>(p.get<std::string> ("Basal Friction Coefficient Variable Name"), layout);
  this->addEvaluatedField(beta);

  printedMu      = -9999.999;
  printedLambda  = -9999.999;
  printedQ       = -9999.999;

  auto is_dist_param = p.isParameter("Dist Param Query Map") ? p.get<Teuchos::RCP<std::map<std::string,bool>>>("Dist Param Query Map") : Teuchos::null;
  if (betaType == "GIVEN CONSTANT") {
    beta_type = GIVEN_CONSTANT;
    given_val = beta_list.get<double>("Constant Given Beta Value");
#ifdef OUTPUT_TO_SCREEN
    *output << "Given constant and uniform beta, value = " << given_val << " (loaded from xml input file).\n";
#endif
  } else if ((betaType == "GIVEN FIELD")|| (betaType == "EXPONENT OF GIVEN FIELD")) {
#ifdef OUTPUT_TO_SCREEN
    *output << "Given constant beta field, loaded from mesh or file.\n";
#endif
    if (betaType == "GIVEN FIELD") {
      beta_type = GIVEN_FIELD;
    } else {
      beta_type = EXP_GIVEN_FIELD;
    }

    auto layout_given_field = layout;
    std::string given_field_name = beta_list.get<std::string> ("Given Field Variable Name");
    is_given_field_param = is_dist_param.is_null() ? false : (*is_dist_param)[given_field_name];
    if (is_side_equation) {
      given_field_name += "_" + basalSideName;
    }
    if( (beta_type == EXP_GIVEN_FIELD) && !interpolate_then_exponentiate ) {
      BF = PHX::MDField<const RealType>(p.get<std::string> ("BF Variable Name"), useCollapsedSidesets ? dl->node_qp_scalar_sideset : dl->node_qp_scalar);
      layout_given_field = useCollapsedSidesets ? dl->node_scalar_sideset : dl->node_scalar;
      this->addDependentField (BF);
    }
    if (is_given_field_param) {
      given_field_param = PHX::MDField<const ParamScalarT>(given_field_name, layout_given_field);
      this->addDependentField (given_field_param);
    } else {
      given_field = PHX::MDField<const RealType>(given_field_name, layout_given_field);
      this->addDependentField (given_field);
    }
  } else if (betaType == "POWER LAW") {
    beta_type = POWER_LAW;

#ifdef OUTPUT_TO_SCREEN
    *output << "Velocity-dependent beta (power law):\n\n"
            << "      beta = mu * N * |u|^{q-1} \n\n"
            << "  with N being the effective pressure, |u| the sliding velocity\n";
#endif

    N              = PHX::MDField<const EffPressureST>(p.get<std::string> ("Effective Pressure Variable Name"), layout);
    u_norm         = PHX::MDField<const VelocityST>(p.get<std::string> ("Sliding Velocity Variable Name"), layout);
    powerParam     = PHX::MDField<const ScalarT,Dim>("Power Exponent", dl->shared_param);

    this->addDependentField (powerParam);
    this->addDependentField (u_norm);
    this->addDependentField (N);

    distributedMu = beta_list.get<bool>("Distributed Mu",false);
    if (distributedMu) {
      muPowerLawField = PHX::MDField<const ParamScalarT>(p.get<std::string> ("Power Law Coefficient Variable Name"), layout);
      this->addDependentField (muPowerLawField);
    } else {
      muPowerLaw     = PHX::MDField<const ScalarT,Dim>("Power Law Coefficient", dl->shared_param);
      this->addDependentField (muPowerLaw);
    }
  } else if (betaType == "REGULARIZED COULOMB") {
    beta_type = REGULARIZED_COULOMB;

#ifdef OUTPUT_TO_SCREEN
    *output << "Velocity-dependent beta (regularized coulomb law):\n\n"
            << "      beta = mu * N * |u|^{q-1} / [|u| + lambda*A*N^n]^q\n\n"
            << "  with N being the effective pressure, |u| the sliding velocity\n";
#endif

    n            = p.get<Teuchos::ParameterList*>("Viscosity Parameter List")->get<double>("Glen's Law n");
    N            = PHX::MDField<const EffPressureST>(p.get<std::string> ("Effective Pressure Variable Name"), layout);
    u_norm       = PHX::MDField<const VelocityST>(p.get<std::string> ("Sliding Velocity Variable Name"), layout);
    powerParam   = PHX::MDField<const ScalarT,Dim>("Power Exponent", dl->shared_param);
    ice_softness = PHX::MDField<const TemperatureST>(p.get<std::string>("Ice Softness Variable Name"), useCollapsedSidesets ? dl->cell_scalar2_sideset : dl->cell_scalar2);

    this->addDependentField (powerParam);
    this->addDependentField (N);
    this->addDependentField (u_norm);
    this->addDependentField (ice_softness);

    distributedLambda = beta_list.get<bool>("Distributed Lambda",false);
    if (distributedLambda) {
      lambdaField = PHX::MDField<const ParamScalarT>(p.get<std::string> ("Bed Roughness Variable Name"), layout);
      this->addDependentField (lambdaField);
    } else {
      lambdaParam    = PHX::MDField<ScalarT,Dim>("Bed Roughness", dl->shared_param);
      this->addDependentField (lambdaParam);
    }

    distributedMu = beta_list.get<bool>("Distributed Mu",false);
    if (distributedMu) {
      muCoulombField = PHX::MDField<const ParamScalarT>(p.get<std::string> ("Coulomb Friction Coefficient Variable Name"), layout);
      this->addDependentField (muCoulombField);
    } else {
      muCoulomb = PHX::MDField<const ScalarT,Dim>("Coulomb Friction Coefficient", dl->shared_param);
      this->addDependentField (muCoulomb);
    }
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
        std::endl << "Error in LandIce::BasalFrictionCoefficient:  \"" << betaType << "\" is not a valid parameter for Beta Type\n");
  }

  if(zero_on_floating) {
    bed_topo_field = PHX::MDField<const MeshScalarT>(p.get<std::string> ("Bed Topography Variable Name"), layout);
    is_thickness_param = is_dist_param.is_null() ? false : (*is_dist_param)[p.get<std::string>("Ice Thickness Variable Name")];
    if (is_thickness_param) {
      thickness_param_field = PHX::MDField<const ParamScalarT>(p.get<std::string> ("Ice Thickness Variable Name"), layout);
      this->addDependentField (thickness_param_field);
    } else {
      thickness_field = PHX::MDField<const MeshScalarT>(p.get<std::string> ("Ice Thickness Variable Name"), layout);
      this->addDependentField (thickness_field);
    }
    Teuchos::ParameterList& phys_param_list = *p.get<Teuchos::ParameterList*>("Physical Parameter List");
    rho_i = phys_param_list.get<double> ("Ice Density");
    rho_w = phys_param_list.get<double> ("Water Density");
    this->addDependentField (bed_topo_field);
  }

  auto& stereographicMapList = p.get<Teuchos::ParameterList*>("Stereographic Map");
  use_stereographic_map = stereographicMapList->get("Use Stereographic Map", false);
  if(use_stereographic_map) {
    if (useCollapsedSidesets) {
      layout = nodal ? dl->node_vector_sideset : dl->qp_coords_sideset;
    } else {
      layout = nodal ? dl->node_vector : dl->qp_coords;
    }
    coordVec = PHX::MDField<MeshScalarT>(p.get<std::string>("Coordinate Vector Variable Name"), layout);

    double R = stereographicMapList->get<double>("Earth Radius", 6371);
    x_0 = stereographicMapList->get<double>("X_0", 0);//-136);
    y_0 = stereographicMapList->get<double>("Y_0", 0);//-2040);
    R2 = std::pow(R,2);

    this->addDependentField(coordVec);
  }

  logParameters = beta_list.get<bool>("Use log scalar parameters",false);

  this->setName("BasalFrictionCoefficient"+PHX::print<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits, typename EffPressureST, typename VelocityST, typename TemperatureST>
void BasalFrictionCoefficient<EvalT, Traits, EffPressureST, VelocityST, TemperatureST>::
postRegistrationSetup (typename Traits::SetupData d,
                       PHX::FieldManager<Traits>& fm)
{
  if (beta_type == GIVEN_CONSTANT)
    beta.deep_copy(ScalarT(given_val));

  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
  if (d.memoizer_active())
    memoizer.enable_memoizer();
}

// *********************************************************************
// Kokkos functor (Cell)
template<typename EvalT, typename Traits, typename EffPressureST, typename VelocityST, typename TemperatureST>
KOKKOS_INLINE_FUNCTION
void BasalFrictionCoefficient<EvalT, Traits, EffPressureST, VelocityST, TemperatureST>::
operator() (const GivenFieldParam_Tag& tag, const int& cell) const {
  for (unsigned int ipt=0; ipt<dim; ++ipt)
    beta(cell,ipt) = given_field_param(cell,ipt);
}
template<typename EvalT, typename Traits, typename EffPressureST, typename VelocityST, typename TemperatureST>
KOKKOS_INLINE_FUNCTION
void BasalFrictionCoefficient<EvalT, Traits, EffPressureST, VelocityST, TemperatureST>::
operator() (const GivenField_Tag& tag, const int& cell) const {
  for (unsigned int ipt=0; ipt<dim; ++ipt)
    beta(cell,ipt) = given_field(cell,ipt);
}
template<typename EvalT, typename Traits, typename EffPressureST, typename VelocityST, typename TemperatureST>
KOKKOS_INLINE_FUNCTION
void BasalFrictionCoefficient<EvalT, Traits, EffPressureST, VelocityST, TemperatureST>::
operator() (const PowerLaw_DistributedMu_Tag& tag, const int& cell) const {
  for (unsigned int ipt=0; ipt<dim; ++ipt)
    beta(cell,ipt) = muPowerLawField(cell,ipt) * N(cell,ipt) * std::pow (u_norm(cell,ipt), power-1);
}
template<typename EvalT, typename Traits, typename EffPressureST, typename VelocityST, typename TemperatureST>
KOKKOS_INLINE_FUNCTION
void BasalFrictionCoefficient<EvalT, Traits, EffPressureST, VelocityST, TemperatureST>::
operator() (const PowerLaw_Tag& tag, const int& cell) const {
  for (unsigned int ipt=0; ipt<dim; ++ipt)
    beta(cell,ipt) = mu * N(cell,ipt) * std::pow (u_norm(cell,ipt), power-1);
}
template<typename EvalT, typename Traits, typename EffPressureST, typename VelocityST, typename TemperatureST>
KOKKOS_INLINE_FUNCTION
void BasalFrictionCoefficient<EvalT, Traits, EffPressureST, VelocityST, TemperatureST>::
operator() (const RegularizedCoulomb_DistributedLambda_DistributedMu_Tag& tag, const int& cell) const {
  for (unsigned int ipt=0; ipt<dim; ++ipt) {
    ScalarT q = u_norm(cell,ipt) / ( u_norm(cell,ipt) + lambdaField(cell,ipt)*ice_softness(cell)*std::pow(N(cell,ipt),n) );
    beta(cell,ipt) = muCoulombField(cell,ipt) * N(cell,ipt) * std::pow( q, power) / u_norm(cell,ipt);
  }
}
template<typename EvalT, typename Traits, typename EffPressureST, typename VelocityST, typename TemperatureST>
KOKKOS_INLINE_FUNCTION
void BasalFrictionCoefficient<EvalT, Traits, EffPressureST, VelocityST, TemperatureST>::
operator() (const RegularizedCoulomb_DistributedLambda_Tag& tag, const int& cell) const {
  for (unsigned int ipt=0; ipt<dim; ++ipt) {
    ScalarT q = u_norm(cell,ipt) / ( u_norm(cell,ipt) + lambdaField(cell,ipt)*ice_softness(cell)*std::pow(N(cell,ipt),n) );
    beta(cell,ipt) = mu * N(cell,ipt) * std::pow( q, power) / u_norm(cell,ipt);
  }
}
template<typename EvalT, typename Traits, typename EffPressureST, typename VelocityST, typename TemperatureST>
KOKKOS_INLINE_FUNCTION
void BasalFrictionCoefficient<EvalT, Traits, EffPressureST, VelocityST, TemperatureST>::
operator() (const RegularizedCoulomb_DistributedMu_Tag& tag, const int& cell) const {
  for (unsigned int ipt=0; ipt<dim; ++ipt) {
    ScalarT q = u_norm(cell,ipt) / ( u_norm(cell,ipt) + lambda*ice_softness(cell)*std::pow(N(cell,ipt),n) );
    beta(cell,ipt) = muCoulombField(cell,ipt) * N(cell,ipt) * std::pow( q, power) / u_norm(cell,ipt);
  }
}
template<typename EvalT, typename Traits, typename EffPressureST, typename VelocityST, typename TemperatureST>
KOKKOS_INLINE_FUNCTION
void BasalFrictionCoefficient<EvalT, Traits, EffPressureST, VelocityST, TemperatureST>::
operator() (const RegularizedCoulomb_Tag& tag, const int& cell) const {
  for (unsigned int ipt=0; ipt<dim; ++ipt) {
    ScalarT q = u_norm(cell,ipt) / ( u_norm(cell,ipt) + lambda*ice_softness(cell)*std::pow(N(cell,ipt),n) );
    beta(cell,ipt) = mu * N(cell,ipt) * std::pow( q, power) / u_norm(cell,ipt);
  }
}
template<typename EvalT, typename Traits, typename EffPressureST, typename VelocityST, typename TemperatureST>
KOKKOS_INLINE_FUNCTION
void BasalFrictionCoefficient<EvalT, Traits, EffPressureST, VelocityST, TemperatureST>::
operator() (const ExpGivenFieldParam_Nodal_Tag& tag, const int& cell) const {
  for (unsigned int ipt=0; ipt<dim; ++ipt) {
    beta(cell,ipt) = std::exp(given_field_param(cell,ipt));
  }
}
template<typename EvalT, typename Traits, typename EffPressureST, typename VelocityST, typename TemperatureST>
KOKKOS_INLINE_FUNCTION
void BasalFrictionCoefficient<EvalT, Traits, EffPressureST, VelocityST, TemperatureST>::
operator() (const ExpGivenField_Nodal_Tag& tag, const int& cell) const {
  for (unsigned int ipt=0; ipt<dim; ++ipt) {
    beta(cell,ipt) = std::exp(given_field(cell,ipt));
  }
}
template<typename EvalT, typename Traits, typename EffPressureST, typename VelocityST, typename TemperatureST>
KOKKOS_INLINE_FUNCTION
void BasalFrictionCoefficient<EvalT, Traits, EffPressureST, VelocityST, TemperatureST>::
operator() (const ExpGivenFieldParam_Tag& tag, const int& cell) const {
  for (unsigned int ipt=0; ipt<dim; ++ipt) {
    beta(cell,ipt) = 0;
    for (unsigned int node=0; node<numNodes; ++node)
      beta(cell,ipt) += std::exp(given_field_param(cell,node))*BF(cell,node,ipt);
  }
}
template<typename EvalT, typename Traits, typename EffPressureST, typename VelocityST, typename TemperatureST>
KOKKOS_INLINE_FUNCTION
void BasalFrictionCoefficient<EvalT, Traits, EffPressureST, VelocityST, TemperatureST>::
operator() (const ExpGivenField_Tag& tag, const int& cell) const {
  for (unsigned int ipt=0; ipt<dim; ++ipt) {
    beta(cell,ipt) = 0;
    for (unsigned int node=0; node<numNodes; ++node)
      beta(cell,ipt) += std::exp(given_field(cell,node))*BF(cell,node,ipt);
  }
}
template<typename EvalT, typename Traits, typename EffPressureST, typename VelocityST, typename TemperatureST>
KOKKOS_INLINE_FUNCTION
void BasalFrictionCoefficient<EvalT, Traits, EffPressureST, VelocityST, TemperatureST>::
operator() (const StereographicMapCorrection_Tag& tag, const int& cell) const {
  for (unsigned int ipt=0; ipt<dim; ++ipt) {
    MeshScalarT x = coordVec(cell,ipt,0) - x_0;
    MeshScalarT y = coordVec(cell,ipt,1) - y_0;
    MeshScalarT h = 4.0*R2/(4.0*R2 + x*x + y*y);
    beta(cell,ipt) *= h*h;
  }
}

// *********************************************************************
// Kokkos functor (Side)
template<typename EvalT, typename Traits, typename EffPressureST, typename VelocityST, typename TemperatureST>
KOKKOS_INLINE_FUNCTION
void BasalFrictionCoefficient<EvalT, Traits, EffPressureST, VelocityST, TemperatureST>::
operator() (const Side_GivenFieldParam_Tag& tag, const int& sideSet_idx) const {
  for (unsigned int ipt=0; ipt<dim; ++ipt)
    beta(sideSet_idx,ipt) = given_field_param(sideSet_idx,ipt);
}
template<typename EvalT, typename Traits, typename EffPressureST, typename VelocityST, typename TemperatureST>
KOKKOS_INLINE_FUNCTION
void BasalFrictionCoefficient<EvalT, Traits, EffPressureST, VelocityST, TemperatureST>::
operator() (const Side_GivenField_Tag& tag, const int& sideSet_idx) const {
  for (unsigned int ipt=0; ipt<dim; ++ipt)
    beta(sideSet_idx,ipt) = given_field(sideSet_idx,ipt);
}
template<typename EvalT, typename Traits, typename EffPressureST, typename VelocityST, typename TemperatureST>
KOKKOS_INLINE_FUNCTION
void BasalFrictionCoefficient<EvalT, Traits, EffPressureST, VelocityST, TemperatureST>::
operator() (const Side_PowerLaw_DistributedMu_Tag& tag, const int& sideSet_idx) const {
  for (unsigned int ipt=0; ipt<dim; ++ipt) {
    ScalarT Nval = KU::max(N(sideSet_idx,ipt),0.0);
    beta(sideSet_idx,ipt) = muPowerLawField(sideSet_idx,ipt) * Nval * std::pow (u_norm(sideSet_idx,ipt), power-1);
  }
}
template<typename EvalT, typename Traits, typename EffPressureST, typename VelocityST, typename TemperatureST>
KOKKOS_INLINE_FUNCTION
void BasalFrictionCoefficient<EvalT, Traits, EffPressureST, VelocityST, TemperatureST>::
operator() (const Side_PowerLaw_Tag& tag, const int& sideSet_idx) const {
  for (unsigned int ipt=0; ipt<dim; ++ipt) {
    ScalarT Nval = KU::max(N(sideSet_idx,ipt),0.0);
    beta(sideSet_idx,ipt) = mu * Nval * std::pow (u_norm(sideSet_idx,ipt), power-1);
  }
}
template<typename EvalT, typename Traits, typename EffPressureST, typename VelocityST, typename TemperatureST>
KOKKOS_INLINE_FUNCTION
void BasalFrictionCoefficient<EvalT, Traits, EffPressureST, VelocityST, TemperatureST>::
operator() (const Side_RegularizedCoulomb_DistributedLambda_DistributedMu_Tag& tag, const int& sideSet_idx) const {
  for (unsigned int ipt=0; ipt<dim; ++ipt) {
    ScalarT Nval = KU::max(N(sideSet_idx,ipt),0.0);
    ScalarT q = u_norm(sideSet_idx,ipt) / ( u_norm(sideSet_idx,ipt) + lambdaField(sideSet_idx,ipt)*ice_softness(sideSet_idx)*std::pow(Nval,n) );
    beta(sideSet_idx,ipt) = muCoulombField(sideSet_idx,ipt) * Nval * std::pow( q, power) / u_norm(sideSet_idx,ipt);
  }
}
template<typename EvalT, typename Traits, typename EffPressureST, typename VelocityST, typename TemperatureST>
KOKKOS_INLINE_FUNCTION
void BasalFrictionCoefficient<EvalT, Traits, EffPressureST, VelocityST, TemperatureST>::
operator() (const Side_RegularizedCoulomb_DistributedLambda_Tag& tag, const int& sideSet_idx) const {
  for (unsigned int ipt=0; ipt<dim; ++ipt) {
    ScalarT Nval = KU::max(N(sideSet_idx,ipt),0.0);
    ScalarT q = u_norm(sideSet_idx,ipt) / ( u_norm(sideSet_idx,ipt) + lambdaField(sideSet_idx,ipt)*ice_softness(sideSet_idx)*std::pow(Nval,n) );
    beta(sideSet_idx,ipt) = mu * Nval * std::pow( q, power) / u_norm(sideSet_idx,ipt);
  }
}
template<typename EvalT, typename Traits, typename EffPressureST, typename VelocityST, typename TemperatureST>
KOKKOS_INLINE_FUNCTION
void BasalFrictionCoefficient<EvalT, Traits, EffPressureST, VelocityST, TemperatureST>::
operator() (const Side_RegularizedCoulomb_DistributedMu_Tag& tag, const int& sideSet_idx) const {
  for (unsigned int ipt=0; ipt<dim; ++ipt) {
    ScalarT Nval = KU::max(N(sideSet_idx,ipt),0.0);
    ScalarT q = u_norm(sideSet_idx,ipt) / ( u_norm(sideSet_idx,ipt) + lambda*ice_softness(sideSet_idx)*std::pow(Nval,n) );
    beta(sideSet_idx,ipt) = muCoulombField(sideSet_idx,ipt) * Nval * std::pow( q, power) / u_norm(sideSet_idx,ipt);
  }
}
template<typename EvalT, typename Traits, typename EffPressureST, typename VelocityST, typename TemperatureST>
KOKKOS_INLINE_FUNCTION
void BasalFrictionCoefficient<EvalT, Traits, EffPressureST, VelocityST, TemperatureST>::
operator() (const Side_RegularizedCoulomb_Tag& tag, const int& sideSet_idx) const {
  for (unsigned int ipt=0; ipt<dim; ++ipt) {
    ScalarT Nval = KU::max(N(sideSet_idx,ipt),0.0);
    ScalarT q = u_norm(sideSet_idx,ipt) / ( u_norm(sideSet_idx,ipt) + lambda*ice_softness(sideSet_idx)*std::pow(Nval,n) );
    beta(sideSet_idx,ipt) = mu * Nval * std::pow( q, power) / u_norm(sideSet_idx,ipt);
  }
}
template<typename EvalT, typename Traits, typename EffPressureST, typename VelocityST, typename TemperatureST>
KOKKOS_INLINE_FUNCTION
void BasalFrictionCoefficient<EvalT, Traits, EffPressureST, VelocityST, TemperatureST>::
operator() (const Side_ExpGivenFieldParam_Nodal_Tag& tag, const int& sideSet_idx) const {
  for (unsigned int ipt=0; ipt<dim; ++ipt)
    beta(sideSet_idx,ipt) = std::exp(given_field_param(sideSet_idx,ipt));
}
template<typename EvalT, typename Traits, typename EffPressureST, typename VelocityST, typename TemperatureST>
KOKKOS_INLINE_FUNCTION
void BasalFrictionCoefficient<EvalT, Traits, EffPressureST, VelocityST, TemperatureST>::
operator() (const Side_ExpGivenFieldParam_Tag& tag, const int& sideSet_idx) const {
  for (unsigned int qp=0; qp<numQPs; ++qp) {
    beta(sideSet_idx,qp) = 0;
    for (unsigned int node=0; node<numNodes; ++node)
      beta(sideSet_idx,qp) += std::exp(given_field_param(sideSet_idx,node))*BF(sideSet_idx,node,qp);  
  }
}
template<typename EvalT, typename Traits, typename EffPressureST, typename VelocityST, typename TemperatureST>
KOKKOS_INLINE_FUNCTION
void BasalFrictionCoefficient<EvalT, Traits, EffPressureST, VelocityST, TemperatureST>::
operator() (const Side_ExpGivenField_Nodal_Tag& tag, const int& sideSet_idx) const {
  for (unsigned int ipt=0; ipt<dim; ++ipt)
    beta(sideSet_idx,ipt) = std::exp(given_field(sideSet_idx,ipt));
}
template<typename EvalT, typename Traits, typename EffPressureST, typename VelocityST, typename TemperatureST>
KOKKOS_INLINE_FUNCTION
void BasalFrictionCoefficient<EvalT, Traits, EffPressureST, VelocityST, TemperatureST>::
operator() (const Side_ExpGivenField_Tag& tag, const int& sideSet_idx) const {
  for (unsigned int qp=0; qp<numQPs; ++qp) {
    beta(sideSet_idx,qp) = 0;
    for (unsigned int node=0; node<numNodes; ++node)
      beta(sideSet_idx,qp) += std::exp(given_field(sideSet_idx,node))*BF(sideSet_idx,node,qp);
  }
}
template<typename EvalT, typename Traits, typename EffPressureST, typename VelocityST, typename TemperatureST>
KOKKOS_INLINE_FUNCTION
void BasalFrictionCoefficient<EvalT, Traits, EffPressureST, VelocityST, TemperatureST>::
operator() (const Side_ZeroOnFloatingParam_Tag& tag, const int& sideSet_idx) const {
  for (unsigned int ipt=0; ipt<dim; ++ipt) {
    ParamScalarT isGrounded = rho_i*thickness_param_field(sideSet_idx,ipt) > -rho_w*bed_topo_field(sideSet_idx,ipt);
    beta(sideSet_idx,ipt) *=  isGrounded;
  }
}
template<typename EvalT, typename Traits, typename EffPressureST, typename VelocityST, typename TemperatureST>
KOKKOS_INLINE_FUNCTION
void BasalFrictionCoefficient<EvalT, Traits, EffPressureST, VelocityST, TemperatureST>::
operator() (const Side_ZeroOnFloating_Tag& tag, const int& sideSet_idx) const {
  for (unsigned int ipt=0; ipt<dim; ++ipt) {
    ParamScalarT isGrounded = rho_i*thickness_field(sideSet_idx,ipt) > -rho_w*bed_topo_field(sideSet_idx,ipt);
    beta(sideSet_idx,ipt) *=  isGrounded;
  }
}
template<typename EvalT, typename Traits, typename EffPressureST, typename VelocityST, typename TemperatureST>
KOKKOS_INLINE_FUNCTION
void BasalFrictionCoefficient<EvalT, Traits, EffPressureST, VelocityST, TemperatureST>::
operator() (const Side_StereographicMapCorrection_Tag& tag, const int& sideSet_idx) const {
  for (unsigned int ipt=0; ipt<dim; ++ipt) {
    MeshScalarT x = coordVec(sideSet_idx,ipt,0) - x_0;
    MeshScalarT y = coordVec(sideSet_idx,ipt,1) - y_0;
    MeshScalarT h = 4.0*R2/(4.0*R2 + x*x + y*y);
    beta(sideSet_idx,ipt) *= h*h;
  }
}

//**********************************************************************
template<typename EvalT, typename Traits, typename EffPressureST, typename VelocityST, typename TemperatureST>
void BasalFrictionCoefficient<EvalT, Traits, EffPressureST, VelocityST, TemperatureST>::
evaluateFields (typename Traits::EvalData workset)
{
  if (memoizer.have_saved_data(workset,this->evaluatedFields()))
    return;

  if (beta_type == POWER_LAW) {
    if (logParameters) {
      power = std::exp(Albany::convertScalar<const ParamScalarT>(powerParam(0)));
      if (!distributedMu)
        mu = std::exp(Albany::convertScalar<const ParamScalarT>(muPowerLaw(0)));
    } else {
      power = Albany::convertScalar<const ParamScalarT>(powerParam(0));
      if (!distributedMu)
        mu = Albany::convertScalar<const ParamScalarT>(muPowerLaw(0));
    }
#ifdef OUTPUT_TO_SCREEN
    Teuchos::RCP<Teuchos::FancyOStream> output(Teuchos::VerboseObjectBase::getDefaultOStream());
    int procRank = Teuchos::GlobalMPISession::getRank();
    int numProcs = Teuchos::GlobalMPISession::getNProc();
    output->setProcRankAndSize (procRank, numProcs);
    output->setOutputToRootOnly (0);

    if (!distributedMu && printedMu!=mu) {
      *output << "[Basal Friction Coefficient" << PHX::print<EvalT>() << "] mu = " << mu << " [kPa yr^q m^{-q}]\n";
      printedMu = mu;
    }

    if (printedQ!=power) {
      *output << "[Basal Friction Coefficient" << PHX::print<EvalT>() << "] power = " << power << "\n";
      printedQ = power;
    }
#endif

    TEUCHOS_TEST_FOR_EXCEPTION (power<0, Teuchos::Exceptions::InvalidParameter,
                                "\nError in LandIce::BasalFrictionCoefficient: 'Power Exponent' must be >= 0.\n");
    TEUCHOS_TEST_FOR_EXCEPTION (!distributedMu && mu<0, Teuchos::Exceptions::InvalidParameter,
                                "\nError in LandIce::BasalFrictionCoefficient: 'Coulomb Friction Coefficient' must be >= 0.\n");
  }

  if (beta_type==REGULARIZED_COULOMB) {
    if (logParameters) {
      power = std::exp(Albany::convertScalar<const ParamScalarT>(powerParam(0)));

      if (!distributedLambda)
        lambda = std::exp(Albany::convertScalar<const ParamScalarT>(lambdaParam(0)));
      if (!distributedMu) 
        mu     = std::exp(Albany::convertScalar<const ParamScalarT>(muCoulomb(0)));
    } else {
      power = Albany::convertScalar<const ParamScalarT>(powerParam(0));

      if (!distributedLambda)
        lambda = Albany::convertScalar<const ParamScalarT>(lambdaParam(0));
      if (!distributedMu) 
        mu     = Albany::convertScalar<const ParamScalarT>(muCoulomb(0));
    }
#ifdef OUTPUT_TO_SCREEN
    Teuchos::RCP<Teuchos::FancyOStream> output(Teuchos::VerboseObjectBase::getDefaultOStream());
    int procRank = Teuchos::GlobalMPISession::getRank();
    int numProcs = Teuchos::GlobalMPISession::getNProc();
    output->setProcRankAndSize (procRank, numProcs);
    output->setOutputToRootOnly (0);

    if (!distributedLambda && printedLambda!=lambda) {
      *output << "[Basal Friction Coefficient" << PHX::print<EvalT>() << "] lambda = " << lambda << "\n";
      printedLambda = lambda;
    }

    if (!distributedMu && printedMu!=mu) {
      *output << "[Basal Friction Coefficient" << PHX::print<EvalT>() << "] mu = " << mu << "\n";
      printedMu = mu;
    }

    if (printedQ!=power) {
      *output << "[Basal Friction Coefficient" << PHX::print<EvalT>() << "] power = " << power << "\n";
      printedQ = power;
    }
#endif

    TEUCHOS_TEST_FOR_EXCEPTION (power<0, Teuchos::Exceptions::InvalidParameter,
                                "\nError in LandIce::BasalFrictionCoefficient: 'Power Exponent' must be >= 0.\n");
    TEUCHOS_TEST_FOR_EXCEPTION (!distributedMu && mu<0, Teuchos::Exceptions::InvalidParameter,
                                "\nError in LandIce::BasalFrictionCoefficient: 'Coulomb Friction Coefficient' must be >= 0.\n");
    TEUCHOS_TEST_FOR_EXCEPTION (!distributedLambda && lambda<0, Teuchos::Exceptions::InvalidParameter,
                                "\nError in LandIce::BasalFrictionCoefficient: \"Bed Roughness\" must be >= 0.\n");
  }

  if (is_side_equation)
    evaluateFieldsSide(workset,mu,lambda,power);
  else
    evaluateFieldsCell(workset,mu,lambda,power);
}

template<typename EvalT, typename Traits, typename EffPressureST, typename VelocityST, typename TemperatureST>
void BasalFrictionCoefficient<EvalT, Traits, EffPressureST, VelocityST, TemperatureST>::
evaluateFieldsSide (typename Traits::EvalData workset, ScalarT mu, ScalarT lambda, ScalarT power)
{
  if (workset.sideSetViews->find(basalSideName)==workset.sideSetViews->end()) return;

  sideSet = workset.sideSetViews->at(basalSideName);

  dim = nodal ? numNodes : numQPs;

  if (useCollapsedSidesets) {
    switch (beta_type) {
      case GIVEN_CONSTANT:
        return;   // We can save ourself some useless iterations

      case GIVEN_FIELD:
        if (is_given_field_param) {
          Kokkos::parallel_for(Side_GivenFieldParam_Policy(0, sideSet.size), *this);
        } else {
          Kokkos::parallel_for(Side_GivenField_Policy(0, sideSet.size), *this);
        }
        break;

      case POWER_LAW:
        if (distributedMu) {
          Kokkos::parallel_for(Side_PowerLaw_DistributedMu_Policy(0, sideSet.size), *this);
        } else {
          Kokkos::parallel_for(Side_PowerLaw_Policy(0, sideSet.size), *this);
        }

        break;

      case REGULARIZED_COULOMB:
        if (distributedLambda) {
          if (distributedMu) {
            Kokkos::parallel_for(Side_RegularizedCoulomb_DistributedLambda_DistributedMu_Policy(0, sideSet.size), *this);
          } else {
            Kokkos::parallel_for(Side_RegularizedCoulomb_DistributedLambda_Policy(0, sideSet.size), *this);
          }
        } else {
          if (distributedMu) {
            Kokkos::parallel_for(Side_RegularizedCoulomb_DistributedMu_Policy(0, sideSet.size), *this);
          } else {
            Kokkos::parallel_for(Side_RegularizedCoulomb_Policy(0, sideSet.size), *this);
          }
        }
        break;

      case EXP_GIVEN_FIELD:
        if(nodal || interpolate_then_exponentiate) {
          if (is_given_field_param) {
            Kokkos::parallel_for(Side_ExpGivenFieldParam_Nodal_Policy(0, sideSet.size), *this);
          } else {
            Kokkos::parallel_for(Side_ExpGivenField_Nodal_Policy(0, sideSet.size), *this);
          }
        } else {
          if (is_given_field_param) {
            Kokkos::parallel_for(Side_ExpGivenFieldParam_Policy(0, sideSet.size), *this);
          } else {
            Kokkos::parallel_for(Side_ExpGivenField_Policy(0, sideSet.size), *this);
          }
        }
        break;

      default:
        break;
     }

     if(zero_on_floating) {
      if (is_thickness_param) {
        Kokkos::parallel_for(Side_ZeroOnFloatingParam_Policy(0, sideSet.size), *this);
      } else {
        Kokkos::parallel_for(Side_ZeroOnFloating_Policy(0, sideSet.size), *this);
      }
    }

    // Correct the value if we are using a stereographic map
    if (use_stereographic_map) {
      Kokkos::parallel_for(Side_StereographicMapCorrection_Policy(0, sideSet.size), *this);
    }
  }
  else {
    for (int sideSet_idx = 0; sideSet_idx < sideSet.size; ++sideSet_idx)
    {
      // Get the local data of side and cell
      const int cell = sideSet.elem_LID(sideSet_idx);
      const int side = sideSet.side_local_id(sideSet_idx);

      switch (beta_type) {
        case GIVEN_CONSTANT:
          return;   // We can save ourself some useless iterations

        case GIVEN_FIELD:
          if (is_given_field_param) {
            for (unsigned int ipt=0; ipt<dim; ++ipt)
              beta(cell,side,ipt) = given_field_param(cell,side,ipt);
          } else {
            for (unsigned int ipt=0; ipt<dim; ++ipt)
              beta(cell,side,ipt) = given_field(cell,side,ipt);
          }
          break;

        case POWER_LAW:
          if (distributedMu) {
            for (unsigned int ipt=0; ipt<dim; ++ipt) {
              ScalarT Nval = std::max(N(cell,side,ipt),0.0);
              beta(cell,side,ipt) = muPowerLawField(cell,side,ipt) * Nval * std::pow (u_norm(cell,side,ipt), power-1);
            }
          } else {
            for (unsigned int ipt=0; ipt<dim; ++ipt) {
              ScalarT Nval = std::max(N(cell,side,ipt),0.0);
              beta(cell,side,ipt) = mu * Nval * std::pow (u_norm(cell,side,ipt), power-1);
            }
          }

          break;

        case REGULARIZED_COULOMB:
          if (distributedLambda) {
            if (distributedMu) {
              for (unsigned int ipt=0; ipt<dim; ++ipt) {
                ScalarT Nval = std::max(N(cell,side,ipt),0.0);
                ScalarT q = u_norm(cell,side,ipt) / ( u_norm(cell,side,ipt) + lambdaField(cell,side,ipt)*ice_softness(cell,side)*std::pow(Nval,n) );
                beta(cell,side,ipt) = muCoulombField(cell,side,ipt) * Nval * std::pow( q, power) / u_norm(cell,side,ipt);
              }
            } else {
              for (unsigned int ipt=0; ipt<dim; ++ipt) {
                ScalarT Nval = std::max(N(cell,side,ipt),0.0);
                ScalarT q = u_norm(cell,side,ipt) / ( u_norm(cell,side,ipt) + lambdaField(cell,side,ipt)*ice_softness(cell,side)*std::pow(Nval,n) );
                beta(cell,side,ipt) = mu * Nval * std::pow( q, power) / u_norm(cell,side,ipt);
              }
            }
          } else {
            if (distributedMu) {
              for (unsigned int ipt=0; ipt<dim; ++ipt) {
                ScalarT Nval = std::max(N(cell,side,ipt),0.0);
                ScalarT q = u_norm(cell,side,ipt) / ( u_norm(cell,side,ipt) + lambda*ice_softness(cell,side)*std::pow(Nval,n) );
                beta(cell,side,ipt) = muCoulombField(cell,side,ipt) * Nval * std::pow( q, power) / u_norm(cell,side,ipt);
              }
            } else {
              for (unsigned int ipt=0; ipt<dim; ++ipt) {
                ScalarT Nval = std::max(N(cell,side,ipt),0.0);
                ScalarT q = u_norm(cell,side,ipt) / ( u_norm(cell,side,ipt) + lambda*ice_softness(cell,side)*std::pow(Nval,n) );
                beta(cell,side,ipt) = mu * Nval * std::pow( q, power) / u_norm(cell,side,ipt);
              }
            }
          }
          break;

        case EXP_GIVEN_FIELD:
          if(nodal || interpolate_then_exponentiate) {
            if (is_given_field_param) {
              for (unsigned int ipt=0; ipt<dim; ++ipt)
                beta(cell,side,ipt) = std::exp(given_field_param(cell,side,ipt));
            } else {
              for (unsigned int ipt=0; ipt<dim; ++ipt)
                beta(cell,side,ipt) = std::exp(given_field(cell,side,ipt));
            }
          } else {
            if (is_given_field_param) {
              for (unsigned int qp=0; qp<numQPs; ++qp) {
                beta(cell,side,qp) = 0;
                for (unsigned int node=0; node<numNodes; ++node)
                  beta(cell,side,qp) += std::exp(given_field_param(cell,side,node))*BF(cell,side,node,qp);
              }
            } else {
              for (unsigned int qp=0; qp<numQPs; ++qp) {
                beta(cell,side,qp) = 0;
                  for (unsigned int node=0; node<numNodes; ++node)
                    beta(cell,side,qp) += std::exp(given_field(cell,side,node))*BF(cell,side,node,qp);
              }
            }
          }
          break;

        default:
          break;
      }

      if(zero_on_floating) {
        if (is_thickness_param) {
          for (unsigned int ipt=0; ipt<dim; ++ipt) {
            ParamScalarT isGrounded = rho_i*thickness_param_field(cell,side,ipt) > -rho_w*bed_topo_field(cell,side,ipt);
            beta(cell,side,ipt) *=  isGrounded;
          }
        } else {
          for (unsigned int ipt=0; ipt<dim; ++ipt) {
            ParamScalarT isGrounded = rho_i*thickness_field(cell,side,ipt) > -rho_w*bed_topo_field(cell,side,ipt);
            beta(cell,side,ipt) *=  isGrounded;
          }
        }
      }

      // Correct the value if we are using a stereographic map
      if (use_stereographic_map) {
        for (unsigned int ipt=0; ipt<dim; ++ipt) {
          MeshScalarT x = coordVec(cell,side,ipt,0) - x_0;
          MeshScalarT y = coordVec(cell,side,ipt,1) - y_0;
          MeshScalarT h = 4.0*R2/(4.0*R2 + x*x + y*y);
          beta(cell,side,ipt) *= h*h;
        }
      }
    }
  }
  
}

template<typename EvalT, typename Traits, typename EffPressureST, typename VelocityST, typename TemperatureST>
void BasalFrictionCoefficient<EvalT, Traits, EffPressureST, VelocityST, TemperatureST>::
evaluateFieldsCell (typename Traits::EvalData workset, ScalarT mu, ScalarT lambda, ScalarT power)
{
  dim = nodal ? numNodes : numQPs;
  switch (beta_type)
  {
    case GIVEN_CONSTANT:
      break;   // We don't have anything to do

    case GIVEN_FIELD:
      if (is_given_field_param) {
        Kokkos::parallel_for(GivenFieldParam_Policy(0, workset.numCells), *this);
      } else {
        Kokkos::parallel_for(GivenField_Policy(0, workset.numCells), *this);
      }
      break;

    case POWER_LAW:
      if (distributedMu) {
        Kokkos::parallel_for(PowerLaw_DistributedMu_Policy(0, workset.numCells), *this);
      } else {
        Kokkos::parallel_for(PowerLaw_Policy(0, workset.numCells), *this);
      }
      break;

    case REGULARIZED_COULOMB:
      if (distributedLambda) {
        if (distributedMu) {
          Kokkos::parallel_for(RegularizedCoulomb_DistributedLambda_DistributedMu_Policy(0, workset.numCells), *this);
        } else {
          Kokkos::parallel_for(RegularizedCoulomb_DistributedLambda_Policy(0, workset.numCells), *this);
        }
      } else {
        if (distributedMu) {
          Kokkos::parallel_for(RegularizedCoulomb_DistributedMu_Policy(0, workset.numCells), *this);
        } else {
          Kokkos::parallel_for(RegularizedCoulomb_Policy(0, workset.numCells), *this);
        }
      }
      break;

    case EXP_GIVEN_FIELD:
      if(nodal || interpolate_then_exponentiate) {
        if (is_given_field_param) {
          Kokkos::parallel_for(ExpGivenFieldParam_Nodal_Policy(0, workset.numCells), *this);
        } else {
          Kokkos::parallel_for(ExpGivenField_Nodal_Policy(0, workset.numCells), *this);
        } 
      } else {
        if (is_given_field_param) {
          Kokkos::parallel_for(ExpGivenFieldParam_Policy(0, workset.numCells), *this);
        } else {
          Kokkos::parallel_for(ExpGivenField_Policy(0, workset.numCells), *this);
        }
      }
      break;
  }

  // Correct the value if we are using a stereographic map
  if (use_stereographic_map)
  {
    Kokkos::parallel_for(StereographicMapCorrection_Policy(0, workset.numCells), *this);
  }
}

} // Namespace LandIce
