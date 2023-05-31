//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_FancyOStream.hpp"

#include "PHAL_FactoryTraits.hpp"
#include "Albany_Utils.hpp"
#include "Albany_BCUtils.hpp"
#include "LandIce_StokesFOHydrology.hpp"

namespace LandIce {

StokesFOHydrology::
StokesFOHydrology (const Teuchos::RCP<Teuchos::ParameterList>& params_,
                   const Teuchos::RCP<Teuchos::ParameterList>& discParams_,
                   const Teuchos::RCP<ParamLib>& paramLib_,
                   const int numDim_)
 : StokesFOBase(params_, discParams_, paramLib_, numDim_)
{
  // Figure out what kind of hydro problem we solve
  eliminate_h = params->sublist("LandIce Hydrology").get<bool>("Eliminate Water Thickness", false);
  has_h_till  = params->sublist("LandIce Hydrology").get<double>("Maximum Till Water Storage",0.0) > 0.0;
  has_p_dot   = params->sublist("LandIce Hydrology").get<double>("Englacial Porosity",0.0) > 0.0;

  std::string sol_method = params->get<std::string>("Solution Method");
  if (sol_method=="Transient") {
    unsteady = true;
  } else {
    unsteady = false;
    hydro_ndofs_dot = 0;
  }

  TEUCHOS_TEST_FOR_EXCEPTION (eliminate_h && unsteady, std::logic_error,
                              "Error! Water Thickness can be eliminated only in the steady case.\n");
  TEUCHOS_TEST_FOR_EXCEPTION (has_h_till && !unsteady, std::logic_error,
                              "Error! Till Water Storage equation only makes sense in the unsteady case.\n");
  TEUCHOS_TEST_FOR_EXCEPTION (has_p_dot && !unsteady, std::logic_error,
                              "Error! Englacial porosity model only makes sense in the unsteady case.\n");

  // Fill the variables names
  auto& vnames = params->sublist("Variables Names");

  water_pressure_name         = vnames.get<std::string>("Water Pressure Name", "water_pressure");
  water_thickness_name        = vnames.get<std::string>("Water Thickness Name", "water_thickness");
  till_water_storage_name     = vnames.get<std::string>("Till Water Storage Name", "till_water_storage");
  water_pressure_dot_name     = vnames.get<std::string>("Water Pressure Dot Name", "water_pressure_dot");
  water_thickness_dot_name    = vnames.get<std::string>("Water Thickness Dot Name", "water_thickness_dot");
  till_water_storage_dot_name = vnames.get<std::string>("Till Water Storage Dot Name", "till_water_storage_dot");

  hydropotential_name       = vnames.get<std::string>("Hydraulic Potential Name", "hydropotential");
  ice_overburden_name       = vnames.get<std::string>("Ice Overburden Name", "ice_overburden");
  beta_name                 = vnames.get<std::string>("Beta Name", "beta");
  melting_rate_name         = vnames.get<std::string>("Melting Rate Name", "melting_rate");
  surface_water_input_name  = vnames.get<std::string>("Surface Water Input Name", "surface_water_input");
  surface_mass_balance_name = vnames.get<std::string>("Surface Mass Balance Name", "surface_mass_balance");
  geothermal_flux_name      = vnames.get<std::string>("Geothermal Flux Name", "geothermal_flux");
  water_discharge_name      = vnames.get<std::string>("Water Discharge Name", "water_discharge");
  sliding_velocity_name     = vnames.get<std::string>("Sliding Velocity Name", "sliding_velocity");
  grav_hydropotential_name  = vnames.get<std::string>("Basal Gravitational Water Potential Name", "basal_gravitational_water_potential");

  // Set the num PDEs depending on the problem specs
  if (eliminate_h) {
    hydro_neq = 1;
  } else if (has_h_till) {
    hydro_neq = 3;
  } else {
    hydro_neq = 2;
  }
  stokes_neq = vecDimFO;
  stokes_ndofs = 1;
  hydro_ndofs = hydro_neq;
  stokes_dof_offset = 0;
  hydro_dof_offset = stokes_neq;

  this->setNumEquations(hydro_neq + stokes_neq);
  rigidBodyModes->setParameters(neq, computeConstantModes, vecDimFO, computeRotationModes);

  // Copy all dof_blah arrays into stokes_blah ones.
  // (so far, dof_blah contains only data relative to stokes)
  stokes_dofs_names.resize(stokes_ndofs);
  stokes_resids_names.resize(stokes_neq);
  stokes_dofs_names.deepCopy(dof_names());
  stokes_resids_names.deepCopy(resid_names());

  // Now resize global dof/resid/scatter/offsets arrays
  dof_names.resize(stokes_ndofs+hydro_ndofs);
  resid_names.resize(stokes_neq+hydro_neq);
  dof_offsets.resize(stokes_ndofs+hydro_ndofs);

  // Hydro-specific arrays
  hydro_dofs_names.resize(hydro_ndofs);
  hydro_resids_names.resize(hydro_neq);

  // We always solve for the water pressure
  hydro_dofs_names[0]   = dof_names[stokes_ndofs] = basal_fname(water_pressure_name);
  hydro_resids_names[0] = resid_names[stokes_ndofs] = "Residual Mass Eqn";

  if (!eliminate_h) {
    hydro_dofs_names[1]   = dof_names[stokes_ndofs+1] = basal_fname(water_thickness_name);
    hydro_resids_names[1] = resid_names[stokes_ndofs+1] = "Residual Cavities Eqn";
  }

  if (has_h_till) {
    hydro_dofs_names[2]   = dof_names[stokes_ndofs+2] = basal_fname(till_water_storage_name);
    hydro_resids_names[2] = resid_names[stokes_ndofs+2] = "Residual Till Storage Eqn";
  }

  // A single scatter op for hydrology (may have multiple resid fields though).
  scatter_names.resize(2);
  scatter_names[1] = "Scatter Hydrology";

  // Figure out which dofs appear under time derivative
  if (unsteady) {
    auto& hy = params->sublist("LandIce Hydrology");
    auto& cav = hy.sublist("Cavities Equation");
    if (cav.get<double>("Englacial Porosity")>0.0) {
      hydro_dofs_dot_names.resize(2);
      hydro_dofs_dot_names[0] = basal_fname(water_pressure_name) + "_dot";
      hydro_dofs_dot_names[1] = basal_fname(water_thickness_name) + "_dot";
      hydro_dof_dot_offset = hydro_dof_offset;
      hydro_ndofs_dot = 2;
    } else {
      hydro_dofs_dot_names.resize(1);
      hydro_dofs_dot_names[0] = basal_fname(water_thickness_name) + "_dot";
      hydro_dof_dot_offset = hydro_dof_offset + 1;
      hydro_ndofs_dot = 1;
    }
    if (has_h_till) {
      hydro_dofs_dot_names.resize(hydro_ndofs_dot+1);
      hydro_dofs_dot_names[hydro_ndofs_dot] = basal_fname(till_water_storage_name) + "_dot";
      ++hydro_ndofs_dot;
    }
  }

  // Set the hydrology equations as side set equations on the basal side
  for (unsigned int eq=stokes_neq; eq<neq; ++eq) {
    this->sideSetEquations[eq].push_back(basalSideName);
  }
}

Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
StokesFOHydrology::
buildEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                 const Albany::MeshSpecs& meshSpecs,
                 Albany::StateManager& stateMgr,
                 Albany::FieldManagerChoice fmchoice,
                 const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructeEvaluators<EvalT>(*fm0, *meshSpecs, stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  Albany::ConstructEvaluatorsOp<StokesFOHydrology> op(
    *this, fm0, meshSpecs, stateMgr, fmchoice, responseList);
  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);
  return *op.tags;
}

void
StokesFOHydrology::buildFields(PHX::FieldManager<PHAL::AlbanyTraits>& fm0)
{
  // Allocate memory for unmanaged fields
  fieldUtils = Teuchos::rcp(new Albany::FieldUtils(fm0, dl));
  buildStokesFOBaseFields(fm0);

  // Call constructFields<EvalT>() for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  Albany::ConstructFieldsOp<StokesFOHydrology> op(*this, fm0);
  Sacado::mpl::for_each_no_kokkos<PHAL::AlbanyTraits::BEvalTypes> fe(op);
}

void StokesFOHydrology::
constructDirichletEvaluators(const Albany::MeshSpecs& meshSpecs)
{
  // Construct Dirichlet evaluators for all nodesets and names
  std::vector<std::string> dir_names(stokes_neq+hydro_neq);
  for (int i=0; i<stokes_neq; i++) {
    std::stringstream s; s << "U" << i;
    dir_names[i] = s.str();
  }
  for (int i=0; i<hydro_neq; ++i) {
    dir_names[stokes_neq+i] = hydro_dofs_names[i];
  }

  Albany::BCUtils<Albany::DirichletTraits> dirUtils;
  dfm = dirUtils.constructBCEvaluators(meshSpecs.nsNames, dir_names, this->params, this->paramLib, neq);
  use_sdbcs_ = dirUtils.useSDBCs(); 
  offsets_ = dirUtils.getOffsets();
  nodeSetIDs_ = dirUtils.getNodeSetIDs();
}

// Neumann BCs
void StokesFOHydrology::
constructNeumannEvaluators (const Teuchos::RCP<Albany::MeshSpecs>& meshSpecs)
{
  Albany::BCUtils<Albany::NeumannTraits> nbcUtils;

  // Check to make sure that Neumann BCs are given in the input file
  if(!nbcUtils.haveBCSpecified(this->params)) {
     return;
  }

  // Construct BC evaluators for all side sets and names
  // Note that the string index sets up the equation offset, so ordering is important
  // Also, note that we only have neumann conditions for the ice. Hydrology can also
  // have neumann BC, but they are homogeneous (do-nothing).

  // Stokes BCs
  std::vector<std::string> stokes_neumann_names(stokes_neq + 1);
  Teuchos::Array<Teuchos::Array<int> > stokes_offsets;
  stokes_offsets.resize(stokes_neq + 1);

  stokes_neumann_names[0] = "U0";
  stokes_offsets[0].resize(1);
  stokes_offsets[0][0] = 0;
  stokes_offsets[stokes_neq].resize(stokes_neq);
  stokes_offsets[stokes_neq][0] = 0;

  if (neq>1) {
    stokes_neumann_names[1] = "U1";
    stokes_offsets[1].resize(1);
    stokes_offsets[1][0] = 1;
    stokes_offsets[stokes_neq][1] = 1;
  }

  stokes_neumann_names[stokes_neq] = "all";

  std::vector<std::string> stokes_cond_names(1);
  stokes_cond_names[0] = "lateral";

  nfm = nbcUtils.constructBCEvaluators(meshSpecs, stokes_neumann_names, stokes_dofs_names, true, 0,
                                       stokes_cond_names, stokes_offsets, dl,
                                       this->params, this->paramLib);
}

Teuchos::RCP<const Teuchos::ParameterList>
StokesFOHydrology::getValidProblemParameters () const
{
  auto validPL = StokesFOBase::getStokesFOBaseProblemParameters();

  validPL->sublist("LandIce Hydrology", false, "");
  validPL->sublist("LandIce Field Norm", false, "");
  validPL->sublist("LandIce Viscosity", false, "");
  validPL->sublist("LandIce Physical Parameters", false, "");
  validPL->sublist("LandIce Basal Friction Coefficient", false, "Parameters needed to compute the basal friction coefficient");

  return validPL;
}

void StokesFOHydrology::setFieldsProperties () {
  StokesFOBase::setFieldsProperties();

  // Set scalar type of hydro dofs (and dot dofs) to ScalarT
  for (const auto& dof : hydro_dofs_names) {
    setSingleFieldProperties(dof, FRT::Scalar, FST::Scalar);
  }
  for (const auto& dof : hydro_dofs_dot_names) {
    setSingleFieldProperties(dof, FRT::Scalar, FST::Scalar);
  }

  // Set dof's properties
  setSingleFieldProperties(water_pressure_name,      FRT::Scalar,   FST::Scalar);
  setSingleFieldProperties(water_thickness_name,     FRT::Scalar,   FST::Scalar);
  setSingleFieldProperties(till_water_storage_name,  FRT::Scalar,   FST::Scalar);

  setSingleFieldProperties(effective_pressure_name,  FRT::Scalar,   FST::Scalar);
  setSingleFieldProperties(water_discharge_name,     FRT::Gradient, FST::Scalar);
  setSingleFieldProperties(hydropotential_name,      FRT::Scalar,   FST::Scalar);
  setSingleFieldProperties(surface_water_input_name, FRT::Scalar);
}

void StokesFOHydrology::setupEvaluatorRequests () {
  StokesFOBase::setupEvaluatorRequests();

  ss_build_interp_ev[basalSideName][water_pressure_name][IReq::QP_VAL] = true; 
  if (!eliminate_h) {
    // If we eliminate h, then we compute water thickness, rather than interpolate the dof
    ss_build_interp_ev[basalSideName][water_thickness_name][IReq::QP_VAL] = true; 
  }
  ss_build_interp_ev[basalSideName][hydropotential_name][IReq::GRAD_QP_VAL] = true; 
  ss_build_interp_ev[basalSideName][water_discharge_name][IReq::CELL_VAL] = true; 
  ss_build_interp_ev[basalSideName][surface_water_input_name][IReq::QP_VAL] = true; 
  ss_build_interp_ev[basalSideName][flow_factor_name][IReq::CELL_TO_SIDE] = true; 
  ss_build_interp_ev[basalSideName][ice_thickness_name][IReq::QP_VAL] = true; 

  ss_utils_needed[basalSideName][UtilityRequest::BFS] = true;
}

} // namespace LandIce
