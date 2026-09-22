//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "LandIce_StokesFOThickness.hpp"

#include "PHAL_FactoryTraits.hpp"
#include "Albany_BCUtils.hpp"
#include "Albany_StringUtils.hpp" // for 'upper_case'
#include "Albany_ThyraUtils.hpp"
#include "Albany_CombineAndScatterManager.hpp"

#include "Thyra_VectorStdOps.hpp"

#include <algorithm>

// Uncomment for some setup output
#define OUTPUT_TO_SCREEN

namespace LandIce {

StokesFOThickness::StokesFOThickness(
            const Teuchos::RCP<Teuchos::ParameterList>& params_,
            const Teuchos::RCP<Teuchos::ParameterList>& discParams_,
            const Teuchos::RCP<ParamLib>& paramLib_,
            const int numDim_) :
  StokesFOBase(params_, discParams_, paramLib_, numDim_)
{
  //Set # of PDEs per node.
  std::string eqnSet = params_->sublist("Equation Set").get<std::string>("Type", "LandIce");
  neq = 3; //LandIce FO Stokes system is a system of 2 PDEs; add one for the thickness.

  // Set the num PDEs for the null space object to pass to ML
  // this->rigidBodyModes->setNumPDEs(neq);

  TEUCHOS_TEST_FOR_EXCEPTION (surfaceSideName=="__INVALID__", std::runtime_error,
    "Error! StokesFOThickness requires a valid surfaceSideName, since the thickness equation is solved on the surface.\n");
  // Defining the thickness equation only in 2D (basal side)

  std::string sol_method = params->get<std::string>("Solution Method");
  if (sol_method=="Transient") {
    unsteady = true;
  } else {
    unsteady = false;
  }

  sideSetEquations[2].push_back(surfaceSideName);

  dof_names.resize(2);
  dof_names[1] = ice_thickness_name + "_change";
  if(unsteady) {
    dof_names_dot.resize(2);
    dof_names_dot[0] = dof_names[0] + "_dot";
    dof_names_dot[1] = ice_thickness_name + "_dot";
  }

  resid_names.resize(2);
  resid_names[1] = dof_names[1] + " Residual";

  scatter_names.resize(2);
  scatter_names[1] = "Scatter " + resid_names[1];

  dof_offsets.resize(2,0);
  dof_offsets[1] = vecDimFO;

  // We have two values for ice_thickness: the initial one, and the updated one.
  initial_ice_thickness_name = ice_thickness_name;

  if(unsteady) {
    surface_height_name += "_computed";
    ice_thickness_name += "_computed";
  }
  lateralSideName = this->params->get<std::string>("Lateral Side Name", "lateralside");

  effectivePressure_from_basalFrictionEval = true;
}

Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
StokesFOThickness::buildEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fmchoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes

  Albany::ConstructEvaluatorsOp<StokesFOThickness> op(
    *this, fm0, meshSpecs, stateMgr, fmchoice, responseList);

  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);

  return *op.tags;
}

void
StokesFOThickness::buildFields(PHX::FieldManager<PHAL::AlbanyTraits>& fm0)
{
  // Allocate memory for unmanaged fields
  fieldUtils = Teuchos::rcp(new Albany::FieldUtils(fm0, dl));
  buildStokesFOBaseFields();

  // Call constructFields<EvalT>() for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  Albany::ConstructFieldsOp<StokesFOThickness> op(*this, fm0);
  Sacado::mpl::for_each_no_kokkos<PHAL::AlbanyTraits::BEvalTypes> fe(op);
}

void StokesFOThickness::constructDirichletEvaluators(const Albany::MeshSpecsStruct& meshSpecs)
{
   // Construct Dirichlet evaluators for all nodesets and names
   std::vector<std::string> dirichletNames(neq);
   for (size_t i=0; i<neq-1; i++) {
     std::stringstream s; s << "U" << i;
     dirichletNames[i] = s.str();
   }
   dirichletNames[neq-1] = "H";
   Albany::BCUtils<Albany::DirichletTraits> dirUtils;
   dfm = dirUtils.constructBCEvaluators(meshSpecs.nsNames, dirichletNames,
                                          this->params, this->paramLib);
   use_sdbcs_ = dirUtils.useSDBCs();
   offsets_ = dirUtils.getOffsets();
   nodeSetIDs_ = dirUtils.getNodeSetIDs();
}

// Neumann BCs
void
LandIce::StokesFOThickness::constructNeumannEvaluators(const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs)
{

   // Note: we only enter this function if sidesets are defined in the mesh file
   // i.e. meshSpecs.ssNames.size() > 0

   Albany::BCUtils<Albany::NeumannTraits> nbcUtils;

   // Check to make sure that Neumann BCs are given in the input file

   if(!nbcUtils.haveBCSpecified(this->params)) {
      return;
   }


   // Construct BC evaluators for all side sets and names
   // Note that the string index sets up the equation offset, so ordering is important

   std::vector<std::string> neumannNames(neq + 1);
   Teuchos::Array<Teuchos::Array<int> > offsets;
   offsets.resize(neq + 1);

   neumannNames[0] = "U0";
   offsets[0].resize(1);
   offsets[0][0] = 0;
   offsets[neq].resize(neq-1);
   offsets[neq][0] = 0;

   if (neq-1>1){
      neumannNames[1] = "U1";
      offsets[1].resize(1);
      offsets[1][0] = 1;
      offsets[neq][1] = 1;
   }

   if (neq-1>2){
     neumannNames[2] = "U2";
      offsets[2].resize(1);
      offsets[2][0] = 2;
      offsets[neq][2] = 2;
   }

   neumannNames[neq-1] = "H";
   offsets[neq-1].resize(1);
   offsets[neq-1][0] = neq-1;

   neumannNames[neq] = "all";

   // Construct BC evaluators for all possible names of conditions
   // Should only specify flux vector components (dCdx, dCdy, dCdz), or dCdn, not both
   std::vector<std::string> condNames(3); //(dCdx, dCdy, dCdz), dCdn, P

   // Note that sidesets are only supported for two and 3D currently
   if(numDim == 2)
    condNames[0] = "(dFluxdx, dFluxdy)";
   else if(numDim == 3)
    condNames[0] = "(dFluxdx, dFluxdy, dFluxdz)";
   else
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
       std::endl << "Error: Sidesets only supported in 2 and 3D." << std::endl);

   condNames[1] = "dFluxdn";
   condNames[2] = "P";

   nfm.resize(1); // LandIce problem only has one element block

   nfm[0] = nbcUtils.constructBCEvaluators(meshSpecs, neumannNames, dof_names, true, 0,
                                          condNames, offsets, dl,
                                          this->params, this->paramLib);
}

Teuchos::RCP<const Teuchos::ParameterList>
StokesFOThickness::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = StokesFOBase::getStokesFOBaseProblemParameters();

  validPL->sublist("Equation Set", false, "");
  validPL->sublist("Body Force", false, "");
  validPL->set<bool>("Allow Loss Of Derivative Terms", false, "Allow loss of derivative terms in mesh coordinates");
  validPL->set<double>("Time Step", 1.0, "Time step for divergence flux (Not used for time integation)");
  validPL->set<Teuchos::RCP<double> >("Time Step Ptr", Teuchos::null, "Time step ptr for divergence flux (Not used for time integation)");
  validPL->set<std::string>("Thickness Stabilization", "SUPG", "Stabilization for the thickness equation");
  validPL->set<bool>("Lump Time Derivative Mass Matrix", false, "Whether to Lump the Mass Matrix for Time Derivative");
  validPL->set<std::string>("Lateral Side Name", "lateral side", "Lateral Side Set Name");

  return validPL;
}

bool StokesFOThickness::
getDAEMasks (const Albany::AbstractDiscretization& disc,
             Teuchos::RCP<Thyra_Vector>& diagnostic_mask,
             Teuchos::RCP<Thyra_Vector>& prognostic_mask) const
{
  if (!unsteady) {
    // Steady runs (including the quasi-static thickness update used in coupled runs)
    // have no time derivative at all: there is nothing to integrate.
    return false;
  }

  const auto  dof_mgr       = disc.getDOFManager();
  const auto  elem_dof_lids = dof_mgr->elem_dof_lids().host();
  const auto& layers_data   = disc.getMeshStruct()->layers_data;
  const int   thk_eq        = dof_offsets[1];

  // Build the masks on the overlapped space (element-based loops), then combine
  auto ov_vs   = disc.getOverlapVectorSpace();
  auto ov_diag = Thyra::createMember(ov_vs);
  auto ov_prog = Thyra::createMember(ov_vs);
  auto ov_dbc  = Thyra::createMember(ov_vs);
  ov_diag->assign(0.0);
  ov_prog->assign(0.0);
  ov_dbc->assign(0.0);
  auto diag_data = Albany::getNonconstLocalData(*ov_diag);
  auto prog_data = Albany::getNonconstLocalData(*ov_prog);
  auto dbc_data  = Albany::getNonconstLocalData(*ov_dbc);

  // 1. Diagnostic (algebraic) dofs: all velocity components, at all nodes
  const int num_elems = dof_mgr->cell_indexer()->getNumLocalElements();
  for (int eq=dof_offsets[0]; eq<thk_eq; ++eq) {
    const auto& offsets = dof_mgr->getGIDFieldOffsets(eq);
    for (int ielem=0; ielem<num_elems; ++ielem) {
      for (auto o : offsets) {
        diag_data[elem_dof_lids(ielem,o)] = 1.0;
      }
    }
  }

  // 2. Prognostic (differential) dofs: the thickness change at the level where its
  //    equation is scattered (same logic as PHAL::ScatterResidual2D, with
  //    'Field Level' = NumLayers). The thickness dofs at the other levels are copies.
  const int field_level = discParams->get<int>("NumLayers");
  const int field_layer = field_level==0 ? 0 : field_level-1;
  const int field_pos   = field_layer==field_level ? layers_data.bot_side_pos : layers_data.top_side_pos;
  const auto& thk_side_offsets = dof_mgr->getGIDFieldOffsetsSide(thk_eq,field_pos);
  for (int ws=0; ws<disc.getNumWorksets(); ++ws) {
    const auto& ssList = disc.getSideSets(ws);
    const auto it_ss = ssList.find(surfaceSideName);
    if (it_ss==ssList.end()) {
      continue;
    }
    const auto elem_lids = disc.getElementLIDs_host(ws);
    for (const auto& side : it_ss->second) {
      const int elem_LID       = elem_lids(side.ws_elem_idx);
      const int basal_elem_LID = layers_data.cell.lid->getColumnId(elem_LID);
      const int field_elem_LID = layers_data.cell.lid->getId(basal_elem_LID,field_layer);
      for (auto o : thk_side_offsets) {
        prog_data[elem_dof_lids(field_elem_LID,o)] = 1.0;
      }
    }
  }

  // 3. Thickness dofs with a Dirichlet condition are prescribed, not integrated
  const auto& node_sets   = disc.getNodeSets();
  const auto& thk_offsets = dof_mgr->getGIDFieldOffsets(thk_eq);
  for (size_t ins=0; ins<nodeSetIDs_.size(); ++ins) {
    const auto& ns_eqs = offsets_[ins];
    if (std::find(ns_eqs.begin(),ns_eqs.end(),thk_eq)==ns_eqs.end()) {
      continue;
    }
    const auto it_ns = node_sets.find(nodeSetIDs_[ins]);
    if (it_ns==node_sets.end()) {
      continue;
    }
    for (const auto& ep : it_ns->second) {
      dbc_data[elem_dof_lids(ep.first,thk_offsets[ep.second])] = 1.0;
    }
  }

  // Overlapped -> owned
  auto vs  = disc.getVectorSpace();
  auto cas = Albany::createCombineAndScatterManager(vs,ov_vs);
  diagnostic_mask = Thyra::createMember(vs);
  prognostic_mask = Thyra::createMember(vs);
  auto dbc        = Thyra::createMember(vs);
  diagnostic_mask->assign(0.0);
  prognostic_mask->assign(0.0);
  dbc->assign(0.0);
  cas->combine(*ov_diag,*diagnostic_mask, Albany::CombineMode::ABSMAX);
  cas->combine(*ov_prog,*prognostic_mask, Albany::CombineMode::ABSMAX);
  cas->combine(*ov_dbc, *dbc,             Albany::CombineMode::ABSMAX);

  auto p_data = Albany::getNonconstLocalData(*prognostic_mask);
  auto b_data = Albany::getLocalData(*dbc);
  for (int i=0; i<static_cast<int>(p_data.size()); ++i) {
    if (b_data[i]!=0.0) {
      p_data[i] = 0.0;
    }
  }

  return true;
}

void StokesFOThickness::setFieldsProperties () {
  StokesFOBase::setFieldsProperties();

  if(unsteady) {
    setSingleFieldProperties(ice_thickness_name, FRT::Scalar, FST::Scalar);
    setSingleFieldProperties(effective_pressure_name, FRT::Scalar, FST::Scalar);
    setSingleFieldProperties(surface_height_name, FRT::Scalar, FST::Scalar);
  } else {
    setSingleFieldProperties(surface_height_name, FRT::Scalar, FST::ParamScalar);
  }
  setSingleFieldProperties(initial_ice_thickness_name, FRT::Scalar, FST::ParamScalar);  
}

} // namespace LandIce
