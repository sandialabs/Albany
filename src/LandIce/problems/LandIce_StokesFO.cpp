//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <string>

#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "Teuchos_FancyOStream.hpp"

#include "PHAL_FactoryTraits.hpp"
#include "Albany_Utils.hpp"
#include "Albany_BCUtils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "LandIce_StokesFO.hpp"

// Uncomment for some setup output
// #define OUTPUT_TO_SCREEN

namespace LandIce {

StokesFO::
StokesFO( const Teuchos::RCP<Teuchos::ParameterList>& params_,
          const Teuchos::RCP<Teuchos::ParameterList>& discParams_,
          const Teuchos::RCP<ParamLib>& paramLib_,
          const int numDim_) :
  StokesFOBase(params_, discParams_, paramLib_, numDim_)
{
  //Set # of PDEs per node based on the Equation Set.
  //Equation Set is LandIce by default (2 dofs / node -- usual LandIce Stokes FO).
  std::string eqnSet = params_->sublist("Equation Set").get<std::string>("Type", "LandIce");
  if (eqnSet == "LandIce")
    neq = 2; //LandIce FO Stokes system is a system of 2 PDEs
  else if (eqnSet == "Poisson" || eqnSet == "LandIce X-Z")
    neq = 1; //1 PDE/node for Poisson or LandIce X-Z physics

  neq =  params_->sublist("Equation Set").get<int>("Num Equations", neq);

  // Set parameters for passing coords/near-null space to preconditioner
  rigidBodyModes->setParameters(neq, computeConstantModes, vecDimFO, computeRotationModes);

  adjustSurfaceHeight = false;
  adjustBedTopo = false;
}

Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
StokesFO::buildEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fmchoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  Albany::ConstructEvaluatorsOp<StokesFO> op(
    *this, fm0, meshSpecs, stateMgr, fmchoice, responseList);
  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);
  return *op.tags;
}

void
StokesFO::buildFields(PHX::FieldManager<PHAL::AlbanyTraits>& fm0)
{
  // Allocate memory for unmanaged fields
  fieldUtils = Teuchos::rcp(new Albany::FieldUtils(fm0, dl));
  buildStokesFOBaseFields(fm0);

  // Call constructFields<EvalT>() for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  Albany::ConstructFieldsOp<StokesFO> op(*this, fm0);
  Sacado::mpl::for_each_no_kokkos<PHAL::AlbanyTraits::BEvalTypes> fe(op);
}

void StokesFO::constructDirichletEvaluators(
        const Albany::MeshSpecsStruct& meshSpecs)
{
   // Construct Dirichlet evaluators for all nodesets and names
   std::vector<std::string> dirichletNames(neq);
   for (unsigned int i=0; i<vecDimFO; i++) {
     std::stringstream s; s << "U" << i;
     dirichletNames[i] = s.str();
   }
   if(static_cast<unsigned>(vecDimFO) < neq)
     dirichletNames[vecDimFO] = "Lapl_L2Proj";
   Albany::BCUtils<Albany::DirichletTraits> dirUtils;
   dfm = dirUtils.constructBCEvaluators(meshSpecs.nsNames, dirichletNames,
                                          this->params, this->paramLib);
   use_sdbcs_ = dirUtils.useSDBCs();
   offsets_ = dirUtils.getOffsets();
   nodeSetIDs_ = dirUtils.getNodeSetIDs();
}

// Neumann BCs
void StokesFO::constructNeumannEvaluators (const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs)
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
   offsets[neq].resize(neq);
   offsets[neq][0] = 0;

   if (vecDimFO>1){
     neumannNames[1] = "U1";
     offsets[1].resize(1);
     offsets[1][0] = 1;
     offsets[neq][1] = 1;
   }

   if (vecDimFO>2){
     neumannNames[2] = "U2";
     offsets[2].resize(1);
     offsets[2][0] = 2;
     offsets[neq][2] = 2;
   }

   neumannNames[neq] = "all";

   // Construct BC evaluators for all possible names of conditions
   // Should only specify flux vector components (dCdx, dCdy, dCdz), or dCdn, not both
   std::vector<std::string> condNames(5); //(dCdx, dCdy, dCdz), dCdn, P, robin, natural

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
   condNames[3] = "robin";
   condNames[4] = "neumann"; // This string could be anything, since it is caught by the last 'else' in PHAL_Neumann

   // All nbc refer to the same dof (the velocity)
   Teuchos::ArrayRCP<std::string> nbc_dof_names(neq+1,dof_names[0]);

   nfm.resize(1); // LandIce problem only has one element block

   nfm[0] = nbcUtils.constructBCEvaluators(meshSpecs, neumannNames, dof_names, true, 0,
                                           condNames, offsets, dl,
                                           this->params, this->paramLib);
}

Teuchos::RCP<const Teuchos::ParameterList>
StokesFO::getValidProblemParameters () const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL = StokesFOBase::getStokesFOBaseProblemParameters();

  validPL->sublist("LandIce L2 Projected Boundary Laplacian", false, "Parameters needed to compute the L2 Projected Boundary Laplacian");
  validPL->sublist("Equation Set", false, "");
  validPL->set<bool>("Adjust Bed Topography to Account for Thickness Changes", false, "");
  validPL->set<bool>("Adjust Surface Height to Account for Thickness Changes", false, "");

  return validPL;
}

void StokesFO::setFieldsProperties () {
  StokesFOBase::setFieldsProperties();

  if (Albany::mesh_depends_on_parameters() && is_dist_param[ice_thickness_name]) {
    adjustBedTopo = params->get("Adjust Bed Topography to Account for Thickness Changes", false);
    adjustSurfaceHeight = params->get("Adjust Surface Height to Account for Thickness Changes", false);
    TEUCHOS_TEST_FOR_EXCEPTION(adjustBedTopo == adjustSurfaceHeight, std::logic_error, "Error! When the ice thickness is a parameter,\n "
        "either 'Adjust Bed Topography to Account for Thickness Changes' or\n"
        " 'Adjust Surface Height to Account for Thickness Changes' needs to be true.\n");

  }

  // UpdateZCoordinate expects the (observed) bed topography and (observed) surface height to have scalar type MeshScalarT.
  setSingleFieldProperties("observed_bed_topography", FRT::Scalar);
  setSingleFieldProperties("observed_surface_height", FRT::Scalar);
}

void StokesFO::setupEvaluatorRequests () {
  StokesFOBase::setupEvaluatorRequests();

  // In addition to the StokesFOBase stuff, add the syntetic tests bc needs
  for (auto pl : landice_bcs[LandIceBC::SynteticTest]) {
    const std::string& ssName = pl->get<std::string>("Side Set Name");

    ss_build_interp_ev[ssName][dof_names[0]][InterpolationRequest::CELL_TO_SIDE] = true; 
    ss_build_interp_ev[ssName][dof_names[0]][InterpolationRequest::QP_VAL      ] = true; 
    ss_build_interp_ev[ssName][dof_names[0]][InterpolationRequest::GRAD_QP_VAL ] = true; 

    ss_utils_needed[ssName][UtilityRequest::BFS      ] = true;
    ss_utils_needed[ssName][UtilityRequest::QP_COORDS] = true;
    ss_utils_needed[ssName][UtilityRequest::NORMALS  ] = true;
  }
  if (viscosity_use_corrected_temperature) {
    build_interp_ev[surface_height_name][InterpolationRequest::CELL_VAL] = true;
  }

  if(neq > static_cast<unsigned>(vecDimFO)) {

    auto& proj_lapl_params = params->sublist("LandIce L2 Projected Boundary Laplacian");
    auto& ssName = proj_lapl_params.get<std::string>("Side Set Name",basalSideName);
    std::string field_name = proj_lapl_params.get<std::string>("Field Name","basal_friction");
    auto dof_name_auxiliary = "L2 Projected Boundary Laplacian";

    ss_build_interp_ev[ssName][dof_name_auxiliary][IReq::CELL_TO_SIDE] = true;
    ss_build_interp_ev[ssName][field_name][IReq::GRAD_QP_VAL] = true;
  }
}

} // namespace LandIce
