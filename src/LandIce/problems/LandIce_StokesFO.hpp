//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_STOKES_FO_PROBLEM_HPP
#define LANDICE_STOKES_FO_PROBLEM_HPP 1

#include <type_traits>

#include "Shards_CellTopology.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "Albany_GeneralPurposeFieldsNames.hpp"

#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_AlbanyTraits.hpp"

#include "PHAL_SaveSideSetStateField.hpp"
#include "PHAL_SaveStateField.hpp"

#include "LandIce_StokesFOBase.hpp"

#include "LandIce_StokesFOSynteticTestBC.hpp"
#include "LandIce_L2ProjectedBoundaryLaplacianResidual.hpp"

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN

namespace LandIce
{

/*!
 * \brief Abstract interface for representing a 1-D finite element
 * problem.
 */
class StokesFO : public StokesFOBase
{
public:

  //! Default constructor
  StokesFO (const Teuchos::RCP<Teuchos::ParameterList>& params,
            const Teuchos::RCP<Teuchos::ParameterList>& discParams,
            const Teuchos::RCP<ParamLib>& paramLib,
            const int numDim_);

  //! Destructor
  ~StokesFO();

  // Build evaluators
  virtual Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
  buildEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                   const Albany::MeshSpecsStruct& meshSpecs,
                   Albany::StateManager& stateMgr,
                   Albany::FieldManagerChoice fmchoice,
                   const Teuchos::RCP<Teuchos::ParameterList>& responseList);

  //! Each problem must generate it's list of valide parameters
  Teuchos::RCP<const Teuchos::ParameterList> getValidProblemParameters() const;

private:

  //! Private to prohibit copying
  StokesFO(const StokesFO&);

  //! Private to prohibit copying
  StokesFO& operator=(const StokesFO&);

public:

  //! Main problem setup routines. 
  template <typename EvalT> Teuchos::RCP<const PHX::FieldTag>
  constructEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                       const Albany::MeshSpecsStruct& meshSpecs,
                       Albany::StateManager& stateMgr,
                       Albany::FieldManagerChoice fmchoice,
                       const Teuchos::RCP<Teuchos::ParameterList>& responseList);

  template <typename EvalT>
  void constructSynteticTestBCEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0);

  template <typename EvalT>
  void constructProjLaplEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                    Albany::FieldManagerChoice FieldManagerChoice,
                                    Teuchos::RCP<std::map<std::string, int> > extruded_params_levels);

protected:

  void constructDirichletEvaluators (const Albany::MeshSpecsStruct& meshSpecs);
  void constructNeumannEvaluators (const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs);
};

// ================================ IMPLEMENTATION ============================ //

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
StokesFO::constructEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                               const Albany::MeshSpecsStruct& meshSpecs,
                               Albany::StateManager& stateMgr,
                               Albany::FieldManagerChoice fieldManagerChoice,
                               const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  Teuchos::RCP<std::map<std::string, int> > extruded_params_levels = Teuchos::rcp(new std::map<std::string, int> ());
  std::map<std::string,bool> is_dist_param;
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  Teuchos::RCP<Teuchos::ParameterList> p;

  int offsetVelocity = 0;

  // --- States/parameters --- //
  constructStokesFOBaseEvaluators<EvalT> (fm0, meshSpecs, stateMgr, fieldManagerChoice, *extruded_params_levels, is_dist_param);

  // --- Geometry --- //

  // Gather solution field
  ev = evalUtils.constructGatherSolutionEvaluator_noTransient(true, dof_names[0], offsetVelocity);
  fm0.template registerEvaluator<EvalT> (ev);

  // Scatter residual
  ev = evalUtils.constructScatterResidualEvaluatorWithExtrudedParams(true, resid_names[0], extruded_params_levels, offsetVelocity, scatter_names[0]);
  fm0.template registerEvaluator<EvalT> (ev);

  // Gather thickness, depending on whether it is a parameter and on whether mesh depends on parameters
  if(!(is_dist_param["ice_thickness"]||is_dist_param["ice_thickness_param"]))
  {
    //----- Gather Coordinate Vector (general parameters)
    ev = evalUtils.constructGatherCoordinateVectorEvaluator();
    fm0.template registerEvaluator<EvalT> (ev);
  }
  else
  {
#ifndef ALBANY_MESH_DEPENDS_ON_PARAMETERS
    //----- Gather Coordinate Vector (general parameters)
    ev = evalUtils.constructGatherCoordinateVectorEvaluator();
    fm0.template registerEvaluator<EvalT> (ev);
#else
    bool adjustBedTopo = params->get("Adjust Bed Topography to Account for Thickness Changes", false);
    bool adjustSurfaceHeight = params->get("Adjust Surface Height to Account for Thickness Changes", false);

    if(adjustBedTopo && !adjustSurfaceHeight) {
      //----- Gather Coordinate Vector (ad hoc parameters)
      p = Teuchos::rcp(new Teuchos::ParameterList("Gather Coordinate Vector"));

      // Output:: Coordindate Vector at vertices
      p->set<std::string>("Coordinate Vector Name", "Coord Vec Old");

      ev = Teuchos::rcp(new PHAL::GatherCoordinateVector<EvalT,PHAL::AlbanyTraits>(*p,dl));
      fm0.template registerEvaluator<EvalT>(ev);

      //------ Update Z Coordinate
      p = Teuchos::rcp(new Teuchos::ParameterList("Update Z Coordinate"));

      p->set<std::string>("Old Coords Name",  "Coord Vec Old");
      p->set<std::string>("New Coords Name",  Albany::coord_vec_name);
      p->set<std::string>("Thickness Name",   "ice_thickness");
      p->set<std::string>("Thickness Lower Bound Name",   "ice_thickness_lowerbound");
      p->set<std::string>("Thickness Upper Bound Name",   "ice_thickness_upperbound");
      p->set<std::string>("Top Surface Name", "observed_surface_height");
      p->set<std::string>("Updated Top Surface Name", "surface_height");
      p->set<std::string>("Bed Topography Name", "observed_bed_topography");
      p->set<std::string>("Updated Bed Topography Name", "bed_topography");
      p->set<Teuchos::ParameterList*>("Physical Parameter List", &params->sublist("LandIce Physical Parameters"));

      ev = Teuchos::rcp(new LandIce::UpdateZCoordinateMovingBed<EvalT,PHAL::AlbanyTraits>(*p, dl));
      fm0.template registerEvaluator<EvalT>(ev);
    }
    else if(adjustSurfaceHeight && !adjustBedTopo) {
      //----- Gather Coordinate Vector (ad hoc parameters)
      p = Teuchos::rcp(new Teuchos::ParameterList("Gather Coordinate Vector"));

      // Output:: Coordindate Vector at vertices
      p->set<std::string>("Coordinate Vector Name", "Coord Vec Old");

      ev = Teuchos::rcp(new PHAL::GatherCoordinateVector<EvalT,PHAL::AlbanyTraits>(*p,dl));
      fm0.template registerEvaluator<EvalT>(ev);

      //------ Update Z Coordinate
      p = Teuchos::rcp(new Teuchos::ParameterList("Update Z Coordinate"));

      p->set<std::string>("Old Coords Name",  "Coord Vec Old");
      p->set<std::string>("New Coords Name",  Albany::coord_vec_name);
      p->set<std::string>("Thickness Name",   "ice_thickness");
      p->set<std::string>("Top Surface Name", "surface_height");
      p->set<std::string>("Bed Topography Name", "bed_topography");
      p->set<Teuchos::ParameterList*>("Physical Parameter List", &params->sublist("LandIce Physical Parameters"));

      ev = Teuchos::rcp(new LandIce::UpdateZCoordinateMovingTop<EvalT,PHAL::AlbanyTraits>(*p, dl));
      fm0.template registerEvaluator<EvalT>(ev);
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION(adjustBedTopo == adjustSurfaceHeight, std::logic_error, "Error! When the ice thickness is a parameter,\n "
          "either 'Adjust Bed Topography to Account for Thickness Changes' or\n"
          " 'Adjust Surface Height to Account for Thickness Changes' needs to be true.\n");
    }
#endif
  }

  // --- Syntetic test BC evaluators (if needed) --- //
  constructSynteticTestBCEvaluators<EvalT> (fm0);

  // --- ProjectedLaplacian-related evaluators (if needed) --- //
  constructProjLaplEvaluators<EvalT> (fm0, fieldManagerChoice, extruded_params_levels);

  // Finally, construct responses, and return the tags
  return constructStokesFOBaseResponsesEvaluators<EvalT> (fm0, meshSpecs, stateMgr, fieldManagerChoice, is_dist_param, responseList);
}

template <typename EvalT>
void StokesFO::constructSynteticTestBCEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0)
{
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  Teuchos::RCP<Teuchos::ParameterList> p;

  const bool enableMemoizer = this->params->get<bool>("Use MDField Memoization", false);

  std::string param_name;

  for (auto pl : landice_bcs[LandIceBC::SynteticTest]) {
    const std::string& ssName = pl->get<std::string>("Side Set Name");

    // We may have more than 1 basal side set. The layout of all the side fields is the
    // same, so we need to differentiate them by name (just like we do for the basis functions already).

    std::string velocity_side = "velocity_" + ssName;
    std::string sliding_velocity_side = "sliding_velocity_" + ssName;
    std::string basal_friction_side = "basal_friction_" + ssName;
    std::string beta_side = "beta_" + ssName;
    std::string ice_thickness_side = "ice_thickness_" + ssName;
    std::string ice_overburden_side = "ice_overburden_" + ssName;
    std::string surface_height_side = "surface_height_" + ssName;
    std::string stiffening_factor_side = "stiffenting_factor_" + ssName;
    std::string effective_pressure_side = "effective_pressure_" + ssName;
    std::string bed_roughness_side = "bed_roughness_" + ssName;
    std::string bed_topography_side = "bed_topography_" + ssName;

    // ------------------- Interpolations and utilities ------------------ //

    //---- Restrict vertex coordinates from cell-based to cell-side-based
    ev = evalUtils.getMSTUtils().constructDOFCellToSideEvaluator(Albany::coord_vec_name,ssName,"Vertex Vector",cellType,Albany::coord_vec_name +" " + ssName, enableMemoizer);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Compute side basis functions
    ev = evalUtils.constructComputeBasisFunctionsSideEvaluator(cellType, sideBasis[ssName], sideCubature[ssName], ssName, enableMemoizer, true);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Restrict velocity from cell-based to cell-side-based
    ev = evalUtils.constructDOFCellToSideEvaluator(dof_names[0],ssName,"Node Vector",cellType,velocity_side);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Interpolate velocity on QP on side
    ev = evalUtils.constructDOFVecInterpolationSideEvaluator(velocity_side, ssName);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Interpolate velocity gradient on QP on side
    ev = evalUtils.constructDOFVecGradInterpolationSideEvaluator(velocity_side, ssName);
    fm0.template registerEvaluator<EvalT>(ev);

    //---- Compute Quad Points coordinates on the side set
    ev = evalUtils.constructMapToPhysicalFrameSideEvaluator(cellType, sideCubature[ssName], ssName, enableMemoizer);
    fm0.template registerEvaluator<EvalT> (ev);

    // -------------------------------- LandIce evaluators ------------------------- //

    // --- Syntetic BC Residual --- //
    p = Teuchos::rcp(new Teuchos::ParameterList("Stokes Syntetic BC"));

    //Input
    p->set<std::string>("BF Side Name", Albany::bf_name + " "+ssName);
    p->set<std::string>("Weighted Measure Name", Albany::weighted_measure_name + " "+ssName);
    p->set<std::string>("Coordinate Vector Name", Albany::coord_vec_name + " "+ssName);
    p->set<std::string>("Side Normal Name", Albany::normal_name + " "+ssName);
    p->set<std::string>("Velocity Side QP Variable Name", velocity_side);
    p->set<std::string>("Side Set Name", ssName);
    p->set<Teuchos::RCP<shards::CellTopology> >("Cell Type", cellType);
    p->set<Teuchos::ParameterList*>("BC Params", &pl->sublist("BC Params"));

    //Output
    p->set<std::string>("Residual Variable Name", resid_names[0]);

    ev = Teuchos::rcp(new LandIce::StokesFOSynteticTestBC<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }
}

template <typename EvalT>
void StokesFO::constructProjLaplEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                            Albany::FieldManagerChoice fieldManagerChoice,
                                            Teuchos::RCP<std::map<std::string, int> > extruded_params_levels)
{
  // Only do something if the number of equations is larger than the FO equations
  if(neq > vecDimFO) {
    Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
    Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
    Teuchos::RCP<Teuchos::ParameterList> p;

    auto& proj_lapl_params = params->sublist("LandIce L2 Projected Boundary Laplacian");
    auto& ssName = proj_lapl_params.get<std::string>("Side Set Name",basalSideName);
    std::string field_name = proj_lapl_params.get<std::string>("Field Name","basal_friction");
    std::string field_name_side = field_name + "_" + ssName;

    // ----------  Define Field Names ----------- //
    Teuchos::ArrayRCP<std::string> dof_name_auxiliary(1);
    Teuchos::ArrayRCP<std::string> resid_name_auxiliary(1);
    dof_name_auxiliary[0] = "L2 Projected Boundary Laplacian";
    resid_name_auxiliary[0] = "L2 Projected Boundary Laplacian Residual";
    std::string aux_resid_scatter_tag = "Scatter Auxiliary Residual";

    // ------------------- Interpolations and utilities ------------------ //

    // Gather solution field
    ev = evalUtils.constructGatherSolutionEvaluator_noTransient(false, dof_name_auxiliary, 2);
    fm0.template registerEvaluator<EvalT> (ev);

    // Scatter residual
    ev = evalUtils.constructScatterResidualEvaluatorWithExtrudedParams(false, resid_name_auxiliary, extruded_params_levels, vecDimFO, aux_resid_scatter_tag);
    fm0.template registerEvaluator<EvalT> (ev);

    // Project to side
    ev = evalUtils.constructDOFCellToSideEvaluator(dof_name_auxiliary[0],ssName,"Node Scalar",cellType, dof_name_auxiliary[0]);
    fm0.template registerEvaluator<EvalT> (ev);

    //---- Interpolate velocity gradient on QP on side
    ev = evalUtils.getPSTUtils().constructDOFGradInterpolationSideEvaluator(field_name_side, ssName);
    fm0.template registerEvaluator<EvalT>(ev);

    // -------------------------------- LandIce evaluators ------------------------- //

    // L2 Projected bd laplacian residual
    p = Teuchos::rcp(new Teuchos::ParameterList("L2 Projected Boundary Laplacian Residual"));

    //Input
    p->set<std::string>("Solution Variable Name", dof_name_auxiliary[0]);
    p->set<std::string>("Coordinate Vector Variable Name", Albany::coord_vec_name);
    p->set<std::string>("Field Name", field_name_side);
    p->set<std::string>("Field Gradient Name", field_name_side + " Gradient");
    p->set<std::string>("Gradient BF Side Name", Albany::grad_bf_name + " " + ssName);
    p->set<std::string>("Weighted Measure Side Name", Albany::weighted_measure_name + " "+ssName);
    p->set<std::string>("Tangents Side Name", Albany::tangents_name + " "+ssName);
    p->set<std::string>("Side Set Name", ssName);
    p->set<std::string>("Boundary Edges Set Name", proj_lapl_params.get<std::string>("Boundary Edges Set Name", "lateralside"));
    p->set<double>("Mass Coefficient",  proj_lapl_params.get<double>("Mass Coefficient",1.0));
    p->set<double>("Robin Coefficient", proj_lapl_params.get<double>("Robin Coefficient",0.0));
    p->set<double>("Laplacian Coefficient", proj_lapl_params.get<double>("Laplacian Coefficient",1.0));
    p->set<Teuchos::RCP<shards::CellTopology> >("Cell Type", cellType);

    //Output
    p->set<std::string>("L2 Projected Boundary Laplacian Residual Name", "L2 Projected Boundary Laplacian Residual");

    ev = Teuchos::rcp(new LandIce::L2ProjectedBoundaryLaplacianResidualParam<EvalT,PHAL::AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);

    if (fieldManagerChoice == Albany::BUILD_RESID_FM) {
      PHX::Tag<typename EvalT::ScalarT> res_tag(aux_resid_scatter_tag, dl->dummy);
      fm0.requireField<EvalT>(res_tag);
    }
  }
}

} // namespace LandIce

#endif // LANDICE_STOKES_FO_PROBLEM_HPP
