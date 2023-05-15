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
#include "PHAL_GatherCoordinateVector.hpp"

#include "LandIce_StokesFOBase.hpp"

#include "LandIce_StokesFOSynteticTestBC.hpp"
#include "LandIce_PressureCorrectedTemperature.hpp"

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
  ~StokesFO() = default;

  //! Build evaluators
  virtual Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
  buildEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                   const Albany::MeshSpecsStruct& meshSpecs,
                   Albany::StateManager& stateMgr,
                   Albany::FieldManagerChoice fmchoice,
                   const Teuchos::RCP<Teuchos::ParameterList>& responseList);

  //! Build unmanaged fields
  virtual void buildFields(PHX::FieldManager<PHAL::AlbanyTraits>& fm0);

  //! Each problem must generate it's list of valid parameters
  Teuchos::RCP<const Teuchos::ParameterList> getValidProblemParameters() const;

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
  void constructFields(PHX::FieldManager<PHAL::AlbanyTraits>& fm0);

  const Teuchos::RCP<IntrepidCubature> getCellCubature() const {
    return this->cellCubature;
  }

protected:

  void setFieldsProperties();
  void setupEvaluatorRequests ();

  void constructDirichletEvaluators (const Albany::MeshSpecsStruct& meshSpecs);
  void constructNeumannEvaluators (const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs);

  bool adjustBedTopo;
  bool adjustSurfaceHeight;
  bool fluxDivIsPartOfSolution;
  bool l2ProjectedBoundaryEquation;
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
  Albany::EvaluatorUtilsImpl<EvalT, PHAL::AlbanyTraits,typename EvalT::ScalarT> evalUtils(dl);
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  Teuchos::RCP<Teuchos::ParameterList> p;

  // Gather solution field
  ev = evalUtils.constructGatherSolutionEvaluator_noTransient(true, dof_names[0], dof_offsets[0]);
  fm0.template registerEvaluator<EvalT> (ev);

  // Scatter residual
  ev = evalUtils.constructScatterResidualEvaluatorWithExtrudedParams(true, resid_names[0], Teuchos::rcpFromRef(extruded_params_levels), dof_offsets[0], scatter_names[0]);
  fm0.template registerEvaluator<EvalT> (ev);

  // If the mesh depends on parameters AND the thickness is a parameter,
  // after gathering the coordinates, we modify the z coordinate of the mesh.
  bool require_old_coords=false;
  if (Albany::mesh_depends_on_parameters() && (is_dist_param[ice_thickness_name])) {
    require_old_coords=true;
    if(adjustBedTopo && !adjustSurfaceHeight) {
      //------ Update Z Coordinate
      p = Teuchos::rcp(new Teuchos::ParameterList("Update Z Coordinate"));

      p->set<std::string>("Old Coords Name",  "Coord Vec Old");
      p->set<std::string>("New Coords Name",  Albany::coord_vec_name);
      p->set<std::string>("Thickness Name",   ice_thickness_name);
      p->set<std::string>("Thickness Lower Bound Name",   ice_thickness_name + "_lowerbound");
      p->set<std::string>("Thickness Upper Bound Name",   ice_thickness_name + "_upperbound");
      p->set<std::string>("Top Surface Name", "observed_surface_height");
      p->set<std::string>("Updated Top Surface Name", surface_height_name);
      p->set<std::string>("Bed Topography Name", "observed_bed_topography");
      p->set<std::string>("Updated Bed Topography Name", bed_topography_name);
      p->set<Teuchos::ParameterList*>("Physical Parameter List", &params->sublist("LandIce Physical Parameters"));

      ev = Teuchos::rcp(new LandIce::UpdateZCoordinateMovingBed<EvalT,PHAL::AlbanyTraits>(*p, dl));
      fm0.template registerEvaluator<EvalT>(ev);
    } else if(adjustSurfaceHeight && !adjustBedTopo) {
      //------ Update Z Coordinate
      p = Teuchos::rcp(new Teuchos::ParameterList("Update Z Coordinate"));

      p->set<std::string>("Old Coords Name",  "Coord Vec Old");
      p->set<std::string>("New Coords Name",  Albany::coord_vec_name);
      p->set<std::string>("Thickness Name",   ice_thickness_name);
      p->set<std::string>("Top Surface Name", surface_height_name);
      p->set<std::string>("Bed Topography Name", bed_topography_name);
      p->set<Teuchos::ParameterList*>("Physical Parameter List", &params->sublist("LandIce Physical Parameters"));

      ev = Teuchos::rcp(new LandIce::UpdateZCoordinateMovingTop<EvalT,PHAL::AlbanyTraits>(*p, dl));
      fm0.template registerEvaluator<EvalT>(ev);
    } else {
          TEUCHOS_TEST_FOR_EXCEPTION(adjustBedTopo == adjustSurfaceHeight, std::logic_error, "Error! When the ice thickness is a parameter,\n "
              "either 'Adjust Bed Topography to Account for Thickness Changes' or\n"
              " 'Adjust Surface Height to Account for Thickness Changes' needs to be true.\n");
    }
  }

  // If the mesh depends on parameters AND the bed topography or the surface height are parameters,
  // after gathering the coordinates, we modify the z coordinate of the mesh.
  else if (Albany::mesh_depends_on_parameters() && (is_dist_param[surface_height_param_name]||is_dist_param[bed_topography_param_name])) {
    //------ Update Z Coordinate
    require_old_coords=true;
    p = Teuchos::rcp(new Teuchos::ParameterList("Update Z Coordinate"));

    p->set<std::string>("Old Coords Name",  "Coord Vec Old");
    p->set<std::string>("New Coords Name",  Albany::coord_vec_name);
    p->set<std::string>("Thickness Name",   ice_thickness_name);
    p->set<std::string>("Top Surface Name", surface_height_name);
    if(is_dist_param[surface_height_param_name])
      p->set<std::string>("Top Surface Parameter Name", surface_height_param_name);
    if(is_dist_param[bed_topography_param_name])
      p->set<std::string>("Bed Topography Parameter Name", bed_topography_param_name);
    p->set<std::string>("Bed Topography Name", bed_topography_name);
    p->set<Teuchos::ParameterList*>("Physical Parameter List", &params->sublist("LandIce Physical Parameters"));

    ev = Teuchos::rcp(new LandIce::UpdateZCoordinateGivenTopAndBedSurfaces<EvalT,PHAL::AlbanyTraits>(*p, dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if(require_old_coords) {
    //----- Gather Coordinate Vector (ad hoc parameters)
    p = Teuchos::rcp(new Teuchos::ParameterList("Gather Coordinate Vector"));

    // Output:: Coordinate Vector at vertices
    p->set<std::string>("Coordinate Vector Name", "Coord Vec Old");

    ev = Teuchos::rcp(new PHAL::GatherCoordinateVector<EvalT,PHAL::AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  } else {
    //----- Gather Coordinate Vector (general parameters)
    ev = evalUtils.constructGatherCoordinateVectorEvaluator();
    fm0.template registerEvaluator<EvalT> (ev);
  }

  if (viscosity_use_corrected_temperature) {
    // --- LandIce pressure-melting temperature
    // Note: this CAN'T go in StokesFOBase, since StokesFOThermoCoupled uses LandIce::Temperature
    //       to compute both temperature and corrected temperature
    p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Pressure Corrected Temperature"));

    //Input
    p->set<std::string>("Surface Height Variable Name", surface_height_name);
    p->set<std::string>("Coordinate Vector Variable Name", Albany::coord_vec_name);
    p->set<Teuchos::ParameterList*>("LandIce Physical Parameters", &params->sublist("LandIce Physical Parameters"));
    p->set<std::string>("Temperature Variable Name", temperature_name);
    p->set<bool>("Use P0 Temperature", viscosity_use_p0_temperature);

    //Output
    p->set<std::string>("Corrected Temperature Variable Name", corrected_temperature_name);

    ev = createEvaluatorWithTwoScalarTypes<LandIce::PressureCorrectedTemperature,EvalT>(p,dl,field_scalar_type[temperature_name],field_scalar_type[surface_height_name]);
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // --- Syntetic test BC evaluators (if needed) --- //
  constructSynteticTestBCEvaluators<EvalT> (fm0);

  // --- ProjectedLaplacian-related evaluators (if needed) --- //
  if(l2ProjectedBoundaryEquation) {
    int eqId = 1;
    constructProjLaplEvaluators<EvalT> (fm0, fieldManagerChoice, eqId);
  }

  // --- FluxDiv-related evaluators (if needed) --- //
  if(fluxDivIsPartOfSolution) {
    int eqId = l2ProjectedBoundaryEquation ? 2 : 1;
    constructFluxDivEvaluators<EvalT> (fm0, fieldManagerChoice, eqId, meshSpecs);
  }

  // --- States/parameters --- //
  constructStokesFOBaseEvaluators<EvalT> (fm0, meshSpecs, stateMgr, fieldManagerChoice);

  // Finally, construct responses, and return the tags
  return constructStokesFOBaseResponsesEvaluators<EvalT> (fm0, meshSpecs, stateMgr, fieldManagerChoice, responseList);
}

template <typename EvalT>
void StokesFO::constructSynteticTestBCEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0)
{
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  Teuchos::RCP<Teuchos::ParameterList> p;

  std::string param_name;

  for (auto pl : landice_bcs[LandIceBC::SynteticTest]) {
    const std::string& ssName = pl->get<std::string>("Side Set Name");

    // -------------------------------- LandIce evaluators ------------------------- //

    // --- Syntetic BC Residual --- //
    p = Teuchos::rcp(new Teuchos::ParameterList("Stokes Syntetic BC"));

    //Input
    p->set<std::string>("BF Side Name", side_fname(Albany::bf_name,ssName));
    p->set<std::string>("Weighted Measure Name", side_fname(Albany::weighted_measure_name,ssName));
    p->set<std::string>("Coordinate Vector Name", side_fname(Albany::coord_vec_name,ssName));
    p->set<std::string>("Side Normal Name", side_fname(Albany::normal_name,ssName));
    p->set<std::string>("Velocity Side QP Variable Name", side_fname(dof_names[0],ssName));
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
void
LandIce::StokesFO::constructFields(PHX::FieldManager<PHAL::AlbanyTraits> &fm0)
{
  constructStokesFOBaseFields<EvalT>(fm0);
}

} // namespace LandIce

#endif // LANDICE_STOKES_FO_PROBLEM_HPP
