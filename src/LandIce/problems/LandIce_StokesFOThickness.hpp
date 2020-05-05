//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_STOKES_FO_THICKNESS_PROBLEM_HPP
#define LANDICE_STOKES_FO_THICKNESS_PROBLEM_HPP

#include "LandIce_GatherVerticallyContractedSolution.hpp"
#include "Shards_CellTopology.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_Utils.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "Albany_GeneralPurposeFieldsNames.hpp"

#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_AlbanyTraits.hpp"

#include "LandIce_StokesFOBase.hpp"

#include "LandIce_MapThickness.hpp"
#include "LandIce_Gather2DField.hpp"
#include "LandIce_PressureCorrectedTemperature.hpp"
#include "LandIce_ScatterResidual2D.hpp"
#include "LandIce_SimpleOperationEvaluator.hpp"
#include "LandIce_UpdateZCoordinate.hpp"
#include "LandIce_ThicknessResid.hpp"
#include "LandIce_StokesFOImplicitThicknessUpdateResid.hpp"

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN

namespace LandIce
{

/*!
 * \brief Abstract interface for representing a 1-D finite element
 * problem.
 */
class StokesFOThickness : public StokesFOBase {
public:

  //! Default constructor
  StokesFOThickness(const Teuchos::RCP<Teuchos::ParameterList>& params,
                    const Teuchos::RCP<Teuchos::ParameterList>& discParams_,
                    const Teuchos::RCP<ParamLib>& paramLib,
                    const int numDim_);

  //! Destructor
  ~StokesFOThickness() = default;

  //! Prohibit copying
  StokesFOThickness (const StokesFOThickness&) = delete;
  StokesFOThickness& operator= (const StokesFOThickness&) = delete;

  // Build evaluators
  Teuchos::Array<Teuchos::RCP<const PHX::FieldTag>>
  buildEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                   const Albany::MeshSpecsStruct& meshSpecs,
                   Albany::StateManager& stateMgr,
                   Albany::FieldManagerChoice fmchoice,
                   const Teuchos::RCP<Teuchos::ParameterList>& responseList);

  //! Each problem must generate it's list of valid parameters
  Teuchos::RCP<const Teuchos::ParameterList> getValidProblemParameters() const;

  //! Main problem setup routine. Not directly called, but indirectly by following functions
  template <typename EvalT>
  Teuchos::RCP<const PHX::FieldTag>
  constructEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                       const Albany::MeshSpecsStruct& meshSpecs,
                       Albany::StateManager& stateMgr,
                       Albany::FieldManagerChoice fmchoice,
                       const Teuchos::RCP<Teuchos::ParameterList>& responseList);

protected:

  template <typename EvalT>
  void constructThicknessEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0, 
                                     const Albany::MeshSpecsStruct& meshSpecs,
                                     Albany::FieldManagerChoice fmchoice);

  void constructDirichletEvaluators(const Albany::MeshSpecsStruct& meshSpecs);
  void constructNeumannEvaluators(const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs);

  void setFieldsProperties ();

  std::string initial_ice_thickness_name;
};

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
StokesFOThickness::constructEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                        const Albany::MeshSpecsStruct& meshSpecs,
                                        Albany::StateManager& stateMgr,
                                        Albany::FieldManagerChoice fieldManagerChoice,
                                        const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  Teuchos::RCP<Teuchos::ParameterList> p;

  // --- StokesFOBase evaluators --- //
  constructStokesFOBaseEvaluators<EvalT> (fm0, meshSpecs, stateMgr, fieldManagerChoice);

  // Gather velocity field
  ev = evalUtils.constructGatherSolutionEvaluator_noTransient(true, dof_names[0], dof_offsets[0]);
  fm0.template registerEvaluator<EvalT> (ev);

  // --- LandIce pressure-melting temperature
  // Note: this CAN'T go in StokesFOBase, since StokesFOThermoCoupled uses LandIce::Temperature
  //       to compute both temperature and corrected temperature
  p = Teuchos::rcp(new Teuchos::ParameterList("LandIce Pressure Corrected Temperature"));

  //Input
  p->set<std::string>("Surface Height Variable Name", surface_height_name);
  p->set<std::string>("Coordinate Vector Variable Name", Albany::coord_vec_name);
  p->set<Teuchos::ParameterList*>("LandIce Physical Parameters", &params->sublist("LandIce Physical Parameters"));
  p->set<std::string>("Temperature Variable Name", temperature_name);

  //Output
  p->set<std::string>("Corrected Temperature Variable Name", "corrected " + temperature_name);

  // The input temperature has been interpolated with NodesToCell, which means we added a MeshScalar st to the initial temperature scalar type
  ev = createEvaluatorWithTwoScalarTypes<LandIce::PressureCorrectedTemperature,EvalT>(p,dl,FieldScalarType::MeshScalar | field_scalar_type[temperature_name],field_scalar_type[surface_height_name]);
  fm0.template registerEvaluator<EvalT>(ev);

  //--- Scatter LandIce Stokes FO Residual With Extruded Field ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("Scatter StokesFO"));

  //Input
  p->set<std::string>("Residual Name", resid_names[0]);
  p->set<int>("Tensor Rank", 1); 
  p->set<int>("Field Level", 0);
  p->set<int>("Offset of First DOF", dof_offsets[0]); 
  p->set<int>("Offset 2D Field", dof_offsets[1]); 

  //Output
  p->set<std::string>("Scatter Field Name", scatter_names[0]);

  ev = Teuchos::rcp(new PHAL::ScatterResidualWithExtrudedField<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  // --- Thickness equation evaluators --- //
  constructThicknessEvaluators<EvalT> (fm0, meshSpecs, fieldManagerChoice);


  // Finally, construct responses, and return the tags
  return constructStokesFOBaseResponsesEvaluators<EvalT> (fm0, meshSpecs, stateMgr, fieldManagerChoice, responseList);
}

template<typename EvalT>
void StokesFOThickness::constructThicknessEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0, 
                                                      const Albany::MeshSpecsStruct& meshSpecs,
                                                      Albany::FieldManagerChoice fieldManagerChoice)
{
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  Teuchos::RCP<Teuchos::ParameterList> p;

  if (surfaceSideName!="__INVALID__")
  {
    //--- LandIce Gather Vertically Averaged Velocity ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("Gather Averaged Velocity"));
    //Input
    p->set<std::string>("Contracted Solution Name", "Averaged Velocity");
    p->set<std::string>("Mesh Part", "upperside");
    p->set<std::string>("Side Set Name", surfaceSideName);
    p->set<Teuchos::RCP<const CellTopologyData> >("Cell Topology",Teuchos::rcp(new CellTopologyData(meshSpecs.ctd)));
    p->set<int>("Solution Offset", dof_offsets[0]);
    p->set<bool>("Is Vector", true);
    p->set<std::string>("Contraction Operator", "Vertical Average");

    ev = Teuchos::rcp(new LandIce::GatherVerticallyContractedSolution<EvalT,PHAL::AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  //--- LandIce Gather 2D Field (Thickness) ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("Gather Thickness"));

  //Input
  p->set<std::string>("2D Field Name", dof_names[1]);
  p->set<int>("Offset of First DOF", dof_offsets[1]);
  p->set<Teuchos::RCP<const CellTopologyData> >("Cell Topology",Teuchos::rcp(new CellTopologyData(meshSpecs.ctd)));

  ev = Teuchos::rcp(new LandIce::Gather2DField<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  //--- LandIce Gather Extruded 2D Field (Thickness) ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("Gather ExtrudedThickness"));

  //Input
  p->set<std::string>("2D Field Name", "Extruded " + dof_names[1]);
  p->set<int>("Offset of First DOF", dof_offsets[1]);
  p->set<Teuchos::RCP<const CellTopologyData> >("Cell Topology",Teuchos::rcp(new CellTopologyData(meshSpecs.ctd)));

  ev = Teuchos::rcp(new LandIce::GatherExtruded2DField<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  // --- FO Stokes Implicit Thickness Update Resid --- //
  p = Teuchos::rcp(new Teuchos::ParameterList("StokesFOImplicitThicknessUpdate Resid"));

  //Input
  p->set<std::string>("Thickness Increment Variable Name", "Extruded " + dof_names[1]);
  p->set<std::string>("Gradient BF Name", Albany::grad_bf_name);
  p->set<std::string>("Weighted BF Name", Albany::weighted_bf_name);

  Teuchos::ParameterList& physParamList = params->sublist("LandIce Physical Parameters");
  p->set<Teuchos::ParameterList*>("Physical Parameter List", &physParamList);

  //Output
  p->set<std::string>("Residual Name", resid_names[0]);

  ev = Teuchos::rcp(new LandIce::StokesFOImplicitThicknessUpdateResid<EvalT,PHAL::AlbanyTraits>(*p, dl));
  fm0.template registerEvaluator<EvalT>(ev);

  // --- Thickness Resid --- //
  p = Teuchos::rcp(new Teuchos::ParameterList("Thickness Resid"));

  //Input
  p->set<std::string>("Averaged Velocity Variable Name", "Averaged Velocity");
  p->set<std::string>("Thickness Increment Variable Name", dof_names[1]);
  p->set<std::string>("Past Thickness Name", initial_ice_thickness_name);
  p->set<std::string>("Side Set Name", surfaceSideName);
  p->set<std::string>("Mesh Part", "upperside");
  p->set<std::string>("Coordinate Vector Name", Albany::coord_vec_name);
  p->set<int>("Cubature Degree",3);
  if (std::find(requirements.begin(),requirements.end(),"apparent_mass_balance")!=requirements.end()) {
    p->set<std::string>("SMB Name", "apparent_mass_balance");
  }
  p->set<Teuchos::RCP<const Albany::MeshSpecsStruct> >("Mesh Specs Struct", Teuchos::rcpFromRef(meshSpecs));
  if(this->params->isParameter("Time Step Ptr")) {
    p->set<Teuchos::RCP<double> >("Time Step Ptr", this->params->get<Teuchos::RCP<double> >("Time Step Ptr"));
  } else {
    Teuchos::RCP<double> dt = Teuchos::rcp(new double(this->params->get<double>("Time Step")));
    p->set<Teuchos::RCP<double> >("Time Step Ptr", dt);
  }

  //Output
  p->set<std::string>("Residual Name", resid_names[1]);

  ev = Teuchos::rcp(new LandIce::ThicknessResid<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  if (Albany::mesh_depends_on_solution()) {
    //--- Gather Coordinates ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("Gather Coordinate Vector"));

    // Output:: Coordindate Vector at vertices
    p->set<std::string>("Coordinate Vector Name", "Coord Vec Old");

    ev =  Teuchos::rcp(new PHAL::GatherCoordinateVector<EvalT,PHAL::AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);

    //--- Update Z Coordinate ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("Update Z Coordinate"));

    // Input
    p->set<std::string>("Old Coords Name", "Coord Vec Old");
    p->set<std::string>("New Coords Name", Albany::coord_vec_name);
    p->set<std::string>("Thickness Increment Name", "Extruded " + dof_names[1]);
    p->set<std::string>("Past Thickness Name", initial_ice_thickness_name);
    p->set<std::string>("Top Surface Name", surface_height_name);
    p->set<std::string>("Bed Topography Name", bed_topography_name);
    p->set<Teuchos::ParameterList*>("Physical Parameter List", &params->sublist("LandIce Physical Parameters"));

    ev = Teuchos::rcp(new LandIce::UpdateZCoordinateMovingTop<EvalT,PHAL::AlbanyTraits>(*p, dl));
    fm0.template registerEvaluator<EvalT>(ev);
  } else {
    //---- Gather coordinates
    ev = evalUtils.constructGatherCoordinateVectorEvaluator();
    fm0.template registerEvaluator<EvalT> (ev);
  }

  //--- Compute actual thickness --- //
  p = Teuchos::rcp(new Teuchos::ParameterList("Update Thickness"));

  // Input
  p->set<std::string> ("Input Field Name",dof_names[1]);
  p->set<std::string> ("Parameter Field 1",initial_ice_thickness_name);
  p->set<Teuchos::RCP<PHX::DataLayout>> ("Field Layout",dl->node_scalar);

  // Output
  p->set<std::string> ("Output Field Name",ice_thickness_name);

  ev = Teuchos::rcp(new LandIce::BinarySumOp<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT, RealType>(*p, dl));
  fm0.template registerEvaluator<EvalT>(ev);

  //--- LandIce Stokes FO Residual Thickness ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("Scatter ResidualH"));

  //Input
  p->set<std::string>("Residual Name", resid_names[1]);
  p->set<int>("Tensor Rank", 0);
  p->set<int>("Field Level", discParams->get<int>("NumLayers"));
  p->set<std::string>("Mesh Part", surfaceSideName);
  p->set<int>("Offset of First DOF", dof_offsets[1]);
  p->set<Teuchos::RCP<const CellTopologyData> >("Cell Topology",Teuchos::rcp(new CellTopologyData(meshSpecs.ctd)));

  //Output
  p->set<std::string>("Scatter Field Name", scatter_names[1]);

  ev = Teuchos::rcp(new PHAL::ScatterResidual2D<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  if (fieldManagerChoice == Albany::BUILD_RESID_FM) {
    // Require scattering of residual
    PHX::Tag<typename EvalT::ScalarT> scatterTag(scatter_names[1], dl->dummy);
    fm0.requireField<EvalT>(scatterTag);
  }
}

} // namespace LandIce

#endif // LANDICE_STOKES_FO_THICKNESS_PROBLEM_HPP
