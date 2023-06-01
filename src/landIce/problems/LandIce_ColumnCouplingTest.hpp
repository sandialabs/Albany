//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_COLUMN_COUPLING_TEST_HPP
#define ALBANY_COLUMN_COUPLING_TEST_HPP 1

#include "LandIce_GatherVerticallyContractedSolution.hpp"
#include "LandIce_ColumnCouplingTestResidual.hpp"
#include "LandIce_ScatterResidual2D.hpp"
#include "LandIce_ResponseUtilities.hpp"

#include "Albany_AbstractProblem.hpp"
#include "Albany_ResponseUtilities.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "Albany_StringUtils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_GeneralPurposeFieldsNames.hpp"

#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"

#include "PHAL_SaveStateField.hpp"
#include "PHAL_LoadStateField.hpp"
#include "PHAL_SaveSideSetStateField.hpp"
#include "PHAL_LoadSideSetStateField.hpp"

#include "Shards_CellTopology.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN

namespace LandIce
{

/*!
 * \brief  A 2D problem for the subglacial hydrology
 */
class ColumnCouplingTest : public Albany::AbstractProblem
{
public:

  //! Default constructor
  ColumnCouplingTest (const Teuchos::RCP<Teuchos::ParameterList>& params,
                      const Teuchos::RCP<Teuchos::ParameterList>& discParams,
                      const Teuchos::RCP<ParamLib>& paramLib);

  //! Destructor
  ~ColumnCouplingTest() = default;

  //! Return number of spatial dimensions
  int spatialDimension () const { return numDim; }
  
  //! Get boolean telling code if SDBCs are utilized  
  bool useSDBCs() const {return use_sdbcs_; }

  //! Build the PDE instantiations, boundary conditions, and initial solution
  void buildProblem (Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecs> >  meshSpecs,
                             Albany::StateManager& stateMgr);

  // Build evaluators
  Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
  buildEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                   const Albany::MeshSpecs& meshSpecs,
                   Albany::StateManager& stateMgr,
                   Albany::FieldManagerChoice fmchoice,
                   const Teuchos::RCP<Teuchos::ParameterList>& responseList);

  //! Each problem must generate it's list of valid parameters
  Teuchos::RCP<const Teuchos::ParameterList> getValidProblemParameters() const;

  //! Main problem setup routine. Not directly called, but indirectly by buildEvaluators
  template <typename EvalT> Teuchos::RCP<const PHX::FieldTag>
  constructEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                       const Albany::MeshSpecs& meshSpecs,
                       Albany::StateManager& stateMgr,
                       Albany::FieldManagerChoice fmchoice,
                       const Teuchos::RCP<Teuchos::ParameterList>& responseList);

  template <typename EvalT>
  void constructStatesEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                  const Albany::MeshSpecs& meshSpecs,
                                  Albany::StateManager& stateMgr,
                                  Albany::FieldManagerChoice fieldManagerChoice);

  void constructDirichletEvaluators (const Albany::MeshSpecs& meshSpecs);
protected:

  int numDim;

  std::string dof_name;
  std::string resid_name;
  std::string scatter_name;

  Teuchos::RCP<Teuchos::ParameterList> discParams;

  Teuchos::RCP<shards::CellTopology>   cellType;

  Teuchos::RCP<Albany::Layouts> dl, dl_side;

  using IntrepidBasis    = Intrepid2::Basis<PHX::Device, RealType, RealType>;
  using IntrepidCubature = Intrepid2::Cubature<PHX::Device>;
  std::map<std::string,Teuchos::RCP<IntrepidBasis>>         sideBasis;
  std::map<std::string,Teuchos::RCP<IntrepidCubature>>      sideCubature;

  std::string sideSetName;

  std::string cellEBName;
  std::map<std::string,std::string> sideEBName;
  
  /// Boolean marking whether SDBCs are used 
  bool use_sdbcs_ = false; 
};

// ===================================== IMPLEMENTATION ======================================= //

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
ColumnCouplingTest::constructEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                    const Albany::MeshSpecs& meshSpecs,
                                    Albany::StateManager& stateMgr,
                                    Albany::FieldManagerChoice fieldManagerChoice,
                                    const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  constructStatesEvaluators<EvalT> (fm0, meshSpecs, stateMgr, fieldManagerChoice);

  // Service variables for registering state variables and evaluators
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  Teuchos::RCP<Teuchos::ParameterList> p;

  // -------------------- Starting evaluators construction and registration ------------------------ //

  //--- LandIce Gather Vertically Averaged Velocity ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("Gather Averaged Velocity"));

  p->set<std::string>("Contracted Solution Name", dof_name+"_av");
  p->set<std::string>("Mesh Part", sideSetName);
  p->set<std::string>("Side Set Name", sideSetName);
  p->set<Teuchos::RCP<const CellTopologyData> >("Cell Topology",Teuchos::rcp(new CellTopologyData(meshSpecs.ctd)));
  p->set<int>("Solution Offset", 0);
  p->set<bool>("Is Vector", false);
  p->set<std::string>("Contraction Operator", "Vertical Sum");

  ev = Teuchos::rcp(new GatherVerticallyContractedSolution<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  //--- LandIce Column Coupling Test Residual ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("Column Coupling Residual"));

  auto sh_name = discParams->get<std::string>("Surface Height Field Name");
  p->set<std::string>("Side Set Name", sideSetName);
  p->set<std::string>("Solution Name", dof_name + "_av");
  p->set<std::string>("Residual Name", resid_name);
  p->set<std::string>("Surface Height Name", sh_name + "_" + sideSetName);

  ev = Teuchos::rcp(new ColumnCouplingTestResidual<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  //--- LandIce Scatter Residual 2D ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("Scatter Residual 2D"));

  p->set<std::string>("Residual Name", resid_name);
  p->set<int>("Tensor Rank", 0);
  p->set<int>("Field Level", sideSetName=="basalside" ? 0 : 1);
  p->set<std::string>("Mesh Part", sideSetName);
  p->set<int>("Offset of First DOF", 0);
  p->set<Teuchos::RCP<const shards::CellTopology> >("Cell Topology",cellType);

  //Output
  p->set<std::string>("Scatter Field Name", scatter_name);

  ev = Teuchos::rcp(new PHAL::ScatterResidual2D<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  // ----------------------------------------------------- //

  Teuchos::RCP<Teuchos::ParameterList> paramList = Teuchos::rcp(new Teuchos::ParameterList("Param List"));
  {
    // response
    Teuchos::RCP<const Albany::MeshSpecs> meshSpecsPtr = Teuchos::rcpFromRef(meshSpecs);
    paramList->set<Teuchos::RCP<const Albany::MeshSpecs> >("Mesh Specs Struct", meshSpecsPtr);
    paramList->set<Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);
  }

  if (fieldManagerChoice == Albany::BUILD_RESID_FM)
  {
    PHX::Tag<typename EvalT::ScalarT> scatter_tag(scatter_name, dl->dummy);
    fm0.requireField<EvalT>(scatter_tag);
  }
  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM)
  {
    Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
    auto dof_side = dof_name + "_" + sideSetName;
    auto sh_side = sh_name + "_" + sideSetName;
    auto coords_side = Albany::coord_vec_name + "_" + sideSetName;

    // --- Solution --- //
    ev = evalUtils.constructGatherSolutionEvaluator_noTransient(false, dof_name, 0);
    fm0.template registerEvaluator<EvalT> (ev);

    ev = evalUtils.getSTUtils().constructDOFCellToSideEvaluator(dof_name, sideSetName, "Node Scalar", cellType, dof_side);
    fm0.template registerEvaluator<EvalT> (ev);

    // --- Coors, basis functions, measure, etc. --- //
    ev = evalUtils.getMSTUtils().constructGatherCoordinateVectorEvaluator();
    fm0.template registerEvaluator<EvalT> (ev);

    ev = evalUtils.getMSTUtils().constructDOFCellToSideEvaluator(Albany::coord_vec_name, sideSetName, "Vertex Vector", cellType, coords_side);
    fm0.template registerEvaluator<EvalT> (ev);

    ev = evalUtils.getSTUtils().constructDOFInterpolationSideEvaluator (dof_side, sideSetName);
    fm0.template registerEvaluator<EvalT> (ev);

    ev = evalUtils.constructComputeBasisFunctionsSideEvaluator(cellType, sideBasis[sideSetName], sideCubature[sideSetName], sideSetName, false);
    fm0.template registerEvaluator<EvalT> (ev);

    // --- Surface height --- //
    ev = evalUtils.getRTUtils().constructDOFInterpolationSideEvaluator (sh_side, sideSetName);
    fm0.template registerEvaluator<EvalT> (ev);

    // ----------------------- Responses --------------------- //

    Teuchos::RCP<Teuchos::ParameterList> paramList = Teuchos::rcp(new Teuchos::ParameterList("Param List"));
    paramList->set<Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);
    paramList->set<Teuchos::RCP<const CellTopologyData> >("Cell Topology",Teuchos::rcp(new CellTopologyData(meshSpecs.ctd)));

    ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
    return respUtils.constructResponses(fm0, *responseList, paramList, stateMgr);
  }

  return Teuchos::null;
}

template <typename EvalT>
void ColumnCouplingTest::
constructStatesEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                           const Albany::MeshSpecs& meshSpecs,
                           Albany::StateManager& stateMgr,
                           Albany::FieldManagerChoice fieldManagerChoice)
{
  using FL  = Albany::FieldLocation;
  using FRT = Albany::FieldRankType;

  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);

  // Temporary variables used numerous times below
  Albany::StateStruct::MeshFieldEntity entity;
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  Teuchos::RCP<Teuchos::ParameterList> p;

  // ---------------------------- Registering state variables ------------------------- //

  std::string stateName, fieldName;
  Teuchos::RCP<PHX::DataLayout> state_dl;

  // Volume requirements
  auto& info = discParams->sublist("Required Fields Info");
  int num_fields = info.get<int>("Number Of Fields",0);

  for (int ifield=0; ifield<num_fields; ++ifield) {
    Teuchos::ParameterList& thisFieldList =  info.sublist(util::strint("Field", ifield));

    // Get current state specs
    auto fieldName = thisFieldList.get<std::string>("Field Name");
    auto stateName = thisFieldList.get<std::string>("State Name", fieldName);
    fieldName = fieldName + "_" + sideSetName;
    auto fieldUsage = thisFieldList.get<std::string>("Field Usage","Input"); // WARNING: assuming Input if not specified

    if (fieldUsage == "Unused") {
      continue;
    }

    auto meshPart = "";

    auto fieldType  = thisFieldList.get<std::string>("Field Type");
    auto loc = fieldType.find("Node")!=std::string::npos ? FL::Node : FL::Cell;

    TEUCHOS_TEST_FOR_EXCEPTION (
        fieldType.find("Scalar")==std::string::npos &&
        fieldType.find("Vector")==std::string::npos &&
        fieldType.find("Gradient")==std::string::npos &&
        fieldType.find("Tensor")==std::string::npos, std::runtime_error,
        "Error! Invalid rank type for state " + stateName + "\n");

    auto rank = fieldType.find("Scalar")!=std::string::npos ? FRT::Scalar :
               (fieldType.find("Vector")!=std::string::npos ? FRT::Vector :
               (fieldType.find("Gradient")!=std::string::npos ? FRT::Gradient : FRT::Tensor));

    // Get data layout
    if (rank == FRT::Scalar) {
      state_dl = loc == FL::Node
               ? dl->node_scalar
               : dl->cell_scalar2;
    } else if (rank == FRT::Vector) {
      state_dl = loc == FL::Node
               ? dl->node_vector
               : dl->cell_vector;
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION (false, std::runtime_error,
          "Error! Only scalar/vector states supported by ColumnCouplingTest.\n");
    }

    // Set entity for state struct
    if(loc==FL::Cell) {
      entity = Albany::StateStruct::ElemData;
    } else {
      entity = Albany::StateStruct::NodalDataToElemNode;
    }

    // Register the state
    p = stateMgr.registerStateVariable(stateName, state_dl, meshSpecs.ebName, true, &entity, meshPart);

    // Create load/save evaluator(s)
    if ( fieldUsage == "Output" || fieldUsage == "Input-Output") {
      // Only save fields in the residual FM (and not in state/response FM)
      if (fieldManagerChoice == Albany::BUILD_RESID_FM) {
        // An output: save it.
        ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT,PHAL::AlbanyTraits>(*p));
        fm0.template registerEvaluator<EvalT>(ev);

        // Only PHAL::AlbanyTraits::Residual evaluates something, others will have empty list of evaluated fields
        if (ev->evaluatedFields().size()>0) {
          fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
        }
      }
    }

    if (fieldUsage == "Input" || fieldUsage == "Input-Output") {
      p->set<std::string>("Field Name", fieldName);
      ev = Teuchos::rcp(new PHAL::LoadStateFieldRT<EvalT,PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }
  }

  auto& ss_discs_params = discParams->sublist("Side Set Discretizations");
  auto& ss_names = ss_discs_params.get<Teuchos::Array<std::string>>("Side Sets");
  for (const auto& ss_name : ss_names) {
    auto& ss_info = ss_discs_params.sublist(ss_name).sublist("Required Fields Info");
    int num_ss_fields = ss_info.get<int>("Number Of Fields",0);

    Teuchos::RCP<Albany::Layouts> ss_dl = dl->side_layouts.at(ss_name);
    for (int ifield=0; ifield<num_ss_fields; ++ifield) {
      Teuchos::ParameterList& thisFieldList =  ss_info.sublist(util::strint("Field", ifield));

      // Get current state specs
      auto fieldName = thisFieldList.get<std::string>("Field Name");
      auto stateName = thisFieldList.get<std::string>("State Name", fieldName);
      fieldName = fieldName + "_" + ss_name;
      auto fieldUsage = thisFieldList.get<std::string>("Field Usage","Input"); // WARNING: assuming Input if not specified

      if (fieldUsage == "Unused") {
        continue;
      }

      auto meshPart = "";

      auto fieldType  = thisFieldList.get<std::string>("Field Type");
      auto loc = fieldType.find("Node")!=std::string::npos ? FL::Node : FL::Cell;

      TEUCHOS_TEST_FOR_EXCEPTION (
          fieldType.find("Scalar")==std::string::npos &&
          fieldType.find("Vector")==std::string::npos &&
          fieldType.find("Gradient")==std::string::npos &&
          fieldType.find("Tensor")==std::string::npos, std::runtime_error,
          "Error! Invalid rank type for state " + stateName + "\n");

      auto rank = fieldType.find("Scalar")!=std::string::npos ? FRT::Scalar :
                 (fieldType.find("Vector")!=std::string::npos ? FRT::Vector :
                 (fieldType.find("Gradient")!=std::string::npos ? FRT::Gradient : FRT::Tensor));

      // Get data layout
      if (rank == FRT::Scalar) {
        state_dl = loc == FL::Node
                 ? ss_dl->node_scalar
                 : ss_dl->cell_scalar2;
      } else if (rank == FRT::Vector) {
        state_dl = loc == FL::Node
                 ? ss_dl->node_vector
                 : ss_dl->cell_vector;
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION (false, std::runtime_error,
            "Error! Only scalar/vector states supported by ColumnCouplingTest.\n");
      }

      // Set entity for state struct
      if(loc==FL::Cell) {
        entity = Albany::StateStruct::ElemData;
      } else {
        entity = Albany::StateStruct::NodalDataToElemNode;
      }

      // Register the state
      p = stateMgr.registerSideSetStateVariable(ss_name, stateName, fieldName, state_dl, sideEBName.at(ss_name), true, &entity, meshPart);

      // Create load/save evaluator(s)
      if ( fieldUsage == "Output" || fieldUsage == "Input-Output") {
        // Only save fields in the residual FM (and not in state/response FM)
        if (fieldManagerChoice == Albany::BUILD_RESID_FM) {
          // An output: save it.
          // SaveSideSetStateField takes the layout from dl, using FRT and FL to determine it.
          // It does so, in order to do J*v if v is a Gradient (covariant), where J is the 2x3
          // matrix of the tangent vectors
          p->set("Field Rank",rank);
          p->set("Field Location",loc);
          ev = Teuchos::rcp(new PHAL::SaveSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p,ss_dl));
          fm0.template registerEvaluator<EvalT>(ev);

          // Only PHAL::AlbanyTraits::Residual evaluates something, others will have empty list of evaluated fields
          if (ev->evaluatedFields().size()>0) {
            fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
          }
        }
      }

      if (fieldUsage == "Input" || fieldUsage == "Input-Output") {
        p->set<std::string>("Field Name", fieldName);
        ev = Teuchos::rcp(new PHAL::LoadSideSetStateFieldRT<EvalT,PHAL::AlbanyTraits>(*p));
        fm0.template registerEvaluator<EvalT>(ev);
      }
    }
  }
}


} // namespace Albany

#endif // ALBANY_COLUMN_COUPLING_TEST_HPP
