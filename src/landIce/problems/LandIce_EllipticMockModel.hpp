//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_ELLIPTIC_MOCK_MODEL_PROBLEM_HPP
#define LANDICE_ELLIPTIC_MOCK_MODEL_PROBLEM_HPP 1

#include "Albany_ResponseUtilities.hpp"
#include "LandIce_EllipticMockModelResidual.hpp"

#include "Albany_AbstractProblem.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "Albany_GeneralPurposeFieldsNames.hpp"
#include "Albany_StringUtils.hpp"

#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_LoadStateField.hpp"
#include "PHAL_LoadSideSetStateField.hpp"

#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include <type_traits>

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN

namespace LandIce
{

/*!
 * \brief Abstract interface for representing a 1-D finite element
 * problem.
 */
class EllipticMockModel : public Albany::AbstractProblem
{
public:

  //! Default constructor
  EllipticMockModel (const Teuchos::RCP<Teuchos::ParameterList>& params,
            const Teuchos::RCP<Teuchos::ParameterList>& discParams,
            const Teuchos::RCP<ParamLib>& paramLib,
            const int numDim_);

  //! Destructor
  ~EllipticMockModel();

  //! Return number of spatial dimensions
  virtual int spatialDimension() const { return numDim; }

  //! Get boolean telling code if SDBCs are utilized
  virtual bool useSDBCs() const {return use_sdbcs_; }

  //! Build the PDE instantiations, boundary conditions, and initial solution
  virtual void buildProblem (Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
                             Albany::StateManager& stateMgr);

  // Build evaluators
  virtual Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
  buildEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                   const Albany::MeshSpecsStruct& meshSpecs,
                   Albany::StateManager& stateMgr,
                   Albany::FieldManagerChoice fmchoice,
                   const Teuchos::RCP<Teuchos::ParameterList>& responseList);

  //! Each problem must generate it's list of valid parameters
  Teuchos::RCP<const Teuchos::ParameterList> getValidProblemParameters() const;

private:

  //! Private to prohibit copying
  EllipticMockModel(const EllipticMockModel&);

  //! Private to prohibit copying
  EllipticMockModel& operator=(const EllipticMockModel&);

public:

  //! Main problem setup routine. Not directly called, but indirectly by following functions
  template <typename EvalT> Teuchos::RCP<const PHX::FieldTag>
  constructEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                       const Albany::MeshSpecsStruct& meshSpecs,
                       Albany::StateManager& stateMgr,
                       Albany::FieldManagerChoice fmchoice,
                       const Teuchos::RCP<Teuchos::ParameterList>& responseList);

  void constructDirichletEvaluators(const Albany::MeshSpecsStruct& meshSpecs);
  void constructNeumannEvaluators(const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs);

protected:

  Teuchos::RCP<shards::CellTopology> cellType;
  Teuchos::RCP<shards::CellTopology> sideType;

  Teuchos::RCP<Intrepid2::Cubature<PHX::Device> >  cellCubature, sideCubature;

  Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> > cellBasis;
  Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> > sideBasis;

  unsigned int numDim;
  Teuchos::RCP<Albany::Layouts> dl, dl_side;

  //! Discretization parameters
  Teuchos::RCP<Teuchos::ParameterList> discParams;

  std::string elementBlockName;
  std::string sideName;
  /// Boolean marking whether SDBCs are used
  bool use_sdbcs_;
};

} // Namespace LandIce

// ================================ IMPLEMENTATION ============================ //

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
LandIce::EllipticMockModel::constructEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                      const Albany::MeshSpecsStruct& meshSpecs,
                                      Albany::StateManager& stateMgr,
                                      Albany::FieldManagerChoice fieldManagerChoice,
                                      const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);

  int offset=0;

  Albany::StateStruct::MeshFieldEntity entity;
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  Teuchos::RCP<Teuchos::ParameterList> p;

  // ---------------------------- Registering state variables ------------------------- //

  std::string stateName, param_name;

  const Teuchos::ParameterList& parameterParams = this->params->sublist("Parameters");
  int total_num_param_vecs, num_param_vecs, num_dist_param_vecs;
  Albany::getParameterSizes(parameterParams, total_num_param_vecs, num_param_vecs, num_dist_param_vecs);

  {
    stateName = "parameter";
    bool isParameter = false;
    for (int p_index=0; p_index< num_dist_param_vecs; ++p_index) {
      std::string parameter_sublist_name = util::strint("Parameter", p_index+num_param_vecs);
      Teuchos::ParameterList param_list = parameterParams.sublist(parameter_sublist_name);
      param_name = param_list.get<std::string>("Name");
      if(param_name == stateName) {
        isParameter = true;
        break;
      }
    }
    if(isParameter) {
      entity = Albany::StateStruct::NodalDistParameter;
      stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity);
      ev = evalUtils.constructGatherScalarNodalParameter(stateName,stateName);
    } else {
      entity = Albany::StateStruct::NodalDataToElemNode;
      p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity);
      p->set<std::string>("Field Name", stateName);
      ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
    }
    fm0.template registerEvaluator<EvalT>(ev);
    ev = evalUtils.getPSTUtils().constructDOFInterpolationEvaluator(stateName);
    fm0.template registerEvaluator<EvalT> (ev);

    stateName = "nominal_parameter";
    isParameter = false;
    for (int p_index=0; p_index< num_dist_param_vecs; ++p_index) {
      std::string parameter_sublist_name = util::strint("Parameter", p_index+num_param_vecs);
      Teuchos::ParameterList param_list = parameterParams.sublist(parameter_sublist_name);
      param_name = param_list.get<std::string>("Name");
      if(param_name == stateName) {
        isParameter = true;
        break;
      }
    }
    if(isParameter) {
      entity = Albany::StateStruct::NodalDistParameter;
      stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity);
      ev = evalUtils.constructGatherScalarNodalParameter(stateName,stateName);
    } else {
      entity = Albany::StateStruct::NodalDataToElemNode;
      p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity);
      p->set<std::string>("Field Name", stateName);
      ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
    }
    fm0.template registerEvaluator<EvalT>(ev);
    ev = evalUtils.getPSTUtils().constructDOFInterpolationEvaluator(stateName);
    fm0.template registerEvaluator<EvalT> (ev);

    stateName = "nominal_solution";
    entity = Albany::StateStruct::NodalDataToElemNode;
    p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity);
    p->set<std::string>("Field Name", stateName);
    ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
    ev = evalUtils.getPSTUtils().constructDOFInterpolationEvaluator(stateName);
    fm0.template registerEvaluator<EvalT> (ev);
    ev = evalUtils.getPSTUtils().constructDOFGradInterpolationEvaluator(stateName);
    fm0.template registerEvaluator<EvalT> (ev);

    stateName = "forcing";
    entity = Albany::StateStruct::NodalDataToElemNode;
    p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity);
    p->set<std::string>("Field Name", stateName);
    ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
    ev = evalUtils.getPSTUtils().constructDOFInterpolationEvaluator(stateName);
    fm0.template registerEvaluator<EvalT> (ev);

    stateName = "obs_solution";
    entity = Albany::StateStruct::NodalDistParameter;
    stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity);
    ev = evalUtils.constructGatherScalarNodalParameter(stateName,stateName);
    fm0.template registerEvaluator<EvalT>(ev);

  }
  
  if(numDim == 3) {
    Teuchos::Array<std::string> ss_names;
    if (discParams->sublist("Side Set Discretizations").isParameter("Side Sets")) {
      ss_names = discParams->sublist("Side Set Discretizations").get<Teuchos::Array<std::string>>("Side Sets");
    }
    for (unsigned int i=0; i<ss_names.size(); ++i) {
      const std::string& ss_name = ss_names[i];
      Teuchos::ParameterList& info = discParams->sublist("Side Set Discretizations").sublist(ss_name).sublist("Required Fields Info");
      int num_fields = info.get<int>("Number Of Fields",0);

      const std::string& sideEBName = meshSpecs.sideSetMeshSpecs.at(ss_name)[0]->ebName;
      Teuchos::RCP<Albany::Layouts> ss_dl = dl->side_layouts.at(ss_name);
      for (int ifield=0; ifield<num_fields; ++ifield) {
        Teuchos::ParameterList& thisFieldList =  info.sublist(util::strint("Field", ifield));

        // Get current state specs
        stateName = thisFieldList.get<std::string>("Field Name");
        std::string fieldName(stateName + "_" + ss_name);

        entity = Albany::StateStruct::NodalDataToElemNode;

        // Register the state
        p = stateMgr.registerSideSetStateVariable(ss_name, stateName, fieldName, ss_dl->node_scalar, sideEBName, true, &entity);
        ev = Teuchos::rcp(new PHAL::LoadSideSetStateFieldRT<EvalT,PHAL::AlbanyTraits>(*p));
        fm0.template registerEvaluator<EvalT>(ev);
      }
    }
  }


  // ----------  Define Field Names ----------- //
  Teuchos::ArrayRCP<std::string> dof_names(1);
  Teuchos::ArrayRCP<std::string> resid_names(1);
  dof_names[0] = "solution";


  // ------------------- Interpolations and utilities ------------------ //

  // Gather solution field
  ev = evalUtils.constructGatherSolutionEvaluator_noTransient(false, dof_names, offset);
  fm0.template registerEvaluator<EvalT> (ev);

  // Interpolate solution field
  ev = evalUtils.constructDOFInterpolationEvaluator(dof_names[0]);
  fm0.template registerEvaluator<EvalT> (ev);

  // Interpolate solution gradient
  ev = evalUtils.constructDOFGradInterpolationEvaluator(dof_names[0]);
  fm0.template registerEvaluator<EvalT> (ev);

  // Scatter residual
  resid_names[0] = "Elliptic Residual";
  ev = evalUtils.constructScatterResidualEvaluator(false, resid_names, offset, "Scatter Mock Elliptic");
  fm0.template registerEvaluator<EvalT> (ev);

  //----- Gather Coordinate Vector (general parameters)
  ev = evalUtils.constructGatherCoordinateVectorEvaluator();
  fm0.template registerEvaluator<EvalT> (ev);


  // Map to physical frame
  ev = evalUtils.constructMapToPhysicalFrameEvaluator(cellType, cellCubature);
  fm0.template registerEvaluator<EvalT> (ev);

  // Compute basis functions
  ev = evalUtils.constructComputeBasisFunctionsEvaluator(cellType, cellBasis, cellCubature);
  fm0.template registerEvaluator<EvalT> (ev);

  if(sideName != "INVALID") {
    //---- Restrict vertex coordinates from cell-based to cell-side-based
    ev = evalUtils.getMSTUtils().constructDOFCellToSideEvaluator(Albany::coord_vec_name,sideName,"Vertex Vector",cellType,Albany::coord_vec_name + "_" + sideName);
    fm0.template registerEvaluator<EvalT> (ev);

    ev = evalUtils.constructDOFCellToSideEvaluator(dof_names[0], sideName, "Node Scalar", cellType, dof_names[0] + "_" + sideName);
    fm0.template registerEvaluator<EvalT> (ev);

    ev = evalUtils.constructDOFInterpolationSideEvaluator (dof_names[0] + "_" + sideName, sideName);
    fm0.template registerEvaluator<EvalT> (ev);

    ev = evalUtils.constructComputeBasisFunctionsSideEvaluator(cellType, sideBasis, sideCubature, sideName);
    fm0.template registerEvaluator<EvalT> (ev);
  }

  // -------------------------------- LandIce evaluators ------------------------- //

  // --- FO Stokes Resid --- //
  {
    p = Teuchos::rcp(new Teuchos::ParameterList("Mock Elliptic Resid"));

    //Input
    p->set<std::string>("Gradient BF Name", Albany::grad_bf_name);
    p->set<std::string>("BF Name", Albany::bf_name);
    p->set<std::string>("Side BF Name", Albany::bf_name + "_" + sideName);
    p->set<std::string>("Field Variable Name", "solution");
    p->set<std::string>("Side Field Variable Name", "solution_" + sideName);
    p->set<std::string>("Field Gradient Variable Name", "solution Gradient");
    p->set<std::string>("Forcing Name", "forcing");
    p->set<std::string>("Nominal Field Name", "nominal_solution");
    p->set<std::string>("Nominal Field Gradient Name", "nominal_solution Gradient");
    p->set<std::string>("Parameter Name", "parameter");
    p->set<double>("Viscosity Regularization", params->sublist("Elliptic Model").get<double>("Viscosity Regularization"));
    p->set<double>("Glen's Law Exponent", params->sublist("Elliptic Model").get<double>("Glen's Law Exponent"));
    p->set<bool>("Linearize Model", params->sublist("Elliptic Model").get<bool>("Linearize Model"));
    p->set<std::string>("Nominal Parameter Name", "nominal_parameter");
    p->set<std::string>("Weighted Measure Name", Albany::weights_name);
    p->set<double>("Laplacian Coefficient", params->sublist("Elliptic Model").get<double>("Laplacian Coefficient"));
    p->set<double>("Robin Coefficient", params->sublist("Elliptic Model").get<double>("Robin Coefficient"));
    p->set<std::string>("Weighted Measure Side Name", Albany::weighted_measure_name + "_" + sideName);
    p->set<std::string>("Side Set Name", sideName);
    p->set<Teuchos::RCP<shards::CellTopology> >("Cell Type", cellType);

    //Output
    p->set<std::string>("Elliptic Mock Model Residual Name", "Elliptic Residual");

    ev = Teuchos::rcp(new LandIce::EllipticMockModelResidual<EvalT,PHAL::AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (fieldManagerChoice == Albany::BUILD_RESID_FM)
  {
    // Require scattering of residual
    PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter Mock Elliptic", dl->dummy);
    fm0.requireField<EvalT>(res_tag);
  }
  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM)
  {
    // ----------------------- Responses --------------------- //
    Teuchos::RCP<Teuchos::ParameterList> paramList = Teuchos::rcp(new Teuchos::ParameterList("Param List"));
    paramList->set<Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

    //LandIce::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
    //return respUtils.constructResponses(fm0, *responseList, paramList);
    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
    return respUtils.constructResponses(fm0, *responseList, paramList);
  }

  return Teuchos::null;
}

#endif // LANDICE_STOKES_FO_PROBLEM_HPP
