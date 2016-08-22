//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_SCHOOF_FIT_HPP
#define FELIX_SCHOOF_FIT_HPP 1

#include <type_traits>

#include "Intrepid2_FieldContainer.hpp"
#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Phalanx.hpp"
#include "Shards_CellTopology.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractProblem.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "Albany_ResponseUtilities.hpp"

#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_AlbanyTraits.hpp"

#include "PHAL_LoadStateField.hpp"
#include "PHAL_DOFVecInterpolationSide.hpp"
#include "PHAL_SaveStateField.hpp"
#include "FELIX_SharedParameter.hpp"
#include "FELIX_ParamEnum.hpp"

#include "FELIX_EffectivePressure.hpp"
#include "FELIX_DummyResidual.hpp"
#include "FELIX_FieldNorm.hpp"
#include "FELIX_BasalFrictionCoefficient.hpp"
#include "FELIX_BasalFrictionCoefficientNode.hpp"

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN

namespace FELIX
{

/*!
 * \brief Abstract interface for representing a 1-D finite element
 * problem.
 */
class SchoofFit : public Albany::AbstractProblem
{
public:

  //! Default constructor
  SchoofFit (const Teuchos::RCP<Teuchos::ParameterList>& params,
             const Teuchos::RCP<ParamLib>& paramLib,
             const int numDim_);

  //! Destructor
  ~SchoofFit();

  //! Return number of spatial dimensions
  virtual int spatialDimension() const { return numDim; }

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

  //! Each problem must generate it's list of valide parameters
  Teuchos::RCP<const Teuchos::ParameterList> getValidProblemParameters() const;

private:

  //! Private to prohibit copying
  SchoofFit(const SchoofFit&);

  //! Private to prohibit copying
  SchoofFit& operator=(const SchoofFit&);

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

  Teuchos::RCP<Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,PHX::Device> > >  cellCubature;

  Teuchos::RCP<Intrepid2::Basis<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > > cellBasis;

  int numDim;
  Teuchos::RCP<Albany::Layouts> dl;

  std::string elementBlockName;
};

} // Namespace FELIX

// ================================ IMPLEMENTATION ============================ //

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
FELIX::SchoofFit::constructEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                      const Albany::MeshSpecsStruct& meshSpecs,
                                      Albany::StateManager& stateMgr,
                                      Albany::FieldManagerChoice fieldManagerChoice,
                                      const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);

  Albany::StateStruct::MeshFieldEntity entity;
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  Teuchos::RCP<Teuchos::ParameterList> p;

  // ---------------------------- Registering state variables ------------------------- //

  std::string stateName, fieldName, param_name, emptyString;
  std::string* meshPart;

  stateName = "effective_pressure";
  TEUCHOS_TEST_FOR_EXCEPTION (!this->params->isSublist("Distributed Parameters"), std::runtime_error, "Error! Missing distributed parameter list.\n");

  Teuchos::ParameterList& dist_params_list =  this->params->sublist("Distributed Parameters");
  Teuchos::ParameterList* param_list;
  int numParams = dist_params_list.get<int>("Number of Parameter Vectors",0);
  for (int p_index=0; p_index< numParams; ++p_index)
  {
    std::string parameter_sublist_name = Albany::strint("Distributed Parameter", p_index);
    if (dist_params_list.isSublist(parameter_sublist_name))
    {
      param_list = &dist_params_list.sublist(parameter_sublist_name);
      if (param_list->get<std::string>("Name", emptyString) == stateName)
      {
        meshPart = &param_list->get<std::string>("Mesh Part",emptyString);
        break;
      }
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION (stateName==dist_params_list.get(Albany::strint("Parameter",p_index),emptyString),
                                  std::logic_error, "Error! '" << stateName << "' must be a distributed parameter.\n");
    }
  }

  TEUCHOS_TEST_FOR_EXCEPTION (std::find(requirements.begin(), requirements.end(), stateName)==requirements.end(), std::logic_error,
                              "Error! '" << stateName << "' is a parameter, but is not listed as requirements.\n");

  // effective_pressure is a distributed 3D parameter
  entity = Albany::StateStruct::NodalDistParameter;
  fieldName = "Effective Pressure";
  p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity, *meshPart);
  p->set<std::string>("Field Name", fieldName);
  ev = evalUtils.constructGatherScalarNodalParameter(stateName,fieldName);
  fm0.template registerEvaluator<EvalT>(ev);

  // We save it to the mesh
  p->set<bool>("Is Vector Field", false);
  p->set<bool>("Nodal State",true);
  ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT,PHAL::AlbanyTraits>(*p));
  fm0.template registerEvaluator<EvalT>(ev);

  // Save the response
  if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM)
    // Only PHAL::AlbanyTraits::Residual evaluates something
    if (ev->evaluatedFields().size()>0)
      fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);

  // basal_friction
  entity = Albany::StateStruct::NodalDataToElemNode;
  stateName = "basal_friction";
  fieldName = "Beta Given";
  p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity);
  p->set<std::string>("Field Name", fieldName);
  ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
  fm0.template registerEvaluator<EvalT>(ev);

  // basal_velocity
  entity = Albany::StateStruct::NodalDataToElemNode;
  stateName = "basal_velocity";
  fieldName = "Basal Velocity";
  p = stateMgr.registerStateVariable(stateName, dl->node_vector, elementBlockName, true, &entity);
  p->set<std::string>("Field Name", fieldName);
  ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
  fm0.template registerEvaluator<EvalT>(ev);

  bool optimize_on_sliding_velocity=false;
  if (std::find(requirements.begin(),requirements.end(),"sliding_velocity")!=requirements.end())
    optimize_on_sliding_velocity = true;

  if (optimize_on_sliding_velocity)
  {
    // sliding_velocity is a distributed 3D parameter
    entity = Albany::StateStruct::NodalDistParameter;
    stateName = "sliding_velocity";
    fieldName = "Sliding Velocity";
    p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity, *meshPart);
    p->set<std::string>("Field Name", fieldName);
    ev = evalUtils.constructGatherScalarNodalParameter(stateName,fieldName);
    fm0.template registerEvaluator<EvalT>(ev);

    // We save it to the mesh
    p->set<bool>("Is Vector Field", false);
    p->set<bool>("Nodal State",true);
    ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    // Save the response
    if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM)
      // Only PHAL::AlbanyTraits::Residual evaluates something
      if (ev->evaluatedFields().size()>0)
        fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
  }

  // beta is an output field
  entity = Albany::StateStruct::NodalDataToElemNode;
  stateName = "beta";
  fieldName = "Beta";
  p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity, *meshPart);
  p->set<std::string>("Field Name", fieldName);
  p->set<bool>("Is Vector Field", false);
  p->set<bool>("Nodal State",true);
  ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT,PHAL::AlbanyTraits>(*p));
  fm0.template registerEvaluator<EvalT>(ev);

  // Save the response
  if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM)
    // Only PHAL::AlbanyTraits::Residual evaluates something
    if (ev->evaluatedFields().size()>0)
      fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);

  // Residual and solution names
  Teuchos::ArrayRCP<std::string> dof_names(1), resid_names(1);
  dof_names[0] = "Solution";
  resid_names[0] = "Residual";
  int offset = 0;

  // ------------------- Interpolations and utilities ------------------ //

  // Gather solution field
  ev = evalUtils.constructGatherSolutionEvaluator_noTransient(false, dof_names);
  fm0.template registerEvaluator<EvalT> (ev);

  // Gather coordinates
  ev = evalUtils.constructGatherCoordinateVectorEvaluator();
  fm0.template registerEvaluator<EvalT> (ev);

  // Interpolate effective pressure
  ev = evalUtils.getPSTUtils().constructDOFInterpolationEvaluator("Effective Pressure");
  fm0.template registerEvaluator<EvalT> (ev);

  if (optimize_on_sliding_velocity)
  {
    // Interpolate sliding velocity
    ev = evalUtils.getPSTUtils().constructDOFInterpolationEvaluator("Sliding Velocity");
    fm0.template registerEvaluator<EvalT> (ev);
  }
  else
  {
    // Interpolate basal velocity
    ev = evalUtils.getPSTUtils().constructDOFVecInterpolationEvaluator("Basal Velocity");
    fm0.template registerEvaluator<EvalT> (ev);
  }

  // Scatter residual
  ev = evalUtils.constructScatterResidualEvaluator(false, resid_names, offset, "Scatter Schoof Fit");
  fm0.template registerEvaluator<EvalT> (ev);

  // Interpolate Beta Given
  ev = evalUtils.getPSTUtils().constructDOFInterpolationEvaluator ("Beta Given");
  fm0.template registerEvaluator<EvalT> (ev);

  // Compute basis funcitons
  ev = evalUtils.constructComputeBasisFunctionsEvaluator(cellType, cellBasis, cellCubature);
  fm0.template registerEvaluator<EvalT> (ev);

  //---- Interpolate velocity on QP on
  ev = evalUtils.constructDOFVecInterpolationEvaluator("Basal Velocity");
  fm0.template registerEvaluator<EvalT>(ev);

  // -------------------------------- FELIX evaluators ------------------------- //

  // --- Schoof Fit Resid --- //
  p = Teuchos::rcp(new Teuchos::ParameterList("Schoof Fit Resid"));

  //Input
  p->set<std::string>("Solution Variable Name", dof_names[0]);

  //Output
  p->set<std::string>("Residual Variable Name", resid_names[0]);

  ev = Teuchos::rcp(new FELIX::DummyResidual<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  if (!optimize_on_sliding_velocity)
  {
    //--- Sliding velocity calculation ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Velocity Norm"));

    // Input
    p->set<std::string>("Field Name","Basal Velocity");
    p->set<std::string>("Field Layout","Cell Node Vector");
    p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("FELIX Field Norm"));

    // Output
    p->set<std::string>("Field Norm Name","Sliding Velocity");

    ev = Teuchos::rcp(new FELIX::FieldNormParam<EvalT,PHAL::AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);

    //--- Sliding velocity calculation ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Velocity Norm"));

    // Input
    p->set<std::string>("Field Name","Basal Velocity");
    p->set<std::string>("Field Layout","Cell QuadPoint Vector");
    p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("FELIX Field Norm"));

    // Output
    p->set<std::string>("Field Norm Name","Sliding Velocity");

    ev = Teuchos::rcp(new FELIX::FieldNormParam<EvalT,PHAL::AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  //--- Shared Parameter for basal friction coefficient: lambda ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("Basal Friction Coefficient: lambda"));

  param_name = "Bed Roughness";
  p->set<std::string>("Parameter Name", param_name);
  p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

  Teuchos::RCP<FELIX::SharedParameter<EvalT,PHAL::AlbanyTraits,FelixParamEnum,FelixParamEnum::Lambda>> ptr_lambda;
  ptr_lambda = Teuchos::rcp(new FELIX::SharedParameter<EvalT,PHAL::AlbanyTraits,FelixParamEnum,FelixParamEnum::Lambda>(*p,dl));
  ptr_lambda->setNominalValue(params->sublist("Parameters"),params->sublist("FELIX Basal Friction Coefficient").get<double>(param_name,-1.0));
  fm0.template registerEvaluator<EvalT>(ptr_lambda);

  //--- Shared Parameter for basal friction coefficient: mu ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("Basal Friction Coefficient: mu"));

  param_name = "Coulomb Friction Coefficient";
  p->set<std::string>("Parameter Name", param_name);
  p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

  Teuchos::RCP<FELIX::SharedParameter<EvalT,PHAL::AlbanyTraits,FelixParamEnum,FelixParamEnum::Mu>> ptr_mu;
  ptr_mu = Teuchos::rcp(new FELIX::SharedParameter<EvalT,PHAL::AlbanyTraits,FelixParamEnum,FelixParamEnum::Mu>(*p,dl));
  ptr_mu->setNominalValue(params->sublist("Parameters"),params->sublist("FELIX Basal Friction Coefficient").get<double>(param_name,-1.0));
  fm0.template registerEvaluator<EvalT>(ptr_mu);

  //--- Shared Parameter for basal friction coefficient: power ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("Basal Friction Coefficient: power"));

  param_name = "Power Exponent";
  p->set<std::string>("Parameter Name", param_name);
  p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

  Teuchos::RCP<FELIX::SharedParameter<EvalT,PHAL::AlbanyTraits,FelixParamEnum,FelixParamEnum::Power>> ptr_power;
  ptr_power = Teuchos::rcp(new FELIX::SharedParameter<EvalT,PHAL::AlbanyTraits,FelixParamEnum,FelixParamEnum::Power>(*p,dl));
  ptr_power->setNominalValue(params->sublist("Parameters"),params->sublist("FELIX Basal Friction Coefficient").get<double>(param_name,-1.0));
  fm0.template registerEvaluator<EvalT>(ptr_power);

  //--- FELIX basal friction coefficient at nodes ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Basal Friction Coefficient"));

  //Input
  p->set<std::string>("Sliding Velocity Variable Name", "Sliding Velocity");
  p->set<std::string>("Effective Pressure Variable Name", "Effective Pressure");
  p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("FELIX Basal Friction Coefficient"));

  //Output
  p->set<std::string>("Basal Friction Coefficient Variable Name", "Beta");

  ev = Teuchos::rcp(new FELIX::BasalFrictionCoefficientNode<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  //--- FELIX basal friction coefficient ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Basal Friction Coefficient"));

  //Input
  p->set<std::string>("Sliding Velocity QP Variable Name", "Sliding Velocity");
  p->set<std::string>("BF Variable Name", "BF");
  p->set<std::string>("Effective Pressure QP Variable Name", "Effective Pressure");
  p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("FELIX Basal Friction Coefficient"));
  p->set<Teuchos::ParameterList*>("Stereographic Map", &params->sublist("Stereographic Map"));

  //Output
  p->set<std::string>("Basal Friction Coefficient Variable Name", "Beta");

  ev = Teuchos::rcp(new FELIX::BasalFrictionCoefficient<EvalT,PHAL::AlbanyTraits,false,false>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  p = Teuchos::rcp(new Teuchos::ParameterList("Gather Averaged Velocity"));

  if (fieldManagerChoice == Albany::BUILD_RESID_FM)
  {
    // Require scattering of residual
    PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter Schoof Fit", dl->dummy);
    fm0.requireField<EvalT>(res_tag);
  }
  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM)
  {
    // ----------------------- Responses --------------------- //
    Teuchos::RCP<Teuchos::ParameterList> paramList = Teuchos::rcp(new Teuchos::ParameterList("Param List"));
    paramList->set<Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
    return respUtils.constructResponses(fm0, *responseList, paramList, stateMgr);
  }

  return Teuchos::null;
}

#endif // FELIX_SCHOOF_FIT_HPP
