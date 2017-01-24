//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_SCHOOF_FIT_HPP
#define FELIX_SCHOOF_FIT_HPP 1

#include <type_traits>

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
#include "FELIX_SimpleOperation.hpp"
#include "FELIX_ParamEnum.hpp"

#include "FELIX_EffectivePressure.hpp"
#include "FELIX_DummyResidual.hpp"
#include "PHAL_Field2Norm.hpp"
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

  Teuchos::RCP<shards::CellTopology>                                cellType;
  Teuchos::RCP<Intrepid2::Cubature<PHX::Device>>                    cellCubature;
  Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>>   cellBasis;

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

  std::string param_name;

  // ---------------------------- Registering state variables ------------------------- //

  // Loading info on states: is a parameter? is it a vector? does it need to be saved?
  Teuchos::Array<std::string> vector_states  = params->get<Teuchos::Array<std::string>>("Required Vector Fields");
  Teuchos::Array<std::string> states_to_save = params->get<Teuchos::Array<std::string>>("Save Fields");
  std::map<std::string,bool> require_state_save;
  std::map<std::string,bool> is_state_a_parameter;
  std::map<std::string,bool> is_vector_state;
  std::map<std::string,bool> save_state;
  std::map<std::string,std::string> state_mesh_part;
  std::map<std::string,std::string> state_to_field;
  for (const std::string& stateName : this->requirements)
  {
    is_state_a_parameter[stateName] = false;
    state_mesh_part[stateName]      = "";
    save_state[stateName]           = false;
    is_vector_state[stateName]      = false;
  }
  for (int i=0; i<vector_states.size(); ++i)
  {
    is_vector_state[vector_states[i]] = true;
  }
  for (int i=0; i<states_to_save.size(); ++i)
  {
    save_state[states_to_save[i]] = true;
  }

  Teuchos::ParameterList& dist_params_list =  this->params->sublist("Distributed Parameters");
  Teuchos::ParameterList* param_list;
  int numParams = dist_params_list.get<int>("Number of Parameter Vectors",0);
  for (int p_index=0; p_index< numParams; ++p_index)
  {
    std::string parameter_sublist_name = Albany::strint("Distributed Parameter", p_index);
    TEUCHOS_TEST_FOR_EXCEPTION (!dist_params_list.isSublist(parameter_sublist_name), std::logic_error,
                                "Error! Missing sublist '" << parameter_sublist_name << "'.\n");
    param_list = &dist_params_list.sublist(parameter_sublist_name);
    std::string stateName = param_list->get<std::string>("Name", "");
    TEUCHOS_TEST_FOR_EXCEPTION (is_state_a_parameter.find(stateName)==is_state_a_parameter.end(), Teuchos::Exceptions::InvalidParameter,
                                "Error! Distributed parameter '" << stateName << "' was not found in the list of required fields.\n");

    is_state_a_parameter[stateName] = true;
    state_mesh_part[stateName]      = param_list->get<std::string>("Mesh Part","");
  }

  for (int i=0; i<this->requirements.size(); ++i)
  {
    const std::string& stateName = this->requirements[i];
    const std::string& fieldName = stateName;
    entity = is_state_a_parameter[stateName] ? Albany::StateStruct::NodalDistParameter : Albany::StateStruct::NodalDataToElemNode;
    Teuchos::RCP<PHX::DataLayout> layout = is_vector_state[stateName] ? dl->node_vector : dl->node_scalar;
    p = stateMgr.registerStateVariable(stateName, layout, elementBlockName, true, &entity, state_mesh_part[stateName]);
    p->set<std::string>("Field Name", fieldName);

    if (is_state_a_parameter[stateName])
    {
      ev = evalUtils.constructGatherScalarNodalParameter(stateName,fieldName);
      fm0.template registerEvaluator<EvalT> (ev);
    }
    else if (!save_state[stateName])
    {
      ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT> (ev);
    }

    if (save_state[stateName])
    {
      // We save it to the mesh
      p->set<bool>("Is Vector Field", is_vector_state[stateName]);
      p->set<bool>("Nodal State",true);
      ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT,PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);

      // Require to save only with response
      if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM)
        // Only PHAL::AlbanyTraits::Residual evaluates something
        if (ev->evaluatedFields().size()>0)
          fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
    }
  }

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
  ev = evalUtils.getPSTUtils().constructDOFInterpolationEvaluator("effective_pressure");
  fm0.template registerEvaluator<EvalT> (ev);

  // Gradient of bed_roughness
  ev = evalUtils.getPSTUtils().constructDOFGradInterpolationEvaluator("bed_roughness");
  fm0.template registerEvaluator<EvalT> (ev);

  // Interpolate basal velocity
  ev = evalUtils.getPSTUtils().constructDOFVecInterpolationEvaluator("basal_velocity");
  fm0.template registerEvaluator<EvalT> (ev);

  // Scatter residual
  ev = evalUtils.constructScatterResidualEvaluator(false, resid_names, offset, "Scatter Schoof Fit");
  fm0.template registerEvaluator<EvalT> (ev);

  // Interpolate Beta Given
  ev = evalUtils.getPSTUtils().constructDOFInterpolationEvaluator ("basal_friction");
  fm0.template registerEvaluator<EvalT> (ev);

  // Compute basis funcitons
  ev = evalUtils.constructComputeBasisFunctionsEvaluator(cellType, cellBasis, cellCubature);
  fm0.template registerEvaluator<EvalT> (ev);

  // -------------------------------- FELIX evaluators ------------------------- //
  if (!is_state_a_parameter["effective_pressure"])
  {
    //--- Effective pressure (surrogate) calculation ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Effective Pressure Surrogate"));

    // Input
    p->set<std::string>("Surface Height Variable Name","surface_height");
    p->set<std::string>("Ice Thickness Variable Name", "thickness");
    p->set<Teuchos::ParameterList*>("FELIX Physical Parameters", &params->sublist("FELIX Physical Parameters"));
    p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("FELIX Effective Pressure Surrogate"));

    // Output
    p->set<std::string>("Effective Pressure Variable Name","effective_pressure");

    ev = Teuchos::rcp(new FELIX::EffectivePressure<EvalT,PHAL::AlbanyTraits,false,true>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);

    //--- Shared Parameter for basal friction coefficient: alpha ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("Basal Friction Coefficient: alpha"));

    param_name = "Hydraulic-Over-Hydrostatic Potential Ratio";
    p->set<std::string>("Parameter Name", param_name);
    p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

    Teuchos::RCP<FELIX::SharedParameter<EvalT,PHAL::AlbanyTraits,FelixParamEnum,FelixParamEnum::Alpha>> ptr_alpha;
    ptr_alpha = Teuchos::rcp(new FELIX::SharedParameter<EvalT,PHAL::AlbanyTraits,FelixParamEnum,FelixParamEnum::Alpha>(*p,dl));
    ptr_alpha->setNominalValue(params->sublist("Parameters"),params->sublist("FELIX Basal Friction Coefficient").get<double>(param_name,-1.0));
    fm0.template registerEvaluator<EvalT>(ptr_alpha);
  }

  // --- Schoof Fit Resid --- //
  p = Teuchos::rcp(new Teuchos::ParameterList("Schoof Fit Resid"));

  //Input
  p->set<std::string>("Solution Variable Name", dof_names[0]);

  //Output
  p->set<std::string>("Residual Variable Name", resid_names[0]);

  ev = Teuchos::rcp(new FELIX::DummyResidual<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  //--- Sliding velocity calculation ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Velocity Norm"));

  // Input
  p->set<std::string>("Field Name","basal_velocity");
  p->set<std::string>("Field Layout","Cell Node Vector");
  p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("FELIX Field Norm"));

  // Output
  p->set<std::string>("Field Norm Name","sliding_velocity");

  ev = Teuchos::rcp(new PHAL::Field2NormParam<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  //--- Sliding velocity calculation ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Velocity Norm"));

  // Input
  p->set<std::string>("Field Name","basal_velocity");
  p->set<std::string>("Field Layout","Cell QuadPoint Vector");
  p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("FELIX Field Norm"));

  // Output
  p->set<std::string>("Field Norm Name","sliding_velocity");

  ev = Teuchos::rcp(new PHAL::Field2NormParam<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

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

  //--- FELIX basal friction coefficient ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Basal Friction Coefficient"));

  //Input
  p->set<std::string>("Sliding Velocity QP Variable Name", "sliding_velocity");
  p->set<std::string>("Effective Pressure QP Variable Name", "effective_pressure");
  p->set<std::string>("Bed Roughness Variable Name", "bed_roughness");
  p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("FELIX Basal Friction Coefficient"));
  p->set<Teuchos::ParameterList*>("Stereographic Map", &params->sublist("Stereographic Map"));

  //Output
  p->set<std::string>("Basal Friction Coefficient Variable Name", "beta");

  ev = Teuchos::rcp(new FELIX::BasalFrictionCoefficient<EvalT,PHAL::AlbanyTraits,false,false>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  //--- Basal Friction log ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Basal Friction Coefficient"));

  //Input
  p->set<std::string>("Input Field Name", "basal_friction");
  p->set<Teuchos::RCP<PHX::DataLayout>>("Field Layout", dl->qp_scalar);

  //Output
  p->set<std::string>("Output Field Name", "log_basal_friction");

  ev = Teuchos::rcp(new FELIX::SimpleOperationLog<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  //--- FELIX basal friction coefficient log ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Basal Friction Coefficient"));

  //Input
  p->set<std::string>("Input Field Name", "beta");
  p->set<Teuchos::RCP<PHX::DataLayout>>("Field Layout", dl->qp_scalar);

  //Output
  p->set<std::string>("Output Field Name", "log_beta");

  ev = Teuchos::rcp(new FELIX::SimpleOperationLog<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  if (save_state["beta"])
  {
    //--- FELIX basal friction coefficient at nodes ---//
    p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Basal Friction Coefficient"));

    //Input
    p->set<std::string>("Sliding Velocity Variable Name", "sliding_velocity");
    p->set<std::string>("Effective Pressure Variable Name", "effective_pressure");
    p->set<std::string>("Bed Roughness Variable Name", "bed_roughness");
    p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("FELIX Basal Friction Coefficient"));

    //Output
    p->set<std::string>("Basal Friction Coefficient Variable Name", "beta");

    ev = Teuchos::rcp(new FELIX::BasalFrictionCoefficientNode<EvalT,PHAL::AlbanyTraits,false,false>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

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
