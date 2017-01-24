//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_HYDROLOGY_PROBLEM_HPP
#define FELIX_HYDROLOGY_PROBLEM_HPP 1

#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Phalanx.hpp"
#include "Shards_CellTopology.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractProblem.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "Albany_ResponseUtilities.hpp"

#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_SaveStateField.hpp"
#include "PHAL_LoadStateField.hpp"

#include "FELIX_BasalFrictionCoefficient.hpp"
#include "FELIX_EffectivePressure.hpp"
#include "FELIX_FieldNorm.hpp"
#include "FELIX_ParamEnum.hpp"
#include "FELIX_SharedParameter.hpp"
#include "FELIX_HydrologyDirichlet.hpp"
#include "FELIX_HydrologyWaterDischarge.hpp"
#include "FELIX_HydrologyWaterSource.hpp"
#include "FELIX_HydrologyMeltingRate.hpp"
#include "FELIX_HydrologyResidualPotentialEqn.hpp"
#include "FELIX_HydrologyResidualThicknessEqn.hpp"

namespace FELIX
{

/*!
 * \brief  A 2D problem for the subglacial hydrology
 */
class Hydrology : public Albany::AbstractProblem
{
public:

  //! Default constructor
  Hydrology (const Teuchos::RCP<Teuchos::ParameterList>& params,
             const Teuchos::RCP<ParamLib>& paramLib,
             const int numDimensions);

  //! Destructor
  virtual ~Hydrology();

  //! Return number of spatial dimensions
  virtual int spatialDimension () const
  {
      return numDim;
  }

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

  //! Main problem setup routine. Not directly called, but indirectly by buildEvaluators
  template <typename EvalT> Teuchos::RCP<const PHX::FieldTag>
  constructEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                       const Albany::MeshSpecsStruct& meshSpecs,
                       Albany::StateManager& stateMgr,
                       Albany::FieldManagerChoice fmchoice,
                       const Teuchos::RCP<Teuchos::ParameterList>& responseList);

  //! Boundary conditions evaluators
  void constructDirichletEvaluators (const Albany::MeshSpecsStruct& meshSpecs);
  void constructNeumannEvaluators   (const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs);

protected:

  struct HydrologyDirOp
  {
    HydrologyDirOp (Hydrology& pb) :
        m_pb (pb) {}

    template<typename EvalT>
    void operator() (EvalT x) const
    {
      m_pb.template constructDirichletEvaluators<EvalT>();
    }

    Hydrology& m_pb;
  };

  template<typename EvalT>
  void constructDirichletEvaluators () const;

  bool has_h_equation;
  bool unsteady;

  int numDim;
  std::string elementBlockName;

  Teuchos::ArrayRCP<std::string> dof_names;
  Teuchos::ArrayRCP<std::string> dof_names_dot;
  Teuchos::ArrayRCP<std::string> resid_names;

  Teuchos::RCP<Albany::Layouts> dl;

  Teuchos::RCP<shards::CellTopology> cellType;

  Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> > intrepidBasis;

  Teuchos::RCP<Intrepid2::Cubature<PHX::Device> > cubature;
};

// ===================================== IMPLEMENTATION ======================================= //

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
Hydrology::constructEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                const Albany::MeshSpecsStruct& meshSpecs,
                                Albany::StateManager& stateMgr,
                                Albany::FieldManagerChoice fieldManagerChoice,
                                const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  using Teuchos::RCP;
  using Teuchos::rcp;

  // Using the utility for the common evaluators
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);

  // Service variables for registering state variables and evaluators
  Albany::StateStruct::MeshFieldEntity entity;
  RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  RCP<Teuchos::ParameterList> p;
  std::string param_name, state_name, field_name;

  // -------------- Registering state variables --------------- //

  // Surface height
  entity = Albany::StateStruct::NodalDataToElemNode;
  p = stateMgr.registerStateVariable("surface_height", dl->node_scalar, elementBlockName, true, &entity);
  p->set<const std::string>("Field Name","Surface Height");
  ev = rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
  fm0.template registerEvaluator<EvalT>(ev);

  // Basal velocity
  entity = Albany::StateStruct::NodalDataToElemNode;
  p = stateMgr.registerStateVariable("basal_velocity", dl->node_vector, elementBlockName, true, &entity);
  p->set<const std::string>("Field Name","Basal Velocity");
  ev = rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
  fm0.template registerEvaluator<EvalT>(ev);

  // Basal Friction
  entity = Albany::StateStruct::NodalDataToElemNode;
  p = stateMgr.registerStateVariable("basal_friction", dl->node_scalar, elementBlockName, true, &entity);
  p->set<const std::string>("Field Name","Beta Given");
  ev = rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
  fm0.template registerEvaluator<EvalT>(ev);

  // If uested, we save an average of Beta Given on cells (for comparison with Beta)
  state_name = "basal_friction_cell_avg";
  field_name = "Beta Given";
  if (std::find(requirements.begin(), requirements.end(), state_name)!=requirements.end())
  {
    // We interpolate the given beta from quad point to side
    ev = evalUtils.getPSTUtils().constructNodesToCellInterpolationEvaluator (field_name, false);
    fm0.template registerEvaluator<EvalT>(ev);

    // We save it on the basal mesh
    entity = Albany::StateStruct::ElemData;
    p = stateMgr.registerStateVariable(state_name, dl->cell_scalar2, elementBlockName, true, &entity);
    p->set<std::string>("Field Name", field_name);
    p->set<bool>("Is Vector Field", false);
    ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    // Only PHAL::AlbanyTraits::Residual evaluates something
    if (ev->evaluatedFields().size()>0)
      fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
  }

  state_name = "beta_cell_avg";
  field_name = "Beta";
  if (std::find(requirements.begin(), requirements.end(), state_name)!=requirements.end())
  {
    // We interpolate beta from quad point to cell
    ev = evalUtils.constructQuadPointsToCellInterpolationEvaluator (field_name, false);
    fm0.template registerEvaluator<EvalT>(ev);

    // We save it on the basal mesh
    entity = Albany::StateStruct::ElemData;
    p = stateMgr.registerStateVariable(state_name, dl->cell_scalar2, elementBlockName, true, &entity);
    p->set<std::string>("Field Name", field_name);
    p->set<bool>("Is Vector Field", false);
    ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    // Only PHAL::AlbanyTraits::Residual evaluates something
    if (ev->evaluatedFields().size()>0)
      fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
  }

  // Ice thickness
  entity = Albany::StateStruct::NodalDataToElemNode;
  p = stateMgr.registerStateVariable("ice_thickness", dl->node_scalar, elementBlockName, true, &entity);
  p->set<const std::string>("Field Name","Ice Thickness");
  ev = rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
  fm0.template registerEvaluator<EvalT>(ev);

  if (params->sublist("FELIX Hydrology").get<bool>("Use SMB To Approximate Water Input",false))
  {
    // Surface Mass Balance
    entity = Albany::StateStruct::NodalDataToElemNode;
    p = stateMgr.registerStateVariable("surface_mass_balance", dl->node_scalar, elementBlockName, true, &entity);
    p->set<const std::string>("Field Name","Surface Mass Balance");
    ev = rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }
  else
  {
    // Surface water input
    entity = Albany::StateStruct::NodalDataToElemNode;
    p = stateMgr.registerStateVariable("surface_water_input", dl->node_scalar, elementBlockName, true, &entity);
    p->set<const std::string>("Field Name","Surface Water Input");
    ev = rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // GeothermaL flux
  entity = Albany::StateStruct::NodalDataToElemNode;
  p = stateMgr.registerStateVariable("geothermal_flux", dl->node_scalar, elementBlockName, true, &entity);
  p->set<const std::string>("Field Name","Geothermal Flux");
  ev = rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
  fm0.template registerEvaluator<EvalT>(ev);

  if (!has_h_equation)
  {
    // Water Thickness is not part of the solution, so we need to register it as a state and load it
    entity = Albany::StateStruct::NodalDataToElemNode;
    p = stateMgr.registerStateVariable("water_thickness", dl->node_scalar, elementBlockName, true, &entity);
    p->set<const std::string>("Field Name","Water Thickness");
    ev = rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (std::find(requirements.begin(),requirements.end(),"effective_pressure")!=requirements.end())
  {
    // Effective Pressure
    entity = Albany::StateStruct::NodalDataToElemNode;
    p = stateMgr.registerStateVariable("effective_pressure", dl->node_scalar, elementBlockName, true, &entity);
    p->set<const std::string>("Field Name","Effective Pressure");
    p->set<bool>("Nodal State",true);
    ev = rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    // Only PHAL::AlbanyTraits::Residual evaluates something
    if (ev->evaluatedFields().size()>0)
      fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
  }

  // -------------------- Interpolation and utilities ------------------------ //

  int offset_phi = 0;
  int offset_h = 1;

  // Gather solution field
  if (unsteady)
  {
    Teuchos::ArrayRCP<std::string> tmp;
    tmp.resize(1);
    tmp[0] = dof_names[0];

    ev = evalUtils.constructGatherSolutionEvaluator_noTransient (false, dof_names, offset_phi);
    fm0.template registerEvaluator<EvalT> (ev);

    tmp[0] = dof_names[1];
    ev = evalUtils.constructGatherSolutionEvaluator (false, tmp, dof_names_dot, offset_h);
    fm0.template registerEvaluator<EvalT> (ev);
  }
  else
  {
    ev = evalUtils.constructGatherSolutionEvaluator_noTransient (false, dof_names);
    fm0.template registerEvaluator<EvalT> (ev);
  }

  // Compute basis functions
  ev = evalUtils.constructComputeBasisFunctionsEvaluator(cellType, intrepidBasis, cubature);
  fm0.template registerEvaluator<EvalT> (ev);

  // Gather coordinates
  ev = evalUtils.constructGatherCoordinateVectorEvaluator();
  fm0.template registerEvaluator<EvalT> (ev);

  // Scatter residual
  int offset = 0;
  ev = evalUtils.constructScatterResidualEvaluator(false, resid_names, offset, "Scatter Hydrology");
  fm0.template registerEvaluator<EvalT> (ev);

  // Get qp coordinates (for source terms)
  ev = evalUtils.constructMapToPhysicalFrameEvaluator(cellType, cubature);
  fm0.template registerEvaluator<EvalT> (ev);

  // Interpolate Hydraulic Potential
  ev = evalUtils.constructDOFInterpolationEvaluator("Hydraulic Potential");
  fm0.template registerEvaluator<EvalT> (ev);

  // Interpolate Beta Given (may not be needed)
  ev = evalUtils.getPSTUtils().constructDOFInterpolationEvaluator("Beta Given");
  fm0.template registerEvaluator<EvalT> (ev);

  // Interpolate Effective Pressure
  ev = evalUtils.constructDOFInterpolationEvaluator("Effective Pressure");
  fm0.template registerEvaluator<EvalT> (ev);

  // Water Thickness
  if (has_h_equation)
  {
    ev = evalUtils.constructDOFInterpolationEvaluator("Water Thickness");
    fm0.template registerEvaluator<EvalT> (ev);
    if (unsteady)
    {
      // Interpolate Water Thickness Time Derivative
      ev = evalUtils.constructDOFInterpolationEvaluator("Water Thickness Dot");
      fm0.template registerEvaluator<EvalT> (ev);
    }
  }
  else
  {
    ev = evalUtils.getPSTUtils().constructDOFInterpolationEvaluator("Water Thickness");
    fm0.template registerEvaluator<EvalT> (ev);
  }

  // Hydraulic Potential Gradient
  ev = evalUtils.constructDOFGradInterpolationEvaluator("Hydraulic Potential");
  fm0.template registerEvaluator<EvalT> (ev);

  // Basal Velocity
  ev = evalUtils.getPSTUtils().constructDOFVecInterpolationEvaluator("Basal Velocity");
  fm0.template registerEvaluator<EvalT> (ev);

  // Surface Water Input
  ev = evalUtils.getPSTUtils().constructDOFInterpolationEvaluator("Surface Water Input");
  fm0.template registerEvaluator<EvalT> (ev);

  // Geothermal Flux
  ev = evalUtils.getPSTUtils().constructDOFInterpolationEvaluator("Geothermal Flux");
  fm0.template registerEvaluator<EvalT> (ev);

  // --------------------------------- FELIX evaluators -------------------------------- //

  if (params->sublist("FELIX Hydrology").get<bool>("Use SMB To Approximate Water Input",false))
  {
    //--- Compute Water Input From SMB
    p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Hydrology Water Input"));

    // Input
    p->set<std::string>("Surface Mass Balance Variable Name","Surface Mass Balance");

    // Output
    p->set<std::string>("Surface Water Input Variable Name","Surface Water Input");

    ev = Teuchos::rcp(new FELIX::HydrologyWaterSource<EvalT,PHAL::AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  //--- Hydraulic Potential Gradient Norm ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Potential Gradient Norm"));

  // Input
  p->set<std::string>("Field Name","Hydraulic Potential Gradient");
  p->set<std::string>("Field Layout","Cell QuadPoint Gradient");
  p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("FELIX Field Norm"));

  // Output
  p->set<std::string>("Field Norm Name","Hydraulic Potential Gradient Norm");

  ev = Teuchos::rcp(new FELIX::FieldNorm<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  // ------- Hydrology Water Discharge -------- //
  p = rcp(new Teuchos::ParameterList("Hydrology Water Discharge"));

  //Input
  p->set<std::string> ("Water Thickness QP Variable Name","Water Thickness");
  p->set<std::string> ("Hydraulic Potential Gradient QP Variable Name","Hydraulic Potential Gradient");
  p->set<std::string> ("Hydraulic Potential Gradient Norm QP Variable Name","Hydraulic Potential Gradient Norm");

  p->set<Teuchos::ParameterList*> ("FELIX Hydrology",&params->sublist("FELIX Hydrology"));
  p->set<Teuchos::ParameterList*> ("FELIX Physical Parameters",&params->sublist("FELIX Physical Parameters"));

  //Output
  p->set<std::string> ("Water Discharge QP Variable Name","Water Discharge");

  if (has_h_equation)
    ev = rcp(new FELIX::HydrologyWaterDischarge<EvalT,PHAL::AlbanyTraits,true,false>(*p,dl));
  else
    ev = rcp(new FELIX::HydrologyWaterDischarge<EvalT,PHAL::AlbanyTraits,false,false>(*p,dl));

  fm0.template registerEvaluator<EvalT>(ev);

  // ------- Hydrology Melting Rate -------- //
  p = rcp(new Teuchos::ParameterList("Hydrology Melting Rate"));

  //Input
  p->set<std::string> ("Geothermal Heat Source QP Variable Name","Geothermal Flux");
  p->set<std::string> ("Sliding Velocity QP Variable Name","Sliding Velocity");
  p->set<std::string> ("Basal Friction Coefficient QP Variable Name","Beta");
  p->set<Teuchos::ParameterList*> ("FELIX Physical Parameters",&params->sublist("FELIX Physical Parameters"));

  //Output
  p->set<std::string> ("Melting Rate QP Variable Name","Melting Rate");

  ev = rcp(new FELIX::HydrologyMeltingRate<EvalT,PHAL::AlbanyTraits,false>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  // ------- Sliding Velocity -------- //
  p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Velocity Norm"));

  // Input
  p->set<std::string>("Field Name","Basal Velocity");
  p->set<std::string>("Field Layout","Cell QuadPoint Vector");
  p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("FELIX Field Norm"));

  // Output
  p->set<std::string>("Field Norm Name","Sliding Velocity");

  ev = Teuchos::rcp(new FELIX::FieldNormParam<EvalT,PHAL::AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  //--- Effective pressure calculation ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Effective Pressure"));

  // Input
  p->set<std::string>("Surface Height Variable Name","Surface Height");
  p->set<std::string>("Ice Thickness Variable Name", "Ice Thickness");
  p->set<std::string>("Hydraulic Potential Variable Name", "Hydraulic Potential");
  p->set<Teuchos::ParameterList*>("FELIX Physical Parameters", &params->sublist("FELIX Physical Parameters"));

  // Output
  p->set<std::string>("Effective Pressure Variable Name","Effective Pressure");

  ev = Teuchos::rcp(new FELIX::EffectivePressure<EvalT,PHAL::AlbanyTraits, true, false>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  //--- FELIX basal friction coefficient ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Basal Friction Coefficient"));

  //Input
  p->set<std::string>("Sliding Velocity QP Variable Name", "Sliding Velocity");
  p->set<std::string>("BF Variable Name", "BF");
  p->set<std::string>("Effective Pressure QP Variable Name", "Effective Pressure");
  p->set<bool>("Hydrology",true);
  p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("FELIX Basal Friction Coefficient"));
  p->set<Teuchos::ParameterList*>("Stereographic Map", &params->sublist("Stereographic Map"));

  //Output
  p->set<std::string>("Basal Friction Coefficient Variable Name", "Beta");

  ev = Teuchos::rcp(new FELIX::BasalFrictionCoefficient<EvalT,PHAL::AlbanyTraits,true,false>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  // ------- Hydrology Residual Potential Eqn-------- //
  p = rcp(new Teuchos::ParameterList("Hydrology Residual Potential Eqn"));

  //Input
  p->set<std::string> ("BF Name", "BF");
  p->set<std::string> ("Gradient BF Name", "Grad BF");
  p->set<std::string> ("Weighted Measure Name", "Weights");
  p->set<std::string> ("Water Discharge QP Variable Name", "Water Discharge");
  p->set<std::string> ("Effective Pressure QP Variable Name", "Effective Pressure");
  p->set<std::string> ("Water Thickness QP Variable Name", "Water Thickness");
  p->set<std::string> ("Melting Rate QP Variable Name","Melting Rate");
  p->set<std::string> ("Surface Water Input QP Variable Name","Surface Water Input");
  p->set<std::string> ("Sliding Velocity QP Variable Name","Sliding Velocity");

  p->set<Teuchos::ParameterList*> ("FELIX Physical Parameters",&params->sublist("FELIX Physical Parameters"));
  p->set<Teuchos::ParameterList*> ("FELIX Hydrology Parameters",&params->sublist("FELIX Hydrology"));

  //Output
  p->set<std::string> ("Potential Eqn Residual Name",resid_names[0]);

  if (has_h_equation)
    ev = rcp(new FELIX::HydrologyResidualPotentialEqn<EvalT,PHAL::AlbanyTraits,true,false>(*p,dl));
  else
    ev = rcp(new FELIX::HydrologyResidualPotentialEqn<EvalT,PHAL::AlbanyTraits,false,false>(*p,dl));

  fm0.template registerEvaluator<EvalT>(ev);

  if (has_h_equation)
  {
    // ------- Hydrology Thickness Residual -------- //
    p = rcp(new Teuchos::ParameterList("Hydrology Residual Thickness"));

    //Input
    p->set<std::string> ("BF Name", "BF");
    p->set<std::string> ("Weighted Measure Name", "Weights");
    p->set<std::string> ("Water Thickness QP Variable Name","Water Thickness");
    p->set<std::string> ("Water Thickness Dot QP Variable Name","Water Thickness Dot");
    p->set<std::string> ("Effective Pressure QP Variable Name","Effective Pressure");
    p->set<std::string> ("Melting Rate QP Variable Name","Melting Rate");
    p->set<std::string> ("Sliding Velocity QP Variable Name","Sliding Velocity");
    p->set<bool> ("Unsteady", unsteady);
    p->set<Teuchos::ParameterList*> ("FELIX Hydrology",&params->sublist("FELIX Hydrology"));
    p->set<Teuchos::ParameterList*> ("FELIX Physical Parameters",&params->sublist("FELIX Physical Parameters"));

    //Output
    p->set<std::string> ("Thickness Eqn Residual Name", resid_names[1]);

    ev = rcp(new FELIX::HydrologyResidualThicknessEqn<EvalT,PHAL::AlbanyTraits,false>(*p,dl));
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

  //--- Shared Parameter for Continuation:  ---//
  p = Teuchos::rcp(new Teuchos::ParameterList("Homotopy Parameter"));

  param_name = "Glen's Law Homotopy Parameter";
  p->set<std::string>("Parameter Name", param_name);
  p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

  Teuchos::RCP<FELIX::SharedParameter<EvalT,PHAL::AlbanyTraits,FelixParamEnum,FelixParamEnum::Homotopy>> ptr_homotopy;
  ptr_homotopy = Teuchos::rcp(new FELIX::SharedParameter<EvalT,PHAL::AlbanyTraits,FelixParamEnum,FelixParamEnum::Homotopy>(*p,dl));
  ptr_homotopy->setNominalValue(params->sublist("Parameters"),params->sublist("FELIX Viscosity").get<double>(param_name,-1.0));
  fm0.template registerEvaluator<EvalT>(ptr_homotopy);

  // ----------------------------------------------------- //

  if (fieldManagerChoice == Albany::BUILD_RESID_FM)
  {
    PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter Hydrology", dl->dummy);
    fm0.template requireField<EvalT>(res_tag);
  }
  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM)
  {
    RCP<Teuchos::ParameterList> paramList = rcp(new Teuchos::ParameterList("Param List"));

    RCP<const Albany::MeshSpecsStruct> meshSpecsPtr = Teuchos::rcpFromRef(meshSpecs);
    paramList->set<RCP<const Albany::MeshSpecsStruct> >("Mesh Specs Struct", meshSpecsPtr);
    paramList->set<RCP<ParamLib> >("Parameter Library", paramLib);

    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
    return respUtils.constructResponses(fm0, *responseList, paramList, stateMgr);
  }

  return Teuchos::null;
}

template<typename EvalT>
void Hydrology::constructDirichletEvaluators () const
{
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;
  Teuchos::RCP<Teuchos::ParameterList> p;

  Teuchos::ParameterList& hydro = params->sublist("FELIX Hydrology");

  Teuchos::Array<std::string> ns_names = hydro.get<Teuchos::Array<std::string>>("Zero Porewater Pressure on Node Sets",Teuchos::Array<std::string>());

  for (int i=0; i<ns_names.size(); ++i)
  {

    // ------- Zero Porewater Pressure on Node Set -------- //
    p = Teuchos::rcp(new Teuchos::ParameterList("Zero Porewater Pressure"));

    //Input
    p->set<std::string> ("Ice Thickness Variable Name","Ice Thickness");
    p->set<std::string> ("Surface Height Variable Name","Surface Height");
    p->set<Teuchos::ParameterList*> ("FELIX Physical Parameters",&params->sublist("FELIX Physical Parameters"));
    p->set<std::string> ("Node Set ID",ns_names[i]);
    p->set<int>("Offset",0);

    ev = Teuchos::rcp(new FELIX::HydrologyDirichlet<EvalT,PHAL::AlbanyTraits>(*p));
    dfm->template registerEvaluator<EvalT>(ev);
    dfm->requireField<EvalT>(*ev->evaluatedFields()[0]);
  }
}

} // Namespace FELIX

#endif // FELIX_HYDROLOGY_PROBLEM_HPP
