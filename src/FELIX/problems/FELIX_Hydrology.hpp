//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_HYDROLOGY_PROBLEM_HPP
#define FELIX_HYDROLOGY_PROBLEM_HPP 1

#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_DefaultCubatureFactory.hpp"
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

#include "FELIX_HydrologyDischarge.hpp"
#include "FELIX_HydrologyHydrostaticPotential.hpp"
#include "FELIX_HydrologyMelting.hpp"
#include "FELIX_HydrologyRhs.hpp"
#include "FELIX_HydrologyResidual.hpp"
//#include "FELIX_EffectivePressure.hpp"

//uncomment the following line if you want debug output to be printed to screen
//#define OUTPUT_TO_SCREEN

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

  template<typename EvalT>
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> >
  buildBasalFrictionCoefficientEvaluator ();

  int numDim;

  Teuchos::ArrayRCP<std::string> dof_names;
  Teuchos::ArrayRCP<std::string> resid_names;

  Teuchos::RCP<Albany::Layouts> dl;

  Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > intrepidBasis;
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
  using PHX::DataLayout;
  using PHX::MDALayout;
  using PHAL::AlbanyTraits;

  // Retrieving FE information (basis and cell type)
  if (intrepidBasis.get()==0)
    intrepidBasis = Albany::getIntrepidBasis(meshSpecs.ctd);

  RCP<shards::CellTopology> cellType = rcp(new shards::CellTopology (&meshSpecs.ctd));

  // Building the right quadrature formula
  Intrepid::DefaultCubatureFactory<RealType> cubFactory;
  RCP <Intrepid::Cubature<RealType> > cubature = cubFactory.create(*cellType, meshSpecs.cubatureDegree);

  // Some constants
  const int numNodes = intrepidBasis->getCardinality();
  const int worksetSize = meshSpecs.worksetSize;
  const int numQPts = cubature->getNumPoints();
  const int numVertices = cellType->getNodeCount();
  const std::string& elementBlockName = meshSpecs.ebName;
  int offset = 0;

#ifdef OUTPUT_TO_SCREEN
  *out << "Field Dimensions: Workset=" << worksetSize
       << ", Vertices= " << numVertices
       << ", Nodes= " << numNodes
       << ", QuadPts= " << numQPts
       << ", Dim= " << numDim << std::endl;
#endif

  // Building the data layout
  dl = rcp(new Albany::Layouts(worksetSize,numVertices,numNodes,numQPts,numDim));

  // Using the utility for the common evaluators
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);

  // Service variables for registering state variables and evaluators
  Albany::StateStruct::MeshFieldEntity entity;
  RCP<PHX::Evaluator<AlbanyTraits> > ev;
  RCP<Teuchos::ParameterList> p;

  // -------------- Starting registration of state variables --------------- //

  // Basal height
  entity = Albany::StateStruct::NodalDataToElemNode;
  p = stateMgr.registerStateVariable("surface_height", dl->node_scalar, elementBlockName, true, &entity);
  p->set<const std::string>("Field Name","z_s");
  ev = rcp(new PHAL::LoadStateField<EvalT,AlbanyTraits>(*p));
  fm0.template registerEvaluator<EvalT>(ev);

  // Basal friction
  entity = Albany::StateStruct::NodalDataToElemNode;
  p = stateMgr.registerStateVariable("basal_friction", dl->node_scalar, elementBlockName, true, &entity);
  p->set<const std::string>("Field Name","beta");
  ev = rcp(new PHAL::LoadStateField<EvalT,AlbanyTraits>(*p));
  fm0.template registerEvaluator<EvalT>(ev);

  // Sliding velocity norm
  entity = Albany::StateStruct::NodalDistParameter;
  p = stateMgr.registerStateVariable("sliding_velocity", dl->node_scalar, elementBlockName, true, &entity);
  p->set<const std::string>("Field Name","u_b");
  ev = rcp(new PHAL::LoadStateField<EvalT,AlbanyTraits>(*p));
  fm0.template registerEvaluator<EvalT>(ev);

  // Drainage sheet effective depth
  entity = Albany::StateStruct::NodalDataToElemNode;
  p = stateMgr.registerStateVariable("drainage_sheet_depth", dl->node_scalar, elementBlockName, true, &entity);
  p->set<const std::string>("Field Name","h");
  ev = rcp(new PHAL::LoadStateField<EvalT,AlbanyTraits>(*p));
  fm0.template registerEvaluator<EvalT>(ev);

  // Ice thickness
  entity = Albany::StateStruct::NodalDataToElemNode;
  p = stateMgr.registerStateVariable("ice_thickness", dl->node_scalar, elementBlockName, true, &entity);
  p->set<const std::string>("Field Name","H");
  ev = rcp(new PHAL::LoadStateField<EvalT,AlbanyTraits>(*p));
  fm0.template registerEvaluator<EvalT>(ev);

  // Ice viscosity
  entity = Albany::StateStruct::ElemData;
  p = stateMgr.registerStateVariable("ice_viscosity", dl->cell_scalar2, elementBlockName, true, &entity);
  p->set<const std::string>("Field Name","mu_i");
  ev = rcp(new PHAL::LoadStateField<EvalT,AlbanyTraits>(*p));
  fm0.template registerEvaluator<EvalT>(ev);

  // Surface Water Input
  entity = Albany::StateStruct::NodalDataToElemNode;
  p = stateMgr.registerStateVariable("surface_water_input", dl->node_scalar, elementBlockName, true, &entity);
  p->set<const std::string>("Field Name","omega");
  ev = rcp(new PHAL::LoadStateField<EvalT,AlbanyTraits>(*p));
  fm0.template registerEvaluator<EvalT>(ev);

  // GeothermaL flux
  entity = Albany::StateStruct::NodalDataToElemNode;
  p = stateMgr.registerStateVariable("geothermal_flux", dl->node_scalar, elementBlockName, true, &entity);
  p->set<const std::string>("Field Name","G");
  ev = rcp(new PHAL::LoadStateField<EvalT,AlbanyTraits>(*p));
  fm0.template registerEvaluator<EvalT>(ev);
/*
  // Effective pressure
  entity = Albany::StateStruct::NodalDataToElemNode;
  p = stateMgr.registerStateVariable("effective_pressure", dl->node_scalar, elementBlockName, true, &entity);
  if (PHX::typeAsString<EvalT>()==PHX::typeAsString<AlbanyTraits::Residual>())
  {
    p->set<const std::string>("Field Name","N");
    p->set<Teuchos::RCP<PHX::DataLayout> >("Dummy Data Layout", dl->dummy);
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    p = rcp(new Teuchos::ParameterList("Effective Pressure"));
    p->set<std::string> ("Hydraulic Potential Variable Name", dof_names[0]);
    p->set<std::string> ("Surface Height Variable Name","z_s");
    p->set<std::string> ("Ice Thickness Variable Name","H");
    p->set<std::string> ("Effective Pressure Name","N");

    p->set<Teuchos::ParameterList*>("Physical Parameters",&params->sublist("FELIX Physical Parameters"));
    ev = rcp(new FELIX::EffectivePressure<EvalT,AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);

    PHX::Tag<typename EvalT::ScalarT> eff_press_tag("N", dl->dummy);
    fm0.requireField<EvalT>(eff_press_tag);
  }
*/
  // -------------------- Starting evaluators construction and registration ------------------------ //

  ev = evalUtils.constructGatherSolutionEvaluator_noTransient(false, dof_names, offset);
  fm0.template registerEvaluator<EvalT> (ev);

  ev = evalUtils.constructComputeBasisFunctionsEvaluator(cellType, intrepidBasis, cubature);
  fm0.template registerEvaluator<EvalT> (ev);

  ev = evalUtils.constructGatherCoordinateVectorEvaluator();
  fm0.template registerEvaluator<EvalT> (ev);

  ev = evalUtils.constructScatterResidualEvaluator(false, resid_names,offset, "Scatter Hydrology");
  fm0.template registerEvaluator<EvalT> (ev);

  ev = evalUtils.constructMapToPhysicalFrameEvaluator(cellType, cubature);
  fm0.template registerEvaluator<EvalT> (ev);

  // ------- DOF interpolations -------- //

  // Hydraulic Potential
  ev = evalUtils.constructDOFInterpolationEvaluator(dof_names[0]);
  fm0.template registerEvaluator<EvalT> (ev);

  // Hydraulic Potential Gradient
  ev = evalUtils.constructDOFGradInterpolationEvaluator(dof_names[0]);
  fm0.template registerEvaluator<EvalT> (ev);

  // Drainage Sheet Depth
  ev = evalUtils.constructDOFInterpolationEvaluator("h");
  fm0.template registerEvaluator<EvalT> (ev);

  // Ice Thickness
  ev = evalUtils.constructDOFInterpolationEvaluator("H");
  fm0.template registerEvaluator<EvalT> (ev);

  // Surface Height
  ev = evalUtils.constructDOFInterpolationEvaluator("z_s");
  fm0.template registerEvaluator<EvalT> (ev);

  // Sliding Velocity Norm
  ev = evalUtils.constructDOFInterpolationEvaluator("u_b");
  fm0.template registerEvaluator<EvalT> (ev);

  // Basal Friction Coefficient
  ev = evalUtils.constructDOFInterpolationEvaluator("beta");
  fm0.template registerEvaluator<EvalT> (ev);

  // Surface Water Input
  ev = evalUtils.constructDOFInterpolationEvaluator("omega");
  fm0.template registerEvaluator<EvalT> (ev);

  // Geothermal Flux
  ev = evalUtils.constructDOFInterpolationEvaluator("G");
  fm0.template registerEvaluator<EvalT> (ev);

  // ----- Hydrology Hydrostatic Potential ---- //

  p = rcp(new Teuchos::ParameterList("Hydrology Hydrostatic Potential"));

  //Input
  p->set<std::string> ("Ice Thickness QP Variable Name","H");
  p->set<std::string> ("Surface Height QP Variable Name","z_s");
  p->set<Teuchos::ParameterList*> ("Physical Parameters",&params->sublist("FELIX Physical Parameters"));


  // Output
  std::string hydrostatic_potential_name = "PhiH";
  p->set<std::string> ("Hydrostatic Potential QP Variable Name",hydrostatic_potential_name);

  ev = rcp(new FELIX::HydrologyHydrostaticPotential<EvalT,AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  // ------- Hydrology Water Discharge -------- //

  p = rcp(new Teuchos::ParameterList("Hydrology Water Discharge"));

  //Input
  p->set<std::string> ("Drainage Sheet Depth QP Variable Name","h");
  p->set<std::string> ("Hydraulic Potential Gradient QP Variable Name",dof_names[0] + " Gradient");

  p->set<Teuchos::ParameterList*> ("Hydrology Parameters",&params->sublist("FELIX Hydrology"));
  p->set<Teuchos::ParameterList*> ("Physical Parameters",&params->sublist("FELIX Physical Parameters"));

  //Output
  p->set<std::string> ("Discharge QP Variable Name","q");

  ev = rcp(new FELIX::HydrologyDischarge<EvalT,AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  // ------- Hydrology Constant Rhs -------- //
  p = rcp(new Teuchos::ParameterList("Hydrology Rhs"));

  //Input
  p->set<std::string> ("Ice Viscosity Variable Name","mu_i");
  p->set<std::string> ("Drainage Sheet Depth QP Variable Name","h");
  p->set<std::string> ("Hydrostatic Potential QP Variable Name",hydrostatic_potential_name);
  p->set<std::string> ("Surface Height QP Variable Name","z_s");
  p->set<std::string> ("Sliding Velocity Norm QP Variable Name","u_b");
  p->set<std::string> ("Basal Friction Coefficient QP Variable Name","beta");
  p->set<std::string> ("Surface Water Input QP Variable Name","omega");
  p->set<std::string> ("Geothermal Heat Source QP Variable Name","G");

  p->set<Teuchos::ParameterList*> ("Hydrology Parameters",&params->sublist("FELIX Hydrology"));

  //Output
  p->set<std::string> ("RHS QP Name","rhs");

  ev = rcp(new FELIX::HydrologyRhs<EvalT,AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  // ------- Hydrology Melting Rate -------- //

  p = rcp(new Teuchos::ParameterList("Hydrology Melting Rate"));

  //Input
  p->set<std::string> ("Hydraulic Potential Gradient QP Variable Name",dof_names[0] + " Gradient");
  p->set<std::string> ("Geothermal Heat Source QP Variable Name","G");
  p->set<std::string> ("Discharge QP Variable Name","q");
  p->set<std::string> ("Sliding Velocity Norm QP Variable Name","u_b");
  p->set<std::string> ("Basal Friction Coefficient QP Variable Name","beta");
  p->set<std::string> ("Geothermal Heat Source QP Variable Name","G");

  p->set<Teuchos::ParameterList*> ("Hydrology Parameters",&params->sublist("FELIX Hydrology"));
  p->set<Teuchos::ParameterList*> ("Physical Parameters",&params->sublist("FELIX Physical Parameters"));

  //Output
  p->set<std::string> ("Melting Rate QP Variable Name","m");

  ev = rcp(new FELIX::HydrologyMelting<EvalT,AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  // ------- Hydrology Residual -------- //
  p = rcp(new Teuchos::ParameterList("Hydrology Residual"));

  //Input
  p->set<std::string> ("BF Name", "BF");
  p->set<std::string> ("Weighted BF Name", "wBF");
  p->set<std::string> ("Weighted Gradient BF Name", "wGrad BF");
  p->set<std::string> ("Hydraulic Potential QP Variable Name", dof_names[0]);
  p->set<std::string> ("Discharge QP Variable Name","q");
  p->set<std::string> ("Melting Rate QP Variable Name","m");
  p->set<std::string> ("Ice Viscosity Variable Name","mu_i");
  p->set<std::string> ("Drainage Sheet Depth QP Variable Name","h");
  p->set<std::string> ("RHS QP Name","rhs");

  p->set<Teuchos::ParameterList*> ("Hydrology Parameters",&params->sublist("FELIX Hydrology"));
  p->set<Teuchos::ParameterList*> ("Physical Parameters",&params->sublist("FELIX Physical Parameters"));

  //Output
  p->set<std::string> ("Residual Name", resid_names[0]);

  ev = rcp(new FELIX::HydrologyResidual<EvalT,AlbanyTraits>(*p,dl));
  fm0.template registerEvaluator<EvalT>(ev);

  RCP<Teuchos::ParameterList> paramList = rcp(new Teuchos::ParameterList("Param List"));
  { // response
    RCP<const Albany::MeshSpecsStruct> meshSpecsPtr = Teuchos::rcpFromRef(meshSpecs);
    paramList->set<RCP<const Albany::MeshSpecsStruct> >("Mesh Specs Struct", meshSpecsPtr);
    paramList->set<RCP<ParamLib> >("Parameter Library", paramLib);
  }

  if (fieldManagerChoice == Albany::BUILD_RESID_FM)
  {
    PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter Hydrology", dl->dummy);
    fm0.requireField<EvalT>(res_tag);
  }
  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM)
  {
    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);

    return respUtils.constructResponses(fm0, *responseList, paramList, stateMgr);
  }

  return Teuchos::null;
}

} // Namespace FELIX

#endif // FELIX_HYDROLOGY_PROBLEM_HPP
