/*
 * FELIX_Enthalpy.hpp
 *
 *  Created on: May 10, 2016
 *      Author: abarone
 */

#ifndef FELIX_ENTHALPY_PROBLEM_HPP
#define FELIX_ENTHALPY_PROBLEM_HPP

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

#include "Phalanx.hpp"
#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_AlbanyTraits.hpp"

#include "FELIX_EnthalpyResid.hpp"
//#include "FELIX_w_ZResid.hpp"
#include "FELIX_ViscosityFO.hpp"
#include "FELIX_Dissipation.hpp"
#include "FELIX_BasalFrictionHeat.hpp"
#include "FELIX_GeoFluxHeat.hpp"
#include "FELIX_HydrostaticPressure.hpp"
#include "FELIX_LiquidWaterFraction.hpp"
#include "FELIX_PressureMeltingEnthalpy.hpp"
#include "FELIX_PressureMeltingTemperature.hpp"
#include "FELIX_Temperature.hpp"

namespace FELIX
{

class Enthalpy : public Albany::AbstractProblem
{
	public:
		//! Default constructor
		Enthalpy(const Teuchos::RCP<Teuchos::ParameterList>& params,
				 const Teuchos::RCP<ParamLib>& paramLib,
				 const int numDim_);

		//! Destructor
		~Enthalpy();

		//! Return number of spatial dimensions
		virtual int spatialDimension() const { return numDim; }

		//! Build the PDE instantiations, boundary conditions, and initial solution
		virtual void buildProblem(Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
								  Albany::StateManager& stateMgr);

		// Build evaluators
		virtual Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> > buildEvaluators(PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
																					const Albany::MeshSpecsStruct& meshSpecs,
																					Albany::StateManager& stateMgr,
																					Albany::FieldManagerChoice fmchoice,
																					const Teuchos::RCP<Teuchos::ParameterList>& responseList);

		//! Each problem must generate its list of valid parameters
		Teuchos::RCP<const Teuchos::ParameterList> getValidProblemParameters() const;

	    //! Main problem setup routine. Not directly called, but indirectly by following functions
	    template <typename EvalT> Teuchos::RCP<const PHX::FieldTag>
	    constructEvaluators(PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
						    const Albany::MeshSpecsStruct& meshSpecs,
						    Albany::StateManager& stateMgr,
						    Albany::FieldManagerChoice fmchoice,
						    const Teuchos::RCP<Teuchos::ParameterList>& responseList);

	    void constructDirichletEvaluators(const Albany::MeshSpecsStruct& meshSpecs);
	    void constructNeumannEvaluators(const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs);

	protected:
	    Teuchos::RCP<shards::CellTopology> cellType;
	    Teuchos::RCP<shards::CellTopology> basalSideType;

	    Teuchos::RCP<Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,PHX::Device> > >  cellCubature;
	    Teuchos::RCP<Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,PHX::Device> > >  basalCubature;

	    Teuchos::RCP<Intrepid2::Basis<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > > cellBasis;
	    Teuchos::RCP<Intrepid2::Basis<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > > basalSideBasis;

		int numDim;
		Teuchos::RCP<Albany::Layouts> dl, dl_basal;
        std::string elementBlockName;

        bool haveSUPG;
        bool needsDiss, needsBasFric;
        bool isGeoFluxConst;

        std::string basalSideName, basalEBName;
};

} // end of the namespace FELIX

// ================================ IMPLEMENTATION ============================ //
template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
FELIX::Enthalpy::constructEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                      const Albany::MeshSpecsStruct& meshSpecs,
                                      Albany::StateManager& stateMgr,
                                      Albany::FieldManagerChoice fieldManagerChoice,
                                      const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
	  using Teuchos::RCP;
	  using Teuchos::rcp;
	  using Teuchos::ParameterList;
	  using PHX::DataLayout;
	  using PHX::MDALayout;
	  using std::vector;
	  using std::string;
	  using std::map;
	  using PHAL::AlbanyTraits;

	  int offset = 0;

	  Albany::StateStruct::MeshFieldEntity entity;

	  Teuchos::RCP<ParameterList> p;

	  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);

	  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;

	  // Here is how to register the field for dirichlet condition.
	  // Enthalpy Dirichlet field on the surface
	  {
		  entity = Albany::StateStruct::NodalDistParameter;
		  std::string stateName = "surface_air_enthalpy";
		  p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity, "");
		  ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
		  fm0.template registerEvaluator<EvalT>(ev);
	  }
/*
	  // --- Enthalpy Dirichlet field on the surface ---
	  {
		  p = rcp(new ParameterList("FELIX Dirichlet Enthalpy Surface"));

		  //Input Dirichlet Temperature
		  p->set<std::string>("Dirichlet Temperature Surface Variable Name", "surface_air_temperature");

		  p->set<ParameterList*>("FELIX Physical Parameters", &params->sublist("FELIX Physical Parameters"));

		  //Output
		  p->set<std::string>("Dirichlet Enthalpy Surface Variable Name", "surface_air_enthalpy");

		  ev = Teuchos::rcp(new FELIX::DirichletEnthalpySurface<EvalT,PHAL::AlbanyTraits>(*p,dl));
		  fm0.template registerEvaluator<EvalT>(ev);

		  if (fieldManagerChoice == Albany::BUILD_RESID_FM)
		  {
		  // Only PHAL::AlbanyTraits::Residual evaluates something
			  if (ev->evaluatedFields().size()>0)
		      {
				  // Require execution of evaluator
				  fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
		      }
		  }

     	  entity = Albany::StateStruct::NodalDistParameter;
		  std::string stateName = "surface_air_enthalpy";
		  p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity, "");
		  ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,AlbanyTraits>(*p));
		  fm0.template registerEvaluator<EvalT>(ev);


		  if (fieldManagerChoice == Albany::BUILD_RESID_FM)
		  {
		  // Only PHAL::AlbanyTraits::Residual evaluates something
			  if (ev->evaluatedFields().size()>0)
		      {
				  // Require execution of evaluator
				  fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
		      }
		  }

	  }
*/
	  // Velocity
	  {
		  entity = Albany::StateStruct::NodalDataToElemNode;
		  std::string stateName = "velocity";
		  p = stateMgr.registerStateVariable(stateName, dl->node_vector, elementBlockName, true, &entity, "");
		  ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
		  fm0.template registerEvaluator<EvalT>(ev);
	  }

	  // Flow factor
	  {
		  entity = Albany::StateStruct::ElemData;
		  std::string stateName = "flow_factor";
		  p = stateMgr.registerStateVariable(stateName, dl->cell_scalar2, elementBlockName, true, &entity);
		  ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
		  fm0.template registerEvaluator<EvalT>(ev);
	  }

	  // Basal friction
	  if(needsBasFric)
	  {
		  entity = Albany::StateStruct::NodalDataToElemNode;
		  std::string stateName = "basal_friction";
		  p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName,true, &entity);
		  ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
		  fm0.template registerEvaluator<EvalT>(ev);
	  }

	  // Geotermal flux
	  if(!isGeoFluxConst)
	  {
		  entity = Albany::StateStruct::NodalDataToElemNode;
		  std::string stateName = "basal_heat_flux";
		  p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName,true, &entity);
		  ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
		  fm0.template registerEvaluator<EvalT>(ev);
	  }

	  // Thickness
	  {
		  entity = Albany::StateStruct::NodalDataToElemNode;
		  std::string stateName = "thickness";
		  p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName,true, &entity);
		  ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
		  fm0.template registerEvaluator<EvalT>(ev);
	  }

	  // Surface Height
	  {
		  entity = Albany::StateStruct::NodalDataToElemNode;
		  std::string stateName = "surface_height";
		  p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName,true, &entity);
		  ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
		  fm0.template registerEvaluator<EvalT>(ev);
	  }


	  // Define Field Names
	  Teuchos::ArrayRCP<string> dof_names(neq);
	  dof_names[0] = "Enthalpy";
	  //dof_names[1] = "w_z";
	  Teuchos::ArrayRCP<string> resid_names(neq);
	  resid_names[0] = "Enthalpy Resid";
	  //resid_names[1] = "w_z Resid";

	  // --- Interpolation and utilities ---

	  // no transient
	  fm0.template registerEvaluator<EvalT> (evalUtils.constructGatherSolutionEvaluator_noTransient(false, dof_names));

	  fm0.template registerEvaluator<EvalT> (evalUtils.constructScatterResidualEvaluator(false, resid_names));

	  fm0.template registerEvaluator<EvalT> (evalUtils.constructGatherCoordinateVectorEvaluator());

	  fm0.template registerEvaluator<EvalT> (evalUtils.constructMapToPhysicalFrameEvaluator(cellType, cellCubature));

	  fm0.template registerEvaluator<EvalT> (evalUtils.constructComputeBasisFunctionsEvaluator(cellType, cellBasis, cellCubature));

	  fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFInterpolationEvaluator(dof_names[0]));

	  //fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFInterpolationEvaluator(dof_names[1]));

	  fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFInterpolationEvaluator("Hydrostatic Pressure"));

	  fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFGradInterpolationEvaluator(dof_names[0]));

	  fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFGradInterpolationEvaluator("thickness"));

	  fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFGradInterpolationEvaluator("surface_height"));

	  fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFVecInterpolationEvaluator("velocity", offset));

	  fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFVecGradInterpolationEvaluator("velocity"));

	  // Interpolate temperature from nodes to cell
	  if(needsDiss)
		  fm0.template registerEvaluator<EvalT> (evalUtils.constructQuadPointsToCellInterpolationEvaluator ("temperature",false));

	  // --- Special evaluators for side handling

	  // --- Restrict vertex coordinates from cell-based to cell-side-based
	  fm0.template registerEvaluator<EvalT> (evalUtils.getMSTUtils().constructDOFCellToSideEvaluator("Coord Vec",basalSideName,"Vertex Vector",cellType,
												  "Coord Vec " + basalSideName));

	  // --- Compute side basis functions
	  fm0.template registerEvaluator<EvalT> (evalUtils.constructComputeBasisFunctionsSideEvaluator(cellType, basalSideBasis, basalCubature, basalSideName));

	  // --- Compute Quad Points coordinates on the side set
	  fm0.template registerEvaluator<EvalT> (evalUtils.constructMapToPhysicalFrameSideEvaluator(cellType,basalCubature,basalSideName));

	  // --- Restrict basal velocity from cell-based to cell-side-based
	  fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFCellToSideEvaluator("velocity",basalSideName,"Node Vector",cellType));

	  // --- Interpolate velocity on QP on side
	  fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFVecInterpolationSideEvaluator("velocity", basalSideName));

	  if(needsBasFric)
	  {
		  // --- Restrict basal friction from cell-based to cell-side-based
		  fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFCellToSideEvaluator("basal_friction",basalSideName,"Node Scalar",cellType));

		  // --- Interpolate Beta Given on QP on side
		  fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator("basal_friction", basalSideName));

		  fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFInterpolationEvaluator("Basal Heat"));

		  fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFInterpolationEvaluator("Basal Heat SUPG"));
	  }

	  // --- Utilities for Geotermal flux
	  if(!isGeoFluxConst)
	  {
		  // --- Restrict geotermal flux from cell-based to cell-side-based
		  fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFCellToSideEvaluator("basal_heat_flux",basalSideName,"Node Scalar",cellType));

	  	  // --- Interpolate geotermal_flux on QP on side
	  	  fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator("basal_heat_flux", basalSideName));

		  fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFInterpolationEvaluator("Geo Flux Heat"));

		  fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFInterpolationEvaluator("Geo Flux Heat SUPG"));
	  }

	  // --- Evaluators for computing the velocity along z direction
	  // fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFGradInterpolationSideEvaluator("thickness", basalSideName));

	  // fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFGradInterpolationSideEvaluator("surface_height", basalSideName));


	  // -------------------------------- FELIX evaluators ------------------------- //
	  // --- Enthalpy Residual ---
	  {
	  p = rcp(new ParameterList("Enthalpy Resid"));

	  //Input
	  p->set<string>("Weighted BF Variable Name", "wBF");
	  p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);

	  p->set<string>("Weighted Gradient BF Variable Name", "wGrad BF");
	  p->set< RCP<DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);

	  p->set<string>("Enthalpy QP Variable Name", dof_names[0]);
	  p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

	  p->set<string>("Enthalpy Gradient QP Variable Name", dof_names[0]+" Gradient");
	  p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_gradient);

	  p->set<std::string>("Enthalpy Hs QP Variable Name", "melting enthalpy");
	  p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

	  p->set<std::string>("Coordinate Vector Name", "Coord Vec");

	  // Velocity field for the convective term (read from the mesh)
	  p->set<string>("Velocity QP Variable Name", "velocity");
	  p->set< RCP<DataLayout> >("QP Vector Data Layout", dl->qp_vector);

	  p->set<string>("Geotermal Flux Heat QP Variable Name","Geo Flux Heat");
	  p->set<string>("Geotermal Flux Heat QP SUPG Variable Name","Geo Flux Heat SUPG");

	  if(needsDiss)
	  {
		  p->set<std::string>("Dissipation QP Variable Name", "FELIX Dissipation");
	  }

	  if(needsBasFric)
	  {
		  p->set<std::string>("Basal Friction Heat QP Variable Name", "Basal Heat");
		  p->set<std::string>("Basal Friction Heat QP SUPG Variable Name", "Basal Heat SUPG");
	  }

	  p->set<bool>("Needs Dissipation", needsDiss);
	  p->set<bool>("Needs Basal Friction", needsBasFric);
	  p->set<bool>("Constant Geotermal Flux", isGeoFluxConst);

	  p->set<RCP<ParamLib> >("Parameter Library", paramLib);

	  p->set<ParameterList*>("FELIX Physical Parameters", &params->sublist("FELIX Physical Parameters"));
	  p->set<ParameterList*>("SUPG Settings", &params->sublist("SUPG Settings"));

	  //Output
	  p->set<string>("Residual Variable Name", resid_names[0]);
	  p->set< RCP<DataLayout> >("Node Scalar Data Layout", dl->node_scalar);

	  ev = rcp(new FELIX::EnthalpyResid<EvalT,AlbanyTraits,typename EvalT::ParamScalarT>(*p,dl));
	  fm0.template registerEvaluator<EvalT>(ev);
	  }
/*
	  // --- w_z Residual ---
	  {
	  p = rcp(new ParameterList("w_Z Resid"));

	  //Input
	  p->set<string>("Weighted BF Variable Name", "wBF");
	  p->set< RCP<DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);

	  p->set<string>("w_z QP Variable Name", dof_names[1]);
	  p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

	  p->set<std::string>("Velocity Gradient QP Variable Name", "velocity Gradient");
	  p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_vecgradient);

	  //Output
	  p->set<string>("w_z Residual Variable Name", resid_names[1]);
	  p->set< RCP<DataLayout> >("Node Scalar Data Layout", dl->node_scalar);

	  ev = rcp(new FELIX::w_ZResid<EvalT,AlbanyTraits,typename EvalT::ParamScalarT>(*p,dl));
	  fm0.template registerEvaluator<EvalT>(ev);
	  }
*/
	  // --- FELIX Dissipation ---
	  if(needsDiss)
	  {
		  {
		  p = rcp(new ParameterList("FELIX Dissipation"));

		  //Input
		  p->set<std::string>("Viscosity QP Variable Name", "FELIX Viscosity");
		  p->set<std::string>("EpsilonSq QP Variable Name", "FELIX EpsilonSq");

		  //Output
		  p->set<std::string>("Dissipation QP Variable Name", "FELIX Dissipation");

		  ev = Teuchos::rcp(new FELIX::Dissipation<EvalT,PHAL::AlbanyTraits>(*p,dl));
		  fm0.template registerEvaluator<EvalT>(ev);
		  }

		  // --- FELIX Viscosity ---
		  {
		  p = rcp(new ParameterList("FELIX Viscosity"));

		  //Input
		  p->set<std::string>("Coordinate Vector Variable Name", "Coord Vec");
		  p->set<std::string>("Velocity QP Variable Name", "velocity");
		  p->set<std::string>("Velocity Gradient QP Variable Name", "velocity Gradient");
		  p->set<std::string>("Temperature Variable Name", "temperature");
		  p->set<std::string>("Flow Factor Variable Name", "flow_factor");
		  p->set<RCP<ParamLib> >("Parameter Library", paramLib);
		  p->set<ParameterList*>("Stereographic Map", &params->sublist("Stereographic Map"));
		  p->set<ParameterList*>("Parameter List", &params->sublist("FELIX Viscosity"));

		  //Output
		  p->set<std::string>("Viscosity QP Variable Name", "FELIX Viscosity");
		  p->set<std::string>("EpsilonSq QP Variable Name", "FELIX EpsilonSq");

		  ev = Teuchos::rcp(new FELIX::ViscosityFO<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT,typename EvalT::ScalarT>(*p,dl));
		  fm0.template registerEvaluator<EvalT>(ev);
		  }
	  }

	  // --- FELIX Basal friction heat ---
	  if(needsBasFric)
	  {
		  p = rcp(new ParameterList("FELIX Basal Friction Heat"));
		  //Input
		  p->set<std::string>("BF Side Name", "BF "+basalSideName);
		  p->set<std::string>("Gradient BF Side Name", "Grad BF "+basalSideName);
		  p->set<std::string>("Weighted Measure Name", "Weighted Measure "+basalSideName);
		  p->set<std::string>("Velocity Side QP Variable Name", "velocity");
		  p->set<std::string>("Basal Friction Coefficient Side QP Variable Name", "basal_friction");
		  p->set<std::string>("Side Set Name", basalSideName);

		  p->set<ParameterList*>("SUPG Settings", &params->sublist("SUPG Settings"));

		  p->set<Teuchos::RCP<shards::CellTopology> >("Cell Type", cellType);

		  //Output
		  p->set<std::string>("Basal Friction Heat Variable Name", "Basal Heat");

		  if(haveSUPG)
			  p->set<std::string>("Basal Friction Heat SUPG Variable Name", "Basal Heat SUPG");

		  ev = Teuchos::rcp(new FELIX::BasalFrictionHeat<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT>(*p,dl));
		  fm0.template registerEvaluator<EvalT>(ev);
	  }

	  // --- FELIX Geotermal flux heat
	  {
		  p = rcp(new ParameterList("FELIX Geotermal Flux Heat"));
		  //Input
		  p->set<std::string>("BF Side Name", "BF "+basalSideName);
		  p->set<std::string>("Gradient BF Side Name", "Grad BF "+basalSideName);
		  p->set<std::string>("Weighted Measure Name", "Weighted Measure "+basalSideName);
		  p->set<std::string>("Velocity Side QP Variable Name", "velocity");
		  p->set<std::string>("Side Set Name", basalSideName);
		  p->set<Teuchos::RCP<shards::CellTopology> >("Cell Type", cellType);

		  p->set<ParameterList*>("SUPG Settings", &params->sublist("SUPG Settings"));

		  if(!isGeoFluxConst)
			  p->set<std::string>("Geotermal Flux Side QP Variable Name", "basal_heat_flux");

		  //p->set<ParameterList*>("Problem", &params->sublist("Problem"));
		  p->set<bool>("Constant Geotermal Flux", isGeoFluxConst);

		  //Output
		  p->set<std::string>("Geotermal Flux Heat Variable Name", "Geo Flux Heat");

		  if(haveSUPG)
			  p->set<std::string>("Geotermal Flux Heat SUPG Variable Name", "Geo Flux Heat SUPG");

		  ev = Teuchos::rcp(new FELIX::GeoFluxHeat<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT>(*p,dl));
		  fm0.template registerEvaluator<EvalT>(ev);
	  }

	  // --- FELIX hydrostatic pressure
	  {
		  p = rcp(new ParameterList("FELIX Hydrostatic Pressure"));

		  //Input
		  p->set<std::string>("Surface Height Variable Name", "surface_height");
		  p->set<std::string>("Coordinate Vector Variable Name", "Coord Vec");

		  p->set<ParameterList*>("FELIX Physical Parameters", &params->sublist("FELIX Physical Parameters"));

		  //Output
		  p->set<std::string>("Hydrostatic Pressure Variable Name", "Hydrostatic Pressure");

		  ev = Teuchos::rcp(new FELIX::HydrostaticPressure<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT>(*p,dl));
		  fm0.template registerEvaluator<EvalT>(ev);
	  }

	  // --- FELIX pressure-melting temperature
	  {
		  p = rcp(new ParameterList("FELIX Pressure Melting Temperature"));

		  //Input
		  p->set<std::string>("Hydrostatic Pressure QP Variable Name", "Hydrostatic Pressure");

		  p->set<ParameterList*>("FELIX Physical Parameters", &params->sublist("FELIX Physical Parameters"));

		  //Output
		  p->set<std::string>("Melting Temperature QP Variable Name", "melting temp");

		  ev = Teuchos::rcp(new FELIX::PressureMeltingTemperature<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT>(*p,dl));
		  fm0.template registerEvaluator<EvalT>(ev);
	  }

	  // --- FELIX pressure-melting enthalpy
	  {
		  p = rcp(new ParameterList("FELIX Pressure Melting Enthalpy"));

		  //Input
		  p->set<std::string>("Melting Temperature QP Variable Name", "melting temp");

		  p->set<ParameterList*>("FELIX Physical Parameters", &params->sublist("FELIX Physical Parameters"));

		  //Output
		  p->set<std::string>("Enthalpy Hs QP Variable Name", "melting enthalpy");
		  ev = Teuchos::rcp(new FELIX::PressureMeltingEnthalpy<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT>(*p,dl));
		  fm0.template registerEvaluator<EvalT>(ev);
	  }

	  // --- FELIX Temperature
	  {
		  p = rcp(new ParameterList("FELIX Temperature"));

		  //Input
		  p->set<std::string>("Melting Temperature QP Variable Name", "melting temp");
		  p->set<std::string>("Enthalpy Hs QP Variable Name", "melting enthalpy");
		  p->set<std::string>("Enthalpy QP Variable Name", dof_names[0]);

		  p->set<ParameterList*>("FELIX Physical Parameters", &params->sublist("FELIX Physical Parameters"));

		  //Output
		  p->set<std::string>("Temperature QP Variable Name", "temperature");
		  ev = Teuchos::rcp(new FELIX::Temperature<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT>(*p,dl));
		  fm0.template registerEvaluator<EvalT>(ev);
	  }

	  // --- FELIX Liquid Water Fraction
	  {
		  p = rcp(new ParameterList("FELIX Liquid Water Fraction"));

		  //Input
		  p->set<std::string>("Enthalpy Hs QP Variable Name", "melting enthalpy");
		  p->set<std::string>("Enthalpy QP Variable Name", dof_names[0]);

		  p->set<ParameterList*>("FELIX Physical Parameters", &params->sublist("FELIX Physical Parameters"));

		  //Output
		  p->set<std::string>("Omega QP Variable Name", "omega");
		  ev = Teuchos::rcp(new FELIX::LiquidWaterFraction<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT>(*p,dl));
		  fm0.template registerEvaluator<EvalT>(ev);
	  }

	  /*
	   *	// --- FELIX Integral 1D w_z
	   *	//Input
	   * 	p->set<string>("w_z QP Variable Name", dof_names[1]);
	   *    p->set<std::string>("Ice Thickness Variable Name", "thickness"); ????????????????? if it is correct, I'd need to interpolate it on QPs
	   *
	   *    //Output
	   *    p->set<std::string>("Integral 1D w_z QP Variable Name", "int1Dw_z");
	   *    ev = Teuchos::rcp(new FELIX::Integral1Dw_Z<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT>(*p,dl));
	   *    fm0.template registerEvaluator<EvalT>(ev);
	   *
	   *    // --- FELIX Velocity Z
	   *    //Input
	   *    p->set<std::string>("Integral 1D w_z QP Variable Name", "int1Dw_z");
	   *
	   */

	  if (fieldManagerChoice == Albany::BUILD_RESID_FM)
	  {
		  PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter", dl->dummy);
	      fm0.requireField<EvalT>(res_tag);
	  }
	  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM)
	  {
		  Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
	      return respUtils.constructResponses(fm0, *responseList, Teuchos::null, stateMgr);
	  }

	  return Teuchos::null;
}



#endif /* FELIX_ENTHALPY_PROBLEM_HPP */
