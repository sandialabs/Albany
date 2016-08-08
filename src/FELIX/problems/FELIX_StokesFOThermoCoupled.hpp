/*
 * FELIX_StokesFOThermoCoupled.hpp
 *
 *  Created on: Jun 23, 2016
 *      Author: abarone
 */

#ifndef FELIX_STOKESFOTHERMOCOUPLED_PROBLEM_HPP
#define FELIX_STOKESFOTHERMOCOUPLED_PROBLEM_HPP

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
#include "PHAL_SaveCellStateField.hpp"
#include "PHAL_LoadSideSetStateField.hpp"
#include "FELIX_SharedParameter.hpp"
#include "FELIX_StokesParamEnum.hpp"

// Include for Velocity
#include "FELIX_StokesFOResid.hpp"
#include "FELIX_StokesFOBasalResid.hpp"
#include "FELIX_StokesFOBodyForce.hpp"

// Include for Enthalpy
#include "FELIX_EnthalpyResid.hpp"
#include "FELIX_w_ZResid.hpp"
#include "FELIX_ViscosityFO.hpp"
#include "FELIX_Dissipation.hpp"
#include "FELIX_BasalFrictionHeat.hpp"
#include "FELIX_GeoFluxHeat.hpp"
#include "FELIX_HydrostaticPressure.hpp"
#include "FELIX_LiquidWaterFraction.hpp"
#include "FELIX_PressureMeltingEnthalpy.hpp"
#include "FELIX_PressureMeltingTemperature.hpp"
#include "FELIX_Temperature.hpp"
#include "FELIX_Integral1Dw_Z.hpp"
#include "FELIX_VerticalVelocity.hpp"
#include "FELIX_BasalMeltRate.hpp"


namespace FELIX
{

class StokesFOThermoCoupled : public Albany::AbstractProblem
{
	public:
		//! Default constructor
		StokesFOThermoCoupled(const Teuchos::RCP<Teuchos::ParameterList>& params,
				 	 	 const Teuchos::RCP<ParamLib>& paramLib,
						 const int numDim_);

		//! Destructor
		~StokesFOThermoCoupled();

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

	private:

	  //! Private to prohibit copying
	    StokesFOThermoCoupled(const StokesFOThermoCoupled&);

	  //! Private to prohibit copying
	    StokesFOThermoCoupled& operator=(const StokesFOThermoCoupled&);


	protected:
	    Teuchos::RCP<shards::CellTopology> cellType;
	    Teuchos::RCP<shards::CellTopology> basalSideType;
	    Teuchos::RCP<shards::CellTopology> surfaceSideType;

	    Teuchos::RCP<Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,PHX::Device> > > cellCubature;
	    Teuchos::RCP<Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,PHX::Device> > > basalCubature;
	    Teuchos::RCP<Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,PHX::Device> > > surfaceCubature;

	    Teuchos::RCP<Intrepid2::Basis<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > > cellBasis;
	    Teuchos::RCP<Intrepid2::Basis<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > > basalSideBasis;
	    Teuchos::RCP<Intrepid2::Basis<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > > surfaceSideBasis;

		int numDim;
		Teuchos::RCP<Albany::Layouts> dl, dl_basal, dl_surface;

		// Flags for Velocity
		bool sliding;

        // Flags for Enthalpy
        bool haveSUPG;
        bool needsDiss, needsBasFric;
        bool isGeoFluxConst;

        std::string basalSideName,surfaceSideName;
        std::string basalEBName, surfaceEBName, elementBlockName;
};

} // end of the namespace FELIX

// ================================ IMPLEMENTATION ============================ //
template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
FELIX::StokesFOThermoCoupled::constructEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                                      const Albany::MeshSpecsStruct& meshSpecs,
                                      Albany::StateManager& stateMgr,
                                      Albany::FieldManagerChoice fieldManagerChoice,
                                      const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
	  Albany::StateStruct::MeshFieldEntity entity;

	  Teuchos::RCP<Teuchos::ParameterList> p;

	  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);

	  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits> > ev;

	  // ---------------------------- Registering state variables ------------------------- //
	  std::string stateName, fieldName;

	  // Enthalpy Dirichlet field on the surface
	  {
		  entity = Albany::StateStruct::NodalDistParameter;
		  stateName = "surface_air_enthalpy";
		  p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity, "");
		  ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
		  fm0.template registerEvaluator<EvalT>(ev);
	  }

	  // Flow factor - actually, this is not used if viscosity is temperature based
	  {
		  entity = Albany::StateStruct::ElemData;
		  stateName = "flow_factor";
		  p = stateMgr.registerStateVariable(stateName, dl->cell_scalar2, elementBlockName, true, &entity);
		  ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
		  fm0.template registerEvaluator<EvalT>(ev);
	  }

	  // Basal friction
	  if(needsBasFric)
	  {
		  stateName = "basal_friction";
		  fieldName = "Beta";
		  entity = Albany::StateStruct::NodalDataToElemNode;

		  // if basal_friction is required at the base
		  if (ss_requirements.find(basalSideName) != ss_requirements.end())
		  {
			  const Albany::AbstractFieldContainer::FieldContainerRequirements& req = ss_requirements.at(basalSideName);
			  if (std::find(req.begin(), req.end(), stateName) != req.end())
			  {
				  p = stateMgr.registerSideSetStateVariable(basalSideName, stateName, fieldName, dl_basal->node_scalar, basalEBName, true, &entity);
			      ev = Teuchos::rcp(new PHAL::LoadSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p));
			      fm0.template registerEvaluator<EvalT>(ev);
			  }
		  }
		  // if basal_friction is required in the whole mesh
		  if (std::find(requirements.begin(),requirements.end(),stateName) != requirements.end())
		  {
		      p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity);
		      p->set<std::string>("Field Name", fieldName);

              ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
		      fm0.template registerEvaluator<EvalT>(ev);
		  }
	  }

	  // Geotermal flux
	  if(!isGeoFluxConst)
	  {
		  stateName = "basal_heat_flux";
		  fieldName = "Basal Heat Flux";
		  entity = Albany::StateStruct::NodalDataToElemNode;

		  // if basal_heat_flux is required at the base
		  if (ss_requirements.find(basalSideName) != ss_requirements.end())
		  {
			  const Albany::AbstractFieldContainer::FieldContainerRequirements& req = ss_requirements.at(basalSideName);
			  if (std::find(req.begin(), req.end(), stateName) != req.end())
			  {
				  p = stateMgr.registerSideSetStateVariable(basalSideName, stateName, fieldName, dl_basal->node_scalar, basalEBName, true, &entity);
			      ev = Teuchos::rcp(new PHAL::LoadSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p));
			      fm0.template registerEvaluator<EvalT>(ev);
			  }
		  }
	  }

	  // Thickness
	  {
		  stateName = "thickness";
		  fieldName = "Ice Thickness";
		  entity = Albany::StateStruct::NodalDataToElemNode;

		  // if thickness is required at the base
		  if (ss_requirements.find(basalSideName) != ss_requirements.end())
		  {
			  const Albany::AbstractFieldContainer::FieldContainerRequirements& req = ss_requirements.at(basalSideName);
			  if (std::find(req.begin(), req.end(), stateName) != req.end())
			  {
				  p = stateMgr.registerSideSetStateVariable(basalSideName, stateName, fieldName, dl_basal->node_scalar, basalEBName, true, &entity);
			      ev = Teuchos::rcp(new PHAL::LoadSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p));
				  fm0.template registerEvaluator<EvalT>(ev);
			  }
		  }

		  // if thickness is required in the whole mesh
		  if (std::find(requirements.begin(),requirements.end(),stateName) != requirements.end())
		  {
		      p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName, true, &entity);
		      p->set<std::string>("Field Name", fieldName);

              ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
		      fm0.template registerEvaluator<EvalT>(ev);
		  }
	  }

	  // Surface Height
	  {
		  stateName = "surface_height";
		  fieldName = "Surface Height";
		  entity = Albany::StateStruct::NodalDataToElemNode;

		  // if surface_height is required at the base
		  if (ss_requirements.find(basalSideName) != ss_requirements.end())
		  {
			  const Albany::AbstractFieldContainer::FieldContainerRequirements& req = ss_requirements.at(basalSideName);
			  if (std::find(req.begin(), req.end(), stateName) != req.end())
			  {
				  p = stateMgr.registerSideSetStateVariable(basalSideName, stateName, fieldName, dl_basal->node_scalar, basalEBName, true, &entity);
			      ev = Teuchos::rcp(new PHAL::LoadSideSetStateField<EvalT,PHAL::AlbanyTraits>(*p));
				  fm0.template registerEvaluator<EvalT>(ev);
			  }
		  }

		  // if surface_height is required in the whole mesh
		  if (std::find(requirements.begin(),requirements.end(),stateName) != requirements.end())
		  {
			  p = stateMgr.registerStateVariable(stateName, dl->node_scalar, elementBlockName,true, &entity);
			  p->set<std::string>("Field Name", fieldName);

			  ev = Teuchos::rcp(new PHAL::LoadStateField<EvalT,PHAL::AlbanyTraits>(*p));
			  fm0.template registerEvaluator<EvalT>(ev);
		  }
	  }

	  // ----------  Define Field Names ----------- //

	  int offset = 0;
	  {	// Velocity
		  Teuchos::ArrayRCP<std::string> dof_names(1);
		  Teuchos::ArrayRCP<std::string> resid_names(1);
		  dof_names[0] = "Velocity";
		  resid_names[0] = "Stokes Residual";

		  // --- Interpolation and utilities ---
		  fm0.template registerEvaluator<EvalT> (evalUtils.constructGatherSolutionEvaluator_noTransient(true, dof_names, offset));

		  fm0.template registerEvaluator<EvalT> (evalUtils.constructScatterResidualEvaluator(true, resid_names, offset, "Scatter Stokes"));

		  fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFVecInterpolationEvaluator(dof_names[0],offset));

		  fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFVecGradInterpolationEvaluator(dof_names[0],offset));

		  if (basalSideName!="INVALID")
		  {
			  //---- Restrict velocity from cell-based to cell-side-based
			  fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFCellToSideEvaluator(dof_names[0],basalSideName,"Node Vector",cellType,"Basal Velocity"));

			  //---- Interpolate velocity on QP on side
			  fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFVecInterpolationSideEvaluator("Basal Velocity", basalSideName));

		  	  //---- Interpolate velocity gradient on QP on side
		  	  fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFVecGradInterpolationSideEvaluator("Basal Velocity", basalSideName));
		  }
		  if (surfaceSideName!="INVALID")
		  {
			  //---- Restrict velocity (the solution) from cell-based to cell-side-based on upper side
			  fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFCellToSideEvaluator(dof_names[0],surfaceSideName,"Node Vector",cellType,"Surface Velocity"));
		  }

		  offset += 2;
	  }

	  {	// Enthalpy
		  Teuchos::ArrayRCP<std::string> dof_names(1);
		  Teuchos::ArrayRCP<std::string> resid_names(1);
		  dof_names[0] = "Enthalpy";
		  resid_names[0] = "Enthalpy Residual";

		  // no transient
		  fm0.template registerEvaluator<EvalT> (evalUtils.constructGatherSolutionEvaluator_noTransient(false, dof_names, offset));

		  fm0.template registerEvaluator<EvalT> (evalUtils.constructScatterResidualEvaluator(false, resid_names, offset, "Scatter Enthalpy"));

		  fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFInterpolationEvaluator(dof_names[0], offset));

		  fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFGradInterpolationEvaluator(dof_names[0], offset));

		  // --- Restrict enthalpy from cell-based to cell-side-based
		  fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFCellToSideEvaluator(dof_names[0],basalSideName,"Node Scalar",cellType));

		  offset++;
	  }

	  {	// w_z
		  Teuchos::ArrayRCP<std::string> dof_names(1);
		  Teuchos::ArrayRCP<std::string> resid_names(1);
		  dof_names[0] = "w_z";
		  resid_names[0] = "w_z Residual";

		  // no transient
		  fm0.template registerEvaluator<EvalT> (evalUtils.constructGatherSolutionEvaluator_noTransient(false, dof_names, offset));

		  fm0.template registerEvaluator<EvalT> (evalUtils.constructScatterResidualEvaluator(false, resid_names, offset, "Scatter w_z"));

		  fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFInterpolationEvaluator(dof_names[0], offset));
	  }

	  // ------------------- Interpolations and utilities ------------------ //

	  fm0.template registerEvaluator<EvalT> (evalUtils.constructGatherCoordinateVectorEvaluator());

	  fm0.template registerEvaluator<EvalT> (evalUtils.constructMapToPhysicalFrameEvaluator(cellType, cellCubature));

	  fm0.template registerEvaluator<EvalT> (evalUtils.constructComputeBasisFunctionsEvaluator(cellType, cellBasis, cellCubature));

	  // Interpolate surface height ---> for Stokes
	  fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFInterpolationEvaluator_noDeriv("Surface Height"));

	  // Interpolate surface height gradient ---> for Stokes
	  fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFGradInterpolationEvaluator_noDeriv("Surface Height"));

	  fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFGradInterpolationEvaluator("Melting Temp"));

	  fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFInterpolationEvaluator("phi"));

	  fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFGradInterpolationEvaluator("phi"));

	  // Interpolate temperature from nodes to cell
	  if(needsDiss)
		  fm0.template registerEvaluator<EvalT> (evalUtils.constructNodesToCellInterpolationEvaluator("Temperature",false));

	  // Interpolate pressure melting temperature gradient from nodes to QPs
	  fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFGradInterpolationSideEvaluator("Melting Temp",basalSideName));

	  fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFInterpolationEvaluator("Melting Temp"));

	  fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFInterpolationEvaluator("Melting Enthalpy"));

	  if(needsBasFric)
	  {
		  fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFInterpolationEvaluator("Basal Heat"));

		  fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFInterpolationEvaluator("Basal Heat SUPG"));
	  }

	  // --- Utilities for Geotermal flux
	  if(!isGeoFluxConst)
	  {
		  fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFInterpolationEvaluator("Geo Flux Heat"));

		  fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFInterpolationEvaluator("Geo Flux Heat SUPG"));
	  }

	  fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFInterpolationEvaluator("w"));

	  fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFSideToCellEvaluator("basal_melt_rate",basalSideName,"Node Scalar",cellType,"basal_melt_rate"));

	  if (basalSideName!="INVALID")
	  {
		  // -------------------- Special evaluators for side handling ----------------- //

	      //---- Restrict vertex coordinates from cell-based to cell-side-based
		  fm0.template registerEvaluator<EvalT> (evalUtils.getMSTUtils().constructDOFCellToSideEvaluator("Coord Vec",basalSideName,"Vertex Vector",cellType,"Coord Vec " + basalSideName));

	      //---- Compute side basis functions
		  fm0.template registerEvaluator<EvalT> (evalUtils.constructComputeBasisFunctionsSideEvaluator(cellType, basalSideBasis, basalCubature, basalSideName));

	      //---- Compute Quad Points coordinates on the side set
		  fm0.template registerEvaluator<EvalT> (evalUtils.constructMapToPhysicalFrameSideEvaluator(cellType,basalCubature,basalSideName));

	      //---- Interpolate thickness gradient on QP on side
		  fm0.template registerEvaluator<EvalT>(evalUtils.getPSTUtils().constructDOFGradInterpolationSideEvaluator("Ice Thickness", basalSideName));

	      //---- Interpolate thickness on QP on side
	      fm0.template registerEvaluator<EvalT>(evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator("Ice Thickness", basalSideName));

	      if(needsBasFric)
		  {
	    	  // --- Interpolate Beta on QP on side
	    	  fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator("Beta", basalSideName));
		  }

	      //---- Interpolate surface height on QP on side
	      fm0.template registerEvaluator<EvalT>(evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator("Surface Height", basalSideName));

		  // --- Restrict enthalpy Hs from cell-based to cell-side-based
		  fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFCellToSideEvaluator("Melting Enthalpy",basalSideName,"Node Scalar",cellType));

		  // --- Utilities for Basal Melt Rate
		  fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFCellToSideEvaluator("Melting Temp",basalSideName,"Node Scalar",cellType));

		  fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFCellToSideEvaluator("phi",basalSideName,"Node Scalar",cellType));

	  	  // --- Interpolate geotermal_flux on QP on side
	  	  fm0.template registerEvaluator<EvalT> (evalUtils.getPSTUtils().constructDOFInterpolationSideEvaluator("Basal Heat Flux", basalSideName));

		  // --- Restrict vertical velocity from cell-based to cell-side-based
		  fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFCellToSideEvaluator("w",basalSideName,"Node Scalar",cellType));

		  fm0.template registerEvaluator<EvalT> (evalUtils.constructDOFInterpolationSideEvaluator("w", basalSideName));
	  }

	  if (surfaceSideName!="INVALID")
	  {
		  //---- Restrict vertex coordinates from cell-based to cell-side-based
		  fm0.template registerEvaluator<EvalT> (evalUtils.getMSTUtils().constructDOFCellToSideEvaluator("Coord Vec",surfaceSideName,"Vertex Vector",cellType,"Coord Vec " + surfaceSideName));

		  //---- Compute side basis functions
		  fm0.template registerEvaluator<EvalT> (evalUtils.constructComputeBasisFunctionsSideEvaluator(cellType, surfaceSideBasis, surfaceCubature, surfaceSideName));

		  //---- Interpolate surface velocity on QP on side
		  fm0.template registerEvaluator<EvalT>(evalUtils.getPSTUtils().constructDOFVecInterpolationSideEvaluator("Observed Surface Velocity", surfaceSideName));

		  //---- Interpolate surface velocity rms on QP on side
		  fm0.template registerEvaluator<EvalT>(evalUtils.getPSTUtils().constructDOFVecInterpolationSideEvaluator("Observed Surface Velocity RMS", surfaceSideName));

		  //---- Interpolate velocity (the solution) on QP on side
		  fm0.template registerEvaluator<EvalT>(evalUtils.constructDOFVecInterpolationSideEvaluator("Surface Velocity", surfaceSideName));
	  }

	  // -------------------------------- FELIX evaluators ------------------------- //

	  // --- FO Stokes Resid --- //
	  {
		  p = Teuchos::rcp(new Teuchos::ParameterList("Stokes Resid"));

		  //Input
		  p->set<std::string>("Weighted BF Variable Name", "wBF");
		  p->set<std::string>("Weighted Gradient BF Variable Name", "wGrad BF");
		  p->set<std::string>("Velocity QP Variable Name", "Velocity");
		  p->set<std::string>("Velocity Gradient QP Variable Name", "Velocity Gradient");
		  p->set<std::string>("Body Force Variable Name", "Body Force");
		  p->set<std::string>("Viscosity QP Variable Name", "FELIX Viscosity");
		  p->set<std::string>("Coordinate Vector Name", "Coord Vec");
		  p->set<std::string>("Basal Residual Variable Name", "Basal Residual");

		  p->set<bool>("Needs Basal Residual", sliding);

		  p->set<Teuchos::ParameterList*>("Stereographic Map", &params->sublist("Stereographic Map"));
		  p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("Equation Set"));

		  //Output
		  p->set<std::string>("Residual Variable Name", "Stokes Residual");

		  ev = Teuchos::rcp(new FELIX::StokesFOResid<EvalT,PHAL::AlbanyTraits>(*p,dl));
		  fm0.template registerEvaluator<EvalT>(ev);
	  }

	  if (sliding)
	  {
		  // --- Basal Residual --- //
		  p = Teuchos::rcp(new Teuchos::ParameterList("Stokes Basal Residual"));

		  //Input
		  p->set<std::string>("BF Side Name", "BF "+basalSideName);
		  p->set<std::string>("Weighted Measure Name", "Weighted Measure "+basalSideName);
		  p->set<std::string>("Basal Friction Coefficient Side QP Variable Name", "Beta");
		  p->set<std::string>("Velocity Side QP Variable Name", "Basal Velocity");
		  p->set<std::string>("Side Set Name", basalSideName);

		  p->set<Teuchos::RCP<shards::CellTopology> >("Cell Type", cellType);
		  p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("FELIX Basal Friction Coefficient"));
		  p->set<Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

		  p->set<std::string>("Continuation Parameter Name","Glen's Law Homotopy Parameter");

		  //Output
		  p->set<std::string>("Basal Residual Variable Name", "Basal Residual");

		  ev = Teuchos::rcp(new FELIX::StokesFOBasalResid<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT>(*p,dl));
		  fm0.template registerEvaluator<EvalT>(ev);
	  }

	  // --- FELIX Viscosity ---
	  {
		  p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Viscosity"));

		  //Input
		  p->set<std::string>("Coordinate Vector Variable Name", "Coord Vec");
		  p->set<std::string>("Velocity QP Variable Name", "Velocity");
		  p->set<std::string>("Velocity Gradient QP Variable Name", "Velocity Gradient");
		  p->set<std::string>("Temperature Variable Name", "Temperature");
		  p->set<std::string>("Flow Factor Variable Name", "flow_factor");

		  p->set<Teuchos::ParameterList*>("Stereographic Map", &params->sublist("Stereographic Map"));
		  p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("FELIX Viscosity"));
		  p->set<Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

		  p->set<std::string>("Continuation Parameter Name","Glen's Law Homotopy Parameter");

		  //Output
		  p->set<std::string>("Viscosity QP Variable Name", "FELIX Viscosity");
		  p->set<std::string>("EpsilonSq QP Variable Name", "FELIX EpsilonSq");

		  ev = Teuchos::rcp(new FELIX::ViscosityFO<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT,typename EvalT::ScalarT>(*p,dl));
		  fm0.template registerEvaluator<EvalT>(ev);
	  }

	  //--- Body Force ---//
	  {
		  p = Teuchos::rcp(new Teuchos::ParameterList("Body Force"));

		  //Input
		  p->set<std::string>("FELIX Viscosity QP Variable Name", "FELIX Viscosity");
		  p->set<std::string>("Surface Height Gradient Name", "Surface Height Gradient");
		  p->set<std::string>("Surface Height Name", "Surface Height");
		  p->set<std::string>("Coordinate Vector Variable Name", "Coord Vec");

		  p->set<Teuchos::ParameterList*>("Parameter List", &params->sublist("Body Force"));
		  p->set<Teuchos::ParameterList*>("Stereographic Map", &params->sublist("Stereographic Map"));
		  p->set<Teuchos::ParameterList*>("Physical Parameter List", &params->sublist("FELIX Physical Parameters"));

		  #ifdef CISM_HAS_FELIX
		  p->set<std::string>("Surface Height Gradient QP Variable Name", "CISM Surface Height Gradient");
		  #endif

		  //Output
		  p->set<std::string>("Body Force Variable Name", "Body Force");

		  ev = Teuchos::rcp(new FELIX::StokesFOBodyForce<EvalT,PHAL::AlbanyTraits>(*p,dl));
		  fm0.template registerEvaluator<EvalT>(ev);
	  }

	  // --- Enthalpy Residual --- //
	  {
		  p = Teuchos::rcp(new Teuchos::ParameterList("Enthalpy Resid"));

		  //Input
		  p->set<std::string>("Weighted BF Variable Name", "wBF");
		  p->set< Teuchos::RCP<PHX::DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);
		  p->set<std::string>("Weighted Gradient BF Variable Name", "wGrad BF");
		  p->set< Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout", dl->node_qp_vector);

		  p->set<std::string>("Enthalpy QP Variable Name", "Enthalpy");
		  p->set< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);
		  p->set<std::string>("Enthalpy Gradient QP Variable Name", "Enthalpy Gradient");
		  p->set< Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout", dl->qp_gradient);
		  p->set<std::string>("Enthalpy Hs QP Variable Name", "Melting Enthalpy");
		  p->set< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

		  p->set<std::string>("Velocity QP Variable Name", "Velocity");
		  p->set< Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout", dl->qp_vector);

		  p->set<std::string>("Coordinate Vector Name", "Coord Vec");

		  // Vertical velocity derived from the continuity equation
		  p->set<std::string>("Vertical Velocity QP Variable Name", "w");

		  p->set<std::string>("Geotermal Flux Heat QP Variable Name","Geo Flux Heat");
		  p->set<std::string>("Geotermal Flux Heat QP SUPG Variable Name","Geo Flux Heat SUPG");

		  p->set<std::string>("Melting Temperature Gradient QP Variable Name","Melting Temp Gradient");

		  if(needsDiss)
		  {
			  p->set<std::string>("Dissipation QP Variable Name", "FELIX Dissipation");
		  }

		  if(needsBasFric)
		  {
			  p->set<std::string>("Basal Friction Heat QP Variable Name", "Basal Heat");
			  p->set<std::string>("Basal Friction Heat QP SUPG Variable Name", "Basal Heat SUPG");
		  }

		  p->set<std::string>("Water Content QP Variable Name","phi");
		  p->set<std::string>("Water Content Gradient QP Variable Name","phi Gradient");

		  p->set<bool>("Needs Dissipation", needsDiss);
		  p->set<bool>("Needs Basal Friction", needsBasFric);
		  p->set<bool>("Constant Geotermal Flux", isGeoFluxConst);

		  p->set<Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);
		  p->set<std::string>("Continuation Parameter Name","Glen's Law Homotopy Parameter");

		  p->set<Teuchos::ParameterList*>("FELIX Physical Parameters", &params->sublist("FELIX Physical Parameters"));
		  p->set<Teuchos::ParameterList*>("SUPG Settings", &params->sublist("SUPG Settings"));

		  //Output
		  p->set<std::string>("Residual Variable Name", "Enthalpy Residual");
		  p->set< Teuchos::RCP<PHX::DataLayout> >("Node Scalar Data Layout", dl->node_scalar);

		  ev = Teuchos::rcp(new FELIX::EnthalpyResid<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT>(*p,dl));
		  fm0.template registerEvaluator<EvalT>(ev);
	  }

	  // --- FELIX Dissipation ---
	  if(needsDiss)
	  {
		  {
		  p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Dissipation"));

		  //Input
		  p->set<std::string>("Viscosity QP Variable Name", "FELIX Viscosity");
		  p->set<std::string>("EpsilonSq QP Variable Name", "FELIX EpsilonSq");

		  //Output
		  p->set<std::string>("Dissipation QP Variable Name", "FELIX Dissipation");

		  ev = Teuchos::rcp(new FELIX::Dissipation<EvalT,PHAL::AlbanyTraits>(*p,dl));
		  fm0.template registerEvaluator<EvalT>(ev);
		  }
	  }

	  // --- FELIX Basal friction heat ---
	  if(needsBasFric)
	  {
		  p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Basal Friction Heat"));
		  //Input
		  p->set<std::string>("BF Side Name", "BF "+basalSideName);
		  p->set<std::string>("Gradient BF Side Name", "Grad BF "+basalSideName);
		  p->set<std::string>("Weighted Measure Name", "Weighted Measure "+basalSideName);
		  p->set<std::string>("Velocity Side QP Variable Name", "Basal Velocity");
		  p->set<std::string>("Vertical Velocity Side QP Variable Name", "w");
		  p->set<std::string>("Basal Friction Coefficient Side QP Variable Name", "Beta");

		  p->set<std::string>("Side Set Name", basalSideName);

		  p->set<Teuchos::ParameterList*>("SUPG Settings", &params->sublist("SUPG Settings"));

		  p->set<Teuchos::RCP<shards::CellTopology> >("Cell Type", cellType);

		  //Output
		  p->set<std::string>("Basal Friction Heat Variable Name", "Basal Heat");

		  if(haveSUPG)
			  p->set<std::string>("Basal Friction Heat SUPG Variable Name", "Basal Heat SUPG");

		  ev = Teuchos::rcp(new FELIX::BasalFrictionHeat<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT>(*p,dl));
		  fm0.template registerEvaluator<EvalT>(ev);
	  }

	  // --- FELIX Geothermal flux heat
	  {
		  p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Geotermal Flux Heat"));
		  //Input
		  p->set<std::string>("BF Side Name", "BF "+basalSideName);
		  p->set<std::string>("Gradient BF Side Name", "Grad BF "+basalSideName);
		  p->set<std::string>("Weighted Measure Name", "Weighted Measure "+basalSideName);
		  p->set<std::string>("Velocity Side QP Variable Name", "Basal Velocity");
		  p->set<std::string>("Vertical Velocity Side QP Variable Name", "w");

		  p->set<std::string>("Side Set Name", basalSideName);
		  p->set<Teuchos::RCP<shards::CellTopology> >("Cell Type", cellType);

		  p->set<Teuchos::ParameterList*>("SUPG Settings", &params->sublist("SUPG Settings"));

		  if(!isGeoFluxConst)
			  p->set<std::string>("Geotermal Flux Side QP Variable Name", "Basal Heat Flux");

		  p->set<bool>("Constant Geotermal Flux", isGeoFluxConst);

		  //Output
		  p->set<std::string>("Geotermal Flux Heat Variable Name", "Geo Flux Heat");

		  if(haveSUPG)
			  p->set<std::string>("Geotermal Flux Heat SUPG Variable Name", "Geo Flux Heat SUPG");

		  ev = Teuchos::rcp(new FELIX::GeoFluxHeat<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT>(*p,dl));
		  fm0.template registerEvaluator<EvalT>(ev);
	  }

	  // --- FELIX hydrostatic pressure
	  {
		  p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Hydrostatic Pressure"));

		  //Input
		  p->set<std::string>("Surface Height Variable Name", "Surface Height");
		  p->set<std::string>("Coordinate Vector Variable Name", "Coord Vec");

		  p->set<Teuchos::ParameterList*>("FELIX Physical Parameters", &params->sublist("FELIX Physical Parameters"));

		  //Output
		  p->set<std::string>("Hydrostatic Pressure Variable Name", "Hydrostatic Pressure");

		  ev = Teuchos::rcp(new FELIX::HydrostaticPressure<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT>(*p,dl));
		  fm0.template registerEvaluator<EvalT>(ev);
	  }

	  // --- FELIX pressure-melting temperature
	  {
		  p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Pressure Melting Temperature"));

		  //Input
		  p->set<std::string>("Hydrostatic Pressure Variable Name", "Hydrostatic Pressure");

		  p->set<Teuchos::ParameterList*>("FELIX Physical Parameters", &params->sublist("FELIX Physical Parameters"));

		  //Output
		  p->set<std::string>("Melting Temperature Variable Name", "Melting Temp");

		  ev = Teuchos::rcp(new FELIX::PressureMeltingTemperature<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT>(*p,dl));
		  fm0.template registerEvaluator<EvalT>(ev);
	  }

	  // --- FELIX pressure-melting enthalpy
	  {
		  p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Pressure Melting Enthalpy"));

		  //Input
		  p->set<std::string>("Melting Temperature Variable Name", "Melting Temp");

		  p->set<Teuchos::ParameterList*>("FELIX Physical Parameters", &params->sublist("FELIX Physical Parameters"));

		  //Output
		  p->set<std::string>("Enthalpy Hs Variable Name", "Melting Enthalpy");
		  ev = Teuchos::rcp(new FELIX::PressureMeltingEnthalpy<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT>(*p,dl));
		  fm0.template registerEvaluator<EvalT>(ev);
	  }

	  // --- FELIX Temperature
	  {
		  p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Temperature"));

		  //Input
		  p->set<std::string>("Melting Temperature Variable Name", "Melting Temp");
		  p->set<std::string>("Enthalpy Hs Variable Name", "Melting Enthalpy");
		  p->set<std::string>("Enthalpy Variable Name", "Enthalpy");

		  p->set<Teuchos::ParameterList*>("FELIX Physical Parameters", &params->sublist("FELIX Physical Parameters"));

		  //Output
		  p->set<std::string>("Temperature Variable Name", "Temperature");
		  p->set<std::string>("Diff Enthalpy Variable Name", "Diff Enth");

		  ev = Teuchos::rcp(new FELIX::Temperature<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT>(*p,dl));
		  fm0.template registerEvaluator<EvalT>(ev);

		  // Saving the temperature in the output mesh
		  {
			  std::string stateName = "Temperature_Cell";
			  p = stateMgr.registerStateVariable(stateName, dl->cell_scalar2, dl->dummy, elementBlockName, "scalar", 0.0, /* save state = */ false, /* write output = */ true);
			  p->set<std::string>("Field Name", "Temperature");
			  p->set<std::string>("Weights Name","Weights");
			  p->set("Weights Layout", dl->qp_scalar);
			  p->set("Field Layout", dl->cell_scalar2);
			  p->set< Teuchos::RCP<PHX::DataLayout> >("Dummy Data Layout",dl->dummy);

			  ev = Teuchos::rcp(new PHAL::SaveCellStateField<EvalT,PHAL::AlbanyTraits>(*p));
			  fm0.template registerEvaluator<EvalT>(ev);
		  }

		  // Saving the diff enthalpy field in the output mesh
		  {
			  fm0.template registerEvaluator<EvalT> (evalUtils.constructNodesToCellInterpolationEvaluator("Diff Enth",false));

			  std::string stateName = "h-hs_Cell";
			  p = stateMgr.registerStateVariable(stateName, dl->cell_scalar2, dl->dummy, elementBlockName, "scalar", 0.0, /* save state = */ false, /* write output = */ true);
			  p->set<std::string>("Field Name", "Diff Enth");
			  p->set<std::string>("Weights Name","Weights");
			  p->set("Weights Layout", dl->qp_scalar);
			  p->set("Field Layout", dl->cell_scalar2);
			  p->set< Teuchos::RCP<PHX::DataLayout> >("Dummy Data Layout",dl->dummy);

			  ev = Teuchos::rcp(new PHAL::SaveCellStateField<EvalT,PHAL::AlbanyTraits>(*p));
			  fm0.template registerEvaluator<EvalT>(ev);
		  }
	  }

	  // --- FELIX Liquid Water Fraction
	  {
		  p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Liquid Water Fraction"));

		  //Input
		  p->set<std::string>("Enthalpy Hs Variable Name", "Melting Enthalpy");
		  p->set<std::string>("Enthalpy Variable Name", "Enthalpy");

		  p->set<Teuchos::ParameterList*>("FELIX Physical Parameters", &params->sublist("FELIX Physical Parameters"));

		  p->set<Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);
		  p->set<std::string>("Continuation Parameter Name","Glen's Law Homotopy Parameter");

		  //Output
		  p->set<std::string>("Water Content Variable Name", "phi");
		  ev = Teuchos::rcp(new FELIX::LiquidWaterFraction<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT>(*p,dl));
		  fm0.template registerEvaluator<EvalT>(ev);

		  // Saving phi in the output mesh
		  {
			  fm0.template registerEvaluator<EvalT> (evalUtils.constructNodesToCellInterpolationEvaluator("phi",false));

			  std::string stateName = "phi";
			  entity = Albany::StateStruct::NodalDataToElemNode;
			  p = stateMgr.registerStateVariable(stateName, dl->cell_scalar2, dl->dummy, elementBlockName, "scalar", 0.0, /* save state = */ false, /* write output = */ true);

			  p->set<std::string>("Weights Name","Weights");
			  p->set("Weights Layout", dl->qp_scalar);
			  p->set("Field Layout", dl->cell_scalar2);
			  p->set< Teuchos::RCP<PHX::DataLayout> >("Dummy Data Layout",dl->dummy);

			  ev = Teuchos::rcp(new PHAL::SaveCellStateField<EvalT,PHAL::AlbanyTraits>(*p));
			  fm0.template registerEvaluator<EvalT>(ev);
		  }

		  // Forcing the execution of the evaluator
	      if (fieldManagerChoice == Albany::BUILD_RESID_FM)
	      {
	    	  if (ev->evaluatedFields().size()>0)
	      	  {
	    		  fm0.template requireField<EvalT>(*ev->evaluatedFields()[0]);
	      	  }
	      }
	  }

	  // --- w_z Residual ---
	  {
		  p = Teuchos::rcp(new Teuchos::ParameterList("w_z Resid"));

		  //Input
		  p->set<std::string>("Weighted BF Variable Name", "wBF");
		  p->set< Teuchos::RCP<PHX::DataLayout> >("Node QP Scalar Data Layout", dl->node_qp_scalar);

		  p->set<std::string>("w_z QP Variable Name", "w_z");
		  p->set< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

		  p->set<std::string>("Velocity Gradient QP Variable Name", "Velocity Gradient");
		  p->set< Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout", dl->qp_vecgradient);

		  //Output
		  p->set<std::string>("Residual Variable Name", "w_z Residual");
		  p->set< Teuchos::RCP<PHX::DataLayout> >("Node Scalar Data Layout", dl->node_scalar);

		  ev = Teuchos::rcp(new FELIX::w_ZResid<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT>(*p,dl));
		  fm0.template registerEvaluator<EvalT>(ev);
	  }

	  // --- FELIX Integral 1D w_z
	  {
		  p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Integral 1D w_z"));

		  p->set<std::string>("Basal Melt Rate Variable Name", "basal_melt_rate");
		  p->set<std::string>("Thickness Variable Name", "Ice Thickness");

		  p->set<Teuchos::RCP<const CellTopologyData> >("Cell Topology",Teuchos::rcp(new CellTopologyData(meshSpecs.ctd)));

		  p->set<bool>("Stokes and Thermo coupled", true);

		  //Output
		  p->set<std::string>("Integral1D w_z Variable Name", "int1Dw_z");
	      ev = Teuchos::rcp(new FELIX::Integral1Dw_Z<EvalT,PHAL::AlbanyTraits>(*p,dl));
	      fm0.template registerEvaluator<EvalT>(ev);
	  }

	  // --- FELIX Vertical Velocity
	  {
		  p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Vertical Velocity"));

		  //Input
		  p->set<std::string>("Thickness Variable Name", "Ice Thickness");
		  p->set<std::string>("Integral1D w_z Variable Name", "int1Dw_z");

		  //Output
		  p->set<std::string>("Vertical Velocity Variable Name", "w");
		  ev = Teuchos::rcp(new FELIX::VerticalVelocity<EvalT,PHAL::AlbanyTraits,typename EvalT::ParamScalarT>(*p,dl));
		  fm0.template registerEvaluator<EvalT>(ev);

		  {
			  fm0.template registerEvaluator<EvalT> (evalUtils.constructNodesToCellInterpolationEvaluator("w",false));

			  std::string stateName = "w";
			  entity = Albany::StateStruct::NodalDataToElemNode;
			  p = stateMgr.registerStateVariable(stateName, dl->cell_scalar2, dl->dummy, elementBlockName, "scalar", 0.0, /* save state = */ false, /* write output = */ true);

			  p->set<std::string>("Weights Name","Weights");
			  p->set("Weights Layout", dl->qp_scalar);
			  p->set("Field Layout", dl->cell_scalar2);
			  p->set< Teuchos::RCP<PHX::DataLayout> >("Dummy Data Layout",dl->dummy);

			  ev = Teuchos::rcp(new PHAL::SaveCellStateField<EvalT,PHAL::AlbanyTraits>(*p));
			  fm0.template registerEvaluator<EvalT>(ev);
		  }
	  }

	  // --- FELIX Basal Melt Rate
	  {
		  p = Teuchos::rcp(new Teuchos::ParameterList("FELIX Basal Melt Rate"));

		  //Input
		  p->set<std::string>("Water Content Side Variable Name", "phi");
		  p->set<std::string>("Geotermal Flux Side Variable Name", "Basal Heat Flux");
		  p->set<std::string>("Velocity Side Variable Name", "Basal Velocity");
		  p->set<std::string>("Basal Friction Coefficient Side Variable Name", "Beta");
		  p->set<std::string>("Enthalpy Hs Side Variable Name", "Melting Enthalpy");
		  p->set<std::string>("Enthalpy Side Variable Name", "Enthalpy");

		  p->set<Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);
		  p->set<std::string>("Continuation Parameter Name","Glen's Law Homotopy Parameter");

		  p->set<Teuchos::ParameterList*>("FELIX Physical Parameters", &params->sublist("FELIX Physical Parameters"));

		  p->set<std::string>("Side Set Name", basalSideName);

		  //Output
		  p->set<std::string>("Basal Melt Rate Variable Name", "basal_melt_rate");
	      ev = Teuchos::rcp(new FELIX::BasalMeltRate<EvalT,PHAL::AlbanyTraits,typename EvalT::ScalarT>(*p,dl_basal));
	      fm0.template registerEvaluator<EvalT>(ev);

		  {
			  fm0.template registerEvaluator<EvalT> (evalUtils.constructNodesToCellInterpolationEvaluator("basal_melt_rate",false));

			  std::string stateName = "basal_melt_rate";
			  entity = Albany::StateStruct::NodalDataToElemNode;
			  p = stateMgr.registerStateVariable(stateName, dl->cell_scalar2, dl->dummy, elementBlockName, "scalar", 0.0, /* save state = */ false, /* write output = */ true);

			  p->set<std::string>("Weights Name","Weights");
			  p->set("Weights Layout", dl->qp_scalar);
			  p->set("Field Layout", dl->cell_scalar2);
			  p->set< Teuchos::RCP<PHX::DataLayout> >("Dummy Data Layout",dl->dummy);

			  ev = Teuchos::rcp(new PHAL::SaveCellStateField<EvalT,PHAL::AlbanyTraits>(*p));
			  fm0.template registerEvaluator<EvalT>(ev);
		  }
	  }

	  {
		  //--- Shared Parameter for homotopy parameter: h ---//
		  p = Teuchos::rcp(new Teuchos::ParameterList("Glen's Law Homotopy Parameter"));

		  std::string param_name = "Glen's Law Homotopy Parameter";
		  p->set<std::string>("Parameter Name", param_name);
		  p->set< Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

		  Teuchos::RCP<FELIX::SharedParameter<EvalT,PHAL::AlbanyTraits,StokesParamEnum,Homotopy>> ptr_h;
		  ptr_h = Teuchos::rcp(new FELIX::SharedParameter<EvalT,PHAL::AlbanyTraits,StokesParamEnum,Homotopy>(*p,dl));
		  ptr_h->setNominalValue(params->sublist("Parameters"),params->sublist("FELIX Viscosity").get<double>(param_name,-1.0));
		  fm0.template registerEvaluator<EvalT>(ptr_h);
	  }

	  if (fieldManagerChoice == Albany::BUILD_RESID_FM)
	  {
		  PHX::Tag<typename EvalT::ScalarT> res_tag0("Scatter Stokes", dl->dummy);
		  fm0.requireField<EvalT>(res_tag0);
		  PHX::Tag<typename EvalT::ScalarT> res_tag1("Scatter Enthalpy", dl->dummy);
	      fm0.requireField<EvalT>(res_tag1);
	      PHX::Tag<typename EvalT::ScalarT> res_tag2("Scatter w_z", dl->dummy);
	      fm0.requireField<EvalT>(res_tag2);
	  }
	  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM)
	  {
		  Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl);
	      return respUtils.constructResponses(fm0, *responseList, Teuchos::null, stateMgr);
	  }

	  return Teuchos::null;
}

#endif /* FELIX_STOKESFOTHERMOCOUPLED_PROBLEM_HPP */
