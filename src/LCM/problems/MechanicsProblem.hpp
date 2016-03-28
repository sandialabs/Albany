//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef MECHANICSPROBLEM_HPP
#define MECHANICSPROBLEM_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "NOX_StatusTest_ModelEvaluatorFlag.h"

#include "Albany_AbstractProblem.hpp"

#include "Phalanx.hpp"
#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_AlbanyTraits.hpp"

#include "AAdapt_RC_Manager.hpp"
#include "MaterialDatabase.h"

static int dir_count = 0; //counter for registration of dirichlet_field 
 
namespace Albany
{

//------------------------------------------------------------------------------
///
/// \brief Definition for the Mechanics Problem
///
class MechanicsProblem: public Albany::AbstractProblem
{
public:

  typedef Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> FC;

  ///
  /// Default constructor
  ///
  MechanicsProblem(const Teuchos::RCP<Teuchos::ParameterList>& params,
      const Teuchos::RCP<ParamLib>& param_lib,
      const int num_dims,
      const Teuchos::RCP<AAdapt::rc::Manager>& rc_mgr,
      Teuchos::RCP<const Teuchos::Comm<int>>& commT);
  ///
  /// Destructor
  ///
  virtual
  ~MechanicsProblem();

  ///
  Teuchos::RCP<std::map<std::string, std::string>>
  constructFieldNameMap(bool surface_flag);

  ///
  /// Return number of spatial dimensions
  ///
  virtual
  int
  spatialDimension() const
  {
    return num_dims_;
  }

  ///
  /// Build the PDE instantiations, boundary conditions, initial solution
  ///
  virtual
  void
  buildProblem(Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct>>
      meshSpecs,
      StateManager& stateMgr);

  ///
  /// Build evaluators
  ///
  virtual Teuchos::Array<Teuchos::RCP<const PHX::FieldTag>>
  buildEvaluators(PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
      const Albany::MeshSpecsStruct& meshSpecs,
      Albany::StateManager& stateMgr,
      Albany::FieldManagerChoice fmchoice,
      const Teuchos::RCP<Teuchos::ParameterList>& responseList);

  ///
  /// Each problem must generate it's list of valid parameters
  ///
  Teuchos::RCP<const Teuchos::ParameterList>
  getValidProblemParameters() const;

  ///
  /// Retrieve the state data
  ///
  void
  getAllocatedStates(Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<FC>>>
      old_state,
      Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<FC>>>
      new_state) const;

  ///
  /// Add a custom NOX Status Test (for example, to trigger a global load step reduction)
  ///
  void
  applyProblemSpecificSolverSettings(Teuchos::RCP<Teuchos::ParameterList> params);

  //----------------------------------------------------------------------------
private:

  ///
  /// Private to prohibit copying
  ///
  MechanicsProblem(const MechanicsProblem&);

  ///
  /// Private to prohibit copying
  ///
  MechanicsProblem& operator=(const MechanicsProblem&);

  //----------------------------------------------------------------------------
public:

  ///
  /// Main problem setup routine.
  /// Not directly called, but indirectly by following functions
  ///
  template<typename EvalT>
  Teuchos::RCP<const PHX::FieldTag>
  constructEvaluators(PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
      const Albany::MeshSpecsStruct& meshSpecs,
      Albany::StateManager& stateMgr,
      Albany::FieldManagerChoice fmchoice,
      const Teuchos::RCP<Teuchos::ParameterList>&
      responseList);

  ///
  /// Setup for the dirichlet BCs
  ///
  void
  constructDirichletEvaluators(const Albany::MeshSpecsStruct& meshSpecs);

  ///
  /// Setup for the traction BCs
  ///
  void
  constructNeumannEvaluators(
      const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs);
  

  //----------------------------------------------------------------------------
protected:

  ///
  /// Enumerated type describing how a variable appears
  ///
  enum MECH_VAR_TYPE
  {
    MECH_VAR_TYPE_NONE,      //! Variable does not appear
    MECH_VAR_TYPE_CONSTANT,  //! Variable is a constant
    MECH_VAR_TYPE_DOF,       //! Variable is a degree-of-freedom
    MECH_VAR_TYPE_TIMEDEP    //! Variable is stepped by LOCA in time
  };

  // Source function type
  enum SOURCE_TYPE
  {
    SOURCE_TYPE_NONE,      //! No source
    SOURCE_TYPE_INPUT,     //! Source is specified in input file
    SOURCE_TYPE_MATERIAL   //! Source is specified in material database
  };

  ///
  /// Accessor for variable type
  ///
  void getVariableType(Teuchos::ParameterList& param_list,
      const std::string& default_type,
      MECH_VAR_TYPE& variable_type,
      bool& have_variable,
      bool& have_equation);

  ///
  /// Conversion from enum to string
  ///
  std::string variableTypeToString(const MECH_VAR_TYPE variable_type);

  ///
  /// Construct a string for consistent output with surface elements
  ///
  //std::string stateString(std::string, bool);

  ///
  /// Boundary conditions on source term
  ///
  bool have_source_;

  // Type of thermal source that is in effect
  SOURCE_TYPE thermal_source_;

  // Has the thermal source been evaluated in this element block?
  bool thermal_source_evaluated_;

  // Is this a contact problem?
  bool have_contact_;

  ///
  /// num of dimensions
  ///
  int num_dims_;

  ///
  /// number of integration points
  ///
  int num_pts_;

  ///
  /// number of element nodes
  ///
  int num_nodes_;

  ///
  /// number of element vertices
  ///
  int num_vertices_;

  ///
  /// Type of mechanics variable (disp or acc)
  ///
  MECH_VAR_TYPE mech_type_;

  ///
  /// Variable types
  ///
  MECH_VAR_TYPE temperature_type_;
  MECH_VAR_TYPE pore_pressure_type_;
  MECH_VAR_TYPE transport_type_;
  MECH_VAR_TYPE hydrostress_type_;
  MECH_VAR_TYPE damage_type_;
  MECH_VAR_TYPE stab_pressure_type_;

  ///
  /// Have mechanics
  ///
  bool have_mech_;

  ///
  /// Have temperature
  ///
  bool have_temperature_;

  ///
  /// Have pore pressure
  ///
  bool have_pore_pressure_;

  ///
  /// Have transport
  ///
  bool have_transport_;

  ///
  /// Have hydrostatic stress
  ///
  bool have_hydrostress_;

  ///
  /// Have damage
  ///
  bool have_damage_;

  ///
  /// Have stabilized pressure
  ///
  bool have_stab_pressure_;

  ///
  /// Have mechanics equation
  ///
  bool have_mech_eq_;

  ///
  /// Have mesh adaptation - both the "Adaptation" sublist exists and the user has specified that the method
  ///    is "RPI Albany Size"
  ///
  bool have_sizefield_adaptation_;

  ///
  /// Have temperature equation
  ///
  bool have_temperature_eq_;

  ///
  /// Have pore pressure equation
  ///
  bool have_pore_pressure_eq_;

  ///
  /// Have transport equation
  ///
  bool have_transport_eq_;

  ///
  /// Have projected hydrostatic stress term
  /// in transport equation
  ///
  bool have_hydrostress_eq_;

  ///
  /// Have damage equation
  ///
  bool have_damage_eq_;

  ///
  /// Have stabilized pressure equation
  ///
  bool have_stab_pressure_eq_;

  ///
  /// Have a Peridynamics block
  ///
  bool have_peridynamics_;

  ///
  /// Topology adaptation (adaptive insertion)
  ///
  bool have_topmod_adaptation_;

  ///
  /// Data layouts
  ///
  Teuchos::RCP<Albany::Layouts> dl_;

  ///
  /// RCP to matDB object
  ///
  Teuchos::RCP<LCM::MaterialDatabase> material_db_;

  ///
  /// old state data
  ///
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<FC>>> old_state_;

  ///
  /// new state data
  ///
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<FC>>> new_state_;

  ///
  /// Reference configuration manager for mesh adaptation with ref config
  /// updating.
  ///
  Teuchos::RCP<AAdapt::rc::Manager> rc_mgr_;

  ///
  /// User defined NOX Status Test that allows model evaluators to set the NOX status to "failed".
  /// This is useful because it forces a global load step reduction.
  ///
  Teuchos::RCP<NOX::StatusTest::Generic> nox_status_test_;
};
//------------------------------------------------------------------------------
}

#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_ResponseUtilities.hpp"
#include "Albany_EvaluatorUtils.hpp"

#include "PHAL_NSMaterialProperty.hpp"
#include "PHAL_Source.hpp"
#include "PHAL_SaveStateField.hpp"

#include "FieldNameMap.hpp"

#include "MechanicsResidual.hpp"
#include "Time.hpp"
#include "SurfaceBasis.hpp"
#include "SurfaceVectorJump.hpp"
#include "SurfaceVectorGradient.hpp"
#include "SurfaceScalarJump.hpp"
#include "SurfaceScalarGradientOperator.hpp"
#include "SurfaceVectorResidual.hpp"
#include "CurrentCoords.hpp"
#include "TvergaardHutchinson.hpp"
#include "MeshSizeField.hpp"
//#include "SurfaceCohesiveResidual.hpp"

// Constitutive Model Interface and parameters
#include "Kinematics.hpp"
#include "ConstitutiveModelInterface.hpp"
#include "ConstitutiveModelParameters.hpp"
#include "FirstPK.hpp"

// Generic Transport Residual
#include "TransportResidual.hpp"

// Thermomechanics specific evaluators
#include "ThermoMechanicalCoefficients.hpp"

// Poromechanics specific evaluators
#include "GradientElementLength.hpp"
#include "BiotCoefficient.hpp"
#include "BiotModulus.hpp"
#include "KCPermeability.hpp"
#include "Porosity.hpp"
#include "TLPoroPlasticityResidMass.hpp"
#include "TLPoroStress.hpp"
#include "SurfaceTLPoroMassResidual.hpp"

// Thermohydromechanics specific evaluators
#include "ThermoPoroPlasticityResidMass.hpp"
#include "ThermoPoroPlasticityResidEnergy.hpp"
#include "MixtureThermalExpansion.hpp"
#include "MixtureSpecificHeat.hpp"

// Hydrogen transport specific evaluators
#include "ScalarL2ProjectionResidual.hpp"
#include "SurfaceL2ProjectionResidual.hpp"
#include "HDiffusionDeformationMatterResidual.hpp"
#include "SurfaceHDiffusionDefResidual.hpp"
#include "LatticeDefGrad.hpp"
#include "TransportCoefficients.hpp"

// Helium bubble specific evaluators
#include "HeliumODEs.hpp"

// Damage equation specific evaluators
#include "DamageCoefficients.hpp"

// Damage equation specific evaluators
#include "StabilizedPressureResidual.hpp"

#ifdef ALBANY_CONTACT
// Contact evaluator
#include "MortarContactConstraints.hpp"
#endif

//------------------------------------------------------------------------------
template<typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
Albany::MechanicsProblem::
constructEvaluators(PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
    const Albany::MeshSpecsStruct& meshSpecs,
    Albany::StateManager& stateMgr,
    Albany::FieldManagerChoice fieldManagerChoice,
    const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  typedef Teuchos::RCP<
      Intrepid2::Basis<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device>>>
  Intrepid2Basis;

  // Collect problem-specific response parameters

  Teuchos::RCP<Teuchos::ParameterList> pFromProb = Teuchos::rcp(
      new Teuchos::ParameterList("Response Parameters from Problem"));

  // get the name of the current element block
  std::string eb_name = meshSpecs.ebName;

  // get the name of the material model to be used (and make sure there is one)
  std::string material_model_name =
      material_db_->
          getElementBlockSublist(eb_name, "Material Model").get<std::string>(
          "Model Name");
  TEUCHOS_TEST_FOR_EXCEPTION(
      material_model_name.length() == 0,
      std::logic_error,
      "A material model must be defined for block: "
          + eb_name);

#ifdef ALBANY_VERBOSE
  *out << "In MechanicsProblem::constructEvaluators" << std::endl;
  *out << "element block name: " << eb_name << std::endl;
  *out << "material model name: " << material_model_name << std::endl;
#endif

  // insert user-defined NOX Status Test for material models that use it
  {
    std::string matName = material_db_->getElementBlockParam<std::string>(eb_name, "material");
    Teuchos::ParameterList& param_list = material_db_->getElementBlockSublist(eb_name, matName);
    std::string materialModelName = param_list.sublist("Material Model").get<std::string>("Model Name");
    if(materialModelName == "CrystalPlasticity"){
      Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag> statusTest =
	Teuchos::rcp_dynamic_cast<NOX::StatusTest::ModelEvaluatorFlag>(nox_status_test_);
      param_list.set< Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag> >("NOX Status Test", statusTest);
    }
  }

  // define cell topologies
  Teuchos::RCP<shards::CellTopology> comp_cellType =
      Teuchos::rcp(
          new shards::CellTopology(
              shards::getCellTopologyData<shards::Tetrahedron<11>>()));
  Teuchos::RCP<shards::CellTopology> cellType =
      Teuchos::rcp(new shards::CellTopology(&meshSpecs.ctd));

  // volume averaging flags
  bool volume_average_j(false);
  bool volume_average_pressure(false);
  RealType volume_average_stabilization_param(0.0);
  if (material_db_->isElementBlockParam(eb_name, "Weighted Volume Average J"))
    volume_average_j = material_db_->getElementBlockParam<bool>(
        eb_name,
        "Weighted Volume Average J");
  if (material_db_->isElementBlockParam(eb_name, "Volume Average Pressure"))
    volume_average_pressure = material_db_->getElementBlockParam<bool>(
        eb_name,
        "Volume Average Pressure");
  if (material_db_->isElementBlockParam(
      eb_name,
      "Average J Stabilization Parameter"))
    volume_average_stabilization_param = material_db_
        ->getElementBlockParam<RealType>(
        eb_name,
        "Average J Stabilization Parameter");

  // Check if we are setting the composite tet flag
  bool composite = false;
  if (material_db_->isElementBlockParam(eb_name, "Use Composite Tet 10"))
    composite =
        material_db_->getElementBlockParam<bool>(eb_name,
            "Use Composite Tet 10");
  pFromProb->set<bool>("Use Composite Tet 10", composite);

  // set flag for small strain option
  bool small_strain(false);
  if (material_model_name == "Linear Elastic") {
    small_strain = true;
  }

  if (material_db_->isElementBlockParam(eb_name, "Strain Flag")) {
    small_strain = true;
  }

//  if (material_db_->isElementBlockParam(eb_name, "Strain Flag")) {
//    small_strain = true;
//  }

  // Surface element checking
  bool surface_element = false;
  bool cohesive_element = false;
  bool compute_membrane_forces = false;
  RealType thickness = 0.0;
  if (material_db_->isElementBlockParam(eb_name, "Surface Element")) {
    surface_element =
        material_db_->getElementBlockParam<bool>(eb_name, "Surface Element");
    if (material_db_->isElementBlockParam(eb_name, "Cohesive Element"))
      cohesive_element =
          material_db_->getElementBlockParam<bool>(eb_name,
              "Cohesive Element");
  }

  if (surface_element) {
    if (material_db_->
        isElementBlockParam(eb_name, "Localization thickness parameter")) {
      thickness =
          material_db_->
              getElementBlockParam<RealType>(eb_name,
              "Localization thickness parameter");
    } else {
      thickness = 0.1;
    }
  }

  if (material_db_->isElementBlockParam(eb_name, "Compute Membrane Forces")) {
    compute_membrane_forces = material_db_->getElementBlockParam<bool>(eb_name,
        "Compute Membrane Forces");
  }

  std::string msg =
      "Surface elements are not yet supported with the composite tet";
  // FIXME, really need to check for WEDGE_12 topologies
  TEUCHOS_TEST_FOR_EXCEPTION(composite && surface_element,
      std::logic_error,
      msg);

  // get the intrepid basis for the given cell topology
  Intrepid2Basis
  intrepidBasis = Albany::getIntrepid2Basis(meshSpecs.ctd, composite);

  if (composite &&
      meshSpecs.ctd.dimension == 3 &&
      meshSpecs.ctd.node_count == 10) cellType = comp_cellType;

  Intrepid2::DefaultCubatureFactory<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > cubFactory;
  Teuchos::RCP<Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> >> cubature =
      cubFactory.create(*cellType, meshSpecs.cubatureDegree);

  // FIXME, this could probably go into the ProblemUtils
  // just like the call to getIntrepid2Basis
  Intrepid2Basis
  surfaceBasis;
  Teuchos::RCP<shards::CellTopology> surfaceTopology;
  Teuchos::RCP<Intrepid2::Cubature<RealType,Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> >> surfaceCubature;
  if (surface_element)
  {
#ifdef ALBANY_VERBOSE
    *out << "In Surface Element Logic" << std::endl;
#endif

    std::string name = meshSpecs.ctd.name;
    if (name == "Triangle_3" || name == "Quadrilateral_4") {
      surfaceBasis =
          Teuchos::rcp(
              new Intrepid2::Basis_HGRAD_LINE_C1_FEM<RealType,
                  Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device>>());
      surfaceTopology =
          Teuchos::rcp(
              new shards::CellTopology(
                  shards::getCellTopologyData<shards::Line<2>>()));
      surfaceCubature =
          cubFactory.create(*surfaceTopology, meshSpecs.cubatureDegree);
    }
    else if (name == "Wedge_6") {
      surfaceBasis =
          Teuchos::rcp(
              new Intrepid2::Basis_HGRAD_TRI_C1_FEM<RealType,
                  Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device>>());
      surfaceTopology =
          Teuchos::rcp(
              new shards::CellTopology(
                  shards::getCellTopologyData<shards::Triangle<3>>()));
      surfaceCubature =
          cubFactory.create(*surfaceTopology, meshSpecs.cubatureDegree);
    }
    else if (name == "Hexahedron_8") {
      surfaceBasis =
          Teuchos::rcp(
              new Intrepid2::Basis_HGRAD_QUAD_C1_FEM<RealType,
                  Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device>>());
      surfaceTopology =
          Teuchos::rcp(
              new shards::CellTopology(
                  shards::getCellTopologyData<shards::Quadrilateral<4>>()));
      surfaceCubature =
          cubFactory.create(*surfaceTopology, meshSpecs.cubatureDegree);
    }

#ifdef ALBANY_VERBOSE
    *out << "surfaceCubature->getNumPoints(): "
    << surfaceCubature->getNumPoints() << std::endl;
    *out << "surfaceCubature->getDimension(): "
    << surfaceCubature->getDimension() << std::endl;
#endif
  }

  // Note that these are the volume element quantities
  num_nodes_ = intrepidBasis->getCardinality();
  const int workset_size = meshSpecs.worksetSize;

#ifdef ALBANY_VERBOSE
  *out << "Setting num_pts_, surface element is "
  << surface_element << std::endl;
#endif
  num_dims_ = cubature->getDimension();
  if (!surface_element) {
    num_pts_ = cubature->getNumPoints();
  } else {
    num_pts_ = surfaceCubature->getNumPoints();
  }
  num_vertices_ = num_nodes_;

#ifdef ALBANY_VERBOSE
  *out << "Field Dimensions: Workset=" << workset_size
  << ", Vertices= " << num_vertices_
  << ", Nodes= " << num_nodes_
  << ", QuadPts= " << num_pts_
  << ", Dim= " << num_dims_ << std::endl;
#endif

  // Construct standard FEM evaluators with standard field names
  dl_ = Teuchos::rcp(new Albany::Layouts(workset_size,
      num_vertices_,
      num_nodes_,
      num_pts_,
      num_dims_));
  msg = "Data Layout Usage in Mechanics problems assume vecDim = num_dims_";
  TEUCHOS_TEST_FOR_EXCEPTION(
      dl_->vectorAndGradientLayoutsAreEquivalent == false,
      std::logic_error,
      msg);
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl_);
  bool supports_transient = true;
  int offset = 0;
  // Temporary variable used numerous times below
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits>> ev;

  // Define Field Names
  // generate the field name map to deal with outputing surface element info
  LCM::FieldNameMap field_name_map(surface_element);
  Teuchos::RCP<std::map<std::string, std::string>> fnm =
      field_name_map.getMap();
  std::string cauchy = (*fnm)["Cauchy_Stress"];
  std::string firstPK = (*fnm)["FirstPK"];
  std::string Fp = (*fnm)["Fp"];
  std::string eqps = (*fnm)["eqps"];
  std::string temperature = (*fnm)["Temperature"];
  std::string pressure = (*fnm)["Pressure"];
  std::string mech_source = (*fnm)["Mechanical_Source"];
  std::string defgrad = (*fnm)["F"];
  std::string J = (*fnm)["J"];
  // Poromechanics variables
  std::string totStress = (*fnm)["Total_Stress"];
  std::string kcPerm = (*fnm)["KCPermeability"];
  std::string biotModulus = (*fnm)["Biot_Modulus"];
  std::string biotCoeff = (*fnm)["Biot_Coefficient"];
  std::string porosity = (*fnm)["Porosity"];
  std::string porePressure = (*fnm)["Pore_Pressure"];
  // Hydrogen diffusion variable
  std::string transport = (*fnm)["Transport"];
  std::string hydroStress = (*fnm)["HydroStress"];
  std::string diffusionCoefficient = (*fnm)["Diffusion_Coefficient"];
  std::string convectionCoefficient = (*fnm)["Tau_Contribution"];
  std::string trappedConcentration = (*fnm)["Trapped_Concentration"];
  std::string totalConcentration = (*fnm)["Total_Concentration"];
  std::string effectiveDiffusivity = (*fnm)["Effective_Diffusivity"];
  std::string trappedSolvent = (*fnm)["Trapped_Solvent"];
  std::string strainRateFactor = (*fnm)["Strain_Rate_Factor"];
  std::string eqilibriumParameter =
      (*fnm)["Concentration_Equilibrium_Parameter"];
  std::string gradient_element_length = (*fnm)["Gradient_Element_Length"];
  // Helium bubble evolution
  std::string he_concentration = (*fnm)["He_Concentration"];
  std::string total_bubble_density = (*fnm)["Total_Bubble_Density"];
  std::string bubble_volume_fraction = (*fnm)["Bubble_Volume_Fraction"];
  

  if (have_mech_eq_) {
    Teuchos::ArrayRCP<std::string> dof_names(1);
    Teuchos::ArrayRCP<std::string> dof_names_dot(1);
    Teuchos::ArrayRCP<std::string> dof_names_dotdot(1);
    Teuchos::ArrayRCP<std::string> resid_names(1);
    dof_names[0] = "Displacement";
    dof_names_dot[0] = "Velocity";
    dof_names_dotdot[0] = "Acceleration";
    resid_names[0] = dof_names[0] + " Residual";

    if (supports_transient) {
      fm0.template registerEvaluator<EvalT>
      (evalUtils.constructGatherSolutionEvaluator_withAcceleration(
          true,
          dof_names,
          dof_names_dot,
          dof_names_dotdot));
    } else {
      fm0.template registerEvaluator<EvalT>
      (evalUtils.constructGatherSolutionEvaluator_noTransient(true,
          dof_names));
    }

    fm0.template registerEvaluator<EvalT>
    (evalUtils.constructGatherCoordinateVectorEvaluator());

    if (!surface_element) {
      fm0.template registerEvaluator<EvalT>
      (evalUtils.constructDOFVecInterpolationEvaluator(dof_names[0]));

      fm0.template registerEvaluator<EvalT>
      (evalUtils.constructDOFVecInterpolationEvaluator(dof_names_dot[0]));

      fm0.template registerEvaluator<EvalT>
      (evalUtils.constructDOFVecInterpolationEvaluator(dof_names_dotdot[0]));

      fm0.template registerEvaluator<EvalT>
      (evalUtils.constructDOFVecGradInterpolationEvaluator(dof_names[0]));

      fm0.template registerEvaluator<EvalT>
      (evalUtils.constructMapToPhysicalFrameEvaluator(cellType,
          cubature));

      fm0.template registerEvaluator<EvalT>
      (evalUtils.constructComputeBasisFunctionsEvaluator(cellType,
          intrepidBasis,
          cubature));
    }

    fm0.template registerEvaluator<EvalT>
    (evalUtils.constructScatterResidualEvaluator(true,
        resid_names));
    offset += num_dims_;
  }
  else if (have_mech_) { // constant configuration
    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
        new Teuchos::ParameterList);

    p->set<std::string>("Material Property Name", "Displacement");
    p->set<Teuchos::RCP<PHX::DataLayout>>("Data Layout", dl_->qp_vector);
    p->set<std::string>("Coordinate Vector Name", "Coord Vec");
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "Coordinate Vector Data Layout",
        dl_->qp_vector);

    p->set<Teuchos::RCP<ParamLib>>("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Displacement");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = Teuchos::rcp(
        new PHAL::NSMaterialProperty<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

  }

  // Output the Velocity and Acceleration
  // Register the states to store the output data in
  if (supports_transient) {

    // store computed xdot in "Velocity" field
    // This is just for testing as it duplicates writing the solution
    pFromProb->set<std::string>("x Field Name", "xField");

    // store computed xdot in "Velocity" field
    pFromProb->set<std::string>("xdot Field Name", "Velocity");

    // store computed xdotdot in "Acceleration" field
    pFromProb->set<std::string>("xdotdot Field Name", "Acceleration");

  }

  if (have_temperature_eq_) { // Gather Solution Temperature
    Teuchos::ArrayRCP<std::string> dof_names(1);
    Teuchos::ArrayRCP<std::string> resid_names(1);
    dof_names[0] = "Temperature";
    resid_names[0] = dof_names[0] + " Residual";
    fm0.template registerEvaluator<EvalT>
    (evalUtils.constructGatherSolutionEvaluator_noTransient(false,
        dof_names,
        offset));

    fm0.template registerEvaluator<EvalT>
    (evalUtils.constructGatherCoordinateVectorEvaluator());

    if (!surface_element) {
      fm0.template registerEvaluator<EvalT>
      (evalUtils.constructDOFInterpolationEvaluator(dof_names[0], offset));

      fm0.template registerEvaluator<EvalT>
      (evalUtils.constructDOFGradInterpolationEvaluator(dof_names[0], offset));

      fm0.template registerEvaluator<EvalT>
      (evalUtils.constructMapToPhysicalFrameEvaluator(cellType,
          cubature));

      fm0.template registerEvaluator<EvalT>
      (evalUtils.constructComputeBasisFunctionsEvaluator(cellType,
          intrepidBasis,
          cubature));
    }

    fm0.template registerEvaluator<EvalT>
    (evalUtils.constructScatterResidualEvaluator(false,
        resid_names,
        offset,
        "Scatter Temperature"));
    offset++;
  }
  else if ((!have_temperature_eq_ && have_temperature_)
      || have_transport_eq_ || have_transport_) {
    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
        new Teuchos::ParameterList);

    p->set<std::string>("Material Property Name", temperature);
    p->set<Teuchos::RCP<PHX::DataLayout>>("Data Layout", dl_->qp_scalar);
    p->set<std::string>("Coordinate Vector Name", "Coord Vec");
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "Coordinate Vector Data Layout",
        dl_->qp_vector);

    p->set<Teuchos::RCP<ParamLib>>("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Temperature");

    // This evaluator is called to set a constant temperature when "Variable Type"
    // is set to "Constant." It is also called when "Variable Type" is set to
    // "Time Dependent." There are two "Type" variables in the PL - "Type" and
    // "Variable Type". For the last case, lets set "Type" to "Time Dependent" to hopefully
    // make the evaluator call a little more general (GAH)

    std::string temp_type = paramList.get<std::string>("Variable Type", "None");
    if ( temp_type == "Time Dependent"){

      paramList.set<std::string>("Type", temp_type);

    }

    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = Teuchos::rcp(
        new PHAL::NSMaterialProperty<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_stab_pressure_eq_) { // Gather Stabilized Pressure
    Teuchos::ArrayRCP<std::string> dof_names(1);
    Teuchos::ArrayRCP<std::string> resid_names(1);
    dof_names[0] = "Pressure";
    resid_names[0] = dof_names[0] + " Residual";
    fm0.template registerEvaluator<EvalT>
    (evalUtils.constructGatherSolutionEvaluator_noTransient(false,
        dof_names,
        offset));

    fm0.template registerEvaluator<EvalT>
    (evalUtils.constructGatherCoordinateVectorEvaluator());

    if (!surface_element) {
      fm0.template registerEvaluator<EvalT>
      (evalUtils.constructDOFInterpolationEvaluator(dof_names[0], offset));

      fm0.template registerEvaluator<EvalT>
      (evalUtils.constructDOFGradInterpolationEvaluator(dof_names[0], offset));

      fm0.template registerEvaluator<EvalT>
      (evalUtils.constructMapToPhysicalFrameEvaluator(cellType,
          cubature));

      fm0.template registerEvaluator<EvalT>
      (evalUtils.constructComputeBasisFunctionsEvaluator(cellType,
          intrepidBasis,
          cubature));
    }

    fm0.template registerEvaluator<EvalT>
    (evalUtils.constructScatterResidualEvaluator(false,
        resid_names,
        offset,
        "Scatter Pressure"));
    offset++;
  }

  if (have_damage_eq_) { // Damage
    Teuchos::ArrayRCP<std::string> dof_names(1);
    Teuchos::ArrayRCP<std::string> resid_names(1);
    dof_names[0] = "Damage";
    resid_names[0] = dof_names[0] + " Residual";
    fm0.template registerEvaluator<EvalT>
    (evalUtils.constructGatherSolutionEvaluator_noTransient(false,
        dof_names,
        offset));

    fm0.template registerEvaluator<EvalT>
    (evalUtils.constructGatherCoordinateVectorEvaluator());

    if (!surface_element) {
      fm0.template registerEvaluator<EvalT>
      (evalUtils.constructDOFInterpolationEvaluator(dof_names[0], offset));

      fm0.template registerEvaluator<EvalT>
      (evalUtils.constructDOFGradInterpolationEvaluator(dof_names[0], offset));

      fm0.template registerEvaluator<EvalT>
      (evalUtils.constructMapToPhysicalFrameEvaluator(cellType,
          cubature));

      fm0.template registerEvaluator<EvalT>
      (evalUtils.constructComputeBasisFunctionsEvaluator(cellType,
          intrepidBasis,
          cubature));
    }

    fm0.template registerEvaluator<EvalT>
    (evalUtils.constructScatterResidualEvaluator(false,
        resid_names,
        offset,
        "Scatter Damage"));
    offset++;
  }
  else if (!have_damage_eq_ && have_damage_) {
    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
        new Teuchos::ParameterList);

    p->set<std::string>("Material Property Name", "Damage");
    p->set<Teuchos::RCP<PHX::DataLayout>>("Data Layout", dl_->qp_scalar);
    p->set<std::string>("Coordinate Vector Name", "Coord Vec");
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "Coordinate Vector Data Layout",
        dl_->qp_vector);

    p->set<Teuchos::RCP<ParamLib>>("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Damage");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = Teuchos::rcp(
        new PHAL::NSMaterialProperty<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_pore_pressure_eq_) {
    Teuchos::ArrayRCP<std::string> dof_names(1);
    Teuchos::ArrayRCP<std::string> resid_names(1);
    dof_names[0] = "Pore_Pressure";
    resid_names[0] = dof_names[0] + " Residual";

    fm0.template registerEvaluator<EvalT>
    (evalUtils.constructGatherSolutionEvaluator_noTransient(false,
        dof_names,
        offset));
    fm0.template registerEvaluator<EvalT>
    (evalUtils.constructGatherCoordinateVectorEvaluator());

    if (!surface_element) {
      fm0.template registerEvaluator<EvalT>
      (evalUtils.constructDOFInterpolationEvaluator(dof_names[0], offset));

      fm0.template registerEvaluator<EvalT>
      (evalUtils.constructDOFGradInterpolationEvaluator(dof_names[0], offset));

      fm0.template registerEvaluator<EvalT>
      (evalUtils.constructMapToPhysicalFrameEvaluator(cellType,
          cubature));

      fm0.template registerEvaluator<EvalT>
      (evalUtils.constructComputeBasisFunctionsEvaluator(cellType,
          intrepidBasis,
          cubature));
    }

    fm0.template registerEvaluator<EvalT>
    (evalUtils.constructScatterResidualEvaluator(false,
        resid_names,
        offset,
        "Scatter Pore_Pressure"));
    offset++;
  }
  else if (have_pore_pressure_) { // constant Pressure
    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
        new Teuchos::ParameterList);

    p->set<std::string>("Material Property Name", "Pressure");
    p->set<Teuchos::RCP<PHX::DataLayout>>("Data Layout", dl_->qp_scalar);
    p->set<std::string>("Coordinate Vector Name", "Coord Vec");
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "Coordinate Vector Data Layout",
        dl_->qp_vector);

    p->set<Teuchos::RCP<ParamLib>>("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Pressure");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = Teuchos::rcp(
        new PHAL::NSMaterialProperty<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_transport_eq_) { // Gather solution for transport problem
    // Lattice Concentration
    Teuchos::ArrayRCP<std::string> dof_names(1);
    Teuchos::ArrayRCP<std::string> resid_names(1);
    dof_names[0] = "Transport";
    resid_names[0] = dof_names[0] + " Residual";
    fm0.template registerEvaluator<EvalT>
    (evalUtils.constructGatherSolutionEvaluator_noTransient(false,
        dof_names,
        offset));
    if (!surface_element) {
      fm0.template registerEvaluator<EvalT>
      (evalUtils.constructDOFInterpolationEvaluator(dof_names[0], offset));

      fm0.template registerEvaluator<EvalT>
      (evalUtils.constructDOFGradInterpolationEvaluator(dof_names[0], offset));

      fm0.template registerEvaluator<EvalT>
      (evalUtils.constructMapToPhysicalFrameEvaluator(cellType,
          cubature));

      fm0.template registerEvaluator<EvalT>
      (evalUtils.constructComputeBasisFunctionsEvaluator(cellType,
          intrepidBasis,
          cubature));
    }

    fm0.template registerEvaluator<EvalT>
    (evalUtils.constructScatterResidualEvaluator(false,
        resid_names,
        offset,
        "Scatter Transport"));
    offset++; // for lattice concentration
  }
  else if (have_transport_) { // Constant transport scalar value
    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
        new Teuchos::ParameterList);

    p->set<std::string>("Material Property Name", "Transport");
    p->set<Teuchos::RCP<PHX::DataLayout>>("Data Layout", dl_->qp_scalar);
    p->set<std::string>("Coordinate Vector Name", "Coord Vec");
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "Coordinate Vector Data Layout",
        dl_->qp_vector);

    p->set<Teuchos::RCP<ParamLib>>("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Transport");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = Teuchos::rcp(
        new PHAL::NSMaterialProperty<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_hydrostress_eq_) { // Gather solution for transport problem
    Teuchos::ArrayRCP<std::string> dof_names(1);
    Teuchos::ArrayRCP<std::string> resid_names(1);
    dof_names[0] = "HydroStress";
    resid_names[0] = dof_names[0] + " Residual";
    fm0.template registerEvaluator<EvalT>
    (evalUtils.constructGatherSolutionEvaluator_noTransient(false,
        dof_names,
        offset));

    if (!surface_element) {
      fm0.template registerEvaluator<EvalT>
      (evalUtils.constructDOFInterpolationEvaluator(dof_names[0], offset));

      fm0.template registerEvaluator<EvalT>
      (evalUtils.constructDOFGradInterpolationEvaluator(dof_names[0], offset));

      fm0.template registerEvaluator<EvalT>
      (evalUtils.constructMapToPhysicalFrameEvaluator(cellType,
          cubature));

      fm0.template registerEvaluator<EvalT>
      (evalUtils.constructComputeBasisFunctionsEvaluator(cellType,
          intrepidBasis,
          cubature));
    }

    fm0.template registerEvaluator<EvalT>
    (evalUtils.constructScatterResidualEvaluator(false,
        resid_names,
        offset,
        "Scatter HydroStress"));
    offset++; // for hydrostatic stress
  }

  { // Time
    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
        new Teuchos::ParameterList("Time"));
    p->set<std::string>("Time Name", "Time");
    p->set<std::string>("Delta Time Name", "Delta Time");
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "Workset Scalar Data Layout",
        dl_->workset_scalar);
    p->set<Teuchos::RCP<ParamLib>>("Parameter Library", paramLib);
    p->set<bool>("Disable Transient", true);
    ev = Teuchos::rcp(new LCM::Time<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
    p = stateMgr.registerStateVariable("Time",
        dl_->workset_scalar,
        dl_->dummy,
        eb_name,
        "scalar",
        0.0,
        true);
    ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }
  // IKT, 3/27/16: register dirichlet_field for specifying Dirichlet data from a field 
  // in the input exodus mesh.
  if (dir_count == 0){ //constructEvaluators gets called multiple times for different specializations.  
                       //Make sure dirichlet_field gets registered only once via counter.
                       //I don't quite understand why this is needed for LCM but not for FELIX... 
    //dirichlet_field
    Albany::StateStruct::MeshFieldEntity entity = Albany::StateStruct::NodalDistParameter;
    stateMgr.registerStateVariable("dirichlet_field", dl_->node_vector, eb_name, true, &entity, "");
    dir_count++; 
  }

  if (have_mech_eq_) { // Current Coordinates
    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
        new Teuchos::ParameterList("Current Coordinates"));
    p->set<std::string>("Reference Coordinates Name", "Coord Vec");
    p->set<std::string>("Displacement Name", "Displacement");
    p->set<std::string>("Current Coordinates Name", "Current Coordinates");
    ev = Teuchos::rcp(
        new LCM::CurrentCoords<EvalT, PHAL::AlbanyTraits>(*p, dl_));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_mech_eq_ && have_sizefield_adaptation_) { // Mesh size field
    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
        new Teuchos::ParameterList("Isotropic Mesh Size Field"));
    p->set<std::string>("IsoTropic MeshSizeField Name", "IsoMeshSizeField");
    p->set<std::string>("Current Coordinates Name", "Current Coordinates");
    p->set<Teuchos::RCP<Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> >>>("Cubature", cubature);

    // Get the Adaptation list and send to the evaluator
    Teuchos::ParameterList& paramList = params->sublist("Adaptation");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    p->set<const Teuchos::RCP<
      Intrepid2::Basis<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device>>>>("Intrepid2 Basis", intrepidBasis);
    ev = Teuchos::rcp(
        new LCM::IsoMeshSizeField<EvalT, PHAL::AlbanyTraits>(*p, dl_));
    fm0.template registerEvaluator<EvalT>(ev);

    // output mesh size field if requested
/*
    bool output_flag = false;
    if (material_db_->isElementBlockParam(eb_name, "Output MeshSizeField"))
      output_flag =
          material_db_->getElementBlockParam<bool>(eb_name, "Output MeshSizeField");
*/
    bool output_flag = true;
    if (output_flag) {
        p = stateMgr.registerStateVariable("IsoMeshSizeField",
            dl_->qp_scalar,
            dl_->dummy,
            eb_name,
            "scalar",
            1.0,
            true,
            output_flag);
        ev = Teuchos::rcp(
            new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
        fm0.template registerEvaluator<EvalT>(ev);
    }
  }

  if (have_temperature_eq_ || have_temperature_) {
    double temp(0.0);
    if (material_db_->isElementBlockParam(eb_name, "Initial Temperature")) {
      temp = material_db_->
          getElementBlockParam<double>(eb_name, "Initial Temperature");
    }
    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
        new Teuchos::ParameterList("Save Temperature"));
    p = stateMgr.registerStateVariable(temperature,
        dl_->qp_scalar,
        dl_->dummy,
        eb_name,
        "scalar",
        temp,
        true,
        false);
    ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_pore_pressure_eq_ || have_pore_pressure_) {
    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
        new Teuchos::ParameterList("Save Pore Pressure"));
    p = stateMgr.registerStateVariable(porePressure,
        dl_->qp_scalar,
        dl_->dummy,
        eb_name,
        "scalar",
        0.0,
        true,
        false);
    ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_transport_eq_ || have_transport_) {
    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
        new Teuchos::ParameterList("Save Transport"));
    bool output_flag(true);
    if (material_db_->isElementBlockParam(eb_name, "Output IP" + transport))
      output_flag =
          material_db_->getElementBlockParam<bool>(
              eb_name,
              "Output IP" + transport);

    RealType ic(0.0);
    if (material_db_->isElementBlockParam(eb_name, "Initial Concentration")) {
      ic = material_db_->
        getElementBlockParam<double>(eb_name, "Initial Concentration");
    }

    p = stateMgr.registerStateVariable(transport,
        dl_->qp_scalar,
        dl_->dummy,
        eb_name,
        "scalar",
        ic,
        true,
        output_flag);
    ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_hydrostress_eq_ || have_hydrostress_) {
    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
        new Teuchos::ParameterList("Save HydroStress"));
    p = stateMgr.registerStateVariable(hydroStress,
        dl_->qp_scalar,
        dl_->dummy,
        eb_name,
        "scalar",
        0.0,
        true,
        true);
    ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // Source list exists and the mechanical source params are defined

  if (have_source_ &&
      params->sublist("Source Functions").isSublist("Mechanical Source")) {

    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
        new Teuchos::ParameterList);

    p->set<std::string>("Source Name", "Source");
    p->set<std::string>("Variable Name", "Displacement");
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "QP Scalar Data Layout",
        dl_->qp_scalar);

    p->set<Teuchos::RCP<ParamLib>>("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList =
        params->sublist("Source Functions").sublist("Mechanical Source");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = Teuchos::rcp(new PHAL::Source<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // Heat Source in Heat Equation

  if (thermal_source_ != SOURCE_TYPE_NONE) {

    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
        new Teuchos::ParameterList);

    p->set<std::string>("Source Name", "Heat Source");
    p->set<std::string>("Variable Name", "Temperature");
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "QP Scalar Data Layout",
        dl_->qp_scalar);

    p->set<Teuchos::RCP<ParamLib>>("Parameter Library", paramLib);

    if (thermal_source_ == SOURCE_TYPE_INPUT) { // Thermal source in input file

      Teuchos::ParameterList& paramList = params->sublist("Source Functions")
          .sublist("Thermal Source");
      p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

      ev = Teuchos::rcp(new PHAL::Source<EvalT, PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);

      thermal_source_evaluated_ = true;

    } else if (thermal_source_ == SOURCE_TYPE_MATERIAL) {

      // There may not be a source in every element block

      if (material_db_->isElementBlockSublist(eb_name, "Source Functions")) { // Thermal source in matDB
          
        Teuchos::ParameterList& srcParamList = material_db_->
            getElementBlockSublist(eb_name, "Source Functions");

        if (srcParamList.isSublist("Thermal Source")) {

          Teuchos::ParameterList& paramList = srcParamList.sublist(
              "Thermal Source");
          p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

          ev = Teuchos::rcp(new PHAL::Source<EvalT, PHAL::AlbanyTraits>(*p));
          fm0.template registerEvaluator<EvalT>(ev);

          thermal_source_evaluated_ = true;
        }
      }
      else // Do not evaluate heat source in TransportResidual
      {
          thermal_source_evaluated_ = false;
      }
    }
    else

      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
          "Unrecognized thermal source specified in input file");

  }

  { // Constitutive Model Parameters
    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
        new Teuchos::ParameterList("Constitutive Model Parameters"));
    std::string matName = material_db_->getElementBlockParam<std::string>(
        eb_name, "material");
    Teuchos::ParameterList& param_list =
        material_db_->getElementBlockSublist(eb_name, matName);
    if (have_temperature_ || have_temperature_eq_) {
      p->set<std::string>("Temperature Name", temperature);
      param_list.set<bool>("Have Temperature", true);
    }

    // optional spatial dependence
    p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");

    // pass through material properties
    p->set<Teuchos::ParameterList*>("Material Parameters", &param_list);

    Teuchos::RCP<LCM::ConstitutiveModelParameters<EvalT, PHAL::AlbanyTraits>>
    cmpEv =
        Teuchos::rcp(
            new LCM::ConstitutiveModelParameters<EvalT, PHAL::AlbanyTraits>(
                *p,
                dl_));
    fm0.template registerEvaluator<EvalT>(cmpEv);
  }

  if (have_mech_eq_) {
    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
        new Teuchos::ParameterList("Constitutive Model Interface"));
    std::string matName = material_db_->getElementBlockParam<std::string>(
        eb_name, "material");
    Teuchos::ParameterList& param_list =
        material_db_->getElementBlockSublist(eb_name, matName);

    // FIXME: figure out how to do this better
    param_list.set<bool>("Have Temperature", false);
    if (have_temperature_ || have_temperature_eq_) {
      p->set<std::string>("Temperature Name", temperature);
      param_list.set<bool>("Have Temperature", true);
    }

    param_list.set<bool>("Have Total Concentration", false);
    if (have_transport_ || have_transport_eq_) {
      p->set<std::string>("Total Concentration Name", totalConcentration);
      param_list.set<bool>("Have Total Concentration", true);
    }

    param_list.set<bool>("Have Bubble Volume Fraction", false);
    param_list.set<bool>("Have Total Bubble Density", false);
    if (param_list.isSublist("Tritium Coefficients")) {
      p->set<std::string>("Bubble Volume Fraction Name", bubble_volume_fraction);
      p->set<std::string>("Total Bubble Density Name", total_bubble_density);
      param_list.set<bool>("Have Bubble Volume Fraction", true);
      param_list.set<bool>("Have Total Bubble Density", true);
      param_list.set<RealType>("Helium Radius",
          param_list.sublist("Tritium Coefficients").get<RealType>("Helium Radius", 0.0));
    }

    param_list.set<Teuchos::RCP<std::map<std::string, std::string>>>(
        "Name Map",
        fnm);
    p->set<Teuchos::ParameterList*>("Material Parameters", &param_list);
    p->set<bool>("Volume Average Pressure", volume_average_pressure);
    if (volume_average_pressure) {
      p->set<std::string>("Weights Name", "Weights");
      p->set<std::string>("J Name", J);
    }

    Teuchos::RCP<LCM::ConstitutiveModelInterface<EvalT, PHAL::AlbanyTraits>>
    cmiEv =
        Teuchos::rcp(
            new LCM::ConstitutiveModelInterface<EvalT, PHAL::AlbanyTraits>(
                *p,
                dl_));
    fm0.template registerEvaluator<EvalT>(cmiEv);

    // register state variables
    for (int sv(0); sv < cmiEv->getNumStateVars(); ++sv) {
      cmiEv->fillStateVariableStruct(sv);
      p = stateMgr.registerStateVariable(cmiEv->getName(),
          cmiEv->getLayout(),
          dl_->dummy,
          eb_name,
          cmiEv->getInitType(),
          cmiEv->getInitValue(),
          cmiEv->getStateFlag(),
          cmiEv->getOutputFlag());
      ev = Teuchos::rcp(
          new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }
  }

  // Surface Element Block
  if (surface_element) {

    { // Surface Basis
      // SurfaceBasis_Def.hpp
      Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
          new Teuchos::ParameterList("Surface Basis"));

      // inputs
      p->set<std::string>("Reference Coordinates Name", "Coord Vec");
      p->set<Teuchos::RCP<Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> >>>(
          "Cubature",
          surfaceCubature);
      p->set<Intrepid2Basis>("Intrepid2 Basis", surfaceBasis);
      if (have_mech_eq_) {
        p->set<std::string>("Current Coordinates Name", "Current Coordinates");
      }

      // outputs
      p->set<std::string>("Reference Basis Name", "Reference Basis");
      p->set<std::string>("Reference Area Name", "Weights");
      p->set<std::string>("Reference Dual Basis Name", "Reference Dual Basis");
      p->set<std::string>("Reference Normal Name", "Reference Normal");
      p->set<std::string>("Current Basis Name", "Current Basis");

      ev = Teuchos::rcp(
          new LCM::SurfaceBasis<EvalT, PHAL::AlbanyTraits>(*p, dl_));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    if (have_mech_eq_) { // Surface Jump
      //SurfaceVectorJump_Def.hpp
      Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
          new Teuchos::ParameterList("Surface Vector Jump"));

      // inputs
      p->set<Teuchos::RCP<Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> >>>(
          "Cubature",
          surfaceCubature);
      p->set<Intrepid2Basis>("Intrepid2 Basis", surfaceBasis);
      p->set<std::string>("Vector Name", "Current Coordinates");

      // outputs
      p->set<std::string>("Vector Jump Name", "Vector Jump");

      ev = Teuchos::rcp(
          new LCM::SurfaceVectorJump<EvalT, PHAL::AlbanyTraits>(*p, dl_));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    if ((have_temperature_eq_ || have_pore_pressure_eq_) ||
        (have_transport_eq_)) { // Surface Temperature Jump
      //SurfaceScalarJump_Def.hpp
      Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
          new Teuchos::ParameterList("Surface Scalar Jump"));

      // inputs
      p->set<Teuchos::RCP<Intrepid2::Cubature<RealType,Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> >>>(
          "Cubature",
          surfaceCubature);
      p->set<Intrepid2Basis>("Intrepid2 Basis", surfaceBasis);
      if (have_temperature_eq_) {
        p->set<std::string>("Nodal Temperature Name", "Temperature");
        // outputs
        p->set<std::string>("Jump of Temperature Name", "Temperature Jump");
        p->set<std::string>("MidPlane Temperature Name", temperature);
      }

      if (have_transport_eq_) {
        p->set<std::string>("Nodal Transport Name", "Transport");
        // outputs
        p->set<std::string>("Jump of Transport Name", "Transport Jump");
        p->set<std::string>("MidPlane Transport Name", transport);
      }

      if (have_pore_pressure_eq_) {
        p->set<std::string>("Nodal Pore Pressure Name", "Pore_Pressure");
        // outputs
        p->set<std::string>("Jump of Pore Pressure Name", "Pore_Pressure Jump");
        p->set<std::string>("MidPlane Pore Pressure Name", porePressure);
      }

      if (have_hydrostress_eq_) {
        p->set<std::string>("Nodal HydroStress Name", "HydroStress");
        // outputs
        p->set<std::string>("Jump of HydroStress Name", "HydroStress Jump");
        p->set<std::string>("MidPlane HydroStress Name", hydroStress);
      }

      ev = Teuchos::rcp(
          new LCM::SurfaceScalarJump<EvalT, PHAL::AlbanyTraits>(*p, dl_));
      fm0.template registerEvaluator<EvalT>(ev);

    }

    if (have_mech_eq_) { // Surface Gradient
      //SurfaceVectorGradient_Def.hpp
      Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
          new Teuchos::ParameterList("Surface Vector Gradient"));

      // inputs
      p->set<RealType>("thickness", thickness);
      // bool WeightedVolumeAverageJ(false);
      // if (material_db_->isElementBlockParam(eb_name,
      //     "Weighted Volume Average J"))
      //   p->set<bool>("Weighted Volume Average J Name",
      //       material_db_->getElementBlockParam<bool>(eb_name,
      //           "Weighted Volume Average J"));
      // if (material_db_->isElementBlockParam(eb_name,
      //     "Average J Stabilization Parameter"))
      //   p->set<RealType>("Averaged J Stabilization Parameter Name",
      //       material_db_->getElementBlockParam<RealType>(eb_name,
      //           "Average J Stabilization Parameter"));
      p->set<bool>("Weighted Volume Average J", volume_average_j);
      p->set<RealType>(
          "Average J Stabilization Parameter",
          volume_average_stabilization_param);
      p->set<Teuchos::RCP<Intrepid2::Cubature<RealType,Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> >>>(
          "Cubature",
          surfaceCubature);
      p->set<std::string>("Weights Name", "Weights");
      p->set<std::string>("Current Basis Name", "Current Basis");
      p->set<std::string>("Reference Dual Basis Name", "Reference Dual Basis");
      p->set<std::string>("Reference Normal Name", "Reference Normal");
      p->set<std::string>("Vector Jump Name", "Vector Jump");

      // outputs
      p->set<std::string>("Surface Vector Gradient Name", defgrad);
      p->set<std::string>("Surface Vector Gradient Determinant Name", J);

      ev = Teuchos::rcp(
          new LCM::SurfaceVectorGradient<EvalT, PHAL::AlbanyTraits>(*p, dl_));
      fm0.template registerEvaluator<EvalT>(ev);

      // optional output of the deformation gradient
      bool output_flag(false);
      if (material_db_->isElementBlockParam(eb_name,
        "Output Deformation Gradient"))
        output_flag = material_db_->getElementBlockParam<bool>(eb_name,
                "Output Deformation Gradient");

      p = stateMgr.registerStateVariable(defgrad,
          dl_->qp_tensor,
          dl_->dummy,
          eb_name,
          "identity",
          1.0,
          false,
          output_flag);
      ev = Teuchos::rcp(
          new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);


      // need J and J_old to perform time integration for poromechanics problem
      output_flag = false;
      if (material_db_->isElementBlockParam(eb_name, "Output J"))
        output_flag =
            material_db_->getElementBlockParam<bool>(eb_name, "Output J");
      if (have_pore_pressure_eq_ || output_flag) {
        p = stateMgr.registerStateVariable(J,
            dl_->qp_scalar,
            dl_->dummy,
            eb_name,
            "scalar",
            1.0,
            true,
            output_flag);
        ev = Teuchos::rcp(
            new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
        fm0.template registerEvaluator<EvalT>(ev);
      }
    }

    // Surface Gradient Operator
    if (have_pore_pressure_eq_) {
      //SurfaceScalarGradientOperator_Def.hpp
      Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
          new Teuchos::ParameterList("Surface Scalar Gradient Operator"));
      // inputs
      p->set<RealType>("thickness", thickness);
      p->set<Teuchos::RCP<Intrepid2::Cubature<RealType,Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> >>>(
          "Cubature",
          surfaceCubature);
      p->set<Intrepid2Basis>("Intrepid2 Basis", surfaceBasis);
      p->set<std::string>("Reference Dual Basis Name", "Reference Dual Basis");
      p->set<std::string>("Reference Normal Name", "Reference Normal");

      // NOTE: NOT surf_Pore_Pressure here
      // NOTE: If you need to compute gradient for more than one scalar field,
      // that could cause trouble
      if (have_pore_pressure_eq_ == true)
        p->set<std::string>("Nodal Scalar Name", "Pore_Pressure");

      // outputs
      p->set<std::string>("Surface Scalar Gradient Operator Name",
          "Surface Scalar Gradient Operator");
      p->set<Teuchos::RCP<PHX::DataLayout>>("Node QP Vector Data Layout",
          dl_->node_qp_vector);
      p->set<std::string>("Surface Scalar Gradient Name",
          "Surface Pressure Gradient");
      p->set<Teuchos::RCP<PHX::DataLayout>>(
          "QP Vector Data Layout",
          dl_->qp_vector);

      ev = Teuchos::rcp(
          new LCM::SurfaceScalarGradientOperator<EvalT, PHAL::AlbanyTraits>(
              *p,
              dl_));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    if (have_transport_eq_) {
      //SurfaceScalarGradientOperator_Def.hpp
      Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
          new Teuchos::ParameterList("Surface Scalar Gradient Operator"));
      // inputs
      p->set<RealType>("thickness", thickness);
      p->set<Teuchos::RCP<Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> >>>(
          "Cubature",
          surfaceCubature);
      p->set<Intrepid2Basis>("Intrepid2 Basis", surfaceBasis);
      p->set<std::string>("Reference Dual Basis Name", "Reference Dual Basis");
      p->set<std::string>("Reference Normal Name", "Reference Normal");

      // NOTE: NOT surf_Pore_Pressure here
      // NOTE: If you need to compute gradient for more than one scalar field,
      // that could cause trouble
      p->set<std::string>("Nodal Scalar Name", "Transport");

      // outputs
      p->set<std::string>("Surface Scalar Gradient Operator Name",
          "Surface Scalar Gradient Operator");
      p->set<Teuchos::RCP<PHX::DataLayout>>("Node QP Vector Data Layout",
          dl_->node_qp_vector);
      if (have_transport_eq_ == true)
        p->set<std::string>("Surface Scalar Gradient Name",
            "Surface Transport Gradient");
      p->set<Teuchos::RCP<PHX::DataLayout>>(
          "QP Vector Data Layout",
          dl_->qp_vector);

      ev = Teuchos::rcp(
          new LCM::SurfaceScalarGradientOperator<EvalT, PHAL::AlbanyTraits>(
              *p,
              dl_));
      fm0.template registerEvaluator<EvalT>(ev);

    }

    if (have_hydrostress_eq_) {
      //SurfaceScalarGradientOperator_Def.hpp
      Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
          new Teuchos::ParameterList("Surface Scalar Gradient Operator"));
      // inputs
      p->set<RealType>("thickness", thickness);
      p->set<Teuchos::RCP<Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> >>>(
          "Cubature",
          surfaceCubature);
      p->set<Intrepid2Basis>("Intrepid2 Basis", surfaceBasis);
      p->set<std::string>("Reference Dual Basis Name", "Reference Dual Basis");
      p->set<std::string>("Reference Normal Name", "Reference Normal");

      // NOTE: NOT surf_Pore_Pressure here
      // NOTE: If you need to compute gradient for more than one scalar field,
      // that could cause trouble
      if (have_transport_eq_ == true)
        p->set<std::string>("Nodal Scalar Name", "HydroStress");

      // outputs
      p->set<std::string>("Surface Scalar Gradient Operator Name",
          "Surface Scalar Gradient Operator");
      p->set<Teuchos::RCP<PHX::DataLayout>>("Node QP Vector Data Layout",
          dl_->node_qp_vector);
      p->set<std::string>("Surface Scalar Gradient Name",
          "Surface HydroStress Gradient");
      p->set<Teuchos::RCP<PHX::DataLayout>>(
          "QP Vector Data Layout",
          dl_->qp_vector);

      ev = Teuchos::rcp(
          new LCM::SurfaceScalarGradientOperator<EvalT, PHAL::AlbanyTraits>(
              *p,
              dl_));
      fm0.template registerEvaluator<EvalT>(ev);
    }
    {
      if (have_mech_eq_) { // Surface Residual
        // SurfaceVectorResidual_Def.hpp
        Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
            new Teuchos::ParameterList("Surface Vector Residual"));

        // inputs
        p->set<RealType>("thickness", thickness);
        p->set<Teuchos::RCP<Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> >>>("Cubature",
            surfaceCubature);
        p->set<Intrepid2Basis>("Intrepid2 Basis", surfaceBasis);

        p->set<bool>("Compute Membrane Forces", compute_membrane_forces);

        p->set<std::string>("Stress Name", firstPK);
        p->set<std::string>("Current Basis Name", "Current Basis");
        p->set<std::string>("Reference Dual Basis Name",
            "Reference Dual Basis");
        p->set<std::string>("Reference Normal Name", "Reference Normal");
        p->set<std::string>("Reference Area Name", "Weights");

        if (cohesive_element) {
          p->set<bool>("Use Cohesive Traction", true);
          p->set<std::string>("Cohesive Traction Name", "Cohesive_Traction");
        }

        // outputs
        p->set<std::string>("Surface Vector Residual Name",
            "Displacement Residual");

        if (have_topmod_adaptation_ == true) {
          // Input
          p->set<std::string>("Jacobian Name", J);
          p->set<bool>("Use Adaptive Insertion", true);
          // Output
          p->set<std::string>("Cauchy Stress Name", cauchy);
        }

        ev = Teuchos::rcp(
            new LCM::SurfaceVectorResidual<EvalT, PHAL::AlbanyTraits>(*p, dl_));
        fm0.template registerEvaluator<EvalT>(ev);
      }
    } // end of coehesive/surface element block
  } else {

    if (have_mech_eq_) { // Kinematics quantities
      Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
          new Teuchos::ParameterList("Kinematics"));

      p->set<bool>("Weighted Volume Average J", volume_average_j);
      p->set<RealType>(
          "Average J Stabilization Parameter",
          volume_average_stabilization_param);

      // strain
      if (small_strain)
        p->set<std::string>("Strain Name", "Strain");

      // set flag for return strain and velocity gradient
      bool have_velocity_gradient(false);
      if (material_db_->isElementBlockParam(eb_name,
          "Velocity Gradient Flag")) {
        p->set<bool>("Velocity Gradient Flag",
            material_db_->
                getElementBlockParam<bool>(eb_name, "Velocity Gradient Flag"));
        have_velocity_gradient = material_db_->
            getElementBlockParam<bool>(eb_name, "Velocity Gradient Flag");
        if (have_velocity_gradient)
          p->set<std::string>("Velocity Gradient Name", "Velocity Gradient");
      }

      // send in integration weights and the displacement gradient
      p->set<std::string>("Weights Name", "Weights");
      p->set<Teuchos::RCP<PHX::DataLayout>>(
          "QP Scalar Data Layout",
          dl_->qp_scalar);
      p->set<std::string>("Gradient QP Variable Name", "Displacement Gradient");
      p->set<Teuchos::RCP<PHX::DataLayout>>(
          "QP Tensor Data Layout",
          dl_->qp_tensor);

      //Outputs: F, J
      p->set<std::string>("DefGrad Name", defgrad); //dl_->qp_tensor also
      p->set<std::string>("DetDefGrad Name", J);
      p->set<Teuchos::RCP<PHX::DataLayout>>(
          "QP Scalar Data Layout",
          dl_->qp_scalar);

      if (Teuchos::nonnull(rc_mgr_)) {
        rc_mgr_->registerField(
            defgrad, dl_->qp_tensor, AAdapt::rc::Init::identity,
            AAdapt::rc::Transformation::right_polar_LieR_LieS, p);
        p->set<std::string>("Displacement Name", "Displacement");
      }

      //ev = Teuchos::rcp(new LCM::DefGrad<EvalT,PHAL::AlbanyTraits>(*p));
      ev = Teuchos::rcp(
          new LCM::Kinematics<EvalT, PHAL::AlbanyTraits>(*p, dl_));
      fm0.template registerEvaluator<EvalT>(ev);

      // optional output
      bool output_flag(false);
      if (material_db_->isElementBlockParam(eb_name,
          "Output Deformation Gradient"))
        output_flag =
            material_db_->getElementBlockParam<bool>(eb_name,
                "Output Deformation Gradient");

      if (output_flag) {
        p = stateMgr.registerStateVariable(defgrad,
            dl_->qp_tensor,
            dl_->dummy,
            eb_name,
            "identity",
            1.0,
            false,
            output_flag);
        ev = Teuchos::rcp(
            new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
        fm0.template registerEvaluator<EvalT>(ev);
      }

            // optional output of the integration weights
      output_flag = false;
      if (material_db_->isElementBlockParam(eb_name,
        "Output Integration Weights"))
        output_flag = material_db_->getElementBlockParam<bool>(eb_name,
                "Output Integration Weights");

      if (output_flag) {
        p = stateMgr.registerStateVariable("Weights",
                                           dl_->qp_scalar,
                                           dl_->dummy,
                                           eb_name,
                                           "scalar",
                                           0.0,
                                           false,
                                           output_flag);
        ev = Teuchos::rcp(
                          new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
        fm0.template registerEvaluator<EvalT>(ev);
      }

      // need J and J_old to perform time integration for poromechanics problem
      output_flag = false;
      if (material_db_->isElementBlockParam(eb_name, "Output J"))
        output_flag =
            material_db_->getElementBlockParam<bool>(eb_name, "Output J");
      if (have_pore_pressure_eq_ || output_flag) {
        p = stateMgr.registerStateVariable(J,
            dl_->qp_scalar,
            dl_->dummy,
            eb_name,
            "scalar",
            1.0,
            true,
            output_flag);
        ev = Teuchos::rcp(
            new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
        fm0.template registerEvaluator<EvalT>(ev);
      }

      // Optional output: strain
      if (small_strain) {
        output_flag = false;
        if (material_db_->isElementBlockParam(eb_name, "Output Strain"))
          output_flag =
              material_db_->getElementBlockParam<bool>(eb_name,
                  "Output Strain");

        if (output_flag) {
          p = stateMgr.registerStateVariable("Strain",
              dl_->qp_tensor,
              dl_->dummy,
              eb_name,
              "scalar",
              0.0,
              false,
              output_flag);
          ev = Teuchos::rcp(
              new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
          fm0.template registerEvaluator<EvalT>(ev);
        }
      }

      // Optional output: velocity gradient
      if (have_velocity_gradient) {
        output_flag = false;
        if (material_db_->isElementBlockParam(eb_name,
            "Output Velocity Gradient"))
          output_flag =
              material_db_->getElementBlockParam<bool>(eb_name,
                  "Output Velocity Gradient");

        if (output_flag) {
          p = stateMgr.registerStateVariable("Velocity Gradient",
              dl_->qp_tensor,
              dl_->dummy,
              eb_name,
              "scalar",
              0.0,
              false,
              output_flag);
          ev = Teuchos::rcp(
              new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
          fm0.template registerEvaluator<EvalT>(ev);
        }
      }
    }
    if (have_mech_eq_)
    { // Residual
      Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
          new Teuchos::ParameterList("Displacement Residual"));
      //Input
      p->set<std::string>("Stress Name", firstPK);
      p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
      p->set<std::string>("Weighted BF Name", "wBF");
      p->set<std::string>("Acceleration Name", "Acceleration");
      if (Teuchos::nonnull(rc_mgr_)) {
        p->set<std::string>("DefGrad Name", defgrad);
        rc_mgr_->registerField(
          defgrad, dl_->qp_tensor, AAdapt::rc::Init::identity,
          AAdapt::rc::Transformation::right_polar_LieR_LieS, p);
      }
      
      // Mechanics residual need value of density for transient analysis.
      // Get it from material. Assumed constant in element block.
            if (material_db_->isElementBlockParam(eb_name,"Density"))
            {
                RealType density =
                    material_db_->getElementBlockParam<RealType>(eb_name, 
                        "Density");
                p->set<RealType>("Density", density);
            }
      
      p->set<Teuchos::RCP<ParamLib>>("Parameter Library", paramLib);
      //Output
      p->set<std::string>("Residual Name", "Displacement Residual");
      ev = Teuchos::rcp(
          new LCM::MechanicsResidual<EvalT, PHAL::AlbanyTraits>(*p, dl_));
      fm0.template registerEvaluator<EvalT>(ev);
    }
  }

  if (have_mech_eq_) {
    // convert Cauchy stress to first Piola-Kirchhoff
    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
        new Teuchos::ParameterList("First PK Stress"));
    //Input
    p->set<std::string>("Stress Name", cauchy);
    p->set<std::string>("DefGrad Name", defgrad);

    // Effective stress theory for poromechanics problem
    if (have_pore_pressure_eq_) {
      p->set<bool>("Have Pore Pressure", true);
      p->set<std::string>("Pore Pressure Name", porePressure);
      p->set<std::string>("Biot Coefficient Name", biotCoeff);
    }

    if (have_stab_pressure_eq_) {
      p->set<bool>("Have Stabilized Pressure", true);
      p->set<std::string>("Pressure Name", pressure);
    }

    if (small_strain) {
      p->set<bool>("Small Strain", true);
    }

    //Output
    p->set<std::string>("First PK Stress Name", firstPK);

    p->set<Teuchos::RCP<ParamLib>>("Parameter Library", paramLib);

    ev = Teuchos::rcp(new LCM::FirstPK<EvalT, PHAL::AlbanyTraits>(*p, dl_));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // Element length in the direction of solution gradient
  bool const
  have_pressure_or_transport =
      have_stab_pressure_eq_ || have_pore_pressure_eq_ || have_transport_eq_;

  if (have_pressure_or_transport) {
    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
        new Teuchos::ParameterList("Gradient_Element_Length"));
    //Input
    if (!surface_element) {  // bulk element length
      if (have_pore_pressure_eq_) {
        p->set<std::string>("Unit Gradient QP Variable Name",
            "Pore_Pressure Gradient");
      } else if (have_transport_eq_) {
        p->set<std::string>("Unit Gradient QP Variable Name",
            "Transport Gradient");
      } else if (have_stab_pressure_eq_) {
        p->set<std::string>("Unit Gradient QP Variable Name",
            "Pressure Gradient");
      }
      p->set<std::string>("Gradient BF Name", "Grad BF");
    }
    else { // surface element length
      if (have_pore_pressure_eq_) {
        p->set<std::string>("Unit Gradient QP Variable Name",
            "surf_Pressure Gradient");
      } else if (have_transport_eq_) {
        p->set<std::string>("Unit Gradient QP Variable Name",
            "surf_Transport Gradient");
      } else if (have_stab_pressure_eq_) {
        p->set<std::string>("Unit Gradient QP Variable Name",
            "surf_Pressure Gradient");
      }
      p->set<std::string>("Gradient BF Name",
          "Surface Scalar Gradient Operator");
      //   p->set<std::string>("Gradient BF Name", "Grad BF");
    }

    //Output
    p->set<std::string>("Element Length Name", gradient_element_length);

    ev = Teuchos::rcp(
        new LCM::GradientElementLength<EvalT, PHAL::AlbanyTraits>(*p, dl_));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_pore_pressure_eq_) {  // Porosity
    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
        new Teuchos::ParameterList);

    p->set<std::string>("Porosity Name", porosity);
    p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");
    // Setting this turns on dependence of strain and pore pressure)
    //p->set<std::string>("Strain Name", "Strain");
    if (have_mech_eq_) p->set<std::string>("DetDefGrad Name", J);
    // porosity update based on Coussy's poromechanics (see p.79)
    p->set<std::string>("QP Pore Pressure Name", porePressure);
    p->set<std::string>("Biot Coefficient Name", biotCoeff);

    p->set<Teuchos::RCP<ParamLib>>("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList =
        material_db_->getElementBlockSublist(eb_name, "Porosity");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = Teuchos::rcp(new LCM::Porosity<EvalT, PHAL::AlbanyTraits>(*p, dl_));
    fm0.template registerEvaluator<EvalT>(ev);

    // Output Porosity
    bool output_flag(false);
    if (material_db_->isElementBlockParam(eb_name, "Output " + porosity))
      output_flag =
          material_db_->getElementBlockParam<bool>(
              eb_name,
              "Output " + porosity);
    if (output_flag) {
      p = stateMgr.registerStateVariable(porosity,
          dl_->qp_scalar,
          dl_->dummy,
          eb_name,
          "scalar",
          0.5, // This is really bad practice. It needs to be fixed
          false,
          true);
      ev = Teuchos::rcp(
          new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }
  }

  if (have_pore_pressure_eq_) { // Biot Coefficient
    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
        new Teuchos::ParameterList);

    p->set<std::string>("Biot Coefficient Name", biotCoeff);
    p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "Node Data Layout",
        dl_->node_scalar);
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "QP Scalar Data Layout",
        dl_->qp_scalar);
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "QP Vector Data Layout",
        dl_->qp_vector);

    p->set<Teuchos::RCP<ParamLib>>("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList =
        material_db_->getElementBlockSublist(eb_name, "Biot Coefficient");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = Teuchos::rcp(new LCM::BiotCoefficient<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_pore_pressure_eq_) { // Biot Modulus
    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
        new Teuchos::ParameterList);

    p->set<std::string>("Biot Modulus Name", biotModulus);
    p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "Node Data Layout",
        dl_->node_scalar);
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "QP Scalar Data Layout",
        dl_->qp_scalar);
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "QP Vector Data Layout",
        dl_->qp_vector);

    p->set<Teuchos::RCP<ParamLib>>("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList =
        material_db_->getElementBlockSublist(eb_name, "Biot Modulus");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    // Setting this turns on linear dependence on porosity and Biot's coeffcient
    p->set<std::string>("Porosity Name", porosity);
    p->set<std::string>("Biot Coefficient Name", biotCoeff);

    ev = Teuchos::rcp(new LCM::BiotModulus<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_pore_pressure_eq_) { // Kozeny-Carman Permeaiblity
    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
        new Teuchos::ParameterList);

    p->set<std::string>("Kozeny-Carman Permeability Name", kcPerm);
    p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "Node Data Layout",
        dl_->node_scalar);
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "QP Scalar Data Layout",
        dl_->qp_scalar);
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "QP Vector Data Layout",
        dl_->qp_vector);

    p->set<Teuchos::RCP<ParamLib>>("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList =
        material_db_->getElementBlockSublist(eb_name,
            "Kozeny-Carman Permeability");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    // Setting this turns on Kozeny-Carman relation
    p->set<std::string>("Porosity Name", porosity);
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "QP Scalar Data Layout",
        dl_->qp_scalar);

    ev = Teuchos::rcp(new LCM::KCPermeability<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    // Output
    bool output_flag(false);
    if (material_db_->isElementBlockParam(eb_name, "Output " + kcPerm))
      output_flag =
          material_db_->getElementBlockParam<bool>(eb_name, "Output " + kcPerm);
    if (output_flag) {
      p = stateMgr.registerStateVariable(kcPerm,
          dl_->qp_scalar,
          dl_->dummy,
          eb_name,
          "scalar",
          0.0,
          false,
          true);
      ev = Teuchos::rcp(
          new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }
  }

  // Pore Pressure Residual (Bulk Element)
  if (have_pore_pressure_eq_ && !surface_element) {
    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
        new Teuchos::ParameterList("Pore_Pressure Residual"));

    //Input

    // Input from nodal points, basis function stuff
    p->set<std::string>("Weights Name", "Weights");
    p->set<std::string>("Weighted BF Name", "wBF");
    p->set<Teuchos::RCP<PHX::DataLayout>>("Node QP Scalar Data Layout",
        dl_->node_qp_scalar);
    p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "Node QP Vector Data Layout",
        dl_->node_qp_vector);

    // Inputs: X, Y at nodes, Cubature, and Basis
    p->set<std::string>("Coordinate Vector Name", "Coord Vec");
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "Coordinate Data Layout",
        dl_->vertices_vector);
    p->set<Teuchos::RCP<Intrepid2::Cubature<RealType,Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> >>>("Cubature", cubature);
    p->set<Teuchos::RCP<shards::CellTopology>>("Cell Type", cellType);

    // DT for  time integration
    p->set<std::string>("Delta Time Name", "Delta Time");
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "Workset Scalar Data Layout",
        dl_->workset_scalar);

    p->set<bool>("Have Source", false);
    p->set<std::string>("Source Name", "Source");
    p->set<bool>("Have Absorption", false);

    // Input from cubature points
    p->set<std::string>("Element Length Name", gradient_element_length);
    p->set<std::string>("QP Pore Pressure Name", porePressure);
    p->set<std::string>("QP Time Derivative Variable Name", porePressure);
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "QP Scalar Data Layout",
        dl_->qp_scalar);

    //p->set<std::string>("Material Property Name", "Stabilization Parameter");
    p->set<std::string>("Porosity Name", "Porosity");
    p->set<std::string>("Thermal Conductivity Name", "Thermal Conductivity");
    p->set<std::string>("Kozeny-Carman Permeability Name", kcPerm);
    p->set<std::string>("Biot Coefficient Name", biotCoeff);
    p->set<std::string>("Biot Modulus Name", biotModulus);

    p->set<std::string>("Gradient QP Variable Name", "Pore_Pressure Gradient");
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "QP Vector Data Layout",
        dl_->qp_vector);

    if (have_mech_eq_) {
      p->set<bool>("Have Mechanics", true);
      p->set<std::string>("DefGrad Name", defgrad);
      p->set<Teuchos::RCP<PHX::DataLayout>>(
          "QP Tensor Data Layout",
          dl_->qp_tensor);
      p->set<std::string>("DetDefGrad Name", J);
      p->set<Teuchos::RCP<PHX::DataLayout>>(
          "QP Scalar Data Layout",
          dl_->qp_scalar);
    }
    RealType stab_param(0.0);
    if (material_db_->isElementBlockParam(eb_name, "Stabilization Parameter")) {
      stab_param =
          material_db_->getElementBlockParam<RealType>(eb_name,
              "Stabilization Parameter");
    }

    p->set<RealType>("Stabilization Parameter", stab_param);

    p->set<Teuchos::RCP<ParamLib>>("Parameter Library", paramLib);

    //Output
    p->set<std::string>("Residual Name", "Pore_Pressure Residual");
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "Node Scalar Data Layout",
        dl_->node_scalar);

    ev = Teuchos::rcp(
        new LCM::TLPoroPlasticityResidMass<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    // Output QP pore pressure
    bool output_flag(false);
    if (material_db_->isElementBlockParam(eb_name, "Output IP" + porePressure))
      output_flag =
          material_db_->getElementBlockParam<bool>(
              eb_name,
              "Output IP" + porePressure);
    p = stateMgr.registerStateVariable(porePressure,
        dl_->qp_scalar,
        dl_->dummy,
        eb_name,
        "scalar",
        0.0,
        true,
        output_flag);
    ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_pore_pressure_eq_ && surface_element) {
    // Pore Pressure Resid for Surface
    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
        new Teuchos::ParameterList("Pore_Pressure Residual"));

    //Input
    p->set<RealType>("thickness", thickness);
    p->set<Teuchos::RCP<Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> >>>(
        "Cubature",
        surfaceCubature);
    p->set<Intrepid2Basis>("Intrepid2 Basis", surfaceBasis);
    p->set<std::string>("Surface Scalar Gradient Operator Name",
        "Surface Scalar Gradient Operator");
    p->set<std::string>("Scalar Gradient Name", "Surface Pressure Gradient");
    p->set<std::string>("Current Basis Name", "Current Basis");
    p->set<std::string>("Reference Dual Basis Name", "Reference Dual Basis");
    p->set<std::string>("Reference Normal Name", "Reference Normal");
    p->set<std::string>("Reference Area Name", "Weights");
    p->set<std::string>("Pore Pressure Name", porePressure);
    // NOTE: NOT surf_Pore_Pressure here
    p->set<std::string>("Nodal Pore Pressure Name", "Pore_Pressure");
    p->set<std::string>("Biot Coefficient Name", biotCoeff);
    p->set<std::string>("Biot Modulus Name", biotModulus);
    p->set<std::string>("Kozeny-Carman Permeability Name", kcPerm);
    p->set<std::string>("Delta Time Name", "Delta Time");
    if (have_mech_eq_) {
      p->set<std::string>("DefGrad Name", defgrad);
      p->set<std::string>("DetDefGrad Name", J);
    }

    //Output
    p->set<std::string>("Residual Name", "Pore_Pressure Residual");
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "Node Scalar Data Layout",
        dl_->node_scalar);

    ev = Teuchos::rcp(
        new LCM::SurfaceTLPoroMassResidual<EvalT, PHAL::AlbanyTraits>(*p, dl_));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_transport_eq_ || have_transport_) { // Transport Coefficients
    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
        new Teuchos::ParameterList("Transport Coefficients"));

    std::string matName = material_db_->getElementBlockParam<std::string>(
        eb_name, "material");
    Teuchos::ParameterList& param_list = material_db_->
        getElementBlockSublist(eb_name, matName).sublist(
        "Transport Coefficients");
    p->set<Teuchos::ParameterList*>("Material Parameters", &param_list);

    //Input
    p->set<std::string>("Lattice Concentration Name", transport);
    p->set<std::string>("Deformation Gradient Name", defgrad);
    p->set<std::string>("Determinant of F Name", J);
    p->set<std::string>("Temperature Name", temperature);
    // FIXME: this creates a circular dependency between the constitutive model and transport
    // see below
    if (material_model_name == "J2" || material_model_name == "Elasto Viscoplastic") {
      p->set<std::string>("Equivalent Plastic Strain Name", eqps);
      p->set<std::string>("Strain Rate Factor Name", strainRateFactor);
    }

    p->set<bool>("Weighted Volume Average J", volume_average_j);
    p->set<RealType>(
        "Average J Stabilization Parameter",
        volume_average_stabilization_param);

    p->set<Teuchos::RCP<std::map<std::string, std::string>>>("Name Map",fnm);

    //Output
    p->set<std::string>("Trapped Concentration Name", trappedConcentration);
    p->set<std::string>("Total Concentration Name", totalConcentration);
    p->set<std::string>("Mechanical Deformation Gradient Name", "Fm");
    p->set<std::string>("Effective Diffusivity Name", effectiveDiffusivity);
    p->set<std::string>("Trapped Solvent Name", trappedSolvent);
    // FIXME: this creates a circular dependency between the constitutive model and transport
    //if (material_model_name == "J2" || material_model_name == "Elasto Viscoplastic") {
    //p->set<std::string>("Strain Rate Factor Name", strainRateFactor);
      //}
    p->set<std::string>("Diffusion Coefficient Name", diffusionCoefficient);
    p->set<std::string>("Tau Contribution Name", convectionCoefficient);
    p->set<std::string>("Concentration Equilibrium Parameter Name",
        eqilibriumParameter);

    ev = Teuchos::rcp(
        new LCM::TransportCoefficients<EvalT, PHAL::AlbanyTraits>(*p, dl_));
    fm0.template registerEvaluator<EvalT>(ev);

    bool output_flag(false);
    // Trapped Concentration
    if (material_db_->isElementBlockParam(
        eb_name,
        "Output " + trappedConcentration))
      output_flag =
          material_db_->getElementBlockParam<bool>(
              eb_name,
              "Output " + trappedConcentration);
    if (output_flag) {
      p = stateMgr.registerStateVariable(trappedConcentration, dl_->qp_scalar,
          dl_->dummy, eb_name, "scalar", 0.0, false, output_flag);
      ev = Teuchos::rcp(
          new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    output_flag = false;
    // Total Concentration
    if (material_db_->isElementBlockParam(
        eb_name,
        "Output " + totalConcentration))
      output_flag =
          material_db_->getElementBlockParam<bool>(
              eb_name,
              "Output " + totalConcentration);
    if (output_flag) {
      p = stateMgr.registerStateVariable(totalConcentration, dl_->qp_scalar,
          dl_->dummy, eb_name, "scalar", 0.0, true, output_flag);
      ev = Teuchos::rcp(
          new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    // Strain Rate Factor
    output_flag = false;
    if (material_db_->isElementBlockParam(
        eb_name,
        "Output " + strainRateFactor))
      output_flag =
          material_db_->getElementBlockParam<bool>(
              eb_name,
              "Output " + strainRateFactor);
    if (output_flag) {
      p = stateMgr.registerStateVariable(strainRateFactor, dl_->qp_scalar,
          dl_->dummy, eb_name, "scalar", 0.0, false, output_flag);
      ev = Teuchos::rcp(
          new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    // Convection Coefficient
    output_flag = false;
    if (material_db_->isElementBlockParam(
        eb_name,
        "Output " + convectionCoefficient))
      output_flag =
          material_db_->getElementBlockParam<bool>(
              eb_name,
              "Output " + convectionCoefficient);
    if (output_flag) {
      p = stateMgr.registerStateVariable(convectionCoefficient, dl_->qp_scalar,
          dl_->dummy, eb_name, "scalar", 0.0, false, output_flag);
      ev = Teuchos::rcp(
          new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    // Diffusion Coefficient
    output_flag = false;
    if (material_db_->isElementBlockParam(
        eb_name,
        "Output " + diffusionCoefficient))
      output_flag =
          material_db_->getElementBlockParam<bool>(
              eb_name,
              "Output " + diffusionCoefficient);
    if (output_flag) {
      p = stateMgr.registerStateVariable(diffusionCoefficient, dl_->qp_scalar,
          dl_->dummy, eb_name, "scalar", 1.0, false, output_flag);
      ev = Teuchos::rcp(
          new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    // Effective Diffusivity
    output_flag = false;
    if (material_db_->isElementBlockParam(
        eb_name,
        "Output " + effectiveDiffusivity))
      output_flag =
          material_db_->getElementBlockParam<bool>(
              eb_name,
              "Output " + effectiveDiffusivity);
    if (output_flag) {
      p = stateMgr.registerStateVariable(effectiveDiffusivity, dl_->qp_scalar,
          dl_->dummy, eb_name, "scalar", 1.0, false, output_flag);
      ev = Teuchos::rcp(
          new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }
  }

  // Helium ODEs
  if (have_transport_eq_ || have_transport_)
      {
    // Get material list prior to establishing a new parameter list
    std::string matName = material_db_->getElementBlockParam<std::string>(
        eb_name, "material");
    Teuchos::ParameterList& param_list =
        material_db_->getElementBlockSublist(eb_name, matName);

    // Check if Tritium Sublist exists. If true, move forward
    if (param_list.isSublist("Tritium Coefficients")) {

      Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
          new Teuchos::ParameterList("Helium ODEs"));

      // Rather than combine lists, we choose to invoke multiple parameter
      // lists and stuff them separately into p.
      // All lists need to be reflected in HeliumODEs_Def.hpp
      Teuchos::ParameterList& transport_param = material_db_->
          getElementBlockSublist(eb_name, matName).sublist(
          "Transport Coefficients");
      Teuchos::ParameterList& tritium_param = material_db_->
          getElementBlockSublist(eb_name, matName).sublist(
          "Tritium Coefficients");
      Teuchos::ParameterList& molar_param = material_db_->
          getElementBlockSublist(eb_name, matName).sublist(
          "Molar Volume");

      p->set<Teuchos::ParameterList*>("Transport Parameters", &transport_param);
      p->set<Teuchos::ParameterList*>("Tritium Parameters", &tritium_param);
      p->set<Teuchos::ParameterList*>("Molar Volume", &molar_param);

      //Input
      p->set<std::string>("Total Concentration Name", totalConcentration);
      p->set<std::string>("Delta Time Name", "Delta Time");
      p->set<std::string>("Diffusion Coefficient Name", diffusionCoefficient);
      // Output
      p->set<std::string>("He Concentration Name", he_concentration);
      p->set<std::string>("Total Bubble Density Name", total_bubble_density);
      p->set<std::string>(
          "Bubble Volume Fraction Name",
          bubble_volume_fraction);

      ev = Teuchos::rcp(
          new LCM::HeliumODEs<EvalT, PHAL::AlbanyTraits>(*p, dl_));
      fm0.template registerEvaluator<EvalT>(ev);

      // Outputting state variables
      //
      // Using field names registered for surface elements
      // (he_concentration, etc.)
      // NOTE: All output variables are stated
      //
      // helium concentration
      bool output_flag(false);
      if (material_db_->isElementBlockParam(
          eb_name,
          "Output " + he_concentration))
        output_flag =
            material_db_->getElementBlockParam<bool>(
                eb_name,
                "Output " + he_concentration);
      if (output_flag) {
        p = stateMgr.registerStateVariable(he_concentration, dl_->qp_scalar,
            dl_->dummy, eb_name, "scalar", 0.0, true, output_flag);
        ev = Teuchos::rcp(
            new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
        fm0.template registerEvaluator<EvalT>(ev);
      }
      // total bubble density
      output_flag = false;
      if (material_db_->isElementBlockParam(
          eb_name,
          "Output " + total_bubble_density))
        output_flag =
            material_db_->getElementBlockParam<bool>(
                eb_name,
                "Output " + total_bubble_density);
      if (output_flag) {
        p = stateMgr.registerStateVariable(total_bubble_density, dl_->qp_scalar,
            dl_->dummy, eb_name, "scalar", 0.0, true, output_flag);
        ev = Teuchos::rcp(
            new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
        fm0.template registerEvaluator<EvalT>(ev);
      }
      // bubble volume fraction
      output_flag = false;
      if (material_db_->isElementBlockParam(
          eb_name,
          "Output " + bubble_volume_fraction))
        output_flag =
            material_db_->getElementBlockParam<bool>(
                eb_name,
                "Output " + bubble_volume_fraction);
      if (output_flag) {
        p = stateMgr.registerStateVariable(
            bubble_volume_fraction,
            dl_->qp_scalar,
            dl_->dummy,
            eb_name,
            "scalar",
            0.0,
            true,
            output_flag);
        ev = Teuchos::rcp(
            new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
        fm0.template registerEvaluator<EvalT>(ev);
      }
    }
  }

  // Transport of the temperature field
  if (have_temperature_eq_ && !surface_element)
      {
    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
        new Teuchos::ParameterList("ThermoMechanical Coefficients"));

    std::string matName =
        material_db_->getElementBlockParam<std::string>(eb_name, "material");
    Teuchos::ParameterList& param_list =
        material_db_->getElementBlockSublist(eb_name, matName);
    p->set<Teuchos::ParameterList*>("Material Parameters", &param_list);

    // Input
    p->set<std::string>("Temperature Name", "Temperature");
    p->set<std::string>("Thermal Conductivity Name", "Thermal Conductivity");
    p->set<std::string>("Thermal Transient Coefficient Name",
        "Thermal Transient Coefficient");
    p->set<std::string>("Delta Time Name", "Delta Time");
    
    // MJJ: Need this here to compute responses later
    RealType heat_capacity = param_list.get<RealType>("Heat Capacity");
    RealType density = param_list.get<RealType>("Density");
    pFromProb->set<RealType>("Heat Capacity",heat_capacity);
    pFromProb->set<RealType>("Density",density);
    
    if (have_mech_eq_) {
      p->set<bool>("Have Mechanics", true);
      p->set<std::string>("Deformation Gradient Name", defgrad);
    }

    // Output
    p->set<std::string>("Thermal Diffusivity Name", "Thermal Diffusivity");
    p->set<std::string>("Temperature Dot Name", "Temperature Dot");

    ev = Teuchos::rcp(
        new LCM::ThermoMechanicalCoefficients<EvalT, PHAL::AlbanyTraits>(
            *p,
            dl_));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // Evaluate contact contributions

  if (have_contact_) { // create the contact evaluator to fill in the
#ifdef ALBANY_CONTACT
    Teuchos::ParameterList& paramList = params->sublist("Contact");
    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
        new Teuchos::ParameterList);

    p->set<Teuchos::Array<std::string>>("Master Side Set Names",
        paramList.get<Teuchos::Array<std::string>>("Master Side Sets"));
    p->set<Teuchos::Array<std::string>>("Slave Side Set Names",
        paramList.get<Teuchos::Array<std::string>>("Slave Side Sets"));
    p->set<Teuchos::Array<std::string>>("Sideset IDs",
        paramList.get<Teuchos::Array<std::string>>("Contact Side Set Pair"));
    p->set<Teuchos::Array<std::string>>("Constrained Field Names",
        paramList.get<Teuchos::Array<std::string>>("Constrained Field Names"));

    p->set<const Albany::MeshSpecsStruct*>("Mesh Specs Struct", &meshSpecs);
    p->set<std::string>("Coordinate Vector Name", "Coord Vec");

    p->set<Teuchos::RCP<ParamLib>>("Parameter Library", paramLib);

    p->set<std::string>("M Name", "M");

    ev = Teuchos::rcp(
        new LCM::MortarContact<EvalT, PHAL::AlbanyTraits>(*p, dl_));
    fm0.template registerEvaluator<EvalT>(ev);
#else // ! defined ALBANY_CONTACT
    TEUCHOS_TEST_FOR_EXCEPTION(
      true, std::logic_error,
      "A contact problem is being created, but ALBANY_CONTACT is not defined. "
      "Use the flag -D ENABLE_CONTACT:BOOL=ON in your Albany configuration.");
#endif // ALBANY_CONTACT
  }

  // Transport of the temperature field
  if (have_temperature_eq_ && !surface_element)
      {
    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
        new Teuchos::ParameterList("Temperature Residual"));

    // Input
    p->set<std::string>("Scalar Variable Name", "Temperature");
    p->set<std::string>("Scalar Gradient Variable Name",
        "Temperature Gradient");
    p->set<std::string>("Weights Name", "Weights");
    p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
    p->set<std::string>("Weighted BF Name", "wBF");

    // Transient
    p->set<bool>("Have Transient", true);
    p->set<std::string>("Scalar Dot Name", "Temperature Dot");
    p->set<std::string>("Transient Coefficient Name",
        "Thermal Transient Coefficient");

    // Diffusion
    p->set<bool>("Have Diffusion", true);
    p->set<std::string>("Diffusivity Name", "Thermal Diffusivity");

    // Source
    if ((have_mech_ || have_mech_eq_) && material_model_name == "J2") {
      p->set<bool>("Have Source", true);
      p->set<std::string>("Source Name", mech_source);
    }

    // Thermal Source (internal energy generation)
    if (thermal_source_evaluated_) {
      p->set<bool>("Have Second Source", true);
      p->set<std::string>("Second Source Name", "Heat Source");
    }

    if (have_contact_) { // Pass M to the heat eqn for thermal fluxes between surfaces
      p->set<bool>("Have Contact", true);
      p->set<std::string>("M Name", "M");
    }

    // Output
    p->set<std::string>("Residual Name", "Temperature Residual");

    ev = Teuchos::rcp(
        new LCM::TransportResidual<EvalT, PHAL::AlbanyTraits>(*p, dl_));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // Hydrogen Transport model proposed in Foulk et al 2014
  if (have_transport_eq_ && !surface_element) {
    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
        new Teuchos::ParameterList("Transport Residual"));

    //Input
    p->set<std::string>("Element Length Name", gradient_element_length);
    p->set<std::string>("Weighted BF Name", "wBF");
    p->set<std::string>("Weights Name", "Weights");
    p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
    p->set<std::string>("Gradient BF Name", "Grad BF");
    if ((have_mech_ || have_mech_eq_) &&
        (material_model_name == "J2" || material_model_name == "Elasto Viscoplastic")) {
      p->set<std::string>("Equivalent Plastic Strain Name", eqps);
      p->set<std::string>("Strain Rate Factor Name", strainRateFactor);
    }
    p->set<std::string>("Tau Contribution Name", convectionCoefficient);
    p->set<std::string>("Tau Contribution Name", convectionCoefficient);
    p->set<std::string>("Trapped Concentration Name", trappedConcentration);
    p->set<std::string>("Trapped Solvent Name", trappedSolvent);
    p->set<std::string>("Deformation Gradient Name", defgrad);
    p->set<std::string>("Effective Diffusivity Name", effectiveDiffusivity);
    p->set<std::string>("Diffusion Coefficient Name", diffusionCoefficient);
    p->set<std::string>("QP Variable Name", "Transport");
    p->set<std::string>("Gradient QP Variable Name", "Transport Gradient");
    p->set<std::string>("Gradient Hydrostatic Stress Name", "HydroStress Gradient");
    p->set<std::string>("Stress Name", cauchy);
    p->set<std::string>("Delta Time Name", "Delta Time");
    RealType stab_param(0.0);

    if (material_db_->isElementBlockParam(eb_name, "Stabilization Parameter")) {
      stab_param =
          material_db_->getElementBlockParam<RealType>(eb_name,
              "Stabilization Parameter");
    }
    p->set<RealType>("Stabilization Parameter", stab_param);
    p->set<Teuchos::RCP<ParamLib>>("Parameter Library", paramLib);

    // Get material list prior to establishing a new parameter list
    std::string matName = material_db_->getElementBlockParam<std::string>(
        eb_name, "material");
    Teuchos::ParameterList& param_list =
        material_db_->getElementBlockSublist(eb_name, matName);

    RealType decay_constant(0.0);
    // Check if Tritium Sublist exists. If true, move forward
    if (param_list.isSublist("Tritium Coefficients")) {
      Teuchos::ParameterList& tritium_param = material_db_->
          getElementBlockSublist(eb_name, matName).sublist(
          "Tritium Coefficients");
      decay_constant = tritium_param.get<RealType>("Tritium Decay Constant",0.0);
    }
    p->set<RealType>("Tritium Decay Constant", decay_constant);

    //Output
    p->set<std::string>("Residual Name", "Transport Residual");

    ev = Teuchos::rcp(new LCM::HDiffusionDeformationMatterResidual<EvalT, PHAL::AlbanyTraits>(*p, dl_));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_transport_eq_ && surface_element) { // Transport Resid for Surface
    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
        new Teuchos::ParameterList("Transport Residual"));

    //Input
    p->set<RealType>("thickness", thickness);
    p->set<Teuchos::RCP<Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> >>>(
        "Cubature",
        surfaceCubature);
    p->set<Intrepid2Basis>("Intrepid2 Basis", surfaceBasis);
    p->set<std::string>("Surface Scalar Gradient Operator Name",
        "Surface Scalar Gradient Operator");
    p->set<std::string>("Surface Transport Gradient Name",
        "Surface Transport Gradient");
    p->set<std::string>("Current Basis Name", "Current Basis");
    p->set<std::string>("Reference Dual Basis Name", "Reference Dual Basis");
    p->set<std::string>("Reference Normal Name", "Reference Normal");
    p->set<std::string>("Reference Area Name", "Weights");
    p->set<std::string>("Transport Name", transport);
    // NOTE: NOT surf_Transport here
    p->set<std::string>("Nodal Transport Name", "Transport");
    p->set<std::string>("Diffusion Coefficient Name", diffusionCoefficient);
    p->set<std::string>("Effective Diffusivity Name", effectiveDiffusivity);
    p->set<std::string>("Tau Contribution Name", convectionCoefficient);
    p->set<std::string>("Strain Rate Factor Name", strainRateFactor);
    // This does not make sense
    p->set<std::string>("Element Length Name", effectiveDiffusivity);
    p->set<std::string>("Surface HydroStress Gradient Name",
        "Surface HydroStress Gradient");
    p->set<std::string>("eqps Name", eqps);
    p->set<std::string>("Delta Time Name", "Delta Time");
    if (have_mech_eq_) {
      p->set<std::string>("DefGrad Name", defgrad);
      p->set<std::string>("DetDefGrad Name", J);
    }

    RealType stab_param(0.0);
    if (material_db_->isElementBlockParam(eb_name, "Stabilization Parameter")) {
      stab_param =
          material_db_->getElementBlockParam<RealType>(eb_name,
              "Stabilization Parameter");
    }
    p->set<RealType>("Stabilization Parameter", stab_param);
    p->set<Teuchos::RCP<ParamLib>>("Parameter Library", paramLib);

    //Output
    p->set<std::string>("Residual Name", "Transport Residual");
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "Node Scalar Data Layout",
        dl_->node_scalar);

    ev = Teuchos::rcp(
        new LCM::SurfaceHDiffusionDefResidual<EvalT, PHAL::AlbanyTraits>(
            *p,
            dl_));
    fm0.template registerEvaluator<EvalT>(ev);

  }

  if (have_hydrostress_eq_ && !surface_element) {
    // L2 hydrostatic stress projection
    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
        new Teuchos::ParameterList("HydroStress Residual"));

    //Input
    p->set<std::string>("Weighted BF Name", "wBF");
    p->set<Teuchos::RCP<PHX::DataLayout>>
    ("Node QP Scalar Data Layout", dl_->node_qp_scalar);

    p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
    p->set<Teuchos::RCP<PHX::DataLayout>>
    ("Node QP Vector Data Layout", dl_->node_qp_vector);

    p->set<bool>("Have Source", false);
    p->set<std::string>("Source Name", "Source");

    p->set<std::string>("Deformation Gradient Name", defgrad);
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "QP Tensor Data Layout",
        dl_->qp_tensor);

    p->set<std::string>("QP Variable Name", hydroStress);
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "QP Scalar Data Layout",
        dl_->qp_scalar);

    p->set<std::string>("Stress Name", cauchy);
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "QP Tensor Data Layout",
        dl_->qp_tensor);

    //Output
    p->set<std::string>("Residual Name", "HydroStress Residual");
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "Node Scalar Data Layout",
        dl_->node_scalar);

    ev = Teuchos::rcp(
        new LCM::ScalarL2ProjectionResidual<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_hydrostress_eq_ && surface_element) {
    // Hydrostress Projection Resid for Surface
    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
        new Teuchos::ParameterList("HydroStress Residual"));

    //Input
    p->set<RealType>("thickness", thickness);
    p->set<Teuchos::RCP<Intrepid2::Cubature<RealType,Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> >>>(
        "Cubature",
        surfaceCubature);
    p->set<Intrepid2Basis>("Intrepid2 Basis", surfaceBasis);
    p->set<std::string>("Surface Scalar Gradient Operator Name",
        "Surface Scalar Gradient Operator");
    p->set<std::string>("Current Basis Name", "Current Basis");
    p->set<std::string>("Reference Dual Basis Name", "Reference Dual Basis");
    p->set<std::string>("Reference Normal Name", "Reference Normal");
    p->set<std::string>("Reference Area Name", "Weights");
    p->set<std::string>("HydoStress Name", hydroStress);
    p->set<std::string>("Cauchy Stress Name", cauchy);
    p->set<std::string>("Jacobian Name", J);

    //Output
    p->set<std::string>("Residual Name", "HydroStress Residual");
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "Node Scalar Data Layout",
        dl_->node_scalar);

    ev = Teuchos::rcp(
        new LCM::SurfaceL2ProjectionResidual<EvalT, PHAL::AlbanyTraits>(
            *p,
            dl_));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_stab_pressure_eq_) {
    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
        new Teuchos::ParameterList("Stabilized Pressure Residual"));
    //Input
    p->set<std::string>("Shear Modulus Name", "Shear Modulus");
    p->set<std::string>("Bulk Modulus Name", "Bulk Modulus");
    p->set<std::string>("Deformation Gradient Name", defgrad);
    p->set<std::string>("Stress Name", cauchy);
    p->set<std::string>("Pressure Name", pressure);
    p->set<std::string>("Pressure Gradient Name", "Pressure Gradient");
    p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
    p->set<std::string>("Weighted BF Name", "wBF");
    p->set<std::string>(
        "Element Characteristic Length Name",
        gradient_element_length);
    RealType stab_param(0.0);
    if (material_db_->isElementBlockParam(eb_name, "Stabilization Parameter")) {
      stab_param =
          material_db_->getElementBlockParam<RealType>(eb_name,
              "Stabilization Parameter");
    }
    p->set<RealType>("Stabilization Parameter", stab_param);
    p->set<bool>("Small Strain", small_strain);

    //Output
    p->set<std::string>("Residual Name", "Pressure Residual");
    ev = Teuchos::rcp(
        new LCM::StabilizedPressureResidual<EvalT, PHAL::AlbanyTraits>(
            *p,
            dl_));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (Teuchos::nonnull(rc_mgr_))
    rc_mgr_->createEvaluators<EvalT>(fm0, dl_);

  if (fieldManagerChoice == Albany::BUILD_RESID_FM) {

    Teuchos::RCP<const PHX::FieldTag> ret_tag;
    if (have_mech_eq_) {
      PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter", dl_->dummy);
      fm0.requireField<EvalT>(res_tag);
      ret_tag = res_tag.clone();
    }
    if (have_pore_pressure_eq_) {
      PHX::Tag<typename EvalT::ScalarT> pore_tag("Scatter Pore_Pressure",
          dl_->dummy);
      fm0.requireField<EvalT>(pore_tag);
      ret_tag = pore_tag.clone();
    }
    if (have_stab_pressure_eq_) {
      PHX::Tag<typename EvalT::ScalarT> pres_tag("Scatter Pressure",
          dl_->dummy);
      fm0.requireField<EvalT>(pres_tag);
      ret_tag = pres_tag.clone();
    }
    if (have_temperature_eq_) {
      PHX::Tag<typename EvalT::ScalarT> temperature_tag("Scatter Temperature",
          dl_->dummy);
      fm0.requireField<EvalT>(temperature_tag);
      ret_tag = temperature_tag.clone();
    }
    if (have_transport_eq_) {
      PHX::Tag<typename EvalT::ScalarT> transport_tag("Scatter Transport",
          dl_->dummy);
      fm0.requireField<EvalT>(transport_tag);
      ret_tag = transport_tag.clone();
    }
    if (have_hydrostress_eq_) {
      PHX::Tag<typename EvalT::ScalarT> l2projection_tag("Scatter HydroStress",
          dl_->dummy);
      fm0.requireField<EvalT>(l2projection_tag);
      ret_tag = l2projection_tag.clone();
    }
    return ret_tag;
  }
  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM) {
    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl_);
    return respUtils.constructResponses(
        fm0, *responseList, pFromProb, stateMgr, &meshSpecs);
  }

  return Teuchos::null;
}

#endif
