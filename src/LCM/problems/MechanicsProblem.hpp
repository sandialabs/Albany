//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef MECHANICSPROBLEM_HPP
#define MECHANICSPROBLEM_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractProblem.hpp"

#include "Phalanx.hpp"
#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_AlbanyTraits.hpp"

namespace Albany
{

//------------------------------------------------------------------------------
///
/// \brief Definition for the Mechanics Problem
///
class MechanicsProblem: public Albany::AbstractProblem
{
public:

  typedef Intrepid::FieldContainer<RealType> FC;

  ///
  /// Default constructor
  ///
  MechanicsProblem(const Teuchos::RCP<Teuchos::ParameterList>& params,
      const Teuchos::RCP<ParamLib>& param_lib,
      const int num_dims,
      const Teuchos::RCP<const Epetra_Comm>& comm);
  ///
  /// Destructor
  ///
  virtual
  ~MechanicsProblem();

  ///
  Teuchos::RCP<std::map<std::string, std::string> >
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
  buildProblem(Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >
      meshSpecs,
      StateManager& stateMgr);

  ///
  /// Build evaluators
  ///
  virtual Teuchos::Array<Teuchos::RCP<const PHX::FieldTag> >
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
  getAllocatedStates(Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<FC> > >
      old_state,
      Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<FC> > >
      new_state) const;

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
    MECH_VAR_TYPE_DOF        //! Variable is a degree-of-freedom
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
  /// Data layouts
  ///
  Teuchos::RCP<Albany::Layouts> dl_;

  ///
  /// RCP to matDB object
  ///
  Teuchos::RCP<QCAD::MaterialDatabase> material_db_;

  ///
  /// old state data
  ///
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<FC> > > old_state_;

  ///
  /// new state data
  ///
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<FC> > > new_state_;

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

// Damage equation specific evaluators
#include "DamageCoefficients.hpp"

// Damage equation specific evaluators
#include "StabilizedPressureResidual.hpp"

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
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::ParameterList;
  using PHX::DataLayout;
  using PHX::MDALayout;
  using std::vector;
  using PHAL::AlbanyTraits;
  using shards::CellTopology;
  using shards::getCellTopologyData;

  // Collect problem-specific response parameters

  RCP<ParameterList> pFromProb = rcp(new ParameterList("Response Parameters from Problem"));

  // get the name of the current element block
  std::string eb_name = meshSpecs.ebName;

  // get the name of the material model to be used (and make sure there is one)
  std::string material_model_name =
      material_db_->
          getElementBlockSublist(eb_name, "Material Model").get<std::string>(
          "Model Name");
  TEUCHOS_TEST_FOR_EXCEPTION(material_model_name.length() == 0, std::logic_error,
      "A material model must be defined for block: "
          + eb_name);

#ifdef ALBANY_VERBOSE
  *out << "In MechanicsProblem::constructEvaluators" << std::endl;
  *out << "element block name: " << eb_name << std::endl;
  *out << "material model name: " << material_model_name << std::endl;
#endif

  // define cell topologies
  RCP<CellTopology> comp_cellType =
      rcp(new CellTopology(getCellTopologyData<shards::Tetrahedron<11> >()));
  RCP<shards::CellTopology> cellType =
      rcp(new CellTopology(&meshSpecs.ctd));

  // volume averaging flags
  bool volume_average_j(false);
  bool volume_average_pressure(false);
  RealType volume_average_stabilization_param(0.0);
  if (material_db_->isElementBlockParam(eb_name, "Weighted Volume Average J"))
    volume_average_j = material_db_->getElementBlockParam<bool>(eb_name,"Weighted Volume Average J");
  if (material_db_->isElementBlockParam(eb_name, "Volume Average Pressure"))
    volume_average_pressure = material_db_->getElementBlockParam<bool>(eb_name,"Volume Average Pressure");
  if (material_db_->isElementBlockParam(eb_name, "Average J Stabilization Parameter"))
    volume_average_stabilization_param = material_db_->getElementBlockParam<RealType>(eb_name,"Average J Stabilization Parameter");

  // Check if we are setting the composite tet flag
  bool composite = false;
  if (material_db_->isElementBlockParam(eb_name, "Use Composite Tet 10"))
    composite =
        material_db_->getElementBlockParam<bool>(eb_name,
            "Use Composite Tet 10");

  // set flag for small strain option
  bool small_strain(false);
  if ( material_model_name == "Linear Elastic" ) {
    small_strain = true;
  }

  if (material_db_->isElementBlockParam(eb_name, "Strain Flag")) {
    small_strain = true;
   }

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
  RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > >
  intrepidBasis = Albany::getIntrepidBasis(meshSpecs.ctd, composite);

  if (composite &&
      meshSpecs.ctd.dimension == 3 &&
      meshSpecs.ctd.node_count == 10) cellType = comp_cellType;

  Intrepid::DefaultCubatureFactory<RealType> cubFactory;
  RCP<Intrepid::Cubature<RealType> > cubature =
      cubFactory.create(*cellType, meshSpecs.cubatureDegree);

  // FIXME, this could probably go into the ProblemUtils
  // just like the call to getIntrepidBasis
  RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > >
  surfaceBasis;
  RCP<shards::CellTopology> surfaceTopology;
  RCP<Intrepid::Cubature<RealType> > surfaceCubature;
  if (surface_element)
  {
#ifdef ALBANY_VERBOSE
    *out << "In Surface Element Logic" << std::endl;
#endif

    std::string name = meshSpecs.ctd.name;
    if (name == "Triangle_3" || name == "Quadrilateral_4") {
      surfaceBasis =
          rcp(
              new Intrepid::Basis_HGRAD_LINE_C1_FEM<RealType,
                  Intrepid::FieldContainer<RealType> >());
      surfaceTopology =
          rcp(
              new shards::CellTopology(
                  shards::getCellTopologyData<shards::Line<2> >()));
      surfaceCubature =
          cubFactory.create(*surfaceTopology, meshSpecs.cubatureDegree);
    }
    else if (name == "Wedge_6") {
      surfaceBasis =
          rcp(
              new Intrepid::Basis_HGRAD_TRI_C1_FEM<RealType,
                  Intrepid::FieldContainer<RealType> >());
      surfaceTopology =
          rcp(
              new shards::CellTopology(
                  shards::getCellTopologyData<shards::Triangle<3> >()));
      surfaceCubature =
          cubFactory.create(*surfaceTopology, meshSpecs.cubatureDegree);
    }
    else if (name == "Hexahedron_8") {
      surfaceBasis =
          rcp(
              new Intrepid::Basis_HGRAD_QUAD_C1_FEM<RealType,
                  Intrepid::FieldContainer<RealType> >());
      surfaceTopology =
          rcp(
              new shards::CellTopology(
                  shards::getCellTopologyData<shards::Quadrilateral<4> >()));
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
  dl_ =rcp( new Albany::Layouts(workset_size,
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
  RCP<PHX::Evaluator<AlbanyTraits> > ev;

  // Define Field Names
  // generate the field name map to deal with outputing surface element info
  LCM::FieldNameMap field_name_map(surface_element);
  RCP<std::map<std::string, std::string> > fnm = field_name_map.getMap();
  std::string cauchy = (*fnm)["Cauchy_Stress"];
  std::string firstPK = (*fnm)["PK1"];
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
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<std::string>("Material Property Name", "Displacement");
    p->set<RCP<DataLayout> >("Data Layout", dl_->qp_vector);
    p->set<std::string>("Coordinate Vector Name", "Coord Vec");
    p->set<RCP<DataLayout> >("Coordinate Vector Data Layout", dl_->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Displacement");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new PHAL::NSMaterialProperty<EvalT, AlbanyTraits>(*p));
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
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<std::string>("Material Property Name", temperature);
    p->set<RCP<DataLayout> >("Data Layout", dl_->qp_scalar);
    p->set<std::string>("Coordinate Vector Name", "Coord Vec");
    p->set<RCP<DataLayout> >("Coordinate Vector Data Layout", dl_->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Temperature");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new PHAL::NSMaterialProperty<EvalT, AlbanyTraits>(*p));
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
      RCP<ParameterList> p = rcp(new ParameterList);

      p->set<std::string>("Material Property Name", "Damage");
      p->set<RCP<DataLayout> >("Data Layout", dl_->qp_scalar);
      p->set<std::string>("Coordinate Vector Name", "Coord Vec");
      p->set<RCP<DataLayout> >("Coordinate Vector Data Layout", dl_->qp_vector);

      p->set<RCP<ParamLib> >("Parameter Library", paramLib);
      Teuchos::ParameterList& paramList = params->sublist("Damage");
      p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

      ev = rcp(new PHAL::NSMaterialProperty<EvalT, AlbanyTraits>(*p));
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
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<std::string>("Material Property Name", "Pressure");
    p->set<RCP<DataLayout> >("Data Layout", dl_->qp_scalar);
    p->set<std::string>("Coordinate Vector Name", "Coord Vec");
    p->set<RCP<DataLayout> >("Coordinate Vector Data Layout", dl_->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Pressure");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new PHAL::NSMaterialProperty<EvalT, AlbanyTraits>(*p));
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
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<std::string>("Material Property Name", "Transport");
    p->set<RCP<DataLayout> >("Data Layout", dl_->qp_scalar);
    p->set<std::string>("Coordinate Vector Name", "Coord Vec");
    p->set<RCP<DataLayout> >("Coordinate Vector Data Layout", dl_->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Transport");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new PHAL::NSMaterialProperty<EvalT, AlbanyTraits>(*p));
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
    RCP<ParameterList> p = rcp(new ParameterList("Time"));
    p->set<std::string>("Time Name", "Time");
    p->set<std::string>("Delta Time Name", "Delta Time");
    p->set<RCP<DataLayout> >("Workset Scalar Data Layout", dl_->workset_scalar);
    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    p->set<bool>("Disable Transient", true);
    ev = rcp(new LCM::Time<EvalT, AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
    p = stateMgr.registerStateVariable("Time",
        dl_->workset_scalar,
        dl_->dummy,
        eb_name,
        "scalar",
        0.0,
        true);
    ev = rcp(new PHAL::SaveStateField<EvalT, AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_mech_eq_) { // Current Coordinates
    RCP<ParameterList> p = rcp(new ParameterList("Current Coordinates"));
    p->set<std::string>("Reference Coordinates Name", "Coord Vec");
    p->set<std::string>("Displacement Name", "Displacement");
    p->set<std::string>("Current Coordinates Name", "Current Coordinates");
    ev = rcp(new LCM::CurrentCoords<EvalT, AlbanyTraits>(*p, dl_));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_temperature_eq_ || have_temperature_) {
    double temp(0.0);
    if (material_db_->isElementBlockParam(eb_name, "Initial Temperature")) {
      temp = material_db_->
       getElementBlockParam<double>(eb_name, "Initial Temperature");
    }
    RCP<ParameterList> p = rcp(new ParameterList("Save Temperature"));
    p = stateMgr.registerStateVariable(temperature,
        dl_->qp_scalar,
        dl_->dummy,
        eb_name,
        "scalar",
        temp,
        true,
        false);
    ev = rcp(new PHAL::SaveStateField<EvalT, AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_pore_pressure_eq_ || have_pore_pressure_) {
    RCP<ParameterList> p = rcp(new ParameterList("Save Pore Pressure"));
    p = stateMgr.registerStateVariable(porePressure,
        dl_->qp_scalar,
        dl_->dummy,
        eb_name,
        "scalar",
        0.0,
        true,
        false);
    ev = rcp(new PHAL::SaveStateField<EvalT, AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_transport_eq_ || have_transport_) {
    RCP<ParameterList> p = rcp(new ParameterList("Save Transport"));
    bool output_flag(true);
    if (material_db_->isElementBlockParam(eb_name, "Output IP"+transport))
      output_flag =
          material_db_->getElementBlockParam<bool>(eb_name, "Output IP"+transport);

    p = stateMgr.registerStateVariable(transport,
        dl_->qp_scalar,
        dl_->dummy,
        eb_name,
        "scalar",
        38.7, // JTO: What sort of Magic is 38.7 !?!
        true,
        output_flag);
    ev = rcp(new PHAL::SaveStateField<EvalT, AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_hydrostress_eq_ || have_hydrostress_) {
    RCP<ParameterList> p = rcp(new ParameterList("Save HydroStress"));
    p = stateMgr.registerStateVariable(hydroStress,
        dl_->qp_scalar,
        dl_->dummy,
        eb_name,
        "scalar",
        0.0,
        true,
        true);
    ev = rcp(new PHAL::SaveStateField<EvalT, AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_source_) { // Source
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<std::string>("Source Name", "Source");
    p->set<std::string>("Variable Name", "Displacement");
    p->set<RCP<DataLayout> >("QP Scalar Data Layout", dl_->qp_scalar);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Source Functions");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new PHAL::Source<EvalT, AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // Constitutive Model Parameters
    RCP<ParameterList> p = rcp(
        new ParameterList("Constitutive Model Parameters"));
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

    RCP<LCM::ConstitutiveModelParameters<EvalT, AlbanyTraits> > cmpEv =
        rcp(new LCM::ConstitutiveModelParameters<EvalT, AlbanyTraits>(*p, dl_));
    fm0.template registerEvaluator<EvalT>(cmpEv);
  }

  if (have_mech_eq_) {
    RCP<ParameterList> p = rcp(
        new ParameterList("Constitutive Model Interface"));
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

    param_list.set<RCP<std::map<std::string, std::string> > >("Name Map", fnm);
    p->set<Teuchos::ParameterList*>("Material Parameters", &param_list);
    p->set<bool>("Volume Average Pressure", volume_average_pressure);
    if (volume_average_pressure) {
      p->set<std::string>("Weights Name", "Weights");
    }

    RCP<LCM::ConstitutiveModelInterface<EvalT, AlbanyTraits> > cmiEv =
        rcp(new LCM::ConstitutiveModelInterface<EvalT, AlbanyTraits>(*p, dl_));
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
      ev = rcp(new PHAL::SaveStateField<EvalT, AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }
  }

  // Surface Element Block
  if (surface_element)
  {

    { // Surface Basis
      // SurfaceBasis_Def.hpp
      RCP<ParameterList> p = rcp(new ParameterList("Surface Basis"));

      // inputs
      p->set<std::string>("Reference Coordinates Name", "Coord Vec");
      p->set<RCP<Intrepid::Cubature<RealType> > >("Cubature", surfaceCubature);
      p->set<RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >(
          "Intrepid Basis", surfaceBasis);
      if (have_mech_eq_) {
        p->set<std::string>("Current Coordinates Name", "Current Coordinates");
      }

      // outputs
      p->set<std::string>("Reference Basis Name", "Reference Basis");
      p->set<std::string>("Reference Area Name", "Weights");
      p->set<std::string>("Reference Dual Basis Name", "Reference Dual Basis");
      p->set<std::string>("Reference Normal Name", "Reference Normal");
      p->set<std::string>("Current Basis Name", "Current Basis");

      ev = rcp(new LCM::SurfaceBasis<EvalT, AlbanyTraits>(*p, dl_));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    if (have_mech_eq_) { // Surface Jump
      //SurfaceVectorJump_Def.hpp
      RCP<ParameterList> p = rcp(new ParameterList("Surface Vector Jump"));

      // inputs
      p->set<RCP<Intrepid::Cubature<RealType> > >("Cubature", surfaceCubature);
      p->set<RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >(
          "Intrepid Basis", surfaceBasis);
      p->set<std::string>("Vector Name", "Current Coordinates");

      // outputs
      p->set<std::string>("Vector Jump Name", "Vector Jump");

      ev = rcp(new LCM::SurfaceVectorJump<EvalT, AlbanyTraits>(*p, dl_));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    if ((have_temperature_eq_ || have_pore_pressure_eq_) ||
        (have_transport_eq_)) { // Surface Temperature Jump
      //SurfaceScalarJump_Def.hpp
      RCP<ParameterList> p = rcp(new ParameterList("Surface Scalar Jump"));

      // inputs
      p->set<RCP<Intrepid::Cubature<RealType> > >("Cubature", surfaceCubature);
      p->set<RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >(
          "Intrepid Basis", surfaceBasis);
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

      ev = rcp(new LCM::SurfaceScalarJump<EvalT, AlbanyTraits>(*p, dl_));
      fm0.template registerEvaluator<EvalT>(ev);

    }

    if (have_mech_eq_) { // Surface Gradient
      //SurfaceVectorGradient_Def.hpp
      RCP<ParameterList> p = rcp(new ParameterList("Surface Vector Gradient"));

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
      p->set<RealType>("Average J Stabilization Parameter", volume_average_stabilization_param);
      p->set<RCP<Intrepid::Cubature<RealType> > >("Cubature", surfaceCubature);
      p->set<std::string>("Weights Name", "Weights");
      p->set<std::string>("Current Basis Name", "Current Basis");
      p->set<std::string>("Reference Dual Basis Name", "Reference Dual Basis");
      p->set<std::string>("Reference Normal Name", "Reference Normal");
      p->set<std::string>("Vector Jump Name", "Vector Jump");

      // outputs
      p->set<std::string>("Surface Vector Gradient Name", defgrad);
      p->set<std::string>("Surface Vector Gradient Determinant Name", J);

      ev = rcp(new LCM::SurfaceVectorGradient<EvalT, AlbanyTraits>(*p, dl_));
      fm0.template registerEvaluator<EvalT>(ev);

      // optional output
      bool output_flag(false);
      if (material_db_->isElementBlockParam(eb_name,
          "Output Deformation Gradient"))
        output_flag =
            material_db_->getElementBlockParam<bool>(eb_name,
                "Output Deformation Gradient");

      p = stateMgr.registerStateVariable(defgrad,
          dl_->qp_tensor,
          dl_->dummy,
          eb_name,
          "identity",
          1.0,
          false,
          output_flag);
      ev = rcp(new PHAL::SaveStateField<EvalT, AlbanyTraits>(*p));
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
        ev = rcp(new PHAL::SaveStateField<EvalT, AlbanyTraits>(*p));
        fm0.template registerEvaluator<EvalT>(ev);
      }
    }

    // Surface Gradient Operator
    if (have_pore_pressure_eq_) {
      //SurfaceScalarGradientOperator_Def.hpp
      RCP<ParameterList> p = rcp(
          new ParameterList("Surface Scalar Gradient Operator"));
      // inputs
      p->set<RealType>("thickness", thickness);
      p->set<RCP<Intrepid::Cubature<RealType> > >("Cubature", surfaceCubature);
      p->set<RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >(
          "Intrepid Basis", surfaceBasis);
      p->set<std::string>("Reference Dual Basis Name", "Reference Dual Basis");
      p->set<std::string>("Reference Normal Name", "Reference Normal");

      // NOTE: NOT surf_Pore_Pressure here
      // NOTE: If you need to compute gradient for more than one scalar field, that could cause troubles
      if (have_pore_pressure_eq_ == true)
        p->set<std::string>("Nodal Scalar Name", "Pore_Pressure");

      // outputs
      p->set<std::string>("Surface Scalar Gradient Operator Name",
          "Surface Scalar Gradient Operator");
      p->set<RCP<DataLayout> >("Node QP Vector Data Layout",
          dl_->node_qp_vector);
        p->set<std::string>("Surface Scalar Gradient Name",
            "Surface Pressure Gradient");
      p->set<RCP<DataLayout> >("QP Vector Data Layout", dl_->qp_vector);

      ev = rcp(
          new LCM::SurfaceScalarGradientOperator<EvalT, AlbanyTraits>(*p, dl_));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    if (have_transport_eq_) {
      //SurfaceScalarGradientOperator_Def.hpp
      RCP<ParameterList> p = rcp(
          new ParameterList("Surface Scalar Gradient Operator"));
      // inputs
      p->set<RealType>("thickness", thickness);
      p->set<RCP<Intrepid::Cubature<RealType> > >("Cubature", surfaceCubature);
      p->set<RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >(
          "Intrepid Basis", surfaceBasis);
      p->set<std::string>("Reference Dual Basis Name", "Reference Dual Basis");
      p->set<std::string>("Reference Normal Name", "Reference Normal");

      // NOTE: NOT surf_Pore_Pressure here
      // NOTE: If you need to compute gradient for more than one scalar field, that could cause troubles
        p->set<std::string>("Nodal Scalar Name", "Transport");

      // outputs
      p->set<std::string>("Surface Scalar Gradient Operator Name",
          "Surface Scalar Gradient Operator");
      p->set<RCP<DataLayout> >("Node QP Vector Data Layout",
          dl_->node_qp_vector);
      if (have_transport_eq_ == true)
        p->set<std::string>("Surface Scalar Gradient Name",
            "Surface Transport Gradient");
      p->set<RCP<DataLayout> >("QP Vector Data Layout", dl_->qp_vector);

      ev = rcp(
          new LCM::SurfaceScalarGradientOperator<EvalT, AlbanyTraits>(*p, dl_));
      fm0.template registerEvaluator<EvalT>(ev);

    }

    if (have_hydrostress_eq_) {
      //SurfaceScalarGradientOperator_Def.hpp
      RCP<ParameterList> p = rcp(
          new ParameterList("Surface Scalar Gradient Operator"));
      // inputs
      p->set<RealType>("thickness", thickness);
      p->set<RCP<Intrepid::Cubature<RealType> > >("Cubature", surfaceCubature);
      p->set<RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >(
          "Intrepid Basis", surfaceBasis);
      p->set<std::string>("Reference Dual Basis Name", "Reference Dual Basis");
      p->set<std::string>("Reference Normal Name", "Reference Normal");

      // NOTE: NOT surf_Pore_Pressure here
      // NOTE: If you need to compute gradient for more than one scalar field, that could cause troubles
      if (have_transport_eq_ == true)
        p->set<std::string>("Nodal Scalar Name", "HydroStress");

      // outputs
      p->set<std::string>("Surface Scalar Gradient Operator Name",
          "Surface Scalar Gradient Operator");
      p->set<RCP<DataLayout> >("Node QP Vector Data Layout",
          dl_->node_qp_vector);
      p->set<std::string>("Surface Scalar Gradient Name",
          "Surface HydroStress Gradient");
      p->set<RCP<DataLayout> >("QP Vector Data Layout", dl_->qp_vector);

      ev = rcp(
          new LCM::SurfaceScalarGradientOperator<EvalT, AlbanyTraits>(*p, dl_));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    {
      if (have_mech_eq_) { // Surface Residual
        // SurfaceVectorResidual_Def.hpp
        RCP<ParameterList> p = rcp(
            new ParameterList("Surface Vector Residual"));

        // inputs
        p->set<RealType>("thickness", thickness);
        p->set<RCP<Intrepid::Cubature<RealType> > >("Cubature",
            surfaceCubature);
        p->set<RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >(
            "Intrepid Basis", surfaceBasis);

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

        ev = rcp(new LCM::SurfaceVectorResidual<EvalT, AlbanyTraits>(*p, dl_));
        fm0.template registerEvaluator<EvalT>(ev);
      }
    } // end of coehesive/surface element block
  } else {

    if (have_mech_eq_) { // Kinematics quantities
      RCP<ParameterList> p = rcp(new ParameterList("Kinematics"));

      p->set<bool>("Weighted Volume Average J", volume_average_j);
      p->set<RealType>("Average J Stabilization Parameter", volume_average_stabilization_param);

      // strain
      if (small_strain) {
          p->set<std::string>("Strain Name", "Strain");
      }

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
      p->set<RCP<DataLayout> >("QP Scalar Data Layout", dl_->qp_scalar);
      p->set<std::string>("Gradient QP Variable Name", "Displacement Gradient");
      p->set<RCP<DataLayout> >("QP Tensor Data Layout", dl_->qp_tensor);

      //Outputs: F, J
      p->set<std::string>("DefGrad Name", defgrad); //dl_->qp_tensor also
      p->set<std::string>("DetDefGrad Name", J);
      p->set<RCP<DataLayout> >("QP Scalar Data Layout", dl_->qp_scalar);

      //ev = rcp(new LCM::DefGrad<EvalT,AlbanyTraits>(*p));
      ev = rcp(new LCM::Kinematics<EvalT, AlbanyTraits>(*p, dl_));
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
        ev = rcp(new PHAL::SaveStateField<EvalT, AlbanyTraits>(*p));
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
        ev = rcp(new PHAL::SaveStateField<EvalT, AlbanyTraits>(*p));
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
          ev = rcp(new PHAL::SaveStateField<EvalT, AlbanyTraits>(*p));
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
          ev = rcp(new PHAL::SaveStateField<EvalT, AlbanyTraits>(*p));
          fm0.template registerEvaluator<EvalT>(ev);
        }
      }
    }
    if (have_mech_eq_)
    { // Residual
      RCP<ParameterList> p = rcp(new ParameterList("Displacement Residual"));
      //Input
      p->set<std::string>("Stress Name", firstPK);
      p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
      p->set<std::string>("Weighted BF Name", "wBF");
      p->set<std::string>("Acceleration Name", "Acceleration");

      p->set<RCP<ParamLib> >("Parameter Library", paramLib);
      //Output
      p->set<std::string>("Residual Name", "Displacement Residual");
      ev = rcp(new LCM::MechanicsResidual<EvalT, AlbanyTraits>(*p, dl_));
      fm0.template registerEvaluator<EvalT>(ev);
    }
  }


  if (have_mech_eq_) {
    // convert Cauchy stress to first Piola-Kirchhoff
    RCP<ParameterList> p = rcp(new ParameterList("First PK Stress"));
    //Input
    p->set<std::string>("Stress Name", cauchy);
    p->set<std::string>("DefGrad Name", defgrad);

    // Effective stress theory for poromechanics problem
    if (have_pore_pressure_eq_) {
      p->set<bool>("Have Pore Pressure", true);
      p->set<std::string>("Pore Pressure Name", porePressure);
      p->set<std::string>("Biot Coefficient Name", biotCoeff);
    }

    if (small_strain) {
      p->set<bool>("Small Strain", true);
    }

    //Output
    p->set<std::string>("First PK Stress Name", firstPK);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);

    ev = rcp(new LCM::FirstPK<EvalT, AlbanyTraits>(*p, dl_));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // Element length in the direction of solution gradient
  if ((have_stab_pressure_eq_ || have_pore_pressure_eq_ || have_transport_eq_)) {
    RCP<ParameterList> p = rcp(new ParameterList("Gradient_Element_Length"));
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

    ev = rcp(new LCM::GradientElementLength<EvalT, AlbanyTraits>(*p, dl_));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_pore_pressure_eq_) {  // Porosity
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<std::string>("Porosity Name", porosity);
    p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");
    // Setting this turns on dependence of strain and pore pressure)
    //p->set<std::string>("Strain Name", "Strain");
    if (have_mech_eq_) p->set<std::string>("DetDefGrad Name", J);
    // porosity update based on Coussy's poromechanics (see p.79)
    p->set<std::string>("QP Pore Pressure Name", porePressure);
    p->set<std::string>("Biot Coefficient Name", biotCoeff);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList =
        material_db_->getElementBlockSublist(eb_name, "Porosity");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new LCM::Porosity<EvalT, AlbanyTraits>(*p, dl_));
    fm0.template registerEvaluator<EvalT>(ev);

    // Output Porosity
    bool output_flag(false);
    if (material_db_->isElementBlockParam(eb_name, "Output "+porosity))
      output_flag =
        material_db_->getElementBlockParam<bool>(eb_name, "Output "+porosity);
    if (output_flag) {
      p = stateMgr.registerStateVariable(porosity,
          dl_->qp_scalar,
          dl_->dummy,
          eb_name,
          "scalar",
          0.5, // This is really bad practice. It needs to be fixed
          false,
          true);
      ev = rcp(new PHAL::SaveStateField<EvalT, AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }
  }

  if (have_pore_pressure_eq_) { // Biot Coefficient
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<std::string>("Biot Coefficient Name", biotCoeff);
    p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");
    p->set<RCP<DataLayout> >("Node Data Layout", dl_->node_scalar);
    p->set<RCP<DataLayout> >("QP Scalar Data Layout", dl_->qp_scalar);
    p->set<RCP<DataLayout> >("QP Vector Data Layout", dl_->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList =
        material_db_->getElementBlockSublist(eb_name, "Biot Coefficient");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = rcp(new LCM::BiotCoefficient<EvalT, AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_pore_pressure_eq_) { // Biot Modulus
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<std::string>("Biot Modulus Name", biotModulus);
    p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");
    p->set<RCP<DataLayout> >("Node Data Layout", dl_->node_scalar);
    p->set<RCP<DataLayout> >("QP Scalar Data Layout", dl_->qp_scalar);
    p->set<RCP<DataLayout> >("QP Vector Data Layout", dl_->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList =
        material_db_->getElementBlockSublist(eb_name, "Biot Modulus");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    // Setting this turns on linear dependence on porosity and Biot's coeffcient
    p->set<std::string>("Porosity Name", porosity);
    p->set<std::string>("Biot Coefficient Name", biotCoeff);

    ev = rcp(new LCM::BiotModulus<EvalT, AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_pore_pressure_eq_) { // Kozeny-Carman Permeaiblity
    RCP<ParameterList> p = rcp(new ParameterList);

    p->set<std::string>("Kozeny-Carman Permeability Name", kcPerm);
    p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");
    p->set<RCP<DataLayout> >("Node Data Layout", dl_->node_scalar);
    p->set<RCP<DataLayout> >("QP Scalar Data Layout", dl_->qp_scalar);
    p->set<RCP<DataLayout> >("QP Vector Data Layout", dl_->qp_vector);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList =
        material_db_->getElementBlockSublist(eb_name,
            "Kozeny-Carman Permeability");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    // Setting this turns on Kozeny-Carman relation
    p->set<std::string>("Porosity Name", porosity);
    p->set<RCP<DataLayout> >("QP Scalar Data Layout", dl_->qp_scalar);

    ev = rcp(new LCM::KCPermeability<EvalT, AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    // Output
    bool output_flag(false);
    if (material_db_->isElementBlockParam(eb_name, "Output "+kcPerm))
      output_flag =
        material_db_->getElementBlockParam<bool>(eb_name, "Output "+kcPerm);
    if (output_flag) {
      p = stateMgr.registerStateVariable(kcPerm,
          dl_->qp_scalar,
          dl_->dummy,
          eb_name,
          "scalar",
          0.0,
          false,
          true);
      ev = rcp(new PHAL::SaveStateField<EvalT, AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }
  }

  // Pore Pressure Residual (Bulk Element)
  if (have_pore_pressure_eq_ && !surface_element) {
    RCP<ParameterList> p = rcp(new ParameterList("Pore_Pressure Residual"));

    //Input

    // Input from nodal points, basis function stuff
    p->set<std::string>("Weights Name", "Weights");
    p->set<std::string>("Weighted BF Name", "wBF");
    p->set<RCP<DataLayout> >("Node QP Scalar Data Layout",
        dl_->node_qp_scalar);
    p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
    p->set<RCP<DataLayout> >("Node QP Vector Data Layout", dl_->node_qp_vector);

    // Inputs: X, Y at nodes, Cubature, and Basis
    p->set<std::string>("Coordinate Vector Name", "Coord Vec");
    p->set<RCP<DataLayout> >("Coordinate Data Layout", dl_->vertices_vector);
    p->set<RCP<Intrepid::Cubature<RealType> > >("Cubature", cubature);
    p->set<RCP<shards::CellTopology> >("Cell Type", cellType);

    // DT for  time integration
    p->set<std::string>("Delta Time Name", "Delta Time");
    p->set<RCP<DataLayout> >("Workset Scalar Data Layout", dl_->workset_scalar);

    p->set<bool>("Have Source", false);
    p->set<std::string>("Source Name", "Source");
    p->set<bool>("Have Absorption", false);

    // Input from cubature points
    p->set<std::string>("Element Length Name", gradient_element_length);
    p->set<std::string>("QP Pore Pressure Name", porePressure);
    p->set<std::string>("QP Time Derivative Variable Name", porePressure);
    p->set<RCP<DataLayout> >("QP Scalar Data Layout", dl_->qp_scalar);

    //p->set<std::string>("Material Property Name", "Stabilization Parameter");
    p->set<std::string>("Porosity Name", "Porosity");
    p->set<std::string>("Thermal Conductivity Name", "Thermal Conductivity");
    p->set<std::string>("Kozeny-Carman Permeability Name", kcPerm);
    p->set<std::string>("Biot Coefficient Name", biotCoeff);
    p->set<std::string>("Biot Modulus Name", biotModulus);

    p->set<std::string>("Gradient QP Variable Name", "Pore_Pressure Gradient");
    p->set<RCP<DataLayout> >("QP Vector Data Layout", dl_->qp_vector);

    if (have_mech_eq_) {
      p->set<bool>("Have Mechanics", true);
      p->set<std::string>("DefGrad Name", defgrad);
      p->set<RCP<DataLayout> >("QP Tensor Data Layout", dl_->qp_tensor);
      p->set<std::string>("DetDefGrad Name", J);
      p->set<RCP<DataLayout> >("QP Scalar Data Layout", dl_->qp_scalar);
    }
    RealType stab_param(0.0);
    if (material_db_->isElementBlockParam(eb_name, "Stabilization Parameter")) {
      stab_param =
          material_db_->getElementBlockParam<RealType>(eb_name,
              "Stabilization Parameter");
    }

    p->set<RealType>("Stabilization Parameter", stab_param);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);

    //Output
    p->set<std::string>("Residual Name", "Pore_Pressure Residual");
    p->set<RCP<DataLayout> >("Node Scalar Data Layout", dl_->node_scalar);

    ev = rcp(new LCM::TLPoroPlasticityResidMass<EvalT, AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    // Output QP pore pressure
    bool output_flag(false);
    if (material_db_->isElementBlockParam(eb_name, "Output IP"+porePressure))
      output_flag =
        material_db_->getElementBlockParam<bool>(eb_name, "Output IP"+porePressure);
    p = stateMgr.registerStateVariable(porePressure,
        dl_->qp_scalar,
        dl_->dummy,
        eb_name,
        "scalar",
        0.0,
        true,
        output_flag);
    ev = rcp(new PHAL::SaveStateField<EvalT, AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_pore_pressure_eq_ && surface_element) { // Pore Pressure Resid for Surface
    RCP<ParameterList> p = rcp(new ParameterList("Pore_Pressure Residual"));

    //Input
    p->set<RealType>("thickness", thickness);
    p->set<RCP<Intrepid::Cubature<RealType> > >("Cubature", surfaceCubature);
    p->set<RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >(
        "Intrepid Basis", surfaceBasis);
    p->set<std::string>("Surface Scalar Gradient Operator Name",
        "Surface Scalar Gradient Operator");
    p->set<std::string>("Scalar Gradient Name", "Surface Pressure Gradient");
    p->set<std::string>("Current Basis Name", "Current Basis");
    p->set<std::string>("Reference Dual Basis Name", "Reference Dual Basis");
    p->set<std::string>("Reference Normal Name", "Reference Normal");
    p->set<std::string>("Reference Area Name", "Weights");
    p->set<std::string>("Pore Pressure Name", porePressure);
    p->set<std::string>("Nodal Pore Pressure Name", "Pore_Pressure"); // NOTE: NOT surf_Pore_Pressure here
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
    p->set<RCP<DataLayout> >("Node Scalar Data Layout", dl_->node_scalar);

    ev = rcp(new LCM::SurfaceTLPoroMassResidual<EvalT, AlbanyTraits>(*p, dl_));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_transport_eq_ || have_transport_) { // Transport Coefficients
    RCP<ParameterList> p = rcp(new ParameterList("Transport Coefficients"));

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
    if (material_model_name == "J2") {
      p->set<std::string>("Equivalent Plastic Strain Name", eqps);
    }

    p->set<bool>("Weighted Volume Average J", volume_average_j);
    p->set<RealType>("Average J Stabilization Parameter", volume_average_stabilization_param);

    //Output
    p->set<std::string>("Trapped Concentration Name", trappedConcentration);
    p->set<std::string>("Mechanical Deformation Gradient Name", trappedConcentration);
    p->set<std::string>("Total Concentration Name", totalConcentration);
    p->set<std::string>("Mechanical Deformation Gradient Name", "Fm");
    p->set<std::string>("Effective Diffusivity Name", effectiveDiffusivity);
    p->set<std::string>("Trapped Solvent Name", trappedSolvent);
    if (material_model_name == "J2") {
       p->set<std::string>("Strain Rate Factor Name", strainRateFactor);
    }
    p->set<std::string>("Diffusion Coefficient Name", diffusionCoefficient);
    p->set<std::string>("Tau Contribution Name", convectionCoefficient);
    p->set<std::string>("Concentration Equilibrium Parameter Name",
        eqilibriumParameter);

    ev = rcp(new LCM::TransportCoefficients<EvalT, AlbanyTraits>(*p, dl_));
    fm0.template registerEvaluator<EvalT>(ev);

    bool output_flag(false);
    // Trapped Concentration
    if (material_db_->isElementBlockParam(eb_name, "Output "+trappedConcentration))
      output_flag =
        material_db_->getElementBlockParam<bool>(eb_name, "Output "+trappedConcentration);
    if (output_flag) {
      p = stateMgr.registerStateVariable(trappedConcentration, dl_->qp_scalar,
          dl_->dummy, eb_name, "scalar", 0.0, false, output_flag);
      ev = rcp(new PHAL::SaveStateField<EvalT, AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    // Strain Rate Factor
    output_flag = false;
    if (material_db_->isElementBlockParam(eb_name, "Output "+strainRateFactor))
      output_flag =
          material_db_->getElementBlockParam<bool>(eb_name, "Output "+strainRateFactor);
    if (output_flag) {
      p = stateMgr.registerStateVariable(strainRateFactor, dl_->qp_scalar,
          dl_->dummy, eb_name, "scalar", 0.0, false, output_flag);
      ev = rcp(new PHAL::SaveStateField<EvalT, AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    // Convection Coefficient
    output_flag = false;
    if (material_db_->isElementBlockParam(eb_name, "Output "+convectionCoefficient))
      output_flag =
          material_db_->getElementBlockParam<bool>(eb_name, "Output "+convectionCoefficient);
    if (output_flag) {
      p = stateMgr.registerStateVariable(convectionCoefficient, dl_->qp_scalar,
          dl_->dummy, eb_name, "scalar", 0.0, false, output_flag);
      ev = rcp(new PHAL::SaveStateField<EvalT, AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    // Diffusion Coefficient
    output_flag = false;
    if (material_db_->isElementBlockParam(eb_name, "Output "+diffusionCoefficient))
      output_flag =
          material_db_->getElementBlockParam<bool>(eb_name, "Output "+diffusionCoefficient);
    if (output_flag) {
      p = stateMgr.registerStateVariable(diffusionCoefficient, dl_->qp_scalar,
          dl_->dummy, eb_name,"scalar", 1.0, false, output_flag);
      ev = rcp(new PHAL::SaveStateField<EvalT, AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    // Effective Diffusivity
    output_flag = false;
    if (material_db_->isElementBlockParam(eb_name, "Output "+effectiveDiffusivity))
      output_flag =
          material_db_->getElementBlockParam<bool>(eb_name, "Output "+effectiveDiffusivity);
    if (output_flag) {
      p = stateMgr.registerStateVariable(effectiveDiffusivity, dl_->qp_scalar,
          dl_->dummy, eb_name,"scalar", 1.0, false, output_flag);
      ev = rcp(new PHAL::SaveStateField<EvalT, AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }
  }

  // Transport of the temperature field
  if (have_temperature_eq_ && !surface_element)
  {
    RCP<ParameterList> p = rcp(
        new ParameterList("ThermoMechanical Coefficients"));

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

    if (have_mech_eq_) {
       p->set<bool>("Have Mechanics", true);
       p->set<std::string>("Deformation Gradient Name", defgrad);
    }

    // Output
    p->set<std::string>("Thermal Diffusivity Name", "Thermal Diffusivity");
    p->set<std::string>("Temperature Dot Name", "Temperature Dot");

    ev = rcp(
        new LCM::ThermoMechanicalCoefficients<EvalT, AlbanyTraits>(*p, dl_));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // Transport of the temperature field
  if (have_temperature_eq_ && !surface_element)
  {
    RCP<ParameterList> p = rcp(new ParameterList("Temperature Residual"));

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

    // Output
    p->set<std::string>("Residual Name", "Temperature Residual");

    ev = rcp(new LCM::TransportResidual<EvalT, AlbanyTraits>(*p, dl_));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // Hydrogen Transport model proposed in Foulk et al 2014
  if (have_transport_eq_ && !surface_element) {
    RCP<ParameterList> p = rcp(new ParameterList("Transport Residual"));

    //Input
    p->set<std::string>("Element Length Name", gradient_element_length);
    p->set<RCP<DataLayout> >("QP Scalar Data Layout", dl_->qp_scalar);

    p->set<std::string>("Weighted BF Name", "wBF");
    p->set<RCP<DataLayout> >("Node QP Scalar Data Layout", dl_->node_qp_scalar);

    p->set<std::string>("Weights Name", "Weights");
    p->set<RCP<DataLayout> >("QP Scalar Data Layout", dl_->qp_scalar);

    p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
    p->set<RCP<DataLayout> >("Node QP Vector Data Layout", dl_->node_qp_vector);

    p->set<std::string>("Gradient BF Name", "Grad BF");
    p->set<RCP<DataLayout> >("Node QP Vector Data Layout", dl_->node_qp_vector);

    if (have_mech_eq_) {
    	p->set<std::string>("eqps Name", eqps);
    	p->set<RCP<DataLayout> >("QP Scalar Data Layout", dl_->qp_scalar);

    	p->set<std::string>("Strain Rate Factor Name", strainRateFactor);
    	p->set<RCP<DataLayout> >("QP Scalar Data Layout", dl_->qp_scalar);

        p->set<std::string>("Tau Contribution Name", convectionCoefficient);
        p->set<RCP<DataLayout> >("QP Scalar Data Layout", dl_->qp_scalar);
    }

    p->set<std::string>("Trapped Concentration Name", trappedConcentration);
    p->set<RCP<DataLayout> >("QP Scalar Data Layout", dl_->qp_scalar);

    p->set<std::string>("Trapped Solvent Name", trappedSolvent);
    p->set<RCP<DataLayout> >("QP Scalar Data Layout", dl_->qp_scalar);

    p->set<std::string>("Deformation Gradient Name", defgrad);
    p->set<RCP<DataLayout> >("QP Tensor Data Layout", dl_->qp_tensor);

    p->set<std::string>("Effective Diffusivity Name", effectiveDiffusivity);
    p->set<RCP<DataLayout> >("QP Scalar Data Layout", dl_->qp_scalar);

    p->set<std::string>("Diffusion Coefficient Name", diffusionCoefficient);
    p->set<RCP<DataLayout> >("QP Scalar Data Layout", dl_->qp_scalar);

    p->set<std::string>("QP Variable Name", "Transport");
    p->set<RCP<DataLayout> >("QP Scalar Data Layout", dl_->qp_scalar);

    p->set<std::string>("Gradient QP Variable Name", "Transport Gradient");
    p->set<RCP<DataLayout> >("QP Vector Data Layout", dl_->qp_vector);

    p->set<std::string>("Gradient Hydrostatic Stress Name",
        "HydroStress Gradient");
    p->set<RCP<DataLayout> >("QP Vector Data Layout", dl_->qp_vector);

    p->set<std::string>("Stress Name", cauchy);
    p->set<RCP<DataLayout> >("QP Tensor Data Layout", dl_->qp_tensor);

    p->set<std::string>("Delta Time Name", "Delta Time");
    p->set<RCP<DataLayout> >("Workset Scalar Data Layout", dl_->workset_scalar);

    RealType stab_param(0.0);
    if (material_db_->isElementBlockParam(eb_name, "Stabilization Parameter")) {
      stab_param =
          material_db_->getElementBlockParam<RealType>(eb_name,
              "Stabilization Parameter");
    }
    p->set<RealType>("Stabilization Parameter", stab_param);
    p->set<RCP<ParamLib> >("Parameter Library", paramLib);

    //Output
    p->set<std::string>("Residual Name", "Transport Residual");
    p->set<RCP<DataLayout> >("Node Scalar Data Layout", dl_->node_scalar);

    ev = rcp(
        new LCM::HDiffusionDeformationMatterResidual<EvalT, AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

  }

  if (have_transport_eq_ && surface_element) { // Transport Resid for Surface
    RCP<ParameterList> p = rcp(new ParameterList("Transport Residual"));

    //Input
    p->set<RealType>("thickness", thickness);
    p->set<RCP<Intrepid::Cubature<RealType> > >("Cubature", surfaceCubature);
    p->set<RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >(
        "Intrepid Basis", surfaceBasis);
    p->set<std::string>("Surface Scalar Gradient Operator Name",
        "Surface Scalar Gradient Operator");
    p->set<std::string>("Surface Transport Gradient Name",
        "Surface Transport Gradient");
    p->set<std::string>("Current Basis Name", "Current Basis");
    p->set<std::string>("Reference Dual Basis Name", "Reference Dual Basis");
    p->set<std::string>("Reference Normal Name", "Reference Normal");
    p->set<std::string>("Reference Area Name", "Weights");
    p->set<std::string>("Transport Name", transport);
    p->set<std::string>("Nodal Transport Name", "Transport"); // NOTE: NOT surf_Transport here
    p->set<std::string>("Diffusion Coefficient Name", diffusionCoefficient);
    p->set<std::string>("Effective Diffusivity Name", effectiveDiffusivity);
    p->set<std::string>("Tau Contribution Name", convectionCoefficient);
    p->set<std::string>("Strain Rate Factor Name", strainRateFactor);
    p->set<std::string>("Element Length Name", effectiveDiffusivity); // This does not make sense
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
    p->set<RCP<ParamLib> >("Parameter Library", paramLib);

    //Output
    p->set<std::string>("Residual Name", "Transport Residual");
    p->set<RCP<DataLayout> >("Node Scalar Data Layout", dl_->node_scalar);

    ev = rcp(new LCM::SurfaceHDiffusionDefResidual<EvalT, AlbanyTraits>(*p, dl_));
    fm0.template registerEvaluator<EvalT>(ev);

  }


  if (have_hydrostress_eq_ && !surface_element) { // L2 hydrostatic stress projection
    RCP<ParameterList> p = rcp(new ParameterList("HydroStress Residual"));

    //Input
    p->set<std::string>("Weighted BF Name", "wBF");
    p->set<RCP<DataLayout> >
    ("Node QP Scalar Data Layout", dl_->node_qp_scalar);

    p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
    p->set<RCP<DataLayout> >
    ("Node QP Vector Data Layout", dl_->node_qp_vector);

    p->set<bool>("Have Source", false);
    p->set<std::string>("Source Name", "Source");

    p->set<std::string>("Deformation Gradient Name", defgrad);
    p->set<RCP<DataLayout> >("QP Tensor Data Layout", dl_->qp_tensor);

    p->set<std::string>("QP Variable Name", hydroStress);
    p->set<RCP<DataLayout> >("QP Scalar Data Layout", dl_->qp_scalar);

    p->set<std::string>("Stress Name", cauchy);
    p->set<RCP<DataLayout> >("QP Tensor Data Layout", dl_->qp_tensor);

    //Output
    p->set<std::string>("Residual Name", "HydroStress Residual");
    p->set<RCP<DataLayout> >("Node Scalar Data Layout", dl_->node_scalar);

    ev = rcp(new LCM::ScalarL2ProjectionResidual<EvalT, AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_hydrostress_eq_ && surface_element) { // Hydrostress Projection Resid for Surface
    RCP<ParameterList> p = rcp(new ParameterList("HydroStress Residual"));

    //Input
    p->set<RealType>("thickness", thickness);
    p->set<RCP<Intrepid::Cubature<RealType> > >("Cubature", surfaceCubature);
    p->set<RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >(
        "Intrepid Basis", surfaceBasis);
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
    p->set<RCP<DataLayout> >("Node Scalar Data Layout", dl_->node_scalar);

    ev = rcp(
        new LCM::SurfaceL2ProjectionResidual<EvalT, AlbanyTraits>(*p, dl_));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_stab_pressure_eq_) {
    RCP<ParameterList> p = rcp(new ParameterList("Stabilized Pressure Residual"));
    //Input
    p->set<std::string>("Shear Modulus Name", "Shear Modulus");
    p->set<std::string>("Bulk Modulus Name", "Bulk Modulus");
    p->set<std::string>("Deformation Gradient Name", defgrad);
    p->set<std::string>("Stress Name", cauchy);
    p->set<std::string>("Pressure Name", pressure);
    p->set<std::string>("Pressure Gradient Name", "Pressure Gradient");
    p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
    p->set<std::string>("Weighted BF Name", "wBF");
    p->set<std::string>("Element Characteristic Length Name", gradient_element_length);
    RealType stab_param(0.0);
    if (material_db_->isElementBlockParam(eb_name, "Stabilization Parameter")) {
      stab_param =
          material_db_->getElementBlockParam<RealType>(eb_name,
              "Stabilization Parameter");
    }
    p->set<RealType>("Stabilization Parameter", stab_param);

    //Output
    p->set<std::string>("Residual Name", "Pressure Residual");
    ev = rcp(new LCM::StabilizedPressureResidual<EvalT, AlbanyTraits>(*p, dl_));
    fm0.template registerEvaluator<EvalT>(ev);
  }

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
    return respUtils.constructResponses(fm0, *responseList, pFromProb, stateMgr);

  }

  return Teuchos::null;
}

#endif
