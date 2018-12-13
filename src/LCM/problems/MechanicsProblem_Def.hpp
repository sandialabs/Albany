//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_EvaluatorUtils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_ResponseUtilities.hpp"
#include "Albany_Utils.hpp"

#include "PHAL_NSMaterialProperty.hpp"
#include "PHAL_SaveStateField.hpp"
#include "PHAL_Source.hpp"

#include "FieldNameMap.hpp"

#include "ACETemperatureResidual.hpp"
#include "AnalyticMassResidual.hpp"
#include "BodyForce.hpp"
#include "CurrentCoords.hpp"
#include "MechanicsResidual.hpp"
#include "SurfaceBasis.hpp"
//#include "SurfaceScalarGradientOperator.hpp"
#include "SurfaceScalarGradientOperatorPorePressure.hpp"
#include "SurfaceScalarGradientOperatorTransport.hpp"
#include "SurfaceScalarGradientOperatorHydroStress.hpp"
#include "SurfaceScalarJump.hpp"
#include "SurfaceVectorGradient.hpp"
#include "SurfaceVectorJump.hpp"
#include "SurfaceVectorResidual.hpp"
#include "Time.hpp"
//#include "TvergaardHutchinson.hpp"
#include "MeshSizeField.hpp"
//#include "SurfaceCohesiveResidual.hpp"

// Constitutive Model Interface and parameters
#include "ConstitutiveModelInterface.hpp"
#include "ConstitutiveModelParameters.hpp"
#include "FirstPK.hpp"
#include "Kinematics.hpp"

// Generic Transport Residual
#include "TransportResidual.hpp"

// Thermomechanics specific evaluators
#include "ThermoMechanicalCoefficients.hpp"

// Poromechanics specific evaluators
#include "BiotCoefficient.hpp"
#include "BiotModulus.hpp"
#include "GradientElementLength.hpp"
#include "KCPermeability.hpp"
#include "Porosity.hpp"
#include "SurfaceTLPoroMassResidual.hpp"
#include "TLPoroPlasticityResidMass.hpp"
#include "TLPoroStress.hpp"

// Thermohydromechanics specific evaluators
#include "MixtureSpecificHeat.hpp"
#include "MixtureThermalExpansion.hpp"
#include "ThermoPoroPlasticityResidEnergy.hpp"
#include "ThermoPoroPlasticityResidMass.hpp"

// Hydrogen transport specific evaluators
#include "HDiffusionDeformationMatterResidual.hpp"
#include "LatticeDefGrad.hpp"
#include "ScalarL2ProjectionResidual.hpp"
#include "SurfaceHDiffusionDefResidual.hpp"
#include "SurfaceL2ProjectionResidual.hpp"
#include "TransportCoefficients.hpp"

// Helium bubble specific evaluators
#include "HeliumODEs.hpp"

// Damage equation specific evaluators
#include "DamageCoefficients.hpp"

// Damage equation specific evaluators
#include "StabilizedPressureResidual.hpp"

#ifdef ALBANY_CONTACT
// Contact evaluator
#include "PHAL_MortarContactResidual.hpp"
#endif

namespace Albany {

///
/// Public methods for MechanicsProblem class
///

Teuchos::Array<Teuchos::RCP<const PHX::FieldTag>>
MechanicsProblem::buildEvaluators(
    PHX::FieldManager<PHAL::AlbanyTraits>&      fm0,
    MeshSpecsStruct const&                      meshSpecs,
    StateManager&                               stateMgr,
    FieldManagerChoice                          fmchoice,
    Teuchos::RCP<Teuchos::ParameterList> const& responseList)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  ConstructEvaluatorsOp<MechanicsProblem> op(
      *this, fm0, meshSpecs, stateMgr, fmchoice, responseList);

  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);

  return *op.tags;
}

//------------------------------------------------------------------------------

Teuchos::RCP<const Teuchos::ParameterList>
MechanicsProblem::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
      this->getGenericProblemParams("ValidMechanicsProblemParams");

  validPL->set<bool>(
      "Register dirichlet_field", true, "Flag to register dirichlet_field");

  validPL->set<std::string>(
      "MaterialDB Filename",
      "materials.xml",
      "Filename of material database xml file");

  for (const std::string& variable : variables_problem_) {
    validPL->sublist(variable, false, "");
  }

  for (const std::string& variable : variables_auxiliary_) {
    validPL->sublist(variable, false, "");
  }

  return validPL;
}

//------------------------------------------------------------------------------

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
MechanicsProblem::constructEvaluators(
    PHX::FieldManager<PHAL::AlbanyTraits>&      fm0,
    MeshSpecsStruct const&                      meshSpecs,
    StateManager&                               stateMgr,
    FieldManagerChoice                          fieldManagerChoice,
    Teuchos::RCP<Teuchos::ParameterList> const& responseList)
{
  using Intrepid2Basis =
      typename Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>>;

  // Collect problem-specific response parameters
  Teuchos::RCP<Teuchos::ParameterList> pFromProb = Teuchos::rcp(
      new Teuchos::ParameterList("Response Parameters from Problem"));

  // get the name of the current element block
  std::string const eb_name = meshSpecs.ebName;

  std::string const matName =
      material_db_->getElementBlockParam<std::string>(eb_name, "material");

  Teuchos::ParameterList& param_list =
      material_db_->getElementBlockSublist(eb_name, matName);

  // get the name of the material model to be used (and make sure there is one)
  std::string const material_model_name =
      param_list.sublist("Material Model").get<std::string>("Model Name");

  TEUCHOS_TEST_FOR_EXCEPTION(
      material_model_name.length() == 0,
      std::logic_error,
      "A material model must be defined for block: " + eb_name);

#ifdef ALBANY_VERBOSE
  *out << "In MechanicsProblem::constructEvaluators" << std::endl;
  *out << "element block name: " << eb_name << std::endl;
  *out << "material model name: " << material_model_name << std::endl;
#endif

  // insert user-defined NOX Status Test for material models that use it
  if (material_model_name == "CrystalPlasticity") {
    Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag> statusTest =
        Teuchos::rcp_dynamic_cast<NOX::StatusTest::ModelEvaluatorFlag>(
            nox_status_test_);

    param_list.set<Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag>>(
        "NOX Status Test", statusTest);
  }

  // volume averaging flags
  bool const volume_average_j = material_db_->getElementBlockParam<bool>(
      eb_name, "Weighted Volume Average J", false);

  bool const volume_average_pressure = material_db_->getElementBlockParam<bool>(
      eb_name, "Volume Average Pressure", false);

  RealType const volume_average_stabilization_param =
      material_db_->getElementBlockParam<RealType>(
          eb_name, "Average J Stabilization Parameter", 0.0);

  // Check if we are setting the composite tet flag
  composite_ = material_db_->getElementBlockParam<bool>(
      eb_name, "Use Composite Tet 10", false);

  pFromProb->set<bool>("Use Composite Tet 10", composite_);

  // set flag for small strain option
  bool small_strain{material_model_name == "Linear Elastic"};

  if (material_db_->isElementBlockParam(eb_name, "Strain Flag")) {
    small_strain = true;
  }

  // Surface element checking
  bool const surface_element = material_db_->getElementBlockParam<bool>(
      eb_name, "Surface Element", false);

  bool const cohesive_element = material_db_->getElementBlockParam<bool>(
      eb_name, "Cohesive Element", false);

  RealType thickness{0.0};

  if (surface_element) {
    thickness = material_db_->getElementBlockParam<RealType>(
        eb_name, "Localization thickness parameter", 0.1);
  }

  bool const compute_membrane_forces = material_db_->getElementBlockParam<bool>(
      eb_name, "Compute Membrane Forces", false);

  // FIXME: really need to check for WEDGE_12 topologies
  TEUCHOS_TEST_FOR_EXCEPTION(
      composite_ && surface_element,
      std::logic_error,
      "Surface elements are not yet supported with the composite tet");

  // Get the intrepid basis for the given cell topology
  Intrepid2Basis intrepidBasis = getIntrepid2Basis(meshSpecs.ctd, composite_);

  // define cell topologies
  Teuchos::RCP<shards::CellTopology> const cellType =
      Teuchos::rcp(new shards::CellTopology(
          composite_ && meshSpecs.ctd.dimension == 3 &&
                  meshSpecs.ctd.node_count == 10 ?
              shards::getCellTopologyData<shards::Tetrahedron<11>>() :
              &meshSpecs.ctd));

  Intrepid2::DefaultCubatureFactory cubFactory;

  Teuchos::RCP<Intrepid2::Cubature<PHX::Device>> cubature =
      cubFactory.create<PHX::Device, RealType, RealType>(
          *cellType, meshSpecs.cubatureDegree);

  if (composite_) {
    ALBANY_ASSERT(
        meshSpecs.cubatureDegree < 4,
        "\n Cannot use Composite Tet 10 elements + Cubature Degree > 3!  You "
        "have "
            << " specified Cubature Degree = " << meshSpecs.cubatureDegree
            << ".\n");
  }

  // TODO: this could probably go into the ProblemUtils
  // just like the call to getIntrepid2Basis
  Intrepid2Basis surfaceBasis;

  Teuchos::RCP<shards::CellTopology> surfaceTopology;

  Teuchos::RCP<Intrepid2::Cubature<PHX::Device>> surfaceCubature;

  if (surface_element) {
#ifdef ALBANY_VERBOSE
    *out << "In Surface Element Logic" << std::endl;
#endif

    std::string name = meshSpecs.ctd.name;

    if (name == "Triangle_3" || name == "Quadrilateral_4") {
      surfaceBasis =
          Teuchos::rcp(new Intrepid2::Basis_HGRAD_LINE_C1_FEM<PHX::Device>());
      surfaceTopology = Teuchos::rcp(new shards::CellTopology(
          shards::getCellTopologyData<shards::Line<2>>()));
      surfaceCubature = cubFactory.create<PHX::Device>(
          *surfaceTopology, meshSpecs.cubatureDegree);
    } else if (name == "Wedge_6") {
      surfaceBasis =
          Teuchos::rcp(new Intrepid2::Basis_HGRAD_TRI_C1_FEM<PHX::Device>());
      surfaceTopology = Teuchos::rcp(new shards::CellTopology(
          shards::getCellTopologyData<shards::Triangle<3>>()));
      surfaceCubature = cubFactory.create<PHX::Device>(
          *surfaceTopology, meshSpecs.cubatureDegree);
    } else if (name == "Hexahedron_8") {
      surfaceBasis =
          Teuchos::rcp(new Intrepid2::Basis_HGRAD_QUAD_C1_FEM<PHX::Device>());
      surfaceTopology = Teuchos::rcp(new shards::CellTopology(
          shards::getCellTopologyData<shards::Quadrilateral<4>>()));
      surfaceCubature = cubFactory.create<PHX::Device>(
          *surfaceTopology, meshSpecs.cubatureDegree);
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

  int const workset_size = meshSpecs.worksetSize;

#ifdef ALBANY_VERBOSE
  *out << "Setting num_pts_, surface element is " << surface_element
       << std::endl;
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
       << ", Vertices= " << num_vertices_ << ", Nodes= " << num_nodes_
       << ", QuadPts= " << num_pts_ << ", Dim= " << num_dims_ << std::endl;
#endif

  // Construct standard FEM evaluators with standard field names
  dl_ = Teuchos::rcp(new Layouts(
      workset_size, num_vertices_, num_nodes_, num_pts_, num_dims_));

  TEUCHOS_TEST_FOR_EXCEPTION(
      dl_->vectorAndGradientLayoutsAreEquivalent == false,
      std::logic_error,
      "Data Layout Usage in Mechanics problems assume vecDim = num_dims_");

  EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl_);

  int offset{0};

  // Temporary variable used numerous times below
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits>> ev;

  // Have to register Lattice_Orientation in the mesh before the discretization
  // is built
  auto find_orientation = std::find(
      this->requirements.begin(),
      this->requirements.end(),
      "Lattice_Orientation");

  if (find_orientation != this->requirements.end()) {
    auto entity = StateStruct::ElemData;

    stateMgr.registerStateVariable(
        "Lattice_Orientation",
        dl_->cell_tensor,
        meshSpecs.ebName,
        false,
        &entity);
  }

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

  std::string ace_temperature = (*fnm)["ACE Temperature"];

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

  // Get the solution method type
  SolutionMethodType SolutionType = getSolutionMethod();

  TEUCHOS_TEST_FOR_EXCEPTION(
      SolutionType == SolutionMethodType::Unknown,
      std::logic_error,
      "Solution Method must be Steady, Transient, "
      "Continuation, Eigensolve, or Aeras Hyperviscosity");

  if (have_mech_eq_) {
    Teuchos::ArrayRCP<std::string> const dof_names(1, "Displacement");

    Teuchos::ArrayRCP<std::string> const dof_names_dot(1, "Velocity");

    Teuchos::ArrayRCP<std::string> const dof_names_dotdot(1, "Acceleration");

    Teuchos::ArrayRCP<std::string> const resid_names(
        1, dof_names[0] + " Residual");

    fm0.template registerEvaluator<EvalT>(
        evalUtils.constructGatherSolutionEvaluator_withAcceleration(
            true, dof_names, dof_names_dot, dof_names_dotdot));

    fm0.template registerEvaluator<EvalT>(
        evalUtils.constructGatherCoordinateVectorEvaluator());

    if (!surface_element) {
      fm0.template registerEvaluator<EvalT>(
          evalUtils.constructDOFVecInterpolationEvaluator(dof_names[0]));

      fm0.template registerEvaluator<EvalT>(
          evalUtils.constructDOFVecInterpolationEvaluator(dof_names_dot[0]));

      fm0.template registerEvaluator<EvalT>(
          evalUtils.constructDOFVecInterpolationEvaluator(dof_names_dotdot[0]));

      fm0.template registerEvaluator<EvalT>(
          evalUtils.constructDOFVecGradInterpolationEvaluator(dof_names[0]));

      fm0.template registerEvaluator<EvalT>(
          evalUtils.constructDOFVecGradInterpolationEvaluator(
              dof_names_dot[0]));

      fm0.template registerEvaluator<EvalT>(
          evalUtils.constructMapToPhysicalFrameEvaluator(cellType, cubature));

      fm0.template registerEvaluator<EvalT>(
          evalUtils.constructComputeBasisFunctionsEvaluator(
              cellType, intrepidBasis, cubature));
    }

    fm0.template registerEvaluator<EvalT>(
        evalUtils.constructScatterResidualEvaluator(true, resid_names));

#ifdef ALBANY_CONTACT
    fm0.template registerEvaluator<EvalT>(
        evalUtils.constructMortarContactResidualEvaluator(resid_names));
#endif

    offset += num_dims_;
  } else if (have_mech_) {  // constant configuration

    Teuchos::RCP<Teuchos::ParameterList> p =
        Teuchos::rcp(new Teuchos::ParameterList);

    p->set<std::string>("Material Property Name", "Displacement");
    p->set<Teuchos::RCP<PHX::DataLayout>>("Data Layout", dl_->qp_vector);
    p->set<std::string>("Coordinate Vector Name", "Coord Vec");
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "Coordinate Vector Data Layout", dl_->qp_vector);

    p->set<Teuchos::RCP<ParamLib>>("Parameter Library", paramLib);

    Teuchos::ParameterList& paramList = params->sublist("Displacement");

    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = Teuchos::rcp(
        new PHAL::NSMaterialProperty<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // Output the Velocity and Acceleration
  // Register the states to store the output data in

  // store computed xdot in "Velocity" field
  // This is just for testing as it duplicates writing the solution
  pFromProb->set<std::string>("x Field Name", "xField");

  // store computed xdot in "Velocity" field
  pFromProb->set<std::string>("xdot Field Name", "Velocity");

  // store computed xdotdot in "Acceleration" field
  pFromProb->set<std::string>("xdotdot Field Name", "Acceleration");

  if (have_temperature_eq_) {  // Gather Solution Temperature

    Teuchos::ArrayRCP<std::string> const dof_names(1, "Temperature");

    Teuchos::ArrayRCP<std::string> const dof_names_dot(1, "Temperature Dot");

    Teuchos::ArrayRCP<std::string> const resid_names(
        1, dof_names[0] + " Residual");

    if (SolutionType == SolutionMethodType::Transient) {
      fm0.template registerEvaluator<EvalT>(
          evalUtils.constructGatherSolutionEvaluator_withAcceleration(
              false, dof_names, dof_names_dot, Teuchos::null, offset));
    } else {
      fm0.template registerEvaluator<EvalT>(
          evalUtils.constructGatherSolutionEvaluator_noTransient(
              false, dof_names, offset));
    }

    if (have_mech_eq_ == false) { 
       fm0.template registerEvaluator<EvalT>(
          evalUtils.constructGatherCoordinateVectorEvaluator());
    }

    if (!surface_element) {
      fm0.template registerEvaluator<EvalT>(
          evalUtils.constructDOFInterpolationEvaluator(dof_names[0], offset));

      if (SolutionType == SolutionMethodType::Transient) {
        fm0.template registerEvaluator<EvalT>(
            evalUtils.constructDOFInterpolationEvaluator(
                dof_names_dot[0], offset));
      }

      fm0.template registerEvaluator<EvalT>(
          evalUtils.constructDOFGradInterpolationEvaluator(
              dof_names[0], offset));

      if (have_mech_eq_ == false) { 
        fm0.template registerEvaluator<EvalT>(
            evalUtils.constructComputeBasisFunctionsEvaluator(
                cellType, intrepidBasis, cubature));
      }

    }

    fm0.template registerEvaluator<EvalT>(
        evalUtils.constructScatterResidualEvaluator(
            false, resid_names, offset, "Scatter Temperature"));
    offset++;
  } else if (
      (!have_temperature_eq_ && have_temperature_) || have_transport_eq_ ||
      have_transport_) {
    Teuchos::RCP<Teuchos::ParameterList> p =
        Teuchos::rcp(new Teuchos::ParameterList);

    p->set<std::string>("Material Property Name", temperature);
    p->set<Teuchos::RCP<PHX::DataLayout>>("Data Layout", dl_->qp_scalar);
    p->set<std::string>("Coordinate Vector Name", "Coord Vec");
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "Coordinate Vector Data Layout", dl_->qp_vector);

    p->set<Teuchos::RCP<ParamLib>>("Parameter Library", paramLib);

    Teuchos::ParameterList& paramList = params->sublist("Temperature");

    // This evaluator is called to set a constant temperature when
    // "Variable Type" is set to "Constant." It is also called when
    // "Variable Type" is set to "Time Dependent." There are two "Type"
    // variables in the PL - "Type" and "Variable Type". For the last case,
    // let's set "Type" to "Time Dependent" to hopefully make the evaluator call
    // a little more general (GAH)

    std::string const temp_type =
        paramList.get<std::string>("Variable Type", "None");

    if (temp_type == "Time Dependent") {
      paramList.set<std::string>("Type", temp_type);
    }

    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = Teuchos::rcp(
        new PHAL::NSMaterialProperty<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_ace_temperature_eq_) {  // Gather Solution ACE Temperature

    Teuchos::ArrayRCP<std::string> const dof_names(1, "ACE Temperature");

    Teuchos::ArrayRCP<std::string> const dof_names_dot(
        1, "ACE Temperature Dot");

    Teuchos::ArrayRCP<std::string> const resid_names(
        1, dof_names[0] + " Residual");

    if (SolutionType == SolutionMethodType::Transient) {
      fm0.template registerEvaluator<EvalT>(
          evalUtils.constructGatherSolutionEvaluator_withAcceleration(
              false, dof_names, dof_names_dot, Teuchos::null, offset));
    } else {
      fm0.template registerEvaluator<EvalT>(
          evalUtils.constructGatherSolutionEvaluator_noTransient(
              false, dof_names, offset));
    }

    if (!surface_element) {
      fm0.template registerEvaluator<EvalT>(
          evalUtils.constructDOFInterpolationEvaluator(dof_names[0], offset));

      if (SolutionType == SolutionMethodType::Transient) {
        fm0.template registerEvaluator<EvalT>(
            evalUtils.constructDOFInterpolationEvaluator(
                dof_names_dot[0], offset));
      }

      fm0.template registerEvaluator<EvalT>(
          evalUtils.constructDOFGradInterpolationEvaluator(
              dof_names[0], offset));
    }

    fm0.template registerEvaluator<EvalT>(
        evalUtils.constructScatterResidualEvaluator(
            false, resid_names, offset, "Scatter ACE Temperature"));
    offset++;
  }

  if (have_stab_pressure_eq_) {  // Gather Stabilized Pressure

    Teuchos::ArrayRCP<std::string> dof_names(1, "Pressure");

    Teuchos::ArrayRCP<std::string> resid_names(1, dof_names[0] + " Residual");

    fm0.template registerEvaluator<EvalT>(
        evalUtils.constructGatherSolutionEvaluator_noTransient(
            false, dof_names, offset));

    if (!surface_element) {
      fm0.template registerEvaluator<EvalT>(
          evalUtils.constructDOFInterpolationEvaluator(dof_names[0], offset));

      fm0.template registerEvaluator<EvalT>(
          evalUtils.constructDOFGradInterpolationEvaluator(
              dof_names[0], offset));
    }

    fm0.template registerEvaluator<EvalT>(
        evalUtils.constructScatterResidualEvaluator(
            false, resid_names, offset, "Scatter Pressure"));
    offset++;
  }

  if (have_damage_eq_) {  // Damage

    Teuchos::ArrayRCP<std::string> const dof_names(1, "Damage");

    Teuchos::ArrayRCP<std::string> const resid_names(
        1, dof_names[0] + " Residual");

    fm0.template registerEvaluator<EvalT>(
        evalUtils.constructGatherSolutionEvaluator_noTransient(
            false, dof_names, offset));

    if (!surface_element) {
      fm0.template registerEvaluator<EvalT>(
          evalUtils.constructDOFInterpolationEvaluator(dof_names[0], offset));

      fm0.template registerEvaluator<EvalT>(
          evalUtils.constructDOFGradInterpolationEvaluator(
              dof_names[0], offset));
    }

    fm0.template registerEvaluator<EvalT>(
        evalUtils.constructScatterResidualEvaluator(
            false, resid_names, offset, "Scatter Damage"));
    offset++;
  } else if (!have_damage_eq_ && have_damage_) {
    Teuchos::RCP<Teuchos::ParameterList> p =
        Teuchos::rcp(new Teuchos::ParameterList);

    p->set<std::string>("Material Property Name", "Damage");
    p->set<Teuchos::RCP<PHX::DataLayout>>("Data Layout", dl_->qp_scalar);
    p->set<std::string>("Coordinate Vector Name", "Coord Vec");
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "Coordinate Vector Data Layout", dl_->qp_vector);

    p->set<Teuchos::RCP<ParamLib>>("Parameter Library", paramLib);

    Teuchos::ParameterList& paramList = params->sublist("Damage");

    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = Teuchos::rcp(
        new PHAL::NSMaterialProperty<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_pore_pressure_eq_) {
    Teuchos::ArrayRCP<std::string> const dof_names(1, "Pore_Pressure");

    Teuchos::ArrayRCP<std::string> const resid_names(
        1, dof_names[0] + " Residual");

    fm0.template registerEvaluator<EvalT>(
        evalUtils.constructGatherSolutionEvaluator_noTransient(
            false, dof_names, offset));

    if (!surface_element) {
      fm0.template registerEvaluator<EvalT>(
          evalUtils.constructDOFInterpolationEvaluator(dof_names[0], offset));

      fm0.template registerEvaluator<EvalT>(
          evalUtils.constructDOFGradInterpolationEvaluator(
              dof_names[0], offset));
    }

    fm0.template registerEvaluator<EvalT>(
        evalUtils.constructScatterResidualEvaluator(
            false, resid_names, offset, "Scatter Pore_Pressure"));
    offset++;
  } else if (have_pore_pressure_) {  // constant Pressure

    Teuchos::RCP<Teuchos::ParameterList> p =
        Teuchos::rcp(new Teuchos::ParameterList);

    p->set<std::string>("Material Property Name", "Pressure");
    p->set<Teuchos::RCP<PHX::DataLayout>>("Data Layout", dl_->qp_scalar);
    p->set<std::string>("Coordinate Vector Name", "Coord Vec");
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "Coordinate Vector Data Layout", dl_->qp_vector);

    p->set<Teuchos::RCP<ParamLib>>("Parameter Library", paramLib);

    Teuchos::ParameterList& paramList = params->sublist("Pressure");

    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = Teuchos::rcp(
        new PHAL::NSMaterialProperty<EvalT, PHAL::AlbanyTraits>(*p));

    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_transport_eq_) {  // Gather solution for transport problem
    // Lattice Concentration
    Teuchos::ArrayRCP<std::string> const dof_names(1, "Transport");

    Teuchos::ArrayRCP<std::string> const resid_names(
        1, dof_names[0] + " Residual");

    fm0.template registerEvaluator<EvalT>(
        evalUtils.constructGatherSolutionEvaluator_noTransient(
            false, dof_names, offset));

    if (!surface_element) {
      fm0.template registerEvaluator<EvalT>(
          evalUtils.constructDOFInterpolationEvaluator(dof_names[0], offset));

      fm0.template registerEvaluator<EvalT>(
          evalUtils.constructDOFGradInterpolationEvaluator(
              dof_names[0], offset));
    }

    fm0.template registerEvaluator<EvalT>(
        evalUtils.constructScatterResidualEvaluator(
            false, resid_names, offset, "Scatter Transport"));

    offset++;                    // for lattice concentration
  } else if (have_transport_) {  // Constant transport scalar value

    Teuchos::RCP<Teuchos::ParameterList> p =
        Teuchos::rcp(new Teuchos::ParameterList);

    p->set<std::string>("Material Property Name", "Transport");
    p->set<Teuchos::RCP<PHX::DataLayout>>("Data Layout", dl_->qp_scalar);
    p->set<std::string>("Coordinate Vector Name", "Coord Vec");
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "Coordinate Vector Data Layout", dl_->qp_vector);

    p->set<Teuchos::RCP<ParamLib>>("Parameter Library", paramLib);

    Teuchos::ParameterList& paramList = params->sublist("Transport");

    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = Teuchos::rcp(
        new PHAL::NSMaterialProperty<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_hydrostress_eq_) {  // Gather solution for transport problem

    Teuchos::ArrayRCP<std::string> const dof_names(1, "HydroStress");

    Teuchos::ArrayRCP<std::string> const resid_names(
        1, dof_names[0] + " Residual");

    fm0.template registerEvaluator<EvalT>(
        evalUtils.constructGatherSolutionEvaluator_noTransient(
            false, dof_names, offset));

    if (!surface_element) {
      fm0.template registerEvaluator<EvalT>(
          evalUtils.constructDOFInterpolationEvaluator(dof_names[0], offset));

      fm0.template registerEvaluator<EvalT>(
          evalUtils.constructDOFGradInterpolationEvaluator(
              dof_names[0], offset));
    }

    fm0.template registerEvaluator<EvalT>(
        evalUtils.constructScatterResidualEvaluator(
            false, resid_names, offset, "Scatter HydroStress"));
    offset++;  // for hydrostatic stress
  }

  Teuchos::RCP<Teuchos::ParameterList> p =
      Teuchos::rcp(new Teuchos::ParameterList("Time"));

  p->set<std::string>("Time Name", "Time");
  p->set<std::string>("Delta Time Name", "Delta Time");
  p->set<Teuchos::RCP<PHX::DataLayout>>(
      "Workset Scalar Data Layout", dl_->workset_scalar);
  p->set<Teuchos::RCP<ParamLib>>("Parameter Library", paramLib);
  p->set<bool>("Disable Transient", true);
  ev = Teuchos::rcp(new LCM::Time<EvalT, PHAL::AlbanyTraits>(*p));
  fm0.template registerEvaluator<EvalT>(ev);
  p = stateMgr.registerStateVariable(
      "Time", dl_->workset_scalar, dl_->dummy, eb_name, "scalar", 0.0, true);
  ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
  fm0.template registerEvaluator<EvalT>(ev);

  bool reg_dir_field{false};

  if (params->isParameter("Register dirichlet_field")) {
    reg_dir_field = params->get<bool>("Register dirichlet_field");
  }
  // IKT, 3/27/16: register dirichlet_field for specifying Dirichlet data from a
  // field
  // in the input exodus mesh, if this is requested in the input file .
  if ((dir_count == 0) && (reg_dir_field == true)) {
    // constructEvaluators gets called multiple times for different
    // specializations.
    // Make sure dirichlet_field gets registered only once via counter.
    // I don't quite understand why this is needed for LCM but not for
    // LANDICE... dirichlet_field
    StateStruct::MeshFieldEntity entity = StateStruct::NodalDistParameter;

    stateMgr.registerStateVariable(
        "dirichlet_field", dl_->node_vector, eb_name, true, &entity, "");
    dir_count++;
  }

  if (have_mech_eq_) {  // Current Coordinates

    Teuchos::RCP<Teuchos::ParameterList> p =
        Teuchos::rcp(new Teuchos::ParameterList("Current Coordinates"));

    p->set<std::string>("Reference Coordinates Name", "Coord Vec");
    p->set<std::string>("Displacement Name", "Displacement");
    p->set<std::string>("Current Coordinates Name", "Current Coordinates");
    ev = Teuchos::rcp(
        new LCM::CurrentCoords<EvalT, PHAL::AlbanyTraits>(*p, dl_));

    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_mech_eq_ && have_sizefield_adaptation_) {  // Mesh size field

    Teuchos::RCP<Teuchos::ParameterList> p =
        Teuchos::rcp(new Teuchos::ParameterList("Isotropic Mesh Size Field"));

    p->set<std::string>("IsoTropic MeshSizeField Name", "IsoMeshSizeField");
    p->set<std::string>("Current Coordinates Name", "Current Coordinates");
    p->set<Teuchos::RCP<Intrepid2::Cubature<PHX::Device>>>(
        "Cubature", cubature);

    // Get the Adaptation list and send to the evaluator
    Teuchos::ParameterList& paramList = params->sublist("Adaptation");

    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    p->set<const Intrepid2Basis>("Intrepid2 Basis", intrepidBasis);
    ev = Teuchos::rcp(
        new LCM::IsoMeshSizeField<EvalT, PHAL::AlbanyTraits>(*p, dl_));
    fm0.template registerEvaluator<EvalT>(ev);

    // output mesh size field if requested
    /*
        bool output_flag = false;
        if (material_db_->isElementBlockParam(eb_name, "Output MeshSizeField"))
          output_flag =
              material_db_->getElementBlockParam<bool>(eb_name, "Output
       MeshSizeField");
    */

    // FIXME: This is unnecessary as written. Should the above code be
    // activated? - CA
    bool output_flag{true};

    if (output_flag) {
      p = stateMgr.registerStateVariable(
          "IsoMeshSizeField",
          dl_->qp_scalar,
          dl_->dummy,
          eb_name,
          "scalar",
          1.0,
          true,
          output_flag);
      ev =
          Teuchos::rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }
  }

  if (have_temperature_eq_ || have_temperature_) {
    RealType const temp = material_db_->getElementBlockParam<RealType>(
        eb_name, "Initial Temperature", 0.0);

    Teuchos::RCP<Teuchos::ParameterList> p =
        Teuchos::rcp(new Teuchos::ParameterList("Save Temperature"));

    p = stateMgr.registerStateVariable(
        temperature,
        dl_->qp_scalar,
        dl_->dummy,
        eb_name,
        "scalar",
        temp,
        true,
        true);

    ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_ace_temperature_eq_ == true) {
    RealType const temp = material_db_->getElementBlockParam<RealType>(
        eb_name, "Initial ACE Temperature", 0.0);

    Teuchos::RCP<Teuchos::ParameterList> p =
        Teuchos::rcp(new Teuchos::ParameterList("Save ACE Temperature"));

    p = stateMgr.registerStateVariable(
        ace_temperature,
        dl_->qp_scalar,
        dl_->dummy,
        eb_name,
        "scalar",
        temp,
        true,
        true);

    ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_pore_pressure_eq_ || have_pore_pressure_) {
    Teuchos::RCP<Teuchos::ParameterList> p =
        Teuchos::rcp(new Teuchos::ParameterList("Save Pore Pressure"));

    p = stateMgr.registerStateVariable(
        porePressure,
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
    bool const output_flag = material_db_->getElementBlockParam<bool>(
        eb_name, "Output IP" + transport, true);

    RealType const ic = material_db_->getElementBlockParam<double>(
        eb_name, "Initial Concentration", 0.0);

    Teuchos::RCP<Teuchos::ParameterList> const p =
        stateMgr.registerStateVariable(
            transport,
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
    Teuchos::RCP<Teuchos::ParameterList> p =
        Teuchos::rcp(new Teuchos::ParameterList("Save HydroStress"));

    p = stateMgr.registerStateVariable(
        hydroStress,
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
    Teuchos::RCP<Teuchos::ParameterList> p =
        Teuchos::rcp(new Teuchos::ParameterList);

    p->set<std::string>("Source Name", "Source");
    p->set<std::string>("Variable Name", "Displacement");
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "QP Scalar Data Layout", dl_->qp_scalar);

    p->set<Teuchos::RCP<ParamLib>>("Parameter Library", paramLib);

    Teuchos::ParameterList& paramList =
        params->sublist("Source Functions").sublist("Mechanical Source");

    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = Teuchos::rcp(new PHAL::Source<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // Heat Source in Heat Equation

  if (thermal_source_ != SOURCE_TYPE_NONE) {
    Teuchos::RCP<Teuchos::ParameterList> p =
        Teuchos::rcp(new Teuchos::ParameterList);

    p->set<std::string>("Source Name", "Heat Source");
    p->set<std::string>("Variable Name", "Temperature");
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "QP Scalar Data Layout", dl_->qp_scalar);

    p->set<Teuchos::RCP<ParamLib>>("Parameter Library", paramLib);

    if (thermal_source_ == SOURCE_TYPE_INPUT) {  // Thermal source in input file

      Teuchos::ParameterList& paramList =
          params->sublist("Source Functions").sublist("Thermal Source");

      p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

      ev = Teuchos::rcp(new PHAL::Source<EvalT, PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);

      thermal_source_evaluated_ = true;

    } else if (thermal_source_ == SOURCE_TYPE_MATERIAL) {
      // There may not be a source in every element block

      if (material_db_->isElementBlockSublist(
              eb_name, "Source Functions")) {  // Thermal source in matDB

        Teuchos::ParameterList& srcParamList =
            material_db_->getElementBlockSublist(eb_name, "Source Functions");

        if (srcParamList.isSublist("Thermal Source")) {
          Teuchos::ParameterList& paramList =
              srcParamList.sublist("Thermal Source");

          p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

          ev = Teuchos::rcp(new PHAL::Source<EvalT, PHAL::AlbanyTraits>(*p));
          fm0.template registerEvaluator<EvalT>(ev);

          thermal_source_evaluated_ = true;
        }
      } else  // Do not evaluate heat source in TransportResidual
      {
        thermal_source_evaluated_ = false;
      }
    } else

      TEUCHOS_TEST_FOR_EXCEPTION(
          true,
          std::logic_error,
          "Unrecognized thermal source specified in input file");
  }

  {  // Constitutive Model Parameters

    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
        new Teuchos::ParameterList("Constitutive Model Parameters"));

    if (have_temperature_ == true) {
      p->set<std::string>("Temperature Name", temperature);
      param_list.set<bool>("Have Temperature", true);
    }

    if (have_ace_temperature_ == true) {
      p->set<std::string>("ACE Temperature Name", ace_temperature);
      param_list.set<bool>("Have ACE Temperature", true);
    }

    // optional spatial dependence
    p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");

    // pass through material properties
    p->set<Teuchos::ParameterList*>("Material Parameters", &param_list);

    Teuchos::RCP<LCM::ConstitutiveModelParameters<EvalT, PHAL::AlbanyTraits>>
        cmpEv = Teuchos::rcp(
            new LCM::ConstitutiveModelParameters<EvalT, PHAL::AlbanyTraits>(
                *p, dl_));
    fm0.template registerEvaluator<EvalT>(cmpEv);
  }

  if (have_mech_eq_) {
    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
        new Teuchos::ParameterList("Constitutive Model Interface"));

    // TODO: figure out how to do this better
    param_list.set<bool>("Have Temperature", false);
    if (have_temperature_) {
      p->set<std::string>("Temperature Name", temperature);
      param_list.set<bool>("Have Temperature", true);
    }

    param_list.set<bool>("Have ACE Temperature", false);
    if (have_ace_temperature_ == true) {
      p->set<std::string>("ACE Temperature Name", ace_temperature);
      param_list.set<bool>("Have ACE Temperature", true);
    }

    param_list.set<bool>("Have Total Concentration", false);
    if (have_transport_) {
      p->set<std::string>("Total Concentration Name", totalConcentration);
      param_list.set<bool>("Have Total Concentration", true);
    }

    param_list.set<bool>("Have Bubble Volume Fraction", false);
    param_list.set<bool>("Have Total Bubble Density", false);
    if (param_list.isSublist("Tritium Coefficients")) {
      p->set<std::string>(
          "Bubble Volume Fraction Name", bubble_volume_fraction);
      p->set<std::string>("Total Bubble Density Name", total_bubble_density);
      param_list.set<bool>("Have Bubble Volume Fraction", true);
      param_list.set<bool>("Have Total Bubble Density", true);
      param_list.set<RealType>(
          "Helium Radius",
          param_list.sublist("Tritium Coefficients")
              .get<RealType>("Helium Radius", 0.0));
    }

    param_list.set<Teuchos::RCP<std::map<std::string, std::string>>>(
        "Name Map", fnm);
    p->set<Teuchos::ParameterList*>("Material Parameters", &param_list);
    p->set<bool>("Volume Average Pressure", volume_average_pressure);
    if (volume_average_pressure) {
      p->set<std::string>("Weights Name", "Weights");
      p->set<std::string>("J Name", J);
    }

    Teuchos::RCP<LCM::ConstitutiveModelInterface<EvalT, PHAL::AlbanyTraits>>
        cmiEv = Teuchos::rcp(
            new LCM::ConstitutiveModelInterface<EvalT, PHAL::AlbanyTraits>(
                *p, dl_));
    fm0.template registerEvaluator<EvalT>(cmiEv);

    // register state variables
    for (int sv(0); sv < cmiEv->getNumStateVars(); ++sv) {
      cmiEv->fillStateVariableStruct(sv);
      p = stateMgr.registerStateVariable(
          cmiEv->getName(),
          cmiEv->getLayout(),
          dl_->dummy,
          eb_name,
          cmiEv->getInitType(),
          cmiEv->getInitValue(),
          cmiEv->getStateFlag(),
          cmiEv->getOutputFlag());
      ev =
          Teuchos::rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }
  }

  // Surface Element Block
  if (surface_element) {
    {  // Surface Basis
      // SurfaceBasis_Def.hpp
      Teuchos::RCP<Teuchos::ParameterList> p =
          Teuchos::rcp(new Teuchos::ParameterList("Surface Basis"));

      // inputs
      p->set<std::string>("Reference Coordinates Name", "Coord Vec");
      p->set<Teuchos::RCP<Intrepid2::Cubature<PHX::Device>>>(
          "Cubature", surfaceCubature);
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

    if (have_mech_eq_) {  // Surface Jump
      // SurfaceVectorJump_Def.hpp

      Teuchos::RCP<Teuchos::ParameterList> p =
          Teuchos::rcp(new Teuchos::ParameterList("Surface Vector Jump"));

      // inputs
      p->set<Teuchos::RCP<Intrepid2::Cubature<PHX::Device>>>(
          "Cubature", surfaceCubature);
      p->set<Intrepid2Basis>("Intrepid2 Basis", surfaceBasis);
      p->set<std::string>("Vector Name", "Current Coordinates");

      // outputs
      p->set<std::string>("Vector Jump Name", "Vector Jump");

      ev = Teuchos::rcp(
          new LCM::SurfaceVectorJump<EvalT, PHAL::AlbanyTraits>(*p, dl_));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    if (have_temperature_eq_ || have_ace_temperature_eq_ ||
        have_pore_pressure_eq_ || have_transport_eq_) {  // Temperature Jump
      // SurfaceScalarJump_Def.hpp
      Teuchos::RCP<Teuchos::ParameterList> p =
          Teuchos::rcp(new Teuchos::ParameterList("Surface Scalar Jump"));

      // inputs
      p->set<Teuchos::RCP<Intrepid2::Cubature<PHX::Device>>>(
          "Cubature", surfaceCubature);
      p->set<Intrepid2Basis>("Intrepid2 Basis", surfaceBasis);
      if (have_temperature_eq_) {
        p->set<std::string>("Nodal Temperature Name", "Temperature");
        // outputs
        p->set<std::string>("Jump of Temperature Name", "Temperature Jump");
        p->set<std::string>("MidPlane Temperature Name", temperature);
      }

      if (have_ace_temperature_eq_) {
        p->set<std::string>("Nodal ACE Temperature Name", "ACE Temperature");
        // outputs
        p->set<std::string>(
            "Jump of ACE Temperature Name", "ACE Temperature Jump");
        p->set<std::string>("MidPlane ACE Temperature Name", temperature);
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

    if (have_mech_eq_) {  // Surface Gradient
      // SurfaceVectorGradient_Def.hpp

      Teuchos::RCP<Teuchos::ParameterList> p =
          Teuchos::rcp(new Teuchos::ParameterList("Surface Vector Gradient"));

      // inputs
      p->set<RealType>("thickness", thickness);
      p->set<bool>("Weighted Volume Average J", volume_average_j);
      p->set<RealType>(
          "Average J Stabilization Parameter",
          volume_average_stabilization_param);
      p->set<Teuchos::RCP<Intrepid2::Cubature<PHX::Device>>>(
          "Cubature", surfaceCubature);
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
      bool const defgrad_flag = material_db_->getElementBlockParam<bool>(
          eb_name, "Output Deformation Gradient", false);

      p = stateMgr.registerStateVariable(
          defgrad,
          dl_->qp_tensor,
          dl_->dummy,
          eb_name,
          "identity",
          1.0,
          true,
          defgrad_flag);

      ev =
          Teuchos::rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);

      bool const j_flag =
          material_db_->getElementBlockParam<bool>(eb_name, "Output J", false);

      // need J and J_old to perform time integration for poromechanics problem
      if (have_pore_pressure_eq_ || j_flag) {
        p = stateMgr.registerStateVariable(
            J,
            dl_->qp_scalar,
            dl_->dummy,
            eb_name,
            "scalar",
            1.0,
            true,
            j_flag);

        ev = Teuchos::rcp(
            new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
        fm0.template registerEvaluator<EvalT>(ev);
      }
    }  // end if (have_mech_eq_)

    // Surface Gradient Operator
    if (have_pore_pressure_eq_ && surface_element) {
      // SurfaceScalarGradientOperatorPorePressure_Def.hpp

      Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
          new Teuchos::ParameterList("Surface Scalar Gradient Operator Pore Pressure"));

      // inputs
      p->set<RealType>("thickness", thickness);
      p->set<Teuchos::RCP<Intrepid2::Cubature<PHX::Device>>>(
          "Cubature", surfaceCubature);
      p->set<Intrepid2Basis>("Intrepid2 Basis", surfaceBasis);
      p->set<std::string>("Reference Dual Basis Name", "Reference Dual Basis");
      p->set<std::string>("Reference Normal Name", "Reference Normal");

      // NOTE: NOT surf_Pore_Pressure here
      // NOTE: If you need to compute gradient for more than one scalar field,
      // that could cause trouble
      p->set<std::string>("Nodal Scalar Name", "Pore_Pressure");

      // outputs
      p->set<std::string>(
          "Surface Scalar Gradient Operator Pore Pressure Name",
          "Surface Scalar Gradient Operator Pore Pressure");
      p->set<Teuchos::RCP<PHX::DataLayout>>(
          "Node QP Vector Data Layout", dl_->node_qp_vector);
      p->set<std::string>(
          "Surface Scalar Gradient Name", "Surface Pressure Gradient");
      p->set<Teuchos::RCP<PHX::DataLayout>>(
          "QP Vector Data Layout", dl_->qp_vector);

      ev = Teuchos::rcp(
          new LCM::SurfaceScalarGradientOperatorPorePressure<EvalT, PHAL::AlbanyTraits>(
              *p, dl_));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    if (have_transport_eq_ && surface_element) {
      // SurfaceScalarGradientOperatorTransport_Def.hpp
      Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
          new Teuchos::ParameterList("Surface Scalar Gradient Operator Transport"));
      // inputs
      p->set<RealType>("thickness", thickness);
      p->set<Teuchos::RCP<Intrepid2::Cubature<PHX::Device>>>(
          "Cubature", surfaceCubature);
      p->set<Intrepid2Basis>("Intrepid2 Basis", surfaceBasis);
      p->set<std::string>("Reference Dual Basis Name", "Reference Dual Basis");
      p->set<std::string>("Reference Normal Name", "Reference Normal");

      // NOTE: NOT surf_Pore_Pressure here
      // NOTE: If you need to compute gradient for more than one scalar field,
      // that could cause trouble
      p->set<std::string>("Nodal Scalar Name", "Transport");

      // outputs
      p->set<std::string>(
          "Surface Scalar Gradient Operator Transport Name",
          "Surface Scalar Gradient Operator Transport");
      p->set<Teuchos::RCP<PHX::DataLayout>>(
          "Node QP Vector Data Layout", dl_->node_qp_vector);
      p->set<std::string>(
            "Surface Scalar Gradient Name", "Surface Transport Gradient");
      p->set<Teuchos::RCP<PHX::DataLayout>>(
          "QP Vector Data Layout", dl_->qp_vector);

      ev = Teuchos::rcp(
          new LCM::SurfaceScalarGradientOperatorTransport<EvalT, PHAL::AlbanyTraits>(
              *p, dl_));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    if (have_hydrostress_eq_ && surface_element) {
      // SurfaceScalarGradientOperatorHydroStress_Def.hpp
      Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
          new Teuchos::ParameterList("Surface Scalar Gradient Operator HydroStress"));
      // inputs
      p->set<RealType>("thickness", thickness);
      p->set<Teuchos::RCP<Intrepid2::Cubature<PHX::Device>>>(
          "Cubature", surfaceCubature);
      p->set<Intrepid2Basis>("Intrepid2 Basis", surfaceBasis);
      p->set<std::string>("Reference Dual Basis Name", "Reference Dual Basis");
      p->set<std::string>("Reference Normal Name", "Reference Normal");

      // NOTE: NOT surf_Pore_Pressure here
      // NOTE: If you need to compute gradient for more than one scalar field,
      // that could cause trouble
      if (have_transport_eq_ == true)
        p->set<std::string>("Nodal Scalar Name", "HydroStress");

      // outputs
      p->set<std::string>(
          "Surface Scalar Gradient Operator HydroStress Name",
          "Surface Scalar Gradient Operator HydroStress");
      p->set<Teuchos::RCP<PHX::DataLayout>>(
          "Node QP Vector Data Layout", dl_->node_qp_vector);
      p->set<std::string>(
          "Surface Scalar Gradient Name", "Surface HydroStress Gradient");
      p->set<Teuchos::RCP<PHX::DataLayout>>(
          "QP Vector Data Layout", dl_->qp_vector);

      ev = Teuchos::rcp(
          new LCM::SurfaceScalarGradientOperatorHydroStress<EvalT, PHAL::AlbanyTraits>(
              *p, dl_));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    if (have_mech_eq_) {  // Surface Residual
      // SurfaceVectorResidual_Def.hpp
      Teuchos::RCP<Teuchos::ParameterList> p =
          Teuchos::rcp(new Teuchos::ParameterList("Surface Vector Residual"));

      // inputs
      p->set<RealType>("thickness", thickness);
      p->set<Teuchos::RCP<Intrepid2::Cubature<PHX::Device>>>(
          "Cubature", surfaceCubature);
      p->set<Intrepid2Basis>("Intrepid2 Basis", surfaceBasis);

      p->set<bool>("Compute Membrane Forces", compute_membrane_forces);

      p->set<std::string>("Stress Name", firstPK);
      p->set<std::string>("Current Basis Name", "Current Basis");
      p->set<std::string>("Reference Dual Basis Name", "Reference Dual Basis");
      p->set<std::string>("Reference Normal Name", "Reference Normal");
      p->set<std::string>("Reference Area Name", "Weights");

      if (cohesive_element) {
        p->set<bool>("Use Cohesive Traction", true);
        p->set<std::string>("Cohesive Traction Name", "Cohesive_Traction");
      }

      // outputs
      p->set<std::string>(
          "Surface Vector Residual Name", "Displacement Residual");

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
    }  // end of coehesive/surface element block

  } else {                // surface_element == False
    if (have_mech_eq_) {  // Kinematics quantities

      Teuchos::RCP<Teuchos::ParameterList> p =
          Teuchos::rcp(new Teuchos::ParameterList("Kinematics"));

      p->set<bool>("Weighted Volume Average J", volume_average_j);
      p->set<RealType>(
          "Average J Stabilization Parameter",
          volume_average_stabilization_param);

      // strain
      if (small_strain) { p->set<std::string>("Strain Name", "Strain"); }

      // set flag for return strain and velocity gradient
      bool have_velocity_gradient(false);

      std::string flag = "Velocity Gradient Flag";

      if (material_db_->isElementBlockParam(eb_name, flag)) {
        p->set<bool>(
            flag, material_db_->getElementBlockParam<bool>(eb_name, flag));

        have_velocity_gradient =
            material_db_->getElementBlockParam<bool>(eb_name, flag);

        if (have_velocity_gradient) {
          p->set<std::string>("Velocity Gradient Name", "Velocity Gradient");
        }
      }

      // set flag for return strain and plastic velocity gradient
      bool have_velocity_gradient_plastic{false};

      flag = "Plastic Velocity Gradient Flag";

      if (material_db_->isElementBlockParam(eb_name, flag)) {
        p->set<bool>(
            flag, material_db_->getElementBlockParam<bool>(eb_name, flag));

        have_velocity_gradient_plastic =
            material_db_->getElementBlockParam<bool>(eb_name, flag);

        if (have_velocity_gradient_plastic) {
          p->set<std::string>(
              "Plastic Velocity Gradient Name", "Plastic Velocity Gradient");
        }
      }

      // send in integration weights and the displacement gradient
      p->set<std::string>("Weights Name", "Weights");
      p->set<Teuchos::RCP<PHX::DataLayout>>(
          "QP Scalar Data Layout", dl_->qp_scalar);
      p->set<std::string>("Gradient QP Variable Name", "Displacement Gradient");
      p->set<Teuchos::RCP<PHX::DataLayout>>(
          "QP Tensor Data Layout", dl_->qp_tensor);

      // Outputs: F, J
      p->set<std::string>("DefGrad Name", defgrad);  // dl_->qp_tensor also
      p->set<std::string>("DetDefGrad Name", J);
      p->set<Teuchos::RCP<PHX::DataLayout>>(
          "QP Scalar Data Layout", dl_->qp_scalar);

      if (Teuchos::nonnull(rc_mgr_)) {
        rc_mgr_->registerField(
            defgrad,
            dl_->qp_tensor,
            AAdapt::rc::Init::identity,
            AAdapt::rc::Transformation::right_polar_LieR_LieS,
            p);
        p->set<std::string>("Displacement Name", "Displacement");
      }

      // ev = Teuchos::rcp(new LCM::DefGrad<EvalT,PHAL::AlbanyTraits>(*p));
      ev =
          Teuchos::rcp(new LCM::Kinematics<EvalT, PHAL::AlbanyTraits>(*p, dl_));
      fm0.template registerEvaluator<EvalT>(ev);

      // optional output
      bool const output_flag = material_db_->getElementBlockParam<bool>(
          eb_name, "Output Deformation Gradient", false);

      // Old values of the deformation gradient
      // optional output
      // FIXME: This currently does nothing - CA
      bool const old_defgrad_flag = material_db_->getElementBlockParam<bool>(
          eb_name, "Old Deformation Gradient", false);

      p = stateMgr.registerStateVariable(
          defgrad,
          dl_->qp_tensor,
          dl_->dummy,
          eb_name,
          "identity",
          1.0,
          true,
          output_flag);

      ev =
          Teuchos::rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));

      fm0.template registerEvaluator<EvalT>(ev);

      // optional output of the integration weights
      bool const weights_flag = material_db_->getElementBlockParam<bool>(
          eb_name, "Output Integration Weights", false);

      if (weights_flag) {
        p = stateMgr.registerStateVariable(
            "Weights",
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

      bool const j_flag =
          material_db_->getElementBlockParam<bool>(eb_name, "Output J", false);

      p = stateMgr.registerStateVariable(
          J, dl_->qp_scalar, dl_->dummy, eb_name, "scalar", 1.0, true, false);

      ev =
          Teuchos::rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));

      fm0.template registerEvaluator<EvalT>(ev);

      // Optional output: strain
      if (small_strain) {
        bool const output_strain = material_db_->getElementBlockParam<bool>(
            eb_name, "Output Strain", false);

        if (output_flag) {
          p = stateMgr.registerStateVariable(
              "Strain",
              dl_->qp_tensor,
              dl_->dummy,
              eb_name,
              "scalar",
              0.0,
              false,
              output_strain);

          ev = Teuchos::rcp(
              new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
          fm0.template registerEvaluator<EvalT>(ev);
        }
      }

      // Optional output: velocity gradient
      if (have_velocity_gradient) {
        bool const output_velgrad = material_db_->getElementBlockParam<bool>(
            eb_name, "Output Velocity Gradient", false);

        if (output_velgrad) {
          p = stateMgr.registerStateVariable(
              "Velocity Gradient",
              dl_->qp_tensor,
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

      // Optional output:plastic velocity gradient
      if (have_velocity_gradient_plastic) {
        bool const output_velgrad_p = material_db_->getElementBlockParam<bool>(
            eb_name, "Output Plastic Velocity Gradient", false);

        if (output_velgrad_p) {
          p = stateMgr.registerStateVariable(
              "Plastic Velocity Gradient",
              dl_->qp_tensor,
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
    }

    if (have_mech_eq_) {  // Analytic Mass residual

      Teuchos::RCP<Teuchos::ParameterList> p =
          Teuchos::rcp(new Teuchos::ParameterList("Analytic Mass Residual"));

      // Input
      p->set<std::string>("Weighted BF Name", "wBF");
      p->set<std::string>("Acceleration Name", "Acceleration");
      p->set<std::string>("Weights Name", "Weights");
      // Mechanics residual need value of density for transient analysis.
      // Get it from material. Assumed constant in element block.
      if (material_db_->isElementBlockParam(eb_name, "Density")) {
        p->set<RealType>(
            "Density",
            material_db_->getElementBlockParam<RealType>(eb_name, "Density"));
      }

      const bool resid_using_cub = material_db_->getElementBlockParam<bool>(
          eb_name, "Residual Computed Using Cubature", false);
      p->set<bool>("Residual Computed Using Cubature", resid_using_cub);

      p->set<bool>("Use Composite Tet 10", composite_);

      const bool use_analytic_mass = material_db_->getElementBlockParam<bool>(
          eb_name, "Use Analytic Mass", false);
      p->set<bool>("Use Analytic Mass", use_analytic_mass);

      const bool lump_analytic_mass = material_db_->getElementBlockParam<bool>(
          eb_name, "Lump Analytic Mass", false);
      p->set<bool>("Lump Analytic Mass", lump_analytic_mass);

      p->set<Teuchos::RCP<ParamLib>>("Parameter Library", paramLib);
      // Output
      p->set<std::string>("Analytic Mass Name", "Analytic Mass Residual");
      ev = Teuchos::rcp(
          new LCM::AnalyticMassResidual<EvalT, PHAL::AlbanyTraits>(*p, dl_));
      fm0.template registerEvaluator<EvalT>(ev);
    }  // end if (have_mech_eq_)

    if (have_mech_eq_) {  // Mechanics Residual

      Teuchos::RCP<Teuchos::ParameterList> p =
          Teuchos::rcp(new Teuchos::ParameterList("Displacement Residual"));

      // Input
      p->set<std::string>("Stress Name", firstPK);
      p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
      p->set<std::string>("Weighted BF Name", "wBF");
      p->set<std::string>("Acceleration Name", "Acceleration");
      p->set<std::string>("Body Force Name", "Body Force");
      p->set<std::string>("Analytic Mass Name", "Analytic Mass Residual");
      const bool use_analytic_mass = material_db_->getElementBlockParam<bool>(
          eb_name, "Use Analytic Mass", false);
      p->set<bool>("Use Analytic Mass", use_analytic_mass);
      if (Teuchos::nonnull(rc_mgr_)) {
        p->set<std::string>("DefGrad Name", defgrad);
        rc_mgr_->registerField(
            defgrad,
            dl_->qp_tensor,
            AAdapt::rc::Init::identity,
            AAdapt::rc::Transformation::right_polar_LieR_LieS,
            p);
      }

      RealType material_density = 0.0;

      // Mechanics residual need value of density for transient analysis.
      // Get it from material. Assumed constant in element block.
      if (material_db_->isElementBlockParam(eb_name, "Density")) {
        material_density =
            material_db_->getElementBlockParam<RealType>(eb_name, "Density");
        p->set<RealType>("Density", material_density);
      }

      // Optional body force
      if (material_db_->isElementBlockSublist(eb_name, "Body Force")) {
        p->set<bool>("Has Body Force", true);

        Teuchos::ParameterList& eb_param =
            material_db_->getElementBlockSublist(eb_name, "Body Force");
        eb_param.set<RealType>("Density", material_density);

        ev = Teuchos::rcp(
            new LCM::BodyForce<EvalT, PHAL::AlbanyTraits>(eb_param, dl_));

        fm0.template registerEvaluator<EvalT>(ev);
      }

      p->set<Teuchos::RCP<ParamLib>>("Parameter Library", paramLib);
      // Output
      p->set<std::string>("Residual Name", "Displacement Residual");
      ev = Teuchos::rcp(
          new LCM::MechanicsResidual<EvalT, PHAL::AlbanyTraits>(*p, dl_));
      fm0.template registerEvaluator<EvalT>(ev);
    }  // end if (have_mech_eq_)
  }    // end if(surface_element)

  if (have_mech_eq_) {
    // convert Cauchy stress to first Piola-Kirchhoff

    Teuchos::RCP<Teuchos::ParameterList> p =
        Teuchos::rcp(new Teuchos::ParameterList("First PK Stress"));

    // Input
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

    if (small_strain) { p->set<bool>("Small Strain", true); }

    // Output
    p->set<std::string>("First PK Stress Name", firstPK);

    p->set<Teuchos::RCP<ParamLib>>("Parameter Library", paramLib);

    ev = Teuchos::rcp(new LCM::FirstPK<EvalT, PHAL::AlbanyTraits>(*p, dl_));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // Element length in the direction of solution gradient
  bool const have_pressure_or_transport =
      have_stab_pressure_eq_ || have_pore_pressure_eq_ || have_transport_eq_;

  if (have_pressure_or_transport) {
    Teuchos::RCP<Teuchos::ParameterList> p =
        Teuchos::rcp(new Teuchos::ParameterList("Gradient_Element_Length"));

    // Input
    if (!surface_element) {  // bulk element length
      if (have_pore_pressure_eq_) {
        p->set<std::string>(
            "Unit Gradient QP Variable Name", "Pore_Pressure Gradient");
      } else if (have_transport_eq_) {
        p->set<std::string>(
            "Unit Gradient QP Variable Name", "Transport Gradient");
      } else if (have_stab_pressure_eq_) {
        p->set<std::string>(
            "Unit Gradient QP Variable Name", "Pressure Gradient");
      }
      p->set<std::string>("Gradient BF Name", "Grad BF");
    } else {  // surface element length
      if (have_pore_pressure_eq_) {
        p->set<std::string>(
            "Unit Gradient QP Variable Name", "surf_Pressure Gradient");
      } else if (have_transport_eq_) {
        p->set<std::string>(
            "Unit Gradient QP Variable Name", "surf_Transport Gradient");
      } else if (have_stab_pressure_eq_) {
        p->set<std::string>(
            "Unit Gradient QP Variable Name", "surf_Pressure Gradient");
      }
      //p->set<std::string>(
      //    "Gradient BF Name", "Surface Scalar Gradient Operator");
      p->set<std::string>(
          "Gradient BF Name", "Surface Scalar Gradient Operator Pore Pressure");
      p->set<std::string>(
          "Gradient BF Name", "Surface Scalar Gradient Operator Transport");
      p->set<std::string>(
          "Gradient BF Name", "Surface Scalar Gradient Operator HydroStress");
      //   p->set<std::string>("Gradient BF Name", "Grad BF");
    }

    // Output
    p->set<std::string>("Element Length Name", gradient_element_length);

    ev = Teuchos::rcp(
        new LCM::GradientElementLength<EvalT, PHAL::AlbanyTraits>(*p, dl_));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_pore_pressure_eq_) {  // Porosity

    Teuchos::RCP<Teuchos::ParameterList> p =
        Teuchos::rcp(new Teuchos::ParameterList);

    p->set<std::string>("Porosity Name", porosity);
    p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");
    // Setting this turns on dependence of strain and pore pressure)
    // p->set<std::string>("Strain Name", "Strain");
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
    bool const porosity_flag = material_db_->getElementBlockParam<bool>(
        eb_name, "Output " + porosity, false);

    if (porosity_flag) {
      p = stateMgr.registerStateVariable(
          porosity,
          dl_->qp_scalar,
          dl_->dummy,
          eb_name,
          "scalar",
          0.5,  // This is really bad practice. It needs to be fixed
          false,
          true);
      ev =
          Teuchos::rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }
  }

  if (have_pore_pressure_eq_) {  // Biot Coefficient

    Teuchos::RCP<Teuchos::ParameterList> p =
        Teuchos::rcp(new Teuchos::ParameterList);

    p->set<std::string>("Biot Coefficient Name", biotCoeff);
    p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");
    p->set<Teuchos::RCP<PHX::DataLayout>>("Node Data Layout", dl_->node_scalar);
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "QP Scalar Data Layout", dl_->qp_scalar);
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "QP Vector Data Layout", dl_->qp_vector);

    p->set<Teuchos::RCP<ParamLib>>("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList =
        material_db_->getElementBlockSublist(eb_name, "Biot Coefficient");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = Teuchos::rcp(new LCM::BiotCoefficient<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_pore_pressure_eq_) {  // Biot Modulus

    Teuchos::RCP<Teuchos::ParameterList> p =
        Teuchos::rcp(new Teuchos::ParameterList);

    p->set<std::string>("Biot Modulus Name", biotModulus);
    p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");
    p->set<Teuchos::RCP<PHX::DataLayout>>("Node Data Layout", dl_->node_scalar);
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "QP Scalar Data Layout", dl_->qp_scalar);
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "QP Vector Data Layout", dl_->qp_vector);

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

  if (have_pore_pressure_eq_) {  // Kozeny-Carman Permeaiblity

    Teuchos::RCP<Teuchos::ParameterList> p =
        Teuchos::rcp(new Teuchos::ParameterList);

    p->set<std::string>("Kozeny-Carman Permeability Name", kcPerm);
    p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");
    p->set<Teuchos::RCP<PHX::DataLayout>>("Node Data Layout", dl_->node_scalar);
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "QP Scalar Data Layout", dl_->qp_scalar);
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "QP Vector Data Layout", dl_->qp_vector);

    p->set<Teuchos::RCP<ParamLib>>("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = material_db_->getElementBlockSublist(
        eb_name, "Kozeny-Carman Permeability");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    // Setting this turns on Kozeny-Carman relation
    p->set<std::string>("Porosity Name", porosity);
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "QP Scalar Data Layout", dl_->qp_scalar);

    ev = Teuchos::rcp(new LCM::KCPermeability<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    // Output
    bool const output_kcperm = material_db_->getElementBlockParam<bool>(
        eb_name, "Output " + kcPerm, false);

    if (output_kcperm) {
      p = stateMgr.registerStateVariable(
          kcPerm,
          dl_->qp_scalar,
          dl_->dummy,
          eb_name,
          "scalar",
          0.0,
          false,
          true);

      ev =
          Teuchos::rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }
  }

  // Pore Pressure Residual (Bulk Element)
  if (have_pore_pressure_eq_ && !surface_element) {
    Teuchos::RCP<Teuchos::ParameterList> p =
        Teuchos::rcp(new Teuchos::ParameterList("Pore_Pressure Residual"));

    // Input

    // Input from nodal points, basis function stuff
    p->set<std::string>("Weights Name", "Weights");
    p->set<std::string>("Weighted BF Name", "wBF");
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "Node QP Scalar Data Layout", dl_->node_qp_scalar);
    p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "Node QP Vector Data Layout", dl_->node_qp_vector);

    // Inputs: X, Y at nodes, Cubature, and Basis
    p->set<std::string>("Coordinate Vector Name", "Coord Vec");
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "Coordinate Data Layout", dl_->vertices_vector);
    p->set<Teuchos::RCP<Intrepid2::Cubature<PHX::Device>>>(
        "Cubature", cubature);
    p->set<Teuchos::RCP<shards::CellTopology>>("Cell Type", cellType);

    // DT for  time integration
    p->set<std::string>("Delta Time Name", "Delta Time");
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "Workset Scalar Data Layout", dl_->workset_scalar);

    p->set<bool>("Have Source", false);
    p->set<std::string>("Source Name", "Source");
    p->set<bool>("Have Absorption", false);

    // Input from cubature points
    p->set<std::string>("Element Length Name", gradient_element_length);
    p->set<std::string>("QP Pore Pressure Name", porePressure);
    p->set<std::string>("QP Time Derivative Variable Name", porePressure);
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "QP Scalar Data Layout", dl_->qp_scalar);

    // p->set<std::string>("Material Property Name", "Stabilization Parameter");
    p->set<std::string>("Porosity Name", "Porosity");
    p->set<std::string>("Thermal Conductivity Name", "Thermal Conductivity");
    p->set<std::string>("Kozeny-Carman Permeability Name", kcPerm);
    p->set<std::string>("Biot Coefficient Name", biotCoeff);
    p->set<std::string>("Biot Modulus Name", biotModulus);

    p->set<std::string>("Gradient QP Variable Name", "Pore_Pressure Gradient");
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "QP Vector Data Layout", dl_->qp_vector);

    if (have_mech_eq_) {
      p->set<bool>("Have Mechanics", true);
      p->set<std::string>("DefGrad Name", defgrad);
      p->set<Teuchos::RCP<PHX::DataLayout>>(
          "QP Tensor Data Layout", dl_->qp_tensor);
      p->set<std::string>("DetDefGrad Name", J);
      p->set<Teuchos::RCP<PHX::DataLayout>>(
          "QP Scalar Data Layout", dl_->qp_scalar);
    }

    RealType const stab_param = material_db_->getElementBlockParam<RealType>(
        eb_name, "Stabilization Parameter", 0.0);

    p->set<RealType>("Stabilization Parameter", stab_param);

    p->set<Teuchos::RCP<ParamLib>>("Parameter Library", paramLib);

    // Output
    p->set<std::string>("Residual Name", "Pore_Pressure Residual");
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "Node Scalar Data Layout", dl_->node_scalar);

    ev = Teuchos::rcp(
        new LCM::TLPoroPlasticityResidMass<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    // Output QP pore pressure
    // IKT: commenting this out b/c it is a duplicate of earlier writing of porePressure
    // to the output.  The current DAG rules in Trilinos as of 12/6/2018 do 
    // not allow for such duplicates.  If Pore_Pressure at IPs is needed,
    // one needs to create a different name for this field and uncomment code below.

    /*bool const output_ip = material_db_->getElementBlockParam<bool>(
        eb_name, "Output IP" + porePressure, false);

    p = stateMgr.registerStateVariable(
        porePressure,
        dl_->qp_scalar,
        dl_->dummy,
        eb_name,
        "scalar",
        0.0,
        true,
        output_ip);

    ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);*/
  }

  if (have_pore_pressure_eq_ && surface_element) {
    // Pore Pressure Resid for Surface

    Teuchos::RCP<Teuchos::ParameterList> p =
        Teuchos::rcp(new Teuchos::ParameterList("Pore_Pressure Residual"));

    // Input
    p->set<RealType>("thickness", thickness);
    p->set<Teuchos::RCP<Intrepid2::Cubature<PHX::Device>>>(
        "Cubature", surfaceCubature);
    p->set<Intrepid2Basis>("Intrepid2 Basis", surfaceBasis);
    p->set<std::string>(
        "Surface Scalar Gradient Operator Pore Pressure Name",
        "Surface Scalar Gradient Operator Pore Pressure");
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

    // Output
    p->set<std::string>("Residual Name", "Pore_Pressure Residual");
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "Node Scalar Data Layout", dl_->node_scalar);

    ev = Teuchos::rcp(
        new LCM::SurfaceTLPoroMassResidual<EvalT, PHAL::AlbanyTraits>(*p, dl_));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_transport_eq_ || have_transport_) {  // Transport Coefficients

    Teuchos::RCP<Teuchos::ParameterList> p =
        Teuchos::rcp(new Teuchos::ParameterList("Transport Coefficients"));

    Teuchos::ParameterList& param_sublist =
        material_db_->getElementBlockSublist(eb_name, matName)
            .sublist("Transport Coefficients");

    p->set<Teuchos::ParameterList*>("Material Parameters", &param_sublist);

    // Input
    p->set<std::string>("Lattice Concentration Name", transport);
    p->set<std::string>("Deformation Gradient Name", defgrad);
    p->set<std::string>("Determinant of F Name", J);
    p->set<std::string>("Temperature Name", temperature);

    if (material_model_name == "J2" ||
        material_model_name == "Elasto Viscoplastic") {
      p->set<std::string>("Equivalent Plastic Strain Name", eqps);
      p->set<std::string>("Strain Rate Factor Name", strainRateFactor);
    }

    p->set<bool>("Weighted Volume Average J", volume_average_j);
    p->set<RealType>(
        "Average J Stabilization Parameter",
        volume_average_stabilization_param);

    p->set<Teuchos::RCP<std::map<std::string, std::string>>>("Name Map", fnm);

    // Output
    p->set<std::string>("Trapped Concentration Name", trappedConcentration);
    p->set<std::string>("Total Concentration Name", totalConcentration);
    p->set<std::string>("Mechanical Deformation Gradient Name", "Fm");
    p->set<std::string>("Effective Diffusivity Name", effectiveDiffusivity);
    p->set<std::string>("Trapped Solvent Name", trappedSolvent);

    p->set<std::string>("Diffusion Coefficient Name", diffusionCoefficient);
    p->set<std::string>("Tau Contribution Name", convectionCoefficient);
    p->set<std::string>(
        "Concentration Equilibrium Parameter Name", eqilibriumParameter);

    ev = Teuchos::rcp(
        new LCM::TransportCoefficients<EvalT, PHAL::AlbanyTraits>(*p, dl_));
    fm0.template registerEvaluator<EvalT>(ev);

    bool const output_totc = material_db_->getElementBlockParam<bool>(
        eb_name, "Output " + trappedConcentration, false);

    if (output_totc) {
      p = stateMgr.registerStateVariable(
          trappedConcentration,
          dl_->qp_scalar,
          dl_->dummy,
          eb_name,
          "scalar",
          0.0,
          false,
          true);

      ev =
          Teuchos::rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    bool const output_trapc = material_db_->getElementBlockParam<bool>(
        eb_name, "Output " + totalConcentration, false);

    if (output_trapc) {
      p = stateMgr.registerStateVariable(
          totalConcentration,
          dl_->qp_scalar,
          dl_->dummy,
          eb_name,
          "scalar",
          0.0,
          true,
          true);

      ev =
          Teuchos::rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    // Strain Rate Factor
    bool const output_srfac = material_db_->getElementBlockParam<bool>(
        eb_name, "Output " + strainRateFactor, false);

    if (output_srfac) {
      p = stateMgr.registerStateVariable(
          strainRateFactor,
          dl_->qp_scalar,
          dl_->dummy,
          eb_name,
          "scalar",
          0.0,
          false,
          true);

      ev =
          Teuchos::rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    // Convection Coefficient
    bool const output_conv = material_db_->getElementBlockParam<bool>(
        eb_name, "Output " + convectionCoefficient, false);

    if (output_conv) {
      p = stateMgr.registerStateVariable(
          convectionCoefficient,
          dl_->qp_scalar,
          dl_->dummy,
          eb_name,
          "scalar",
          0.0,
          false,
          true);

      ev =
          Teuchos::rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    // Diffusion Coefficient
    bool const output_diff = material_db_->getElementBlockParam<bool>(
        eb_name, "Output " + diffusionCoefficient, false);

    if (output_diff) {
      p = stateMgr.registerStateVariable(
          diffusionCoefficient,
          dl_->qp_scalar,
          dl_->dummy,
          eb_name,
          "scalar",
          1.0,
          false,
          true);

      ev =
          Teuchos::rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    // Effective Diffusivity
    bool const output_effd = material_db_->getElementBlockParam<bool>(
        eb_name, "Output " + effectiveDiffusivity, false);

    if (output_effd) {
      p = stateMgr.registerStateVariable(
          effectiveDiffusivity,
          dl_->qp_scalar,
          dl_->dummy,
          eb_name,
          "scalar",
          1.0,
          false,
          true);

      ev =
          Teuchos::rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }
  }

  // Helium ODEs
  if (have_transport_) {
    // Get material list prior to establishing a new parameter list

    // Check if Tritium Sublist exists. If true, move forward
    if (param_list.isSublist("Tritium Coefficients")) {
      Teuchos::RCP<Teuchos::ParameterList> p =
          Teuchos::rcp(new Teuchos::ParameterList("Helium ODEs"));

      // Rather than combine lists, we choose to invoke multiple parameter
      // lists and stuff them separately into p.
      // All lists need to be reflected in HeliumODEs_Def.hpp
      Teuchos::ParameterList& transport_param =
          material_db_->getElementBlockSublist(eb_name, matName)
              .sublist("Transport Coefficients");

      Teuchos::ParameterList& tritium_param =
          material_db_->getElementBlockSublist(eb_name, matName)
              .sublist("Tritium Coefficients");

      Teuchos::ParameterList& molar_param =
          material_db_->getElementBlockSublist(eb_name, matName)
              .sublist("Molar Volume");

      p->set<Teuchos::ParameterList*>("Transport Parameters", &transport_param);
      p->set<Teuchos::ParameterList*>("Tritium Parameters", &tritium_param);
      p->set<Teuchos::ParameterList*>("Molar Volume", &molar_param);

      // Input
      p->set<std::string>("Total Concentration Name", totalConcentration);
      p->set<std::string>("Delta Time Name", "Delta Time");
      p->set<std::string>("Diffusion Coefficient Name", diffusionCoefficient);
      // Output
      p->set<std::string>("He Concentration Name", he_concentration);
      p->set<std::string>("Total Bubble Density Name", total_bubble_density);
      p->set<std::string>(
          "Bubble Volume Fraction Name", bubble_volume_fraction);

      ev =
          Teuchos::rcp(new LCM::HeliumODEs<EvalT, PHAL::AlbanyTraits>(*p, dl_));
      fm0.template registerEvaluator<EvalT>(ev);

      // Outputting state variables
      //
      // Using field names registered for surface elements
      // (he_concentration, etc.)
      // NOTE: All output variables are stated
      //
      // helium concentration

      bool const output_he = material_db_->getElementBlockParam<bool>(
          eb_name, "Output " + he_concentration, false);

      if (output_he) {
        p = stateMgr.registerStateVariable(
            he_concentration,
            dl_->qp_scalar,
            dl_->dummy,
            eb_name,
            "scalar",
            0.0,
            true,
            true);

        ev = Teuchos::rcp(
            new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));

        fm0.template registerEvaluator<EvalT>(ev);
      }
      // total bubble density
      bool const output_bubb_tot = material_db_->getElementBlockParam<bool>(
          eb_name, "Output " + total_bubble_density, false);

      if (output_bubb_tot) {
        Teuchos::RCP<Teuchos::ParameterList> const p =
            stateMgr.registerStateVariable(
                total_bubble_density,
                dl_->qp_scalar,
                dl_->dummy,
                eb_name,
                "scalar",
                0.0,
                true,
                true);

        ev = Teuchos::rcp(
            new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
        fm0.template registerEvaluator<EvalT>(ev);
      }
      // bubble volume fraction
      bool const output_bubb_frac = material_db_->getElementBlockParam<bool>(
          eb_name, "Output " + bubble_volume_fraction, false);

      if (output_bubb_frac) {
        Teuchos::RCP<Teuchos::ParameterList> const p =
            stateMgr.registerStateVariable(
                bubble_volume_fraction,
                dl_->qp_scalar,
                dl_->dummy,
                eb_name,
                "scalar",
                0.0,
                true,
                true);

        ev = Teuchos::rcp(
            new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
        fm0.template registerEvaluator<EvalT>(ev);
      }
    }
  }

  // Transport of the temperature field
  if (have_temperature_eq_ && !surface_element) {
    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
        new Teuchos::ParameterList("ThermoMechanical Coefficients"));

    p->set<Teuchos::ParameterList*>("Material Parameters", &param_list);

    // Input
    p->set<std::string>("Temperature Name", "Temperature");
    p->set<std::string>("Temperature Dot Name", "Temperature Dot");
    if (SolutionType == SolutionMethodType::Continuation) {
      p->set<std::string>("Solution Method Type", "Continuation");
    } else {
      p->set<std::string>("Solution Method Type", "No Continuation");
    }
    p->set<std::string>("Thermal Conductivity Name", "Thermal Conductivity");
    p->set<std::string>(
        "Thermal Transient Coefficient Name", "Thermal Transient Coefficient");
    p->set<std::string>("Delta Time Name", "Delta Time");

    // MJJ: Need this here to compute responses later
    RealType const heat_capacity = param_list.get<RealType>("Heat Capacity");

    RealType const density = param_list.get<RealType>("Density");

    pFromProb->set<RealType>("Heat Capacity", heat_capacity);
    pFromProb->set<RealType>("Density", density);

    if (have_mech_eq_) {
      p->set<bool>("Have Mechanics", true);
      p->set<std::string>("Deformation Gradient Name", defgrad);
    }

    // Output
    p->set<std::string>("Thermal Diffusivity Name", "Thermal Diffusivity");
    //    p->set<std::string>("Temperature Dot Name", "Temperature Dot");

    ev = Teuchos::rcp(
        new LCM::ThermoMechanicalCoefficients<EvalT, PHAL::AlbanyTraits>(
            *p, dl_));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // Equation for ACE temperature
  if (have_ace_temperature_eq_ == true && surface_element == false) {
    Teuchos::RCP<Teuchos::ParameterList> p =
        Teuchos::rcp(new Teuchos::ParameterList("ACE Temperature Residual"));

    p->set<Teuchos::ParameterList*>("Material Parameters", &param_list);

    // Input
    p->set<std::string>("Weighted BF Name", "wBF");
    p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
    p->set<std::string>("ACE Temperature Name", "ACE Temperature");
    p->set<std::string>("ACE Temperature Dot Name", "ACE Temperature Dot");
    p->set<std::string>(
        "ACE Temperature Gradient Name", "ACE Temperature Gradient");
    p->set<std::string>(
        "ACE Thermal Conductivity Name", "ACE Thermal Conductivity");
    p->set<std::string>("ACE Thermal Inertia Name", "ACE Thermal Inertia");
    p->set<std::string>("ACE Residual Name", "ACE Temperature Residual");
    if (SolutionType == SolutionMethodType::Continuation) {
      p->set<std::string>("Solution Method Type", "Continuation");
    } else {
      p->set<std::string>("Solution Method Type", "No Continuation");
    }
    p->set<std::string>("Delta Time Name", "Delta Time");

    if (have_mech_eq_) {
      p->set<bool>("Have Mechanics", true);
      p->set<std::string>("Deformation Gradient Name", defgrad);
    }
    ev = Teuchos::rcp(
        new LCM::ACETemperatureResidual<EvalT, PHAL::AlbanyTraits>(*p, dl_));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // Transport of the temperature field
  if (have_temperature_eq_ && !surface_element) {
    Teuchos::RCP<Teuchos::ParameterList> p =
        Teuchos::rcp(new Teuchos::ParameterList("Temperature Residual"));

    // Input
    p->set<std::string>("Scalar Variable Name", "Temperature");
    p->set<std::string>(
        "Scalar Gradient Variable Name", "Temperature Gradient");
    if (have_mech_eq_) {
      p->set<std::string>(
          "Velocity Gradient Variable Name", "Velocity Gradient");
      p->set<std::string>("Stress Name", firstPK);
      p->set<bool>("Have Mechanics", true);
    }
    p->set<std::string>("Weights Name", "Weights");
    p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
    p->set<std::string>("Weighted BF Name", "wBF");

    // Transient
    p->set<bool>("Have Transient", true);
    p->set<std::string>("Scalar Dot Name", "Temperature Dot");
    p->set<std::string>(
        "Transient Coefficient Name", "Thermal Transient Coefficient");

    if (SolutionType == SolutionMethodType::Continuation) {
      p->set<std::string>("Solution Method Type", "Continuation");
    } else {
      p->set<std::string>("Solution Method Type", "No Continuation");
    }

    // Diffusion
    p->set<bool>("Have Diffusion", true);
    p->set<std::string>("Diffusivity Name", "Thermal Diffusivity");

    // Source
    // TODO: Make this more general
    if ((have_mech_ || have_mech_eq_) &&
        (material_model_name == "J2" ||
         material_model_name == "CrystalPlasticity")) {
      p->set<bool>("Have Source", true);
      p->set<std::string>("Source Name", mech_source);
    }

    // Thermal Source (internal energy generation)
    if (thermal_source_evaluated_) {
      p->set<bool>("Have Second Source", true);
      p->set<std::string>("Second Source Name", "Heat Source");
    }

    // Output
    p->set<std::string>("Residual Name", "Temperature Residual");

    p->set<std::string>("Delta Time Name", "Delta Time");

    ev = Teuchos::rcp(
        new LCM::TransportResidual<EvalT, PHAL::AlbanyTraits>(*p, dl_));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // Hydrogen Transport model proposed in Foulk et al 2014
  if (have_transport_eq_ && !surface_element) {
    Teuchos::RCP<Teuchos::ParameterList> p =
        Teuchos::rcp(new Teuchos::ParameterList("Transport Residual"));

    // Input
    p->set<std::string>("Element Length Name", gradient_element_length);
    p->set<std::string>("Weighted BF Name", "wBF");
    p->set<std::string>("Weights Name", "Weights");
    p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
    p->set<std::string>("Gradient BF Name", "Grad BF");
    if ((have_mech_ || have_mech_eq_) &&
        (material_model_name == "J2" ||
         material_model_name == "Elasto Viscoplastic")) {
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
    p->set<std::string>(
        "Gradient Hydrostatic Stress Name", "HydroStress Gradient");
    p->set<std::string>("Stress Name", cauchy);
    p->set<std::string>("Delta Time Name", "Delta Time");
    RealType stab_param(0.0);

    if (material_db_->isElementBlockParam(eb_name, "Stabilization Parameter")) {
      stab_param = material_db_->getElementBlockParam<RealType>(
          eb_name, "Stabilization Parameter");
    }
    p->set<RealType>("Stabilization Parameter", stab_param);
    p->set<Teuchos::RCP<ParamLib>>("Parameter Library", paramLib);

    RealType decay_constant{0.0};

    // Check if Tritium Sublist exists. If true, move forward
    if (param_list.isSublist("Tritium Coefficients")) {
      Teuchos::ParameterList& tritium_param =
          material_db_->getElementBlockSublist(eb_name, matName)
              .sublist("Tritium Coefficients");
      decay_constant =
          tritium_param.get<RealType>("Tritium Decay Constant", 0.0);
    }
    p->set<RealType>("Tritium Decay Constant", decay_constant);

    // Output
    p->set<std::string>("Residual Name", "Transport Residual");

    ev = Teuchos::rcp(
        new LCM::HDiffusionDeformationMatterResidual<EvalT, PHAL::AlbanyTraits>(
            *p, dl_));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_transport_eq_ && surface_element) {  // Transport Resid for Surface

    Teuchos::RCP<Teuchos::ParameterList> p =
        Teuchos::rcp(new Teuchos::ParameterList("Transport Residual"));

    // Input
    p->set<RealType>("thickness", thickness);
    p->set<Teuchos::RCP<Intrepid2::Cubature<PHX::Device>>>(
        "Cubature", surfaceCubature);
    p->set<Intrepid2Basis>("Intrepid2 Basis", surfaceBasis);
    p->set<std::string>(
        "Surface Scalar Gradient Operator Transport Name",
        "Surface Scalar Gradient Operator Transport");
    p->set<std::string>(
        "Surface Transport Gradient Name", "Surface Transport Gradient");
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
    p->set<std::string>(
        "Surface HydroStress Gradient Name", "Surface HydroStress Gradient");
    p->set<std::string>("eqps Name", eqps);
    p->set<std::string>("Delta Time Name", "Delta Time");
    if (have_mech_eq_) {
      p->set<std::string>("DefGrad Name", defgrad);
      p->set<std::string>("DetDefGrad Name", J);
    }

    RealType stab_param(0.0);
    if (material_db_->isElementBlockParam(eb_name, "Stabilization Parameter")) {
      stab_param = material_db_->getElementBlockParam<RealType>(
          eb_name, "Stabilization Parameter");
    }
    p->set<RealType>("Stabilization Parameter", stab_param);
    p->set<Teuchos::RCP<ParamLib>>("Parameter Library", paramLib);

    // Output
    p->set<std::string>("Residual Name", "Transport Residual");
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "Node Scalar Data Layout", dl_->node_scalar);

    ev = Teuchos::rcp(
        new LCM::SurfaceHDiffusionDefResidual<EvalT, PHAL::AlbanyTraits>(
            *p, dl_));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_hydrostress_eq_ && !surface_element) {
    // L2 hydrostatic stress projection

    Teuchos::RCP<Teuchos::ParameterList> p =
        Teuchos::rcp(new Teuchos::ParameterList("HydroStress Residual"));

    // Input
    p->set<std::string>("Weighted BF Name", "wBF");
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "Node QP Scalar Data Layout", dl_->node_qp_scalar);

    p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "Node QP Vector Data Layout", dl_->node_qp_vector);

    p->set<bool>("Have Source", false);
    p->set<std::string>("Source Name", "Source");

    p->set<std::string>("Deformation Gradient Name", defgrad);
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "QP Tensor Data Layout", dl_->qp_tensor);

    p->set<std::string>("QP Variable Name", hydroStress);
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "QP Scalar Data Layout", dl_->qp_scalar);

    p->set<std::string>("Stress Name", cauchy);
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "QP Tensor Data Layout", dl_->qp_tensor);

    // Output
    p->set<std::string>("Residual Name", "HydroStress Residual");
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "Node Scalar Data Layout", dl_->node_scalar);

    ev = Teuchos::rcp(
        new LCM::ScalarL2ProjectionResidual<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_hydrostress_eq_ && surface_element) {
    // Hydrostress Projection Resid for Surface
    Teuchos::RCP<Teuchos::ParameterList> p =
        Teuchos::rcp(new Teuchos::ParameterList("HydroStress Residual"));

    // Input
    p->set<RealType>("thickness", thickness);
    p->set<Teuchos::RCP<Intrepid2::Cubature<PHX::Device>>>(
        "Cubature", surfaceCubature);
    p->set<Intrepid2Basis>("Intrepid2 Basis", surfaceBasis);
    p->set<std::string>(
        "Surface Scalar Gradient Operator HydroStress Name",
        "Surface Scalar Gradient Operator HydroStress");
    p->set<std::string>("Current Basis Name", "Current Basis");
    p->set<std::string>("Reference Dual Basis Name", "Reference Dual Basis");
    p->set<std::string>("Reference Normal Name", "Reference Normal");
    p->set<std::string>("Reference Area Name", "Weights");
    p->set<std::string>("HydoStress Name", hydroStress);
    p->set<std::string>("Cauchy Stress Name", cauchy);
    p->set<std::string>("Jacobian Name", J);

    // Output
    p->set<std::string>("Residual Name", "HydroStress Residual");
    p->set<Teuchos::RCP<PHX::DataLayout>>(
        "Node Scalar Data Layout", dl_->node_scalar);

    ev = Teuchos::rcp(
        new LCM::SurfaceL2ProjectionResidual<EvalT, PHAL::AlbanyTraits>(
            *p, dl_));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_stab_pressure_eq_) {
    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
        new Teuchos::ParameterList("Stabilized Pressure Residual"));

    // Input
    p->set<std::string>("Shear Modulus Name", "Shear Modulus");
    p->set<std::string>("Bulk Modulus Name", "Bulk Modulus");
    p->set<std::string>("Deformation Gradient Name", defgrad);
    p->set<std::string>("Stress Name", cauchy);
    p->set<std::string>("Pressure Name", pressure);
    p->set<std::string>("Pressure Gradient Name", "Pressure Gradient");
    p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
    p->set<std::string>("Weighted BF Name", "wBF");
    p->set<std::string>(
        "Element Characteristic Length Name", gradient_element_length);

    RealType stab_param{0.0};

    if (material_db_->isElementBlockParam(eb_name, "Stabilization Parameter")) {
      stab_param = material_db_->getElementBlockParam<RealType>(
          eb_name, "Stabilization Parameter");
    }
    p->set<RealType>("Stabilization Parameter", stab_param);
    p->set<bool>("Small Strain", small_strain);

    // Output
    p->set<std::string>("Residual Name", "Pressure Residual");
    ev = Teuchos::rcp(
        new LCM::StabilizedPressureResidual<EvalT, PHAL::AlbanyTraits>(
            *p, dl_));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (Teuchos::nonnull(rc_mgr_)) rc_mgr_->createEvaluators<EvalT>(fm0, dl_);

  if (fieldManagerChoice == BUILD_RESID_FM) {
    Teuchos::RCP<const PHX::FieldTag> ret_tag;

    if (have_mech_eq_) {
      PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter", dl_->dummy);

      fm0.requireField<EvalT>(res_tag);
      ret_tag = res_tag.clone();
    }
    if (have_pore_pressure_eq_) {
      PHX::Tag<typename EvalT::ScalarT> pore_tag(
          "Scatter Pore_Pressure", dl_->dummy);

      fm0.requireField<EvalT>(pore_tag);
      ret_tag = pore_tag.clone();
    }
    if (have_stab_pressure_eq_) {
      PHX::Tag<typename EvalT::ScalarT> pres_tag(
          "Scatter Pressure", dl_->dummy);

      fm0.requireField<EvalT>(pres_tag);
      ret_tag = pres_tag.clone();
    }
    if (have_temperature_eq_) {
      PHX::Tag<typename EvalT::ScalarT> temperature_tag(
          "Scatter Temperature", dl_->dummy);

      fm0.requireField<EvalT>(temperature_tag);
      ret_tag = temperature_tag.clone();
    }
    if (have_ace_temperature_eq_) {
      PHX::Tag<typename EvalT::ScalarT> ace_temperature_tag(
          "Scatter ACE Temperature", dl_->dummy);

      fm0.requireField<EvalT>(ace_temperature_tag);
      ret_tag = ace_temperature_tag.clone();
    }
    if (have_transport_eq_) {
      PHX::Tag<typename EvalT::ScalarT> transport_tag(
          "Scatter Transport", dl_->dummy);

      fm0.requireField<EvalT>(transport_tag);
      ret_tag = transport_tag.clone();
    }
    if (have_hydrostress_eq_) {
      PHX::Tag<typename EvalT::ScalarT> l2projection_tag(
          "Scatter HydroStress", dl_->dummy);

      fm0.requireField<EvalT>(l2projection_tag);
      ret_tag = l2projection_tag.clone();
    }
    return ret_tag;
  } else if (fieldManagerChoice == BUILD_RESPONSE_FM) {
    ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl_);

    return respUtils.constructResponses(
        fm0, *responseList, pFromProb, stateMgr, &meshSpecs);
  }

  return Teuchos::null;

}  // constructEvaluators

}  // namespace Albany
