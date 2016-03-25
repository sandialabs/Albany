//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#if 0
#if !defined(LCM_SolidMechanics_h)
#define LCM_SolidMechanics_h

#include "Albany_AbstractProblem.hpp"
#include "NOX_StatusTest_ModelEvaluatorFlag.h"
#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_Workset.hpp"
#include "Phalanx.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

namespace Albany
{

///
/// Definition for the Mechanics Problem
///
class SolidMechanics: public Albany::AbstractProblem
{
public:

  using FieldContainer =
  Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device>;

  ///
  /// Default constructor
  ///
  SolidMechanics(
      Teuchos::RCP<Teuchos::ParameterList> const & params,
      Teuchos::RCP<ParamLib> const & param_lib,
      int const num_dims,
      Teuchos::RCP<Teuchos::Comm<int> const> & comm);
  ///
  /// Destructor
  ///
  virtual
  ~SolidMechanics();

  ///
  /// Return number of spatial dimensions
  ///
  virtual
  int
  spatialDimension() const final
  {
    return num_dims_;
  }

  ///
  /// Build the PDE instantiations, boundary conditions, initial solution
  ///
  virtual
  void
  buildProblem(
      Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct>> mesh_specs,
      StateManager & state_mgr);

  ///
  /// Build evaluators
  ///
  virtual Teuchos::Array<Teuchos::RCP<PHX::FieldTag const>>
  buildEvaluators(
      PHX::FieldManager<PHAL::AlbanyTraits> & field_mgr,
      Albany::MeshSpecsStruct const & mesh_specs,
      Albany::StateManager & state_mgr,
      Albany::FieldManagerChoice fm_choice,
      Teuchos::RCP<Teuchos::ParameterList> const & response_list);

  ///
  /// Each problem must generate its list of valid parameters
  ///
  Teuchos::RCP<Teuchos::ParameterList const>
  getValidProblemParameters() const;

  ///
  /// Retrieve the state data
  ///
  virtual
  void
  getAllocatedStates(
      Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<FieldContainer>>>
      old_state,
      Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<FieldContainer>>>
      new_state) const;

  ///
  /// Add a custom NOX Status Test,
  /// for example, to trigger a global load step reduction.
  ///
  virtual
  void
  applyProblemSpecificSolverSettings(
      Teuchos::RCP<Teuchos::ParameterList> params);

  ///
  /// No copy constructor
  ///
  SolidMechanics(SolidMechanics const &) = delete;

  ///
  /// No copy assignment
  ///
  SolidMechanics& operator=(SolidMechanics const &) = delete;

  ///
  /// Main problem setup routine.
  /// Not directly called, but indirectly by following functions
  ///
  template<typename EvalT>
  Teuchos::RCP<PHX::FieldTag const>
      constructEvaluators(
          PHX::FieldManager<PHAL::AlbanyTraits> & field_mgr,
          Albany::MeshSpecsStruct const & mesh_specs,
          Albany::StateManager & state_mgr,
          Albany::FieldManagerChoice fm_choice,
          Teuchos::RCP<Teuchos::ParameterList> const & response_list);

      ///
      /// Setup for the dirichlet BCs
      ///
      void
      constructDirichletEvaluators(
          Albany::MeshSpecsStruct const & mesh_specs);

      ///
      /// Setup for the traction BCs
      ///
      void
      constructNeumannEvaluators(
          Albany::MeshSpecsStruct const & mesh_specs);

      //----------------------------------------------------------------------------
    protected:

      ///
      /// num of dimensions
      ///
      int num_dims_;

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
      Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<FieldContainer>>> old_state_;

      ///
      /// new state data
      ///
      Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<FieldContainer>>> new_state_;

      ///
      /// Reference configuration manager for mesh adaptation with ref config
      /// updating.
      ///
      Teuchos::RCP<AAdapt::rc::Manager> rc_mgr_;

      ///
      /// User defined NOX Status Test that allows model evaluators to set the NOX status to "failed".
      /// This is useful because it forces a global load step reduction.
      ///
      Teuchos::RCP<NOX::StatusTest::Generic> userDefinedNOXStatusTest;
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
      Albany::SolidMechanics::
      constructEvaluators(PHX::FieldManager<PHAL::AlbanyTraits>& field_mgr,
          const Albany::MeshSpecsStruct& mesh_specs,
          Albany::StateManager& state_mgr,
          Albany::FieldManagerChoice fieldManagerChoice,
          const Teuchos::RCP<Teuchos::ParameterList>& response_list)
      {
        typedef Teuchos::RCP<
            Intrepid2::Basis<RealType,
                Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,
                    PHX::Device>>>
      Intrepid2Basis;

      // Collect problem-specific response parameters

      Teuchos::RCP<Teuchos::ParameterList> pFromProb = Teuchos::rcp(
          new Teuchos::ParameterList("Response Parameters from Problem"));

      // get the name of the current element block
      std::string eb_name = mesh_specs.ebName;

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
      *out << "In SolidMechanics::constructEvaluators" << std::endl;
      *out << "element block name: " << eb_name << std::endl;
      *out << "material model name: " << material_model_name << std::endl;
#endif

      // insert user-defined NOX Status Test for material models that use it
      {
        std::string matName = material_db_->getElementBlockParam<std::string>(
            eb_name,
            "material");
        Teuchos::ParameterList& param_list = material_db_
        ->getElementBlockSublist(eb_name, matName);
        std::string materialModelName = param_list.sublist("Material Model")
        .get<std::string>("Model Name");
        if (materialModelName == "CrystalPlasticity") {
          Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag> statusTest =
          Teuchos::rcp_dynamic_cast<NOX::StatusTest::ModelEvaluatorFlag>(
              userDefinedNOXStatusTest);
          param_list.set<Teuchos::RCP<NOX::StatusTest::ModelEvaluatorFlag> >(
              "NOX Status Test",
              statusTest);
        }
      }

      // define cell topologies
      Teuchos::RCP<shards::CellTopology> comp_cellType =
      Teuchos::rcp(
          new shards::CellTopology(
              shards::getCellTopologyData<shards::Tetrahedron<11>>()));
      Teuchos::RCP<shards::CellTopology> cellType =
      Teuchos::rcp(new shards::CellTopology(&mesh_specs.ctd));

      // volume averaging flags
      bool volume_average_j(false);
      bool volume_average_pressure(false);
      RealType volume_average_stabilization_param(0.0);
      if (material_db_->isElementBlockParam(
              eb_name,
              "Weighted Volume Average J"))
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
        material_db_->getElementBlockParam<bool>(
            eb_name,
            "Surface Element");
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

      if (material_db_->isElementBlockParam(
              eb_name,
              "Compute Membrane Forces")) {
        compute_membrane_forces = material_db_->getElementBlockParam<bool>(
            eb_name,
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
      intrepidBasis = Albany::getIntrepid2Basis(mesh_specs.ctd, composite);

      if (composite &&
          mesh_specs.ctd.dimension == 3 &&
          mesh_specs.ctd.node_count == 10) cellType = comp_cellType;

      Intrepid2::DefaultCubatureFactory<RealType,
      Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device> > cubFactory;
      Teuchos::RCP<
      Intrepid2::Cubature<RealType,
      Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,
      PHX::Device>>>cubature =
      cubFactory.create(*cellType, mesh_specs.cubatureDegree);

      // FIXME, this could probably go into the ProblemUtils
      // just like the call to getIntrepid2Basis
      Intrepid2Basis
      surfaceBasis;
      Teuchos::RCP<shards::CellTopology> surfaceTopology;
      Teuchos::RCP<
      Intrepid2::Cubature<RealType,
      Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,
      PHX::Device>>>surfaceCubature;
      if (surface_element)
      {
#ifdef ALBANY_VERBOSE
      *out << "In Surface Element Logic" << std::endl;
#endif

      std::string name = mesh_specs.ctd.name;
      if (name == "Triangle_3" || name == "Quadrilateral_4") {
        surfaceBasis =
        Teuchos::rcp(
            new Intrepid2::Basis_HGRAD_LINE_C1_FEM<RealType,
            Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,
            PHX::Device>>());
        surfaceTopology =
        Teuchos::rcp(
            new shards::CellTopology(
                shards::getCellTopologyData<shards::Line<2>>()));
        surfaceCubature =
        cubFactory.create(*surfaceTopology, mesh_specs.cubatureDegree);
      }
      else if (name == "Wedge_6") {
        surfaceBasis =
        Teuchos::rcp(
            new Intrepid2::Basis_HGRAD_TRI_C1_FEM<RealType,
            Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,
            PHX::Device>>());
        surfaceTopology =
        Teuchos::rcp(
            new shards::CellTopology(
                shards::getCellTopologyData<shards::Triangle<3>>()));
        surfaceCubature =
        cubFactory.create(*surfaceTopology, mesh_specs.cubatureDegree);
      }
      else if (name == "Hexahedron_8") {
        surfaceBasis =
        Teuchos::rcp(
            new Intrepid2::Basis_HGRAD_QUAD_C1_FEM<RealType,
            Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,
            PHX::Device>>());
        surfaceTopology =
        Teuchos::rcp(
            new shards::CellTopology(
                shards::getCellTopologyData<shards::Quadrilateral<4>>()));
        surfaceCubature =
        cubFactory.create(*surfaceTopology, mesh_specs.cubatureDegree);
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
    const int workset_size = mesh_specs.worksetSize;

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
          field_mgr.template registerEvaluator<EvalT>
          (evalUtils.constructGatherSolutionEvaluator_withAcceleration(
                  true,
                  dof_names,
                  dof_names_dot,
                  dof_names_dotdot));
        } else {
          field_mgr.template registerEvaluator<EvalT>
          (evalUtils.constructGatherSolutionEvaluator_noTransient(true,
                  dof_names));
        }

        field_mgr.template registerEvaluator<EvalT>
        (evalUtils.constructGatherCoordinateVectorEvaluator());

        if (!surface_element) {
          field_mgr.template registerEvaluator<EvalT>
          (evalUtils.constructDOFVecInterpolationEvaluator(dof_names[0]));

          field_mgr.template registerEvaluator<EvalT>
          (evalUtils.constructDOFVecInterpolationEvaluator(dof_names_dot[0]));

          field_mgr.template registerEvaluator<EvalT>
          (
              evalUtils.constructDOFVecInterpolationEvaluator(
                  dof_names_dotdot[0]));

          field_mgr.template registerEvaluator<EvalT>
          (evalUtils.constructDOFVecGradInterpolationEvaluator(dof_names[0]));

          field_mgr.template registerEvaluator<EvalT>
          (evalUtils.constructMapToPhysicalFrameEvaluator(cellType,
                  cubature));

          field_mgr.template registerEvaluator<EvalT>
          (evalUtils.constructComputeBasisFunctionsEvaluator(cellType,
                  intrepidBasis,
                  cubature));
        }

        field_mgr.template registerEvaluator<EvalT>
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
        field_mgr.template registerEvaluator<EvalT>(ev);

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
        field_mgr.template registerEvaluator<EvalT>(ev);
        p = state_mgr.registerStateVariable("Time",
            dl_->workset_scalar,
            dl_->dummy,
            eb_name,
            "scalar",
            0.0,
            true);
        ev = Teuchos::rcp(
            new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
        field_mgr.template registerEvaluator<EvalT>(ev);
      }

      if (have_mech_eq_) { // Current Coordinates
        Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
            new Teuchos::ParameterList("Current Coordinates"));
        p->set<std::string>("Reference Coordinates Name", "Coord Vec");
        p->set<std::string>("Displacement Name", "Displacement");
        p->set<std::string>("Current Coordinates Name", "Current Coordinates");
        ev = Teuchos::rcp(
            new LCM::CurrentCoords<EvalT, PHAL::AlbanyTraits>(*p, dl_));
        field_mgr.template registerEvaluator<EvalT>(ev);
      }

      if (have_mech_eq_ && have_sizefield_adaptation_) { // Mesh size field
        Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
            new Teuchos::ParameterList("Isotropic Mesh Size Field"));
        p->set<std::string>("IsoTropic MeshSizeField Name", "IsoMeshSizeField");
        p->set<std::string>("Current Coordinates Name", "Current Coordinates");
        p
        ->set<
        Teuchos::RCP<
        Intrepid2::Cubature<RealType,
        Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,
        PHX::Device>>>>("Cubature", cubature);

        // Get the Adaptation list and send to the evaluator
        Teuchos::ParameterList& paramList = params->sublist("Adaptation");
        p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

        p
        ->set<
        const Teuchos::RCP<
        Intrepid2::Basis<RealType,
        Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,
        PHX::Device>>>>("Intrepid2 Basis", intrepidBasis);
        ev = Teuchos::rcp(
            new LCM::IsoMeshSizeField<EvalT, PHAL::AlbanyTraits>(*p, dl_));
        field_mgr.template registerEvaluator<EvalT>(ev);

        // output mesh size field if requested
        /*
         bool output_flag = false;
         if (material_db_->isElementBlockParam(eb_name, "Output MeshSizeField"))
         output_flag =
         material_db_->getElementBlockParam<bool>(eb_name, "Output MeshSizeField");
         */
        bool output_flag = true;
        if (output_flag) {
          p = state_mgr.registerStateVariable("IsoMeshSizeField",
              dl_->qp_scalar,
              dl_->dummy,
              eb_name,
              "scalar",
              1.0,
              true,
              output_flag);
          ev = Teuchos::rcp(
              new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
          field_mgr.template registerEvaluator<EvalT>(ev);
        }
      }

      { // Constitutive Model Parameters
        Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
            new Teuchos::ParameterList("Constitutive Model Parameters"));
        std::string matName = material_db_->getElementBlockParam<std::string>(
            eb_name, "material");
        Teuchos::ParameterList& param_list =
        material_db_->getElementBlockSublist(eb_name, matName);

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
        field_mgr.template registerEvaluator<EvalT>(cmpEv);
      }

      if (have_mech_eq_) {
        Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
            new Teuchos::ParameterList("Constitutive Model Interface"));
        std::string matName = material_db_->getElementBlockParam<std::string>(
            eb_name, "material");
        Teuchos::ParameterList& param_list =
        material_db_->getElementBlockSublist(eb_name, matName);

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
        field_mgr.template registerEvaluator<EvalT>(cmiEv);

        // register state variables
        for (int sv(0); sv < cmiEv->getNumStateVars(); ++sv) {
          cmiEv->fillStateVariableStruct(sv);
          p = state_mgr.registerStateVariable(cmiEv->getName(),
              cmiEv->getLayout(),
              dl_->dummy,
              eb_name,
              cmiEv->getInitType(),
              cmiEv->getInitValue(),
              cmiEv->getStateFlag(),
              cmiEv->getOutputFlag());
          ev = Teuchos::rcp(
              new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
          field_mgr.template registerEvaluator<EvalT>(ev);
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
          p
          ->set<
          Teuchos::RCP<
          Intrepid2::Cubature<RealType,
          Intrepid2::FieldContainer_Kokkos<RealType,
          PHX::Layout, PHX::Device>>>>(
              "Cubature",
              surfaceCubature);
          p->set<Intrepid2Basis>("Intrepid2 Basis", surfaceBasis);
          if (have_mech_eq_) {
            p->set<std::string>(
                "Current Coordinates Name",
                "Current Coordinates");
          }

          // outputs
          p->set<std::string>("Reference Basis Name", "Reference Basis");
          p->set<std::string>("Reference Area Name", "Weights");
          p->set<std::string>(
              "Reference Dual Basis Name",
              "Reference Dual Basis");
          p->set<std::string>("Reference Normal Name", "Reference Normal");
          p->set<std::string>("Current Basis Name", "Current Basis");

          ev = Teuchos::rcp(
              new LCM::SurfaceBasis<EvalT, PHAL::AlbanyTraits>(*p, dl_));
          field_mgr.template registerEvaluator<EvalT>(ev);
        }

        if (have_mech_eq_) { // Surface Jump
          //SurfaceVectorJump_Def.hpp
          Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
              new Teuchos::ParameterList("Surface Vector Jump"));

          // inputs
          p
          ->set<
          Teuchos::RCP<
          Intrepid2::Cubature<RealType,
          Intrepid2::FieldContainer_Kokkos<RealType,
          PHX::Layout, PHX::Device>>>>(
              "Cubature",
              surfaceCubature);
          p->set<Intrepid2Basis>("Intrepid2 Basis", surfaceBasis);
          p->set<std::string>("Vector Name", "Current Coordinates");

          // outputs
          p->set<std::string>("Vector Jump Name", "Vector Jump");

          ev = Teuchos::rcp(
              new LCM::SurfaceVectorJump<EvalT, PHAL::AlbanyTraits>(*p, dl_));
          field_mgr.template registerEvaluator<EvalT>(ev);
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
          p
          ->set<
          Teuchos::RCP<
          Intrepid2::Cubature<RealType,
          Intrepid2::FieldContainer_Kokkos<RealType,
          PHX::Layout, PHX::Device>>>>(
              "Cubature",
              surfaceCubature);
          p->set<std::string>("Weights Name", "Weights");
          p->set<std::string>("Current Basis Name", "Current Basis");
          p->set<std::string>(
              "Reference Dual Basis Name",
              "Reference Dual Basis");
          p->set<std::string>("Reference Normal Name", "Reference Normal");
          p->set<std::string>("Vector Jump Name", "Vector Jump");

          // outputs
          p->set<std::string>("Surface Vector Gradient Name", defgrad);
          p->set<std::string>("Surface Vector Gradient Determinant Name", J);

          ev = Teuchos::rcp(
              new LCM::SurfaceVectorGradient<EvalT, PHAL::AlbanyTraits>(
                  *p,
                  dl_));
          field_mgr.template registerEvaluator<EvalT>(ev);

          // optional output of the deformation gradient
          bool output_flag(false);
          if (material_db_->isElementBlockParam(eb_name,
                  "Output Deformation Gradient"))
          output_flag = material_db_->getElementBlockParam<bool>(eb_name,
              "Output Deformation Gradient");

          p = state_mgr.registerStateVariable(defgrad,
              dl_->qp_tensor,
              dl_->dummy,
              eb_name,
              "identity",
              1.0,
              false,
              output_flag);
          ev = Teuchos::rcp(
              new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
          field_mgr.template registerEvaluator<EvalT>(ev);
        }

        if (have_mech_eq_) { // Surface Residual
          // SurfaceVectorResidual_Def.hpp
          Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
              new Teuchos::ParameterList("Surface Vector Residual"));

          // inputs
          p->set<RealType>("thickness", thickness);
          p
          ->set<
          Teuchos::RCP<
          Intrepid2::Cubature<RealType,
          Intrepid2::FieldContainer_Kokkos<RealType,
          PHX::Layout, PHX::Device>>>>("Cubature",
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
            p->set<std::string>(
                "Cohesive Traction Name",
                "Cohesive_Traction");
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
              new LCM::SurfaceVectorResidual<EvalT, PHAL::AlbanyTraits>(
                  *p,
                  dl_));
          field_mgr.template registerEvaluator<EvalT>(ev);
        }
      } else {

        if (have_mech_eq_) { // Kinematics quantities
          Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
              new Teuchos::ParameterList("Kinematics"));

          // set flag for return strain and velocity gradient
          bool have_velocity_gradient(false);
          if (material_db_->isElementBlockParam(eb_name,
                  "Velocity Gradient Flag")) {
            p->set<bool>(
                "Velocity Gradient Flag",
                material_db_->
                getElementBlockParam<bool>(
                    eb_name,
                    "Velocity Gradient Flag"));
            have_velocity_gradient = material_db_->
            getElementBlockParam<bool>(eb_name, "Velocity Gradient Flag");
            if (have_velocity_gradient)
            p->set<std::string>(
                "Velocity Gradient Name",
                "Velocity Gradient");
          }

          // send in integration weights and the displacement gradient
          p->set<std::string>("Weights Name", "Weights");
          p->set<Teuchos::RCP<PHX::DataLayout>>(
              "QP Scalar Data Layout",
              dl_->qp_scalar);
          p->set<std::string>(
              "Gradient QP Variable Name",
              "Displacement Gradient");
          p->set<Teuchos::RCP<PHX::DataLayout>>(
              "QP Tensor Data Layout",
              dl_->qp_tensor);

          //Outputs: F, J
          p->set<std::string>("DefGrad Name", defgrad);//dl_->qp_tensor also
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
          field_mgr.template registerEvaluator<EvalT>(ev);

          // optional output
          bool output_flag(false);
          if (material_db_->isElementBlockParam(eb_name,
                  "Output Deformation Gradient"))
          output_flag =
          material_db_->getElementBlockParam<bool>(eb_name,
              "Output Deformation Gradient");

          if (output_flag) {
            p = state_mgr.registerStateVariable(defgrad,
                dl_->qp_tensor,
                dl_->dummy,
                eb_name,
                "identity",
                1.0,
                false,
                output_flag);
            ev = Teuchos::rcp(
                new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
            field_mgr.template registerEvaluator<EvalT>(ev);
          }

          // optional output of the integration weights
          output_flag = false;
          if (material_db_->isElementBlockParam(eb_name,
                  "Output Integration Weights"))
          output_flag = material_db_->getElementBlockParam<bool>(eb_name,
              "Output Integration Weights");

          if (output_flag) {
            p = state_mgr.registerStateVariable("Weights",
                dl_->qp_scalar,
                dl_->dummy,
                eb_name,
                "scalar",
                0.0,
                false,
                output_flag);
            ev = Teuchos::rcp(
                new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
            field_mgr.template registerEvaluator<EvalT>(ev);
          }

          // Optional output: strain
          if (small_strain) {
            output_flag = false;
            if (material_db_->isElementBlockParam(eb_name, "Output Strain"))
            output_flag =
            material_db_->getElementBlockParam<bool>(eb_name,
                "Output Strain");

            if (output_flag) {
              p = state_mgr.registerStateVariable("Strain",
                  dl_->qp_tensor,
                  dl_->dummy,
                  eb_name,
                  "scalar",
                  0.0,
                  false,
                  output_flag);
              ev = Teuchos::rcp(
                  new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
              field_mgr.template registerEvaluator<EvalT>(ev);
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
              p = state_mgr.registerStateVariable("Velocity Gradient",
                  dl_->qp_tensor,
                  dl_->dummy,
                  eb_name,
                  "scalar",
                  0.0,
                  false,
                  output_flag);
              ev = Teuchos::rcp(
                  new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
              field_mgr.template registerEvaluator<EvalT>(ev);
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
          if (material_db_->isElementBlockParam(eb_name, "Density"))
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
          field_mgr.template registerEvaluator<EvalT>(ev);
        }
      }

      if (have_mech_eq_) {
        // convert Cauchy stress to first Piola-Kirchhoff
        Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
            new Teuchos::ParameterList("First PK Stress"));
        //Input
        p->set<std::string>("Stress Name", cauchy);
        p->set<std::string>("DefGrad Name", defgrad);

        p->set<std::string>("First PK Stress Name", firstPK);

        p->set<Teuchos::RCP<ParamLib>>("Parameter Library", paramLib);

        ev = Teuchos::rcp(new LCM::FirstPK<EvalT, PHAL::AlbanyTraits>(*p, dl_));
        field_mgr.template registerEvaluator<EvalT>(ev);
      }

      if (Teuchos::nonnull(rc_mgr_))
      rc_mgr_->createEvaluators<EvalT>(field_mgr, dl_);

      if (fieldManagerChoice == Albany::BUILD_RESID_FM) {

        Teuchos::RCP<const PHX::FieldTag> ret_tag;
        if (have_mech_eq_) {
          PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter", dl_->dummy);
          field_mgr.requireField<EvalT>(res_tag);
          ret_tag = res_tag.clone();
        }
        return ret_tag;
      }
      else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM) {
        Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl_);
        return respUtils.constructResponses(
            field_mgr, *response_list, pFromProb, state_mgr, &mesh_specs);
      }

      return Teuchos::null;
    }

#endif // LCM_SolidMechanics_h
#endif
