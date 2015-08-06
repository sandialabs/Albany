//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#if !defined(Projection_Problem_hpp)
#define Projection_Problem_hpp

#include "Albany_AbstractProblem.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_ResponseUtilities.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_Workset.hpp"
#include "Phalanx.hpp"
#include "Projection.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

using Intrepid::FieldContainer;
using PHAL::AlbanyTraits;
using PHX::DataLayout;
using PHX::MDALayout;
using Teuchos::ArrayRCP;
using Teuchos::ParameterList;
using Teuchos::rcp;
using Teuchos::RCP;

namespace Albany {

///
/// Problem definition for solid mechanics problem with projection
///
class ProjectionProblem: public Albany::AbstractProblem
{
public:

  ///
  /// Default constructor
  ///
  ProjectionProblem(
      RCP<ParameterList> const & parameter_list,
      RCP<ParamLib> const & parameter_library,
      int const number_equations);

  ///
  /// Destructor
  ///
  virtual
  ~ProjectionProblem();

  ///
  /// Return number of spatial dimensions
  ///
  virtual
  int
  spatialDimension() const
  {
    return number_dimensions_;
  }

  ///
  /// Build the PDE instantiations, boundary conditions, and initial solution
  ///
  virtual
  void
  buildProblem(
      ArrayRCP<RCP<Albany::MeshSpecsStruct>> mesh_specs,
      StateManager & state_manager);

  ///
  /// Build evaluators
  ///
  virtual
  Teuchos::Array<RCP<const PHX::FieldTag>>
  buildEvaluators(
      PHX::FieldManager<AlbanyTraits> & field_manager,
      Albany::MeshSpecsStruct const & mesh_specs,
      Albany::StateManager & state_manager,
      Albany::FieldManagerChoice field_manager_choice,
      RCP<ParameterList> const & response_list);

  ///
  /// Each problem must generate its list of valid parameters
  ///
  RCP<ParameterList const>
  getValidProblemParameters() const;

  ///
  /// Internal variables
  ///
  void
  getAllocatedStates(
      ArrayRCP<ArrayRCP<RCP<FieldContainer<RealType>>>> old_state,
      ArrayRCP<ArrayRCP<RCP<FieldContainer<RealType>>>> new_state) const;

  ///
  /// Main problem setup routine. Not directly called,
  /// but indirectly by following functions
  ///
  template<typename Evaluator>
  RCP<const PHX::FieldTag>
  constructEvaluators(
      PHX::FieldManager<AlbanyTraits> & field_manager,
      Albany::MeshSpecsStruct const & mesh_specs,
      Albany::StateManager & state_manager,
      Albany::FieldManagerChoice field_manager_choice,
      RCP<ParameterList> const & response_list);

  void
  constructDirichletEvaluators(Albany::MeshSpecsStruct const & mesh_specs);

private:

  ///
  /// Private to prohibit copying
  ///
  ProjectionProblem(ProjectionProblem const &);
  ProjectionProblem & operator=(ProjectionProblem const &);

protected:

  /// Boundary conditions on source term
  bool
  have_boundary_source_;

  /// Position of T unknown in nodal DOFs
  int
  target_offset_;

  /// Position of X unknown in nodal DOFs, followed by Y,Z
  int
  source_offset_;

  /// Number of spatial dimensions and displacement variable
  int
  number_dimensions_;

  std::string
  material_model_name_;

  std::string
  projected_field_name_;

  bool
  is_field_vector_;

  bool
  is_field_tensor_;

  int
  projection_rank_;

  LCM::Projection
  projection_;

  std::string
  insertion_criterion_;

  ArrayRCP<ArrayRCP<RCP<FieldContainer<RealType>>>>
  old_state_;

  ArrayRCP<ArrayRCP<RCP<FieldContainer<RealType>>>>
  new_state_;
};

} // namespace Albany

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

#include "Albany_AbstractProblem.hpp"
#include "Albany_EvaluatorUtils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_ResponseUtilities.hpp"
#include "Albany_Utils.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_Workset.hpp"
#include "Phalanx.hpp"

#include "DefGrad.hpp"
#include "PHAL_SaveStateField.hpp"
#include "Porosity.hpp"
#include "Strain.hpp"
#include "Time.hpp"

#include "ElasticModulus.hpp"
#include "PoissonsRatio.hpp"
#include "ShearModulus.hpp"

#include "L2ProjectionResidual.hpp"
#include "PHAL_NSMaterialProperty.hpp"
#include "PHAL_Source.hpp"
#include "TLElasResid.hpp"

#include "DislocationDensity.hpp"
#include "FaceAverage.hpp"
#include "FaceFractureCriteria.hpp"
#include "HardeningModulus.hpp"
#include "J2Stress.hpp"
#include "Neohookean.hpp"
#include "PisdWdF.hpp"
#include "SaturationExponent.hpp"
#include "SaturationModulus.hpp"
#include "YieldStrength.hpp"

template<typename Evaluator>
RCP<const PHX::FieldTag>
Albany::ProjectionProblem::constructEvaluators(
    PHX::FieldManager<AlbanyTraits> & field_manager,
    Albany::MeshSpecsStruct const & mesh_specs,
    Albany::StateManager & state_manager,
    Albany::FieldManagerChoice field_manager_choice,
    RCP<ParameterList> const & response_list)
{
  std::string
  element_block_name = mesh_specs.ebName;

  RCP<shards::CellTopology>
  cell_type = rcp(new shards::CellTopology(&mesh_specs.ctd));

  RCP<Intrepid::Basis<RealType, FieldContainer<RealType>>>
  intrepid_basis = Albany::getIntrepidBasis(mesh_specs.ctd);

  int const
  number_nodes = intrepid_basis->getCardinality();

  int const
  workset_size = mesh_specs.worksetSize;

  Intrepid::DefaultCubatureFactory<RealType>
  cubature_factory;

  RCP<Intrepid::Cubature<RealType>>
  cubature = cubature_factory.create(*cell_type, mesh_specs.cubatureDegree);

  // Create intrepid basis and cubature for the face averaging. Not the best
  // way of defining the basis functions: requires to know the face type at
  // compile time
  RCP<Intrepid::Basis<RealType, FieldContainer<RealType>>>
  face_intrepid_basis;

  face_intrepid_basis = rcp(
      new Intrepid::Basis_HGRAD_QUAD_C1_FEM<RealType,
      FieldContainer<RealType>>());

  // the quadrature is general to the
  // topology of the faces of the volume elements
  RCP<Intrepid::Cubature<RealType>>
  face_cubature = cubature_factory.create(
      cell_type->getCellTopologyData()->side->topology,
      mesh_specs.cubatureDegree);

  int const
  number_cubature_points = cubature->getNumPoints();

  int const
  number_vertices = cell_type->getNodeCount();

  int const
  number_faces = cell_type->getFaceCount();

  *out << "Field Dimensions: Workset = " << workset_size;
  *out << ", Number Vertices = " << number_vertices;
  *out << ", Number Nodes = " << number_nodes;
  *out << ", Number Cubature Points = " << number_cubature_points;
  *out << ", Number Faces = " << number_faces;
  *out << ", Dimensions = " << number_dimensions_;
  *out << std::endl;

  // Construct standard FEM evaluators with standard field names
  RCP<Albany::Layouts>
  layout = rcp(new Albany::Layouts(
      workset_size,
      number_vertices,
      number_nodes,
      number_cubature_points,
      number_dimensions_));

  TEUCHOS_TEST_FOR_EXCEPTION(
      layout->vectorAndGradientLayoutsAreEquivalent == false,
      std::logic_error,
      "Data Layout assumes vecDim = number_dimensions_");

  Albany::EvaluatorUtils<Evaluator, AlbanyTraits>
  evaluator_utils(layout);

  // Create a separate set of evaluators with their own data layout
  // for use by the projection
  RCP<Albany::Layouts>
  projection_layout = rcp(
      new Albany::Layouts(
          workset_size,
          number_vertices,
          number_nodes,
          number_cubature_points,
          number_dimensions_,
          number_dimensions_ * number_dimensions_,
          number_faces));

  Albany::EvaluatorUtils<Evaluator, AlbanyTraits>
  projection_evaluator_utils(projection_layout);

  std::string
  scatter_name = "Scatter Projection";

  //
  // Solution field setup
  //

  // Displacement Variable
  ArrayRCP<std::string>
  dof_names(1);

  dof_names[0] = "Displacement";

  ArrayRCP<std::string>
  residual_names(1);

  residual_names[0] = dof_names[0] + " Residual";

  field_manager.template
  registerEvaluator<Evaluator>(
      evaluator_utils.constructDOFVecInterpolationEvaluator(dof_names[0], source_offset_));

  field_manager.template
  registerEvaluator<Evaluator>(
      evaluator_utils.constructDOFVecGradInterpolationEvaluator(dof_names[0], source_offset_));

  field_manager.template
  registerEvaluator<Evaluator>(
      evaluator_utils.constructGatherSolutionEvaluator_noTransient(
          true,
          dof_names,
          source_offset_));

  field_manager.template
  registerEvaluator<Evaluator>(
      evaluator_utils.constructScatterResidualEvaluator(
          true,
          residual_names,
          source_offset_));

  // Projected Field Variable
  ArrayRCP<std::string>
  projected_dof_names(1);

  projected_dof_names[0] = "Projected Field";

  ArrayRCP<std::string>
  projected_residual_names(1);

  projected_residual_names[0] = projected_dof_names[0] + " Residual";

  field_manager.template
  registerEvaluator<Evaluator>(
      projection_evaluator_utils.constructDOFVecInterpolationEvaluator(
          projected_dof_names[0], target_offset_));

  field_manager.template
  registerEvaluator<Evaluator>(
      projection_evaluator_utils.constructDOFVecGradInterpolationEvaluator(
          projected_dof_names[0], target_offset_));

  // Need to use different arguments depending on the rank of the
  // projected variables. See the Albany_EvaluatorUtil class for specifics
  field_manager.template
  registerEvaluator<Evaluator>(
      projection_evaluator_utils.constructGatherSolutionEvaluator_noTransient(
          is_field_vector_,
          projected_dof_names,
          target_offset_));

  field_manager.template
  registerEvaluator<Evaluator>(
      projection_evaluator_utils.constructScatterResidualEvaluator(
          is_field_vector_,
          projected_residual_names,
          target_offset_,
          scatter_name));

  //
  // Evaluator setup
  //

  field_manager.template
  registerEvaluator<Evaluator>(
      evaluator_utils.constructGatherCoordinateVectorEvaluator());

  field_manager.template
  registerEvaluator<Evaluator>(
      evaluator_utils.constructMapToPhysicalFrameEvaluator(
          cell_type,
          cubature));

  field_manager.template
  registerEvaluator<Evaluator>(
      evaluator_utils.constructComputeBasisFunctionsEvaluator(
          cell_type,
          intrepid_basis,
          cubature));

  //
  // Time
  //
  {
    RCP<ParameterList>
    p = rcp(new ParameterList);

    p->set<std::string>("Time Name", "Time");
    p->set<std::string>("Delta Time Name", " Delta Time");

    p->set<RCP<DataLayout>>(
        "Workset Scalar Data Layout",
        layout->workset_scalar);

    p->set<RCP<ParamLib>>("Parameter Library", paramLib);
    p->set<bool>("Disable Transient", true);

    RCP<PHX::Evaluator<AlbanyTraits>>
    evaluator = rcp(new LCM::Time<Evaluator, AlbanyTraits>(*p));

    field_manager.template
    registerEvaluator<Evaluator>(evaluator);

    p = state_manager.registerStateVariable(
        "Time",
        layout->workset_scalar,
        layout->dummy,
        element_block_name,
        "scalar",
        0.0,
        true);

    evaluator = rcp(new PHAL::SaveStateField<Evaluator, AlbanyTraits>(*p));

    field_manager.template
    registerEvaluator<Evaluator>(evaluator);
  }

  //
  // Strain
  //
  {
    RCP<ParameterList>
    p = rcp(new ParameterList("Strain"));

    //Input
    p->set<std::string>("Gradient QP Variable Name", "Displacement Gradient");

    //Output
    p->set<std::string>("Strain Name", "Strain");

    RCP<PHX::Evaluator<AlbanyTraits>>
    evaluator = rcp(new LCM::Strain<Evaluator, AlbanyTraits>(*p, layout));

    field_manager.template
    registerEvaluator<Evaluator>(evaluator);

    p = state_manager.registerStateVariable(
        "Strain",
        layout->qp_tensor,
        layout->dummy,
        element_block_name,
        "scalar",
        0.0,
        true);

    evaluator = rcp(new PHAL::SaveStateField<Evaluator, AlbanyTraits>(*p));

    field_manager.template
    registerEvaluator<Evaluator>(evaluator);
  }

  //
  // Elastic Modulus
  //
  {
    RCP<ParameterList>
    p = rcp(new ParameterList);

    p->set<std::string>("QP Variable Name", "Elastic Modulus");
    p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");
    p->set<RCP<DataLayout>>("Node Data Layout", layout->node_scalar);
    p->set<RCP<DataLayout>>("QP Scalar Data Layout", layout->qp_scalar);
    p->set<RCP<DataLayout>>("QP Vector Data Layout", layout->qp_vector);

    p->set<RCP<ParamLib>>("Parameter Library", paramLib);

    ParameterList &
    parameter_list = params->sublist("Elastic Modulus");

    p->set<ParameterList*>("Parameter List", &parameter_list);

    RCP<PHX::Evaluator<AlbanyTraits>>
    evaluator = rcp(new LCM::ElasticModulus<Evaluator, AlbanyTraits>(*p));

    field_manager.template
    registerEvaluator<Evaluator>(evaluator);
  }

  //
  // Shear Modulus
  //
  {
    RCP<ParameterList>
    p = rcp(new ParameterList);

    p->set<std::string>("QP Variable Name", "Shear Modulus");
    p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");
    p->set<RCP<DataLayout>>("Node Data Layout", layout->node_scalar);
    p->set<RCP<DataLayout>>("QP Scalar Data Layout", layout->qp_scalar);
    p->set<RCP<DataLayout>>("QP Vector Data Layout", layout->qp_vector);

    p->set<RCP<ParamLib>>("Parameter Library", paramLib);

    ParameterList &
    parameter_list = params->sublist("Shear Modulus");

    p->set<ParameterList*>("Parameter List", &parameter_list);

    RCP<PHX::Evaluator<AlbanyTraits>>
    evaluator = rcp(new LCM::ShearModulus<Evaluator, AlbanyTraits>(*p));

    field_manager.template
    registerEvaluator<Evaluator>(evaluator);
  }

  //
  // Poisson's Ratio
  //
  {
    RCP<ParameterList>
    p = rcp(new ParameterList);

    p->set<std::string>("QP Variable Name", "Poissons Ratio");
    p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");
    p->set<RCP<DataLayout>>("Node Data Layout", layout->node_scalar);
    p->set<RCP<DataLayout>>("QP Scalar Data Layout", layout->qp_scalar);
    p->set<RCP<DataLayout>>("QP Vector Data Layout", layout->qp_vector);

    p->set<RCP<ParamLib>>("Parameter Library", paramLib);

    ParameterList &
    parameter_list = params->sublist("Poissons Ratio");

    p->set<ParameterList*>("Parameter List", &parameter_list);

    // Setting this turns on linear dependence of nu on T, nu = nu_ + dnudT*T)
    //p->set<std::string>("QP Projected Field Name", "Projected Field");

    RCP<PHX::Evaluator<AlbanyTraits>>
    evaluator = rcp(new LCM::PoissonsRatio<Evaluator, AlbanyTraits>(*p));

    field_manager.template
    registerEvaluator<Evaluator>(evaluator);
  }

  if (material_model_name_ == "Neohookean") {
    //
    // Stress
    //
    {
      RCP<ParameterList>
      p = rcp(new ParameterList("Stress"));

      // Input
      p->set<std::string>("DefGrad Name", "Deformation Gradient");
      p->set<std::string>("Elastic Modulus Name", "Elastic Modulus");
      p->set<std::string>("Poissons Ratio Name", "Poissons Ratio");
      p->set<std::string>("DetDefGrad Name", "Jacobian");

      // Output
      p->set<std::string>("Stress Name", material_model_name_);

      RCP<PHX::Evaluator<AlbanyTraits>>
      evaluator = rcp(new LCM::Neohookean<Evaluator, AlbanyTraits>(*p, layout));

      field_manager.template
      registerEvaluator<Evaluator>(evaluator);

      p = state_manager.registerStateVariable(
          material_model_name_,
          layout->qp_tensor,
          layout->dummy,
          element_block_name,
          "scalar",
          0.0);

      evaluator = rcp(new PHAL::SaveStateField<Evaluator, AlbanyTraits>(*p));

      field_manager.template
      registerEvaluator<Evaluator>(evaluator);
    }
  }
  else if (material_model_name_ == "Neohookean AD") {
    //
    // Stress
    //
    {
      RCP<ParameterList>
      p = rcp(new ParameterList("Stress"));

      //Input
      p->set<std::string>("Elastic Modulus Name", "Elastic Modulus");
      p->set<RCP<DataLayout>>("QP Scalar Data Layout", layout->qp_scalar);
      p->set<std::string>("Poissons Ratio Name", "Poissons Ratio");

      p->set<std::string>("DefGrad Name", "Deformation Gradient");
      p->set<RCP<DataLayout>>("QP Tensor Data Layout", layout->qp_tensor);

      //Output
      p->set<std::string>("Stress Name", material_model_name_);

      RCP<PHX::Evaluator<AlbanyTraits>>
      evaluator = rcp(new LCM::PisdWdF<Evaluator, AlbanyTraits>(*p));

      field_manager.template
      registerEvaluator<Evaluator>(evaluator);

      p = state_manager.registerStateVariable(
          material_model_name_,
          layout->qp_tensor,
          layout->dummy,
          element_block_name,
          "scalar",
          0.0);

      evaluator = rcp(new PHAL::SaveStateField<Evaluator, AlbanyTraits>(*p));

      field_manager.template
      registerEvaluator<Evaluator>(evaluator);
    }
  }
  else if (material_model_name_ == "J2" || material_model_name_ == "J2Fiber") {
    //
    // Hardening Modulus
    //
    {
      RCP<ParameterList>
      p = rcp(new ParameterList);

      p->set<std::string>("QP Variable Name", "Hardening Modulus");
      p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");
      p->set<RCP<DataLayout>>("Node Data Layout", layout->node_scalar);
      p->set<RCP<DataLayout>>("QP Scalar Data Layout", layout->qp_scalar);
      p->set<RCP<DataLayout>>("QP Vector Data Layout", layout->qp_vector);

      p->set<RCP<ParamLib>>("Parameter Library", paramLib);

      ParameterList &
      parameter_list = params->sublist("Hardening Modulus");

      p->set<ParameterList*>("Parameter List", &parameter_list);

      RCP<PHX::Evaluator<AlbanyTraits>>
      evaluator = rcp(new LCM::HardeningModulus<Evaluator, AlbanyTraits>(*p));

      field_manager.template
      registerEvaluator<Evaluator>(evaluator);
    }

    //
    // Yield Strength
    //
    {
      RCP<ParameterList>
      p = rcp(new ParameterList);

      p->set<std::string>("QP Variable Name", "Yield Strength");
      p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");
      p->set<RCP<DataLayout>>("Node Data Layout", layout->node_scalar);
      p->set<RCP<DataLayout>>("QP Scalar Data Layout", layout->qp_scalar);
      p->set<RCP<DataLayout>>("QP Vector Data Layout", layout->qp_vector);

      p->set<RCP<ParamLib>>("Parameter Library", paramLib);

      ParameterList &
      parameter_list = params->sublist("Yield Strength");

      p->set<ParameterList*>("Parameter List", &parameter_list);

      RCP<PHX::Evaluator<AlbanyTraits>>
      evaluator = rcp(new LCM::YieldStrength<Evaluator, AlbanyTraits>(*p));

      field_manager.template registerEvaluator<Evaluator>(evaluator);
    }

    //
    // Saturation Modulus
    //
    {
      RCP<ParameterList>
      p = rcp(new ParameterList);

      p->set<std::string>("Saturation Modulus Name", "Saturation Modulus");
      p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");
      p->set<RCP<DataLayout>>("Node Data Layout", layout->node_scalar);
      p->set<RCP<DataLayout>>("QP Scalar Data Layout", layout->qp_scalar);
      p->set<RCP<DataLayout>>("QP Vector Data Layout", layout->qp_vector);

      p->set<RCP<ParamLib>>("Parameter Library", paramLib);

      ParameterList &
      parameter_list = params->sublist("Saturation Modulus");

      p->set<ParameterList*>("Parameter List", &parameter_list);

      RCP<PHX::Evaluator<AlbanyTraits>>
      evaluator = rcp(new LCM::SaturationModulus<Evaluator, AlbanyTraits>(*p));

      field_manager.template
      registerEvaluator<Evaluator>(evaluator);
    }
    //
    // Saturation Exponent
    //
    {
      RCP<ParameterList>
      p = rcp(new ParameterList);

      p->set<std::string>("Saturation Exponent Name", "Saturation Exponent");
      p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");
      p->set<RCP<DataLayout>>("Node Data Layout", layout->node_scalar);
      p->set<RCP<DataLayout>>("QP Scalar Data Layout", layout->qp_scalar);
      p->set<RCP<DataLayout>>("QP Vector Data Layout", layout->qp_vector);

      p->set<RCP<ParamLib>>("Parameter Library", paramLib);

      ParameterList &
      parameter_list = params->sublist("Saturation Exponent");

      p->set<ParameterList*>("Parameter List", &parameter_list);

      RCP<PHX::Evaluator<AlbanyTraits>>
      evaluator = rcp(new LCM::SaturationExponent<Evaluator, AlbanyTraits>(*p));

      field_manager.template
      registerEvaluator<Evaluator>(evaluator);
    }

    bool const
    compute_dislocation_density =
        params->get("Compute Dislocation Density Tensor", false) &&
        number_dimensions_ == 3;

    if (compute_dislocation_density == true) {
      //
      // Dislocation Density Tensor
      //
      {
        RCP<ParameterList>
        p = rcp(new ParameterList("Dislocation Density"));

        // Input
        p->set<std::string>("Fp Name", "Fp");
        p->set<RCP<DataLayout>>("QP Tensor Data Layout", layout->qp_tensor);
        p->set<std::string>("BF Name", "BF");

        p->set<RCP<DataLayout>>("Node QP Scalar Data Layout",
            layout->node_qp_scalar);

        p->set<std::string>("Gradient BF Name", "Grad BF");

        p->set<RCP<DataLayout>>("Node QP Vector Data Layout",
            layout->node_qp_vector);

        // Output
        p->set<std::string>("Dislocation Density Name", "G");

        // Declare what state data will need to be saved
        // (name, layout, init_type)
        RCP<PHX::Evaluator<AlbanyTraits>>
        evaluator =
            rcp(new LCM::DislocationDensity<Evaluator, AlbanyTraits>(*p));

        field_manager.template
        registerEvaluator<Evaluator>(evaluator);

        p = state_manager.registerStateVariable(
            "G",
            layout->qp_tensor,
            layout->dummy,
            element_block_name,
            "scalar",
            0.0);

        evaluator = rcp(new PHAL::SaveStateField<Evaluator, AlbanyTraits>(*p));

        field_manager.template
        registerEvaluator<Evaluator>(evaluator);
      }
    }

    if (material_model_name_ == "J2") {
      //
      // Stress
      //
      {
        RCP<ParameterList>
        p = rcp(new ParameterList("Stress"));

        // Input
        p->set<std::string>("DefGrad Name", "Deformation Gradient");
        p->set<RCP<DataLayout>>("QP Tensor Data Layout", layout->qp_tensor);

        p->set<std::string>("Elastic Modulus Name", "Elastic Modulus");
        p->set<RCP<DataLayout>>("QP Scalar Data Layout", layout->qp_scalar);

        p->set<std::string>("Poissons Ratio Name", "Poissons Ratio");
        p->set<std::string>("Hardening Modulus Name", "Hardening Modulus");
        p->set<std::string>("Saturation Modulus Name", "Saturation Modulus");
        p->set<std::string>("Saturation Exponent Name", "Saturation Exponent");
        p->set<std::string>("Yield Strength Name", "Yield Strength");
        p->set<std::string>("DetDefGrad Name", "Jacobian");

        // Output
        p->set<std::string>("Stress Name", material_model_name_);
        p->set<std::string>("Fp Name", "Fp");
        p->set<std::string>("Eqps Name", "eqps");

        // Declare what state data will need to be saved
        //(name, layout, init_type)
        RCP<PHX::Evaluator<AlbanyTraits>>
        evaluator = rcp(new LCM::J2Stress<Evaluator, AlbanyTraits>(*p));

        field_manager.template
        registerEvaluator<Evaluator>(evaluator);

        p = state_manager.registerStateVariable(
            material_model_name_,
            layout->qp_tensor,
            layout->dummy,
            element_block_name,
            "scalar",
            0.0);

        evaluator = rcp(new PHAL::SaveStateField<Evaluator, AlbanyTraits>(*p));

        field_manager.template
        registerEvaluator<Evaluator>(evaluator);

        p = state_manager.registerStateVariable(
            "Fp",
            layout->qp_tensor,
            layout->dummy,
            element_block_name,
            "identity",
            1.0,
            true);

        evaluator = rcp(new PHAL::SaveStateField<Evaluator, AlbanyTraits>(*p));

        field_manager.template
        registerEvaluator<Evaluator>(evaluator);

        p = state_manager.registerStateVariable(
            "eqps",
            layout->qp_scalar,
            layout->dummy,
            element_block_name,
            "scalar",
            0.0,
            true);

        evaluator = rcp(new PHAL::SaveStateField<Evaluator, AlbanyTraits>(*p));

        field_manager.template
        registerEvaluator<Evaluator>(evaluator);
      }
    }
  }
  //   else
  //     TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
  // 			       "Unrecognized Material Name: " << material_model_name_
  // 			       << "  Recognized names are : Neohookean and J2");

  if (have_boundary_source_ == true) {
    //
    // Boundary Source
    //
    {
      RCP<ParameterList>
      p = rcp(new ParameterList);

      p->set<std::string>("Source Name", "Source");
      p->set<std::string>("QP Variable Name", "Projected Field");
      p->set<RCP<DataLayout>>("QP Scalar Data Layout", layout->qp_scalar);

      p->set<RCP<ParamLib>>("Parameter Library", paramLib);

      ParameterList &
      parameter_list = params->sublist("Source Functions");

      p->set<ParameterList*>("Parameter List", &parameter_list);

      RCP<PHX::Evaluator<AlbanyTraits>>
      evaluator = rcp(new PHAL::Source<Evaluator, AlbanyTraits>(*p));

      field_manager.template
      registerEvaluator<Evaluator>(evaluator);
    }
  }

  //
  // Deformation Gradient
  //
  {
    RCP<ParameterList>
    p = rcp(new ParameterList("Deformation Gradient"));

    //Inputs: flags, weights, GradU
    bool const
    J_average = params->get("avgJ", false);

    p->set<bool>("avgJ Name", J_average);

    bool const
    J_volume_average = params->get("volavgJ", false);

    p->set<bool>("volavgJ Name", J_volume_average);

    bool const
    J_weighted_volume_average =
        params->get("weighted_Volume_Averaged_J", false);

    p->set<bool>("weighted_Volume_Averaged_J Name", J_weighted_volume_average);

    p->set<std::string>("Weights Name", "Weights");
    p->set<RCP<DataLayout>>("QP Scalar Data Layout", layout->qp_scalar);
    p->set<std::string>("Gradient QP Variable Name", "Displacement Gradient");
    p->set<RCP<DataLayout>>("QP Tensor Data Layout", layout->qp_tensor);

    //Outputs: F, J
    p->set<std::string>("DefGrad Name", "Deformation Gradient");
    p->set<std::string>("DetDefGrad Name", "Jacobian");
    p->set<RCP<DataLayout>>("QP Scalar Data Layout", layout->qp_scalar);

    RCP<PHX::Evaluator<AlbanyTraits>>
    evaluator = rcp(new LCM::DefGrad<Evaluator, AlbanyTraits>(*p));

    field_manager.template
    registerEvaluator<Evaluator>(evaluator);

    p = state_manager.registerStateVariable(
        "Displacement Gradient",
        layout->qp_tensor,
        layout->dummy,
        element_block_name,
        "identity",
        true);

    evaluator = rcp(new PHAL::SaveStateField<Evaluator, AlbanyTraits>(*p));

    field_manager.template
    registerEvaluator<Evaluator>(evaluator);

    p = state_manager.registerStateVariable(
        "Jacobian",
        layout->qp_scalar,
        layout->dummy,
        element_block_name,
        "scalar",
        1.0,
        true);

    evaluator = rcp(new PHAL::SaveStateField<Evaluator, AlbanyTraits>(*p));

    field_manager.template
    registerEvaluator<Evaluator>(evaluator);
  }

  //
  // Displacement Residual
  //
  {
    RCP<ParameterList>
    p = rcp(new ParameterList("Displacement Resid"));

    // Input
    p->set<std::string>("Stress Name", material_model_name_);
    p->set<RCP<DataLayout>>("QP Tensor Data Layout", layout->qp_tensor);

    p->set<std::string>("DefGrad Name", "Deformation Gradient");

    p->set<std::string>("DetDefGrad Name", "Jacobian");
    p->set<RCP<DataLayout>>("QP Scalar Data Layout", layout->qp_scalar);

    p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
    p->set<RCP<DataLayout>>(
        "Node QP Vector Data Layout",
        layout->node_qp_vector);

    p->set<std::string>("Weighted BF Name", "wBF");

    p->set<RCP<DataLayout>>(
        "Node QP Scalar Data Layout",
        layout->node_qp_scalar);

    p->set<RCP<ParamLib>>("Parameter Library", paramLib);

    p->set<bool>("Disable Transient", true);

    //Output
    p->set<std::string>("Residual Name", "Displacement Residual");
    p->set<RCP<DataLayout>>("Node Vector Data Layout", layout->node_vector);

    RCP<PHX::Evaluator<AlbanyTraits>>
    evaluator = rcp(new LCM::TLElasResid<Evaluator, AlbanyTraits>(*p));

    field_manager.template
    registerEvaluator<Evaluator>(evaluator);
  }

  //
  // L2 projection
  //
  {
    RCP<ParameterList>
    p = rcp(new ParameterList("Projected Field Resid"));

    // Input
    p->set<std::string>("Weighted BF Name", "wBF");

    p->set<RCP<DataLayout>>(
        "Node QP Scalar Data Layout",
        projection_layout->node_qp_scalar);

    p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");

    p->set<RCP<DataLayout>>(
        "Node QP Vector Data Layout",
        projection_layout->node_qp_vector);

    p->set<bool>("Have Source", false);
    p->set<std::string>("Source Name", "Source");

    p->set<std::string>("Projected Field Name", "Projected Field");

    p->set<RCP<DataLayout>>(
        "QP Vector Data Layout",
        projection_layout->qp_vector);

    p->set<std::string>("Projection Field Name", projected_field_name_);

    p->set<RCP<DataLayout>>(
        "QP Tensor Data Layout",
        projection_layout->qp_tensor);

    // Output
    p->set<std::string>("Residual Name", "Projected Field Residual");

    p->set<RCP<DataLayout>>(
        "Node Vector Data Layout",
        projection_layout->node_vector);

    RCP<PHX::Evaluator<AlbanyTraits>>
    evaluator = rcp(new LCM::L2ProjectionResidual<Evaluator, AlbanyTraits>(*p));

    field_manager.template
    registerEvaluator<Evaluator>(evaluator);

    p = state_manager.registerStateVariable(
        "Projected Field",
        projection_layout->qp_vector,
        projection_layout->dummy,
        element_block_name,
        "scalar",
        0.0,
        true);

    evaluator = rcp(new PHAL::SaveStateField<Evaluator, AlbanyTraits>(*p));

    field_manager.template
    registerEvaluator<Evaluator>(evaluator);
  }

  //
  // Fracture Criterion
  //
  {
    RCP<ParameterList>
    p = rcp(new ParameterList("Face Fracture Criteria"));

    // Input
    // Nodal coordinates in the reference configuration
    p->set<std::string>("Coordinate Vector Name", "Coord Vec");

    p->set<RCP<DataLayout>>(
        "Vertex Vector Data Layout",
        layout->vertices_vector);

    p->set<std::string>("Face Average Name", "Face Average");

    p->set<RCP<DataLayout>>(
        "Face Vector Data Layout",
        projection_layout->face_vector);

    p->set<RCP<shards::CellTopology>>("Cell Type", cell_type);

    RealType const
    yield_strength = params->sublist("Yield Strength").get("Value", 0.0);

    p->set<RealType>("Yield Name", yield_strength);

    RealType const
    fracture_limit =
        params->sublist("Insertion Criteria").get("Fracture Limit", 0.0);

    p->set<RealType>("Fracture Limit Name", fracture_limit);

    p->set<std::string>("Insertion Criteria Name",
        params->sublist("Insertion Criteria").get("Insertion Criteria", ""));

    // Output
    p->set<std::string>("Criteria Met Name", "Criteria Met");

    p->set<RCP<DataLayout>>(
        "Face Scalar Data Layout",
        projection_layout->face_scalar);

    // This is in here to trick the code to run the evaluator
    // does absolutely nothing
    p->set<std::string>("Temp2 Name", "Temp2");
    p->set<RCP<DataLayout>>(
        "Cell Scalar Data Layout",
        projection_layout->cell_scalar);

    RCP<PHX::Evaluator<AlbanyTraits>>
    evaluator = rcp(new LCM::FaceFractureCriteria<Evaluator, AlbanyTraits>(*p));

    field_manager.template
    registerEvaluator<Evaluator>(evaluator);

    p = state_manager.registerStateVariable(
        "Temp2",
        projection_layout->cell_scalar,
        projection_layout->dummy,
        element_block_name,
        "scalar",
        0.0,
        true);

    evaluator = rcp(new PHAL::SaveStateField<Evaluator, AlbanyTraits>(*p));

    field_manager.template
    registerEvaluator<Evaluator>(evaluator);
  }

  //
  // Face Average
  //
  {
    RCP<ParameterList>
    p = rcp(new ParameterList("Face Average"));

    // Input
    // Nodal coordinates in the reference configuration
    p->set<std::string>("Coordinate Vector Name", "Coord Vec");

    p->set<RCP<DataLayout>>(
        "Vertex Vector Data Layout",
        layout->vertices_vector);

    // The solution of the projection at the nodes
    p->set<std::string>("Projected Field Name", "Projected Field");

    p->set<RCP<DataLayout>>(
        "Node Vector Data Layout",
        projection_layout->node_vector);

    // the cubature and basis function information
    p->set<RCP<Intrepid::Cubature<RealType>>>(
        "Face Cubature",
        face_cubature);

    p->set<RCP<Intrepid::Basis<RealType, FieldContainer<RealType>>>>(
        "Face Intrepid Basis",
        face_intrepid_basis);

    p->set<RCP<shards::CellTopology>>("Cell Type", cell_type);

    // Output
    p->set<std::string>("Face Average Name", "Face Average");

    p->set<RCP<DataLayout>>(
        "Face Vector Data Layout",
        projection_layout->face_vector);

    // This is in here to trick the code to run the evaluator
    // does absolutely nothing
    p->set<std::string>("Temp Name", "Temp");

    p->set<RCP<DataLayout>>(
        "Cell Scalar Data Layout",
        projection_layout->cell_scalar);

    RCP<PHX::Evaluator<AlbanyTraits>>
    evaluator = rcp(new LCM::FaceAverage<Evaluator, AlbanyTraits>(*p));

    field_manager.template
    registerEvaluator<Evaluator>(evaluator);

    p = state_manager.registerStateVariable(
        "Temp",
        projection_layout->cell_scalar,
        projection_layout->dummy,
        element_block_name,
        "scalar",
        0.0,
        true);

    evaluator = rcp(new PHAL::SaveStateField<Evaluator, AlbanyTraits>(*p));

    field_manager.template
    registerEvaluator<Evaluator>(evaluator);
  }

  if (field_manager_choice == Albany::BUILD_RESID_FM) {

    PHX::Tag<typename Evaluator::ScalarT>
    res_tag("Scatter", layout->dummy);

    field_manager.requireField<Evaluator>(res_tag);

    PHX::Tag<typename Evaluator::ScalarT>
    res_tag2(scatter_name, layout->dummy);

    field_manager.requireField<Evaluator>(res_tag2);

    return res_tag.clone();
  }
  else if (field_manager_choice == Albany::BUILD_RESPONSE_FM) {

    Albany::ResponseUtilities<Evaluator, AlbanyTraits>
    respUtils(layout);

    return respUtils.constructResponses(
        field_manager,
        *response_list,
        state_manager);
  }

  return Teuchos::null;
}

#endif // Projection_Problem_hpp
