//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_SchwarzMultiscaleProblem_hpp)
#define LCM_SchwarzMultiscaleProblem_hpp

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractProblem.hpp"

#include "Phalanx.hpp"
#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_AlbanyTraits.hpp"

namespace Albany {

///
/// \brief Definition for the SchwarzMultiscale Problem
///
class SchwarzMultiscaleProblem : public Albany::AbstractProblem {
public:

  typedef Intrepid::FieldContainer<RealType> FC;

  ///
  /// Default constructor
  ///
  SchwarzMultiscaleProblem(
      Teuchos::RCP<Teuchos::ParameterList> const & params,
      Teuchos::RCP<ParamLib> const & param_lib,
      int const num_dims,
      Teuchos::RCP<const Epetra_Comm> const & comm);

  ///
  /// Destructor
  ///
  virtual
  ~SchwarzMultiscaleProblem();

  ///
  Teuchos::RCP<std::map<std::string, std::string> >
  constructFieldNameMap(bool surface_flag);

  ///
  /// Return number of spatial dimensions
  ///
  virtual
  int
  spatialDimension() const {return num_dims_;}

  ///
  /// Build the PDE instantiations, boundary conditions, initial solution
  ///
  virtual
  void
  buildProblem(
      Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> > mesh_specs,
      StateManager & state_mgr);

  ///
  /// Build evaluators
  ///
  virtual
  Teuchos::Array< Teuchos::RCP<const PHX::FieldTag> >
  buildEvaluators(
      PHX::FieldManager<PHAL::AlbanyTraits> & fm0,
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
  void
  getAllocatedStates(
      Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<FC> > > old_state,
      Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<FC> > > new_state
  ) const;

private:

  ///
  /// Private to prohibit copying
  ///
  SchwarzMultiscaleProblem(SchwarzMultiscaleProblem const &);

  ///
  /// Private to prohibit copying
  ///
  SchwarzMultiscaleProblem &
  operator=(SchwarzMultiscaleProblem const &);

  QCAD::MaterialDatabase &
  matDB()
  {return *material_db_;}

public:

  ///
  /// Main problem setup routine.
  /// Not directly called, but indirectly by following functions
  ///
  template <typename EvalT>
  Teuchos::RCP<const PHX::FieldTag>
  constructEvaluators(
      PHX::FieldManager<PHAL::AlbanyTraits> & fm0,
      Albany::MeshSpecsStruct const & mesh_specs,
      Albany::StateManager & state_mgr,
      Albany::FieldManagerChoice fm_choice,
      Teuchos::RCP<Teuchos::ParameterList> & response_list);

  ///
  /// Setup for the dirichlet BCs
  ///
  void
  constructDirichletEvaluators(Albany::MeshSpecsStruct const & mesh_specs);

protected:

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
  ///  Map to indicate overlap block
  ///
  std::map< std::string, bool > overlap_map_;

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

} // namespace Albany


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

// Constitutive Model Interface and parameters
#include "Kinematics.hpp"
#include "ConstitutiveModelInterface.hpp"
#include "ConstitutiveModelParameters.hpp"

//
//
//
template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
Albany::SchwarzMultiscaleProblem::
constructEvaluators(
    PHX::FieldManager<PHAL::AlbanyTraits> & fm0,
    Albany::MeshSpecsStruct const & mesh_specs,
    Albany::StateManager & state_mgr,
    Albany::FieldManagerChoice fm_choice,
    Teuchos::RCP<Teuchos::ParameterList> & response_list)
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

  // get the name of the current element block
  std::string
  eb_name = mesh_specs.ebName;

  // get the name of the material model to be used (and make sure there is one)
  Teuchos::ParameterList &
  material_model_sublist =
      matDB().getElementBlockSublist(eb_name,"Material Model");

  std::string
  material_model_name = material_model_sublist.get<std::string>("Model Name");

  TEUCHOS_TEST_FOR_EXCEPTION(
      material_model_name.length() == 0, std::logic_error,
      "A material model must be defined for block: " + eb_name);

#ifdef ALBANY_VERBOSE
  *out << "In SchwarzMultiscaleProblem::constructEvaluators" << '\n';
  *out << "element block name: " << eb_name << '\n';
  *out << "material model name: " << material_model_name << '\n';
#endif

  // define cell topologies
  RCP<CellTopology>
  composite_cell_type =
    rcp(new CellTopology(getCellTopologyData<shards::Tetrahedron<11> >()));

  RCP<shards::CellTopology>
  cell_type = rcp(new CellTopology (&mesh_specs.ctd));

  // Check if we are setting the composite tet flag
  bool
  is_composite = false;

  bool const
  is_composite_block_present =
      matDB().isElementBlockParam(eb_name, "Use Composite Tet 10");

  if (is_composite_block_present == true) {
    is_composite =
        matDB().getElementBlockParam<bool>(eb_name, "Use Composite Tet 10");
  }

  // get the intrepid basis for the given cell topology
  RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > >
  intrepid_basis = Albany::getIntrepidBasis(mesh_specs.ctd, is_composite);

  bool const
  is_composite_cell_type =
      is_composite &&
      mesh_specs.ctd.dimension == 3 &&
      mesh_specs.ctd.node_count == 10;

  if (is_composite_cell_type == true) {
    cell_type = composite_cell_type;
  }

  Intrepid::DefaultCubatureFactory<RealType>
  cubature_factory;

  RCP <Intrepid::Cubature<RealType> >
  cubature = cubature_factory.create(*cell_type, mesh_specs.cubatureDegree);

  // Note that these are the volume element quantities
  num_nodes_ = intrepid_basis->getCardinality();
  int const
  workset_size = mesh_specs.worksetSize;

  num_dims_ = cubature->getDimension();
  num_pts_ = cubature->getNumPoints();
  num_vertices_ = num_nodes_;

#ifdef ALBANY_VERBOSE
  *out << "Field Dimensions: Workset=" << workset_size
       << ", Vertices= " << num_vertices_
       << ", Nodes= " << num_nodes_
       << ", QuadPts= " << num_pts_
       << ", Dim= " << num_dims_ << '\n';
#endif

  // Construct standard FEM evaluators with standard field names
  RCP<Albany::Layouts>
  dl =
      rcp(
          new Albany::Layouts(
              workset_size, num_vertices_, num_nodes_, num_pts_, num_dims_
          )
      );

  std::string const
  msg = "Data Layout Usage in Mechanics problems assume vecDim = num_dims_";

  TEUCHOS_TEST_FOR_EXCEPTION(
      dl->vectorAndGradientLayoutsAreEquivalent == false,
      std::logic_error,
      msg);

  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits>
  eu(dl);

  int
  offset = 0;

  // Temporary variable used numerous times below
  RCP<PHX::Evaluator<AlbanyTraits> > ev;

  // Define Field Names
  {
    Teuchos::ArrayRCP<std::string>
    dof_names(1);

    Teuchos::ArrayRCP<std::string>
    dof_names_dot(1);

    Teuchos::ArrayRCP<std::string>
    dof_names_dotdot(1);

    Teuchos::ArrayRCP<std::string>
    resid_names(1);

    dof_names[0] = "Displacement";
    dof_names_dot[0] = "Velocity";
    dof_names_dotdot[0] = "Acceleration";
    resid_names[0] = dof_names[0] + " Residual";

    fm0.template registerEvaluator<EvalT>(
        eu.constructGatherSolutionEvaluator_noTransient(true, dof_names)
    );

    fm0.template registerEvaluator<EvalT>(
        eu.constructGatherCoordinateVectorEvaluator()
    );

    fm0.template registerEvaluator<EvalT>(
        eu.constructDOFVecInterpolationEvaluator(dof_names[0])
    );

    fm0.template registerEvaluator<EvalT>(
        eu.constructDOFVecGradInterpolationEvaluator(dof_names[0])
    );

    fm0.template registerEvaluator<EvalT>(
        eu.constructMapToPhysicalFrameEvaluator(cell_type, cubature)
    );

    fm0.template registerEvaluator<EvalT>(
        eu.constructComputeBasisFunctionsEvaluator(
            cell_type, intrepid_basis, cubature
        )
    );

    fm0.template registerEvaluator<EvalT>(
        eu.constructScatterResidualEvaluator(true, resid_names)
    );

    offset += num_dims_;
  }

  // generate the field name map to deal with outputting surface element info
  LCM::FieldNameMap
  field_name_map(false);

  RCP<std::map<std::string, std::string> >
  fnm = field_name_map.getMap();

  std::string cauchy       = (*fnm)["Cauchy_Stress"];
  std::string Fp           = (*fnm)["Fp"];
  std::string eqps         = (*fnm)["eqps"];

  { // Time
    RCP<ParameterList>
    p = rcp(new ParameterList("Time"));
    p->set<std::string>("Time Name", "Time");
    p->set<std::string>("Delta Time Name", "Delta Time");
    p->set<RCP<DataLayout> >("Workset Scalar Data Layout", dl->workset_scalar);
    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    p->set<bool>("Disable Transient", true);
    ev = rcp(new LCM::Time<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
    p = state_mgr.registerStateVariable(
        "Time",
        dl->workset_scalar,
        dl->dummy,
        eb_name,
        "scalar",
        0.0,
        true
    );
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (have_source_ == true) { // Source
    RCP<ParameterList>
    p = rcp(new ParameterList);
    p->set<std::string>("Source Name", "Source");
    p->set<std::string>("Variable Name", "Displacement");
    p->set< RCP<DataLayout> >("QP Scalar Data Layout", dl->qp_scalar);

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    Teuchos::ParameterList &
    param_list = params->sublist("Source Functions");
    p->set<Teuchos::ParameterList*>("Parameter List", &param_list);

    ev = rcp(new PHAL::Source<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // Constitutive Model Parameters
    RCP<ParameterList>
    p = rcp(new ParameterList("Constitutive Model Parameters"));

    std::string const
    name = matDB().getElementBlockParam<std::string>(eb_name,"material");

    Teuchos::ParameterList &
    param_list = matDB().getElementBlockSublist(eb_name, name);

    p->set<Teuchos::ParameterList*>("Material Parameters", &param_list);

    RCP<LCM::ConstitutiveModelParameters<EvalT,AlbanyTraits> >
    cmp_ev = rcp(
        new LCM::ConstitutiveModelParameters<EvalT, AlbanyTraits>(*p, dl)
    );
    fm0.template registerEvaluator<EvalT>(cmp_ev);
  }

  {
    RCP<ParameterList>
    p = rcp(new ParameterList("Constitutive Model Interface"));

    std::string const
    name = matDB().getElementBlockParam<std::string>(eb_name, "material");

    Teuchos::ParameterList &
    param_list = matDB().getElementBlockSublist(eb_name, name);

    param_list.set<RCP<std::map<std::string, std::string> > >("Name Map", fnm);
    p->set<Teuchos::ParameterList*>("Material Parameters", &param_list);

    RCP<LCM::ConstitutiveModelInterface<EvalT,AlbanyTraits> >
    cmi_ev = rcp(
        new LCM::ConstitutiveModelInterface<EvalT, AlbanyTraits>(*p, dl)
    );
    fm0.template registerEvaluator<EvalT>(cmi_ev);

    // register state variables
    for (int sv(0); sv < cmi_ev->getNumStateVars(); ++sv) {
      cmi_ev->fillStateVariableStruct(sv);
      p = state_mgr.registerStateVariable(
          cmi_ev->getName(),
          cmi_ev->getLayout(),
          dl->dummy,
          eb_name,
          cmi_ev->getInitType(),
          cmi_ev->getInitValue(),
          cmi_ev->getStateFlag(),
          cmi_ev->getOutputFlag()
      );
      ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }
  }


  { // Kinematics quantities
    RCP<ParameterList>
    p = rcp(new ParameterList("Kinematics"));

    std::string const
    wva_str = "Weighted Volume Average J";

    bool const
    is_wva = matDB().isElementBlockParam(eb_name, wva_str);

    // set flags to optionally volume average J with a weighted average
    if (is_wva == true) {
      bool const
      ebp_wva = matDB().getElementBlockParam<bool>(eb_name, wva_str);
      p->set<bool>(wva_str, ebp_wva);
    }

    std::string const
    asp_str = "Average J Stabilization Parameter";

    bool const
    is_asp = matDB().isElementBlockParam(eb_name, asp_str);

    if (is_asp == true) {
      bool const
      ebp_asp = matDB().getElementBlockParam<RealType>(eb_name, asp_str);
      p->set<RealType>(asp_str, ebp_asp);
    }

    // set flag for return strain and velocity gradient
    bool have_strain(false), have_velocity_gradient(false);

    std::string const
    str_str = "Strain Flag";

    bool const
    is_str = matDB().isElementBlockParam(eb_name, str_str);

    if (is_str == true) {
      bool const
      ebp_str = matDB().getElementBlockParam<bool>(eb_name, str_str);
      p->set<bool>(str_str, ebp_str);

      if (ebp_str == true) {
        p->set<std::string>("Strain Name", "Strain");
      }
    }

    std::string const
    vgf_str = "Velocity Gradient Flag";

    bool const
    is_vgf = matDB().isElementBlockParam(eb_name, vgf_str);

    if (is_vgf == true) {
      bool const
      ebp_vgf = matDB().getElementBlockParam<bool>(eb_name, vgf_str);
      p->set<bool>(vgf_str, ebp_vgf);

      if (ebp_vgf == true) {
        p->set<std::string>("Velocity Gradient Name", "Velocity Gradient");
      }
    }

    // send in integration weights and the displacement gradient
    p->set<std::string>("Weights Name","Weights");
    p->set<std::string>("Gradient QP Variable Name", "Displacement Gradient");

    //Outputs: F, J
    p->set<std::string>("DefGrad Name", "F"); //dl->qp_tensor also
    p->set<std::string>("DetDefGrad Name", "J");

    ev = rcp(new LCM::Kinematics<EvalT,AlbanyTraits>(*p, dl));
    fm0.template registerEvaluator<EvalT>(ev);


    // optional output
    bool output_flag(false);
    if (matDB().isElementBlockParam(eb_name, "Output Deformation Gradient")) {
      output_flag = matDB().getElementBlockParam<bool>(
          eb_name,
          "Output Deformation Gradient"
      );
    }

    p = state_mgr.registerStateVariable(
        "F",
        dl->qp_tensor,
        dl->dummy,
        eb_name,
        "identity",
        1.0,
        output_flag
    );
    ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);

    // need J and J_old to perform time integration for poromechanics problem
    output_flag = false;

    if (matDB().isElementBlockParam(eb_name, "Output J")) {
      output_flag = matDB().getElementBlockParam<bool>(
          eb_name,
          "Output J"
      );
    }

    if (output_flag == true) {
      p = state_mgr.registerStateVariable(
          "J",
          dl->qp_scalar,
          dl->dummy,
          eb_name,
          "scalar",
          1.0,
          true
      );
      ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    // Optional output: strain
    if (have_strain == true) {
      output_flag = false;
      if (matDB(). isElementBlockParam(eb_name, "Output Strain")) {
        output_flag =
          matDB().getElementBlockParam<bool>(eb_name, "Output Strain");
      }

      p = state_mgr.registerStateVariable(
          "Strain",
          dl->qp_tensor,
          dl->dummy,
          eb_name,
          "scalar",
          0.0,
          output_flag
      );
      ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }

    // Optional output: velocity gradient
    if (have_velocity_gradient == true) {
      output_flag = false;
      if(matDB().isElementBlockParam(eb_name, "Output Velocity Gradient")) {
        output_flag = matDB().getElementBlockParam<bool>(
            eb_name,
            "Output Velocity Gradient"
        );
      }

      p = state_mgr.registerStateVariable(
          "Velocity Gradient",
          dl->qp_tensor,
          dl->dummy,
          eb_name,
          "scalar",
          0.0,
          output_flag
      );
      ev = rcp(new PHAL::SaveStateField<EvalT,AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }

  }

  { // Residual
    RCP<ParameterList>
    p = rcp(new ParameterList("Displacement Residual"));
    //Input
    p->set<std::string>("Stress Name", cauchy);
    p->set<std::string>("DefGrad Name", "F");
    p->set<std::string>("DetDefGrad Name", "J");
    p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
    p->set<std::string>("Weighted BF Name", "wBF");

    // Strain flag for small deformation problem
    if (matDB().isElementBlockParam(eb_name,"Strain Flag")) {
      p->set<bool>("Strain Flag","Strain Flag");
    }

    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    //Output
    p->set<std::string>("Residual Name", "Displacement Residual");

    // Disable dynamics for now
    p->set<bool>("Disable Dynamics", true);

    ev = rcp(new LCM::MechanicsResidual<EvalT, AlbanyTraits>(*p,dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  Teuchos::RCP<const PHX::FieldTag>
  ret_tag = Teuchos::null;

  if (fm_choice == Albany::BUILD_RESID_FM) {
    PHX::Tag<typename EvalT::ScalarT>
    res_tag("Scatter", dl->dummy);
    fm0.requireField<EvalT>(res_tag);
    ret_tag = res_tag.clone();
  }
  else if (fm_choice == Albany::BUILD_RESPONSE_FM) {
    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits>
    respUtils(dl);
    ret_tag = respUtils.constructResponses(fm0, *response_list, state_mgr);
  }

  return ret_tag;
}

#endif // LCM_SchwarzMultiscaleProblem_hpp
