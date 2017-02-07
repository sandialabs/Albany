//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


//
//
//
template<typename EvalT>
Teuchos::RCP<PHX::FieldTag const>
Albany::SolidMechanics::
constructEvaluators(
    PHX::FieldManager<PHAL::AlbanyTraits> & field_mgr,
    Albany::MeshSpecsStruct const & mesh_specs,
    Albany::StateManager & state_mgr,
    Albany::FieldManagerChoice fm_choice,
    Teuchos::RCP<Teuchos::ParameterList> const & response_list)
{
  using Basis = Teuchos::RCP<Intrepid2::Basis<PHX::Device, RealType, RealType>>;

  // Collect problem-specific response parameters

  Teuchos::RCP<Teuchos::ParameterList>
  response_param_list = Teuchos::rcp(
      new Teuchos::ParameterList("Response Parameters from Problem"));

  // get the name of the current element block
  std::string const &
  eb_name = mesh_specs.ebName;

  // get the name of the material model to be used (and make sure there is one)
  Teuchos::ParameterList const &
  eb_sublist = material_db_->getElementBlockSublist(eb_name, "Material Model");

  std::string const &
  material_model_name = eb_sublist.get<std::string>("Model Name");

  TEUCHOS_TEST_FOR_EXCEPTION(
      material_model_name.length() == 0,
      std::logic_error,
      "A material model must be defined for block: " + eb_name);

#ifdef ALBANY_VERBOSE
  *out << "In SolidMechanics::constructEvaluators" << std::endl;
  *out << "element block name: " << eb_name << std::endl;
  *out << "material model name: " << material_model_name << std::endl;
#endif

// define cell topologies
  Teuchos::RCP<shards::CellTopology> comp_cellType =
      Teuchos::rcp(
          new shards::CellTopology(
              shards::getCellTopologyData<shards::Tetrahedron<11>>()));
  Teuchos::RCP<shards::CellTopology> cellType =
      Teuchos::rcp(new shards::CellTopology(&mesh_specs.ctd));


  Intrepid2::DefaultCubatureFactory cubFactory;

  Teuchos::RCP<
      Intrepid2::Cubature<PHX::Device > cubature =
      cubFactory.create<PHX::Device, RealType, RealType>(*cellType, mesh_specs.cubatureDegree);


// Note that these are the volume element quantities
  int const
  workset_size = mesh_specs.worksetSize;

  Basis
  intrepid_basis = Albany::getIntrepid2Basis(mesh_specs.ctd, false);

  int const
  num_vertices = intrepid_basis->getCardinality();

  int const
  num_nodes = num_vertices;

  int const
  num_pts = cubature->getNumPoints();

  num_dims_ = cubature->getDimension();

#ifdef ALBANY_VERBOSE
  *out << "Field Dimensions: Workset=" << workset_size
  << ", Dim= " << num_dims_ << std::endl;
#endif

  // Construct standard FEM evaluators with standard field names
  dl_ = Teuchos::rcp(new Albany::Layouts(
      workset_size,
      num_vertices,
      num_nodes,
      num_pts,
      num_dims_));

  TEUCHOS_TEST_FOR_EXCEPTION(
      dl_->vectorAndGradientLayoutsAreEquivalent == false,
      std::logic_error,
      "Data Layout Usage in Solid Mechanics assume vecDim = num_dims_");

  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits>
  evalUtils(dl_);

  bool
  supports_transient{true};

  int
  offset{0};

  // Temporary variable used numerous times below
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits>>
  ev;

  // Define Field Names
  std::string cauchy{"Cauchy_Stress"};
  std::string firstPK{"FirstPK"};
  std::string Fp{"Fp"};
  std::string eqps{"eqps"};
  std::string temperature{"Temperature"};
  std::string pressure{"Pressure"};
  std::string mech_source{"Mechanical_Source"};
  std::string defgrad{"F"};
  std::string J{"J"};

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
          intrepid_basis,
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
    response_param_list->set<std::string>("x Field Name", "xField");

    // store computed xdot in "Velocity" field
    response_param_list->set<std::string>("xdot Field Name", "Velocity");

    // store computed xdotdot in "Acceleration" field
    response_param_list->set<std::string>("xdotdot Field Name", "Acceleration");

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
                Intrepid2::Cubature<PHX::Device>>>("Cubature", cubature);

    // Get the Adaptation list and send to the evaluator
    Teuchos::ParameterList& paramList = params->sublist("Adaptation");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    p
        ->set<
            const Teuchos::RCP<
                Intrepid2::Basis<PHX::Device, RealType, RealType>>>("Intrepid2 Basis", intrepid_basis);
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

  if (fm_choice == Albany::BUILD_RESID_FM) {

    Teuchos::RCP<const PHX::FieldTag> ret_tag;
    if (have_mech_eq_) {
      PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter", dl_->dummy);
      field_mgr.requireField<EvalT>(res_tag);
      ret_tag = res_tag.clone();
    }
    return ret_tag;
  }
  else if (fm_choice == Albany::BUILD_RESPONSE_FM) {
    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl_);
    return respUtils.constructResponses(
        field_mgr, *response_list, response_param_list, state_mgr, &mesh_specs);
  }

  return Teuchos::null;
}
