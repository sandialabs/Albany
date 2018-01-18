//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ELECTROMECHANICSPROBLEM_HPP
#define ELECTROMECHANICSPROBLEM_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractProblem.hpp"
#include "ConstitutiveModelInterface.hpp"

#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_AlbanyTraits.hpp"

#include "AAdapt_RC_Manager.hpp"

namespace Albany
{

//------------------------------------------------------------------------------
///
/// \brief Definition for the Mechanics Problem
///
class ElectroMechanicsProblem: public Albany::AbstractProblem
{
public:

  typedef Kokkos::DynRankView<RealType, PHX::Device> FC;

  ///
  /// Default constructor
  ///
  ElectroMechanicsProblem(const Teuchos::RCP<Teuchos::ParameterList>& params,
      const Teuchos::RCP<ParamLib>& param_lib,
      const int num_dims,
      Teuchos::RCP<const Teuchos::Comm<int>>& commT);
  ///
  /// Destructor
  ///
  virtual
  ~ElectroMechanicsProblem();

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
  /// Get boolean telling code if SDBCs are utilized
  ///
  virtual bool
  useSDBCs() const {
    return false;
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

  //----------------------------------------------------------------------------
private:

  ///
  /// Private to prohibit copying
  ///
  ElectroMechanicsProblem(const ElectroMechanicsProblem&);

  ///
  /// Private to prohibit copying
  ///
  ElectroMechanicsProblem& operator=(const ElectroMechanicsProblem&);

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
  /// Data layouts
  ///
  Teuchos::RCP<Albany::Layouts> dl_;

  ///
  /// RCP to matDB object
  ///
  Teuchos::RCP<Albany::MaterialDatabase> material_db_;

  ///
  /// old state data
  ///
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<FC>>> old_state_;

  ///
  /// new state data
  ///
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<FC>>> new_state_;

  template <typename EvalT>
  void registerStateVariables(
    Teuchos::RCP<LCM::ConstitutiveModelInterface<EvalT, PHAL::AlbanyTraits>> cmiEv,
    PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
    Albany::StateManager& stateMgr, std::string eb_name, int numDim);

};
//------------------------------------------------------------------------------
}

#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"
#include "Albany_ResponseUtilities.hpp"
#include "Albany_EvaluatorUtils.hpp"

#include "PHAL_NSMaterialProperty.hpp"
#include "PHAL_SaveStateField.hpp"
#include "PHAL_SaveCellStateField.hpp"

#include "FieldNameMap.hpp"

#include "MechanicsResidual.hpp"
#include "ElectrostaticResidual.hpp"
#include "Time.hpp"
#include "CurrentCoords.hpp"

// Constitutive Model Interface and parameters
#include "Kinematics.hpp"
#include "ConstitutiveModelInterface.hpp"
#include "ConstitutiveModelParameters.hpp"
#include "FirstPK.hpp"

//------------------------------------------------------------------------------
template<typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
Albany::ElectroMechanicsProblem::
constructEvaluators(PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
    const Albany::MeshSpecsStruct& meshSpecs,
    Albany::StateManager& stateMgr,
    Albany::FieldManagerChoice fieldManagerChoice,
    const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  typedef Teuchos::RCP<
      Intrepid2::Basis<PHX::Device, RealType, RealType>>
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
  *out << "In ElectroMechanicsProblem::constructEvaluators" << std::endl;
  *out << "element block name: " << eb_name << std::endl;
  *out << "material model name: " << material_model_name << std::endl;
#endif

  // define cell topologies
  Teuchos::RCP<shards::CellTopology> comp_cellType =
      Teuchos::rcp(
          new shards::CellTopology(
              shards::getCellTopologyData<shards::Tetrahedron<11>>()));
  Teuchos::RCP<shards::CellTopology> cellType =
      Teuchos::rcp(new shards::CellTopology(&meshSpecs.ctd));

  // volume averaging flags
  bool volume_average_pressure(false);
  RealType volume_average_stabilization_param(0.0);
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

  // get the intrepid basis for the given cell topology
  Intrepid2Basis intrepidBasis = Albany::getIntrepid2Basis(meshSpecs.ctd);
  Intrepid2::DefaultCubatureFactory cubFactory;
  Teuchos::RCP<Intrepid2::Cubature<PHX::Device>> cubature =
      cubFactory.create<PHX::Device, RealType, RealType>(*cellType, meshSpecs.cubatureDegree);

  // Note that these are the volume element quantities
  num_nodes_ = intrepidBasis->getCardinality();
  const int workset_size = meshSpecs.worksetSize;

  num_dims_     = cubature->getDimension();
  num_pts_      = cubature->getNumPoints();
  num_vertices_ = cellType->getNodeCount();

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
  TEUCHOS_TEST_FOR_EXCEPTION(
      dl_->vectorAndGradientLayoutsAreEquivalent == false, std::logic_error,
      "Data Layout Usage in ElectroMechanics problems assume vecDim = num_dims_");

  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl_);
  bool supports_transient = false;
  int offset = 0;

  // Temporary variable used numerous times below
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits>> ev;

  // Define Field Names
  LCM::FieldNameMap field_name_map(false);
  Teuchos::RCP<std::map<std::string, std::string>> fnm = field_name_map.getMap();

  { // Mechanical
    Teuchos::ArrayRCP<std::string> dof_names(1);
    Teuchos::ArrayRCP<std::string> resid_names(1);
    dof_names[0] = "Displacement";
    resid_names[0] = dof_names[0] + " Residual";

    fm0.template registerEvaluator<EvalT>
      (evalUtils.constructGatherSolutionEvaluator_noTransient(true, dof_names));

    fm0.template registerEvaluator<EvalT>
      (evalUtils.constructGatherCoordinateVectorEvaluator());

    fm0.template registerEvaluator<EvalT>
      (evalUtils.constructDOFVecInterpolationEvaluator(dof_names[0]));

    fm0.template registerEvaluator<EvalT>
      (evalUtils.constructDOFVecGradInterpolationEvaluator(dof_names[0]));

    fm0.template registerEvaluator<EvalT>
      (evalUtils.constructMapToPhysicalFrameEvaluator(cellType, cubature));

    fm0.template registerEvaluator<EvalT>
      (evalUtils.constructComputeBasisFunctionsEvaluator(cellType, intrepidBasis, cubature));

    fm0.template registerEvaluator<EvalT>
      (evalUtils.constructScatterResidualEvaluator(true, resid_names));
    offset += num_dims_;
  }


  { // Electrical
    Teuchos::ArrayRCP<std::string> dof_names(1);
    Teuchos::ArrayRCP<std::string> resid_names(1);
    dof_names[0] = "Electric Potential";
    resid_names[0] = dof_names[0] + " Residual";
    fm0.template registerEvaluator<EvalT>
      (evalUtils.constructGatherSolutionEvaluator_noTransient(false, dof_names, offset));

    fm0.template registerEvaluator<EvalT>
      (evalUtils.constructGatherCoordinateVectorEvaluator());

    fm0.template registerEvaluator<EvalT>
      (evalUtils.constructDOFInterpolationEvaluator(dof_names[0], offset));

    fm0.template registerEvaluator<EvalT>
      (evalUtils.constructDOFGradInterpolationEvaluator(dof_names[0], offset));

    fm0.template registerEvaluator<EvalT>
      (evalUtils.constructMapToPhysicalFrameEvaluator(cellType,
          cubature));

    fm0.template registerEvaluator<EvalT>
      (evalUtils.constructComputeBasisFunctionsEvaluator(cellType, intrepidBasis, cubature));

    fm0.template registerEvaluator<EvalT>
      (evalUtils.constructScatterResidualEvaluator(false, resid_names, offset, "Scatter Electric Potential"));
    offset++;
  }



  { // Temperature
    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(new Teuchos::ParameterList);

    p->set<std::string>("Material Property Name", "Temperature");
    p->set<Teuchos::RCP<PHX::DataLayout>>("Data Layout", dl_->qp_scalar);
    p->set<std::string>("Coordinate Vector Name", "Coord Vec");
    p->set<Teuchos::RCP<PHX::DataLayout>>( "Coordinate Vector Data Layout", dl_->qp_vector);

    p->set<Teuchos::RCP<ParamLib>>("Parameter Library", paramLib);
    Teuchos::ParameterList& paramList = params->sublist("Temperature");
    p->set<Teuchos::ParameterList*>("Parameter List", &paramList);

    ev = Teuchos::rcp(new PHAL::NSMaterialProperty<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
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

  { // Current Coordinates
    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
        new Teuchos::ParameterList("Current Coordinates"));
    p->set<std::string>("Reference Coordinates Name", "Coord Vec");
    p->set<std::string>("Displacement Name", "Displacement");
    p->set<std::string>("Current Coordinates Name", "Current Coordinates");
    ev = Teuchos::rcp(
        new LCM::CurrentCoords<EvalT, PHAL::AlbanyTraits>(*p, dl_));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  std::string temperature("Temperature");
  std::string stress("Stress");
  std::string edisp("Electric Displacement");
  std::string defgrad("Deformation Gradient");
  std::string J("Jacobian Determinant");
  {
    double temp(0.0);
    if (material_db_->isElementBlockParam(eb_name, "Initial Temperature")) {
      temp = material_db_->
          getElementBlockParam<double>(eb_name, "Initial Temperature");
    }
    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
        new Teuchos::ParameterList("Save Temperature"));
    p = stateMgr.registerStateVariable(temperature,
        dl_->qp_scalar, dl_->dummy, eb_name, "scalar", temp, true, false);
    ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // Constitutive Model Parameters
    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
        new Teuchos::ParameterList("Constitutive Model Parameters"));
    std::string matName = material_db_->getElementBlockParam<std::string>(
        eb_name, "material");
    Teuchos::ParameterList& param_list = material_db_->getElementBlockSublist(eb_name, matName);
    p->set<std::string>("Temperature Name", temperature);
    param_list.set<bool>("Have Temperature", true);

    // optional spatial dependence
    p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");

    // pass through material properties
    p->set<Teuchos::ParameterList*>("Material Parameters", &param_list);

    Teuchos::RCP<LCM::ConstitutiveModelParameters<EvalT, PHAL::AlbanyTraits>>
      cmpEv = Teuchos::rcp( new LCM::ConstitutiveModelParameters<EvalT, PHAL::AlbanyTraits>( *p, dl_));
    fm0.template registerEvaluator<EvalT>(cmpEv);
  }

  {
    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
        new Teuchos::ParameterList("Constitutive Model Interface"));
    std::string matName = material_db_->getElementBlockParam<std::string>( eb_name, "material");
    Teuchos::ParameterList& param_list = material_db_->getElementBlockSublist(eb_name, matName);

    p->set<std::string>("Temperature Name", temperature);
    param_list.set<bool>("Have Temperature", true);

    param_list.set<Teuchos::RCP<std::map<std::string, std::string>>>( "Name Map", fnm);
    p->set<Teuchos::ParameterList*>("Material Parameters", &param_list);
    p->set<bool>("Volume Average Pressure", volume_average_pressure);
    if (volume_average_pressure) {
      p->set<std::string>("Weights Name", "Weights");
      p->set<std::string>("J Name", J);
    }

    Teuchos::RCP<LCM::ConstitutiveModelInterface<EvalT, PHAL::AlbanyTraits>>
      cmiEv = Teuchos::rcp( new LCM::ConstitutiveModelInterface<EvalT, PHAL::AlbanyTraits>( *p, dl_));
    fm0.template registerEvaluator<EvalT>(cmiEv);

    registerStateVariables(cmiEv, fm0, stateMgr, eb_name, num_dims_);
  }

  { // Kinematics quantities
    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(new Teuchos::ParameterList("Kinematics"));

    // send in integration weights and the displacement gradient
    p->set<std::string>("Weights Name", "Weights");
    p->set<Teuchos::RCP<PHX::DataLayout>>( "QP Scalar Data Layout", dl_->qp_scalar);

    p->set<std::string>("Gradient QP Variable Name", "Displacement Gradient");
    p->set<Teuchos::RCP<PHX::DataLayout>>( "QP Tensor Data Layout", dl_->qp_tensor);

    //Outputs: F, J, strain
    p->set<std::string>("DefGrad Name", defgrad); //dl_->qp_tensor also
    p->set<std::string>("DetDefGrad Name", J);
    p->set<std::string>("Strain Name", "Strain");
    p->set<Teuchos::RCP<PHX::DataLayout>>( "QP Scalar Data Layout", dl_->qp_scalar);

    ev = Teuchos::rcp( new LCM::Kinematics<EvalT, PHAL::AlbanyTraits>(*p, dl_));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // Mechanical Residual
    Teuchos::RCP<Teuchos::ParameterList>
      p = Teuchos::rcp(new Teuchos::ParameterList("Displacement Residual"));
    //Input
    p->set<std::string>("Stress Name", stress);
    p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
    p->set<std::string>("Weighted BF Name", "wBF");
    p->set<bool>("Disable Dynamics", true);
    p->set<Teuchos::RCP<ParamLib>>("Parameter Library", paramLib);
    //Output
    p->set<std::string>("Residual Name", "Displacement Residual");
    ev = Teuchos::rcp(
        new LCM::MechanicsResidual<EvalT, PHAL::AlbanyTraits>(*p, dl_));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // Electrical Residual
    Teuchos::RCP<Teuchos::ParameterList>
      p = Teuchos::rcp(new Teuchos::ParameterList("Electric Potential Residual"));
    //Input
    p->set<std::string>("Electric Displacement Name", edisp);
    p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
    //Output
    p->set<std::string>("Residual Name", "Electric Potential Residual");
    ev = Teuchos::rcp(
        new LCM::ElectrostaticResidual<EvalT, PHAL::AlbanyTraits>(*p, dl_));
    fm0.template registerEvaluator<EvalT>(ev);
  }



  if (fieldManagerChoice == Albany::BUILD_RESID_FM) {

    Teuchos::RCP<const PHX::FieldTag> ret_tag;
    PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter", dl_->dummy);
    fm0.requireField<EvalT>(res_tag);
    ret_tag = res_tag.clone();

    PHX::Tag<typename EvalT::ScalarT> potential_tag("Scatter Electric Potential", dl_->dummy);
    fm0.requireField<EvalT>(potential_tag);
    ret_tag = potential_tag.clone();
    return ret_tag;

  }


  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM) {
    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl_);
    return respUtils.constructResponses(fm0, *responseList, pFromProb, stateMgr, &meshSpecs);
  }

  return Teuchos::null;
}


/******************************************************************************/
template <typename EvalT>
void Albany::ElectroMechanicsProblem::registerStateVariables(
  Teuchos::RCP<LCM::ConstitutiveModelInterface<EvalT, PHAL::AlbanyTraits>> cmiEv,
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  Albany::StateManager& stateMgr, std::string eb_name, int numDim)
/******************************************************************************/
{
  Teuchos::RCP<Teuchos::ParameterList> p;
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits>> ev;

  // register state variables
  for (int sv(0); sv < cmiEv->getNumStateVars(); ++sv) {
    cmiEv->fillStateVariableStruct(sv);

    //
    // QUAD POINT SCALARS
    if( (cmiEv->getLayout() == dl_->qp_scalar) &&
        (cmiEv->getOutputFlag() == true) ){

      // save cell average for output
      p = stateMgr.registerStateVariable(cmiEv->getName()+"_ave",
          dl_->cell_scalar, dl_->dummy, eb_name, cmiEv->getInitType(),
          cmiEv->getInitValue(), /* save state = */ false, /* write output = */ true);
      p->set("Field Layout", dl_->qp_scalar);
      p->set("Field Name", cmiEv->getName());
      p->set("Weights Layout", dl_->qp_scalar);
      p->set("Weights Name", "Weights");
      ev = Teuchos::rcp(new PHAL::SaveCellStateField<EvalT, PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);

      // save state w/o output
      p = stateMgr.registerStateVariable(cmiEv->getName(),
          cmiEv->getLayout(), dl_->dummy, eb_name, cmiEv->getInitType(),
          cmiEv->getInitValue(), cmiEv->getStateFlag(), /* write output = */ false);
      ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);

    } else

    //
    // QUAD POINT VECTORS
    if( (cmiEv->getLayout() == dl_->qp_vector) &&
        (cmiEv->getOutputFlag() == true) ){

      std::string cn[3] = {"x","y","z"};

      // save cell average for output
      for(int i=0; i< numDim; i++){
        std::string varname(cmiEv->getName());
        varname += " ";
        varname += cn[i];
        varname += "_ave ";
        p = stateMgr.registerStateVariable(varname,
            dl_->cell_scalar, dl_->dummy, eb_name, cmiEv->getInitType(),
            cmiEv->getInitValue(), /* save state = */ false, /* write output = */ true);
        p->set("Field Layout", dl_->qp_vector);
        p->set("Field Name", cmiEv->getName());
        p->set("Weights Layout", dl_->qp_scalar);
        p->set("Weights Name", "Weights");
        p->set("component i", i);
        ev = Teuchos::rcp(new PHAL::SaveCellStateField<EvalT, PHAL::AlbanyTraits>(*p));
        fm0.template registerEvaluator<EvalT>(ev);
      }

      // save state w/o output
      p = stateMgr.registerStateVariable(cmiEv->getName(),
          cmiEv->getLayout(), dl_->dummy, eb_name, cmiEv->getInitType(),
          cmiEv->getInitValue(), cmiEv->getStateFlag(), /* write output = */ false);
      ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);

    } else

    //
    // QUAD POINT TENSORS
    if( (cmiEv->getLayout() == dl_->qp_tensor) &&
        (cmiEv->getOutputFlag() == true) ){

      std::string cn[3] = {"x","y","z"};

      // save cell average for output
      for(int i=0; i< numDim; i++)
        for(int j=0; j< numDim; j++){
          std::string varname(cmiEv->getName());
          varname += " ";
          varname += cn[i];
          varname += cn[j];
          varname += "_ave ";
          p = stateMgr.registerStateVariable(varname,
              dl_->cell_scalar, dl_->dummy, eb_name, cmiEv->getInitType(),
              cmiEv->getInitValue(), /* save state = */ false, /* write output = */ true);
          p->set("Field Layout", dl_->qp_tensor);
          p->set("Field Name", cmiEv->getName());
          p->set("Weights Layout", dl_->qp_scalar);
          p->set("Weights Name", "Weights");
          p->set("component i", i);
          p->set("component j", j);
          ev = Teuchos::rcp(new PHAL::SaveCellStateField<EvalT, PHAL::AlbanyTraits>(*p));
          fm0.template registerEvaluator<EvalT>(ev);
        }

      // save state w/o output
      p = stateMgr.registerStateVariable(cmiEv->getName(),
          cmiEv->getLayout(), dl_->dummy, eb_name, cmiEv->getInitType(),
          cmiEv->getInitValue(), cmiEv->getStateFlag(), /* write output = */ false);
      ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);

    } else

    //
    // QUAD POINT THIRD RANK TENSORS
    if( (cmiEv->getLayout() == dl_->qp_tensor3) &&
        (cmiEv->getOutputFlag() == true) ){

      std::string cn[3] = {"x","y","z"};

      // save cell average for output
      for(int i=0; i< numDim; i++)
        for(int j=0; j< numDim; j++)
          for(int k=0; k< numDim; k++){
            std::string varname(cmiEv->getName());
            varname += " ";
            varname += cn[i];
            varname += cn[j];
            varname += cn[k];
            varname += "_ave ";
            p = stateMgr.registerStateVariable(varname,
                dl_->cell_scalar, dl_->dummy, eb_name, cmiEv->getInitType(),
                cmiEv->getInitValue(), /* save state = */ false, /* write output = */ true);
            p->set("Field Layout", dl_->qp_tensor3);
            p->set("Field Name", cmiEv->getName());
            p->set("Weights Layout", dl_->qp_scalar);
            p->set("Weights Name", "Weights");
            p->set("component i", i);
            p->set("component j", j);
            p->set("component k", k);
            ev = Teuchos::rcp(new PHAL::SaveCellStateField<EvalT, PHAL::AlbanyTraits>(*p));
            fm0.template registerEvaluator<EvalT>(ev);
          }

      // save state w/o output
      p = stateMgr.registerStateVariable(cmiEv->getName(),
          cmiEv->getLayout(), dl_->dummy, eb_name, cmiEv->getInitType(),
          cmiEv->getInitValue(), cmiEv->getStateFlag(), /* write output = */ false);
      ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);

    } else {
      p = stateMgr.registerStateVariable(cmiEv->getName(),
          cmiEv->getLayout(),
          dl_->dummy,
          eb_name,
          cmiEv->getInitType(),
          cmiEv->getInitValue(),
          cmiEv->getStateFlag(),
          cmiEv->getOutputFlag());
      ev = Teuchos::rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }
  }
}
#endif
