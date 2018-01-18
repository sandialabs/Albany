//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_ConstitutiveDriverProblem_hpp)
#define LCM_ConstitutiveDriverProblem_hpp

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractProblem.hpp"

#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"
#include "PHAL_AlbanyTraits.hpp"

namespace Albany
{

//------------------------------------------------------------------------------
///
/// \brief Definition for the Constitutive Model Driver Problem
///
class ConstitutiveDriverProblem: public Albany::AbstractProblem
{
public:

  typedef Kokkos::DynRankView<RealType, PHX::Device> FC;

  ///
  /// Default constructor
  ///
  ConstitutiveDriverProblem(const Teuchos::RCP<Teuchos::ParameterList>& params,
      const Teuchos::RCP<ParamLib>& param_lib,
      const int num_dims,
      Teuchos::RCP<const Teuchos::Comm<int>>& commT);

  ///
  /// Destructor
  ///
  virtual
  ~ConstitutiveDriverProblem();

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
    return use_sdbcs_;
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
  ConstitutiveDriverProblem(const ConstitutiveDriverProblem&);

  ///
  /// Private to prohibit copying
  ///
  ConstitutiveDriverProblem& operator=(const ConstitutiveDriverProblem&);

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
  ///Boolean marking whether SDBCs are used
  bool use_sdbcs_;

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
  /// number of element vertices
  ///
  bool have_temperature_;

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
#include "PHAL_ScatterResidual.hpp"
#include "FieldNameMap.hpp"

#include "Time.hpp"

// Constitutive Model Interface and parameters
#include "Kinematics.hpp"
#include "ConstitutiveModelInterface.hpp"
#include "ConstitutiveModelParameters.hpp"
#include "ConstitutiveModelDriver.hpp"
#include "ConstitutiveModelDriverPre.hpp"
#include "FirstPK.hpp"

//------------------------------------------------------------------------------
template<typename EvalT>
Teuchos::RCP<const PHX::FieldTag>
Albany::ConstitutiveDriverProblem::
constructEvaluators(PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
    const Albany::MeshSpecsStruct& meshSpecs,
    Albany::StateManager& stateMgr,
    Albany::FieldManagerChoice fieldManagerChoice,
    const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Collect problem-specific response parameters
  Teuchos::RCP<Teuchos::ParameterList> pFromProb = Teuchos::rcp(
      new Teuchos::ParameterList("Response Parameters from Problem"));

  // get the name of the current element block
  std::string eb_name = meshSpecs.ebName;

  // get the name of the material model to be used (and make sure there is one)
  std::string material_model_name =
    material_db_->
    getElementBlockSublist(eb_name, "Material Model").
    get<std::string>("Model Name");
  TEUCHOS_TEST_FOR_EXCEPTION(
      material_model_name.length() == 0,
      std::logic_error,
      "A material model must be defined for block: "
          + eb_name);

  // Note that these are the volume element quantities
  num_nodes_ = 1;
  const int workset_size = meshSpecs.worksetSize;
  num_pts_ = 1;
  num_vertices_ = num_nodes_;

  *out << "Field Dimensions: Workset=" << workset_size
       << ", Vertices= " << num_vertices_
       << ", Nodes= " << num_nodes_
       << ", QuadPts= " << num_pts_
       << ", Dim= " << num_dims_ << std::endl;

  // Construct standard FEM evaluators with standard field names
  dl_ = Teuchos::rcp(new Albany::Layouts(workset_size,
                                         num_vertices_,
                                         num_nodes_,
                                         num_pts_,
                                         num_dims_));
  std::string msg = "Data Layout Usage in Mechanics problems assume vecDim = num_dims_";
  TEUCHOS_TEST_FOR_EXCEPTION(dl_->vectorAndGradientLayoutsAreEquivalent == false,
                             std::logic_error,
                             msg);
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl_);
  bool supports_transient = true;
  int offset = 0;

  // Define Field Names
  // generate the field name map to deal with outputing surface element info
  LCM::FieldNameMap field_name_map(false);
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

  // Temporary variable used numerous times below
  Teuchos::RCP<PHX::Evaluator<PHAL::AlbanyTraits>> ev;

  // Register the solution and residual fields
  Teuchos::ArrayRCP<std::string> dof_names(1);
  Teuchos::ArrayRCP<std::string> resid_names(1);
  dof_names[0] = "driver deformation gradient";
  resid_names[0] = "driver scatter";

  { // Gather Solution
    Teuchos::RCP<Teuchos::ParameterList> p =
      Teuchos::rcp(new Teuchos::ParameterList("Gather Solution"));
    p->set< Teuchos::ArrayRCP<std::string>>("Solution Names", dof_names);

    p->set<int>("Tensor Rank", 2);

    p->set<int>("Offset of First DOF", 0);
    p->set<bool>("Disable Transient", true);

    ev = Teuchos::rcp(new PHAL::GatherSolution<EvalT,PHAL::AlbanyTraits>(*p,dl_));
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

  { // Constitutive Model Driver Preprocessor
    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
        new Teuchos::ParameterList("Constitutive Model Driver Preprocessor"));

    p->set<Teuchos::ParameterList>("Driver Params", params->sublist("Constitutive Model Driver Parameters"));
    p->set<std::string>("Solution Name", dof_names[0]);
    p->set<std::string>("Prescribed F Name", "Prescribed F");
    p->set<std::string>("Time Name", "Time");
    p->set<std::string>("F Name", defgrad);
    p->set<std::string>("J Name", J);

    ev=Teuchos::rcp(new LCM::ConstitutiveModelDriverPre<EvalT, PHAL::AlbanyTraits>(*p,dl_));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // Constitutive Model Parameters
    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
        new Teuchos::ParameterList("Constitutive Model Parameters"));
    std::string matName = material_db_->getElementBlockParam<std::string>(
        eb_name, "material");
    Teuchos::ParameterList& param_list =
        material_db_->getElementBlockSublist(eb_name, matName);
    if (have_temperature_) {
      p->set<std::string>("Temperature Name", temperature);
      param_list.set<bool>("Have Temperature", true);
    }

    // pass through material properties
    p->set<Teuchos::ParameterList*>("Material Parameters", &param_list);

    Teuchos::RCP<LCM::ConstitutiveModelParameters<EvalT, PHAL::AlbanyTraits>>
    cmpEv =
      Teuchos::rcp(new LCM::ConstitutiveModelParameters<EvalT, PHAL::AlbanyTraits>(*p,dl_));
    fm0.template registerEvaluator<EvalT>(cmpEv);
  }

  { // Constitutive Model Interface
    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
        new Teuchos::ParameterList("Constitutive Model Interface"));
    std::string matName = material_db_->getElementBlockParam<std::string>(
        eb_name, "material");
    Teuchos::ParameterList& param_list =
        material_db_->getElementBlockSublist(eb_name, matName);

    // FIXME: figure out how to do this better
    param_list.set<bool>("Have Temperature", false);
    if (have_temperature_) {
      p->set<std::string>("Temperature Name", temperature);
      param_list.set<bool>("Have Temperature", true);
    }

    param_list.set<Teuchos::RCP<std::map<std::string, std::string>>>(
        "Name Map",
        fnm);
    p->set<Teuchos::ParameterList*>("Material Parameters", &param_list);

    Teuchos::RCP<LCM::ConstitutiveModelInterface<EvalT, PHAL::AlbanyTraits>>
    cmiEv =
        Teuchos::rcp(new LCM::ConstitutiveModelInterface<EvalT, PHAL::AlbanyTraits>(*p,dl_));
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

  { // Constitutive Model Driver
    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(
        new Teuchos::ParameterList("Constitutive Model Driver"));
    p->set<std::string>("Residual Name", resid_names[0]);
    p->set<std::string>("F Name", defgrad);
    p->set<std::string>("Prescribed F Name", "Prescribed F");
    p->set<std::string>("Stress Name", cauchy);
    Teuchos::RCP<LCM::ConstitutiveModelDriver<EvalT, PHAL::AlbanyTraits>>
    cmdEv =
        Teuchos::rcp(new LCM::ConstitutiveModelDriver<EvalT, PHAL::AlbanyTraits>(*p,dl_));
    fm0.template registerEvaluator<EvalT>(cmdEv);
  }


  { // Scatter Residual
    Teuchos::RCP<Teuchos::ParameterList> p =
      Teuchos::rcp(new Teuchos::ParameterList("Scatter Residual"));
    p->set< Teuchos::ArrayRCP<std::string>>("Residual Names", resid_names);
    p->set<int>("Tensor Rank", 2);
    p->set<int>("Offset of First DOF", 0);
    p->set<bool>("Disable Transient", true);
    ev = Teuchos::rcp(new PHAL::ScatterResidual<EvalT, PHAL::AlbanyTraits>(*p,dl_));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (fieldManagerChoice == Albany::BUILD_RESID_FM) {

    Teuchos::RCP<const PHX::FieldTag> ret_tag;

    PHX::Tag<typename EvalT::ScalarT> res_tag("Scatter", dl_->dummy);
    fm0.requireField<EvalT>(res_tag);
    ret_tag = res_tag.clone();

    return ret_tag;
  }
  else if (fieldManagerChoice == Albany::BUILD_RESPONSE_FM) {

    Albany::ResponseUtilities<EvalT, PHAL::AlbanyTraits> respUtils(dl_);
    return
    respUtils.constructResponses(fm0, *responseList, pFromProb, stateMgr);

  }

  return Teuchos::null;
}

#endif
