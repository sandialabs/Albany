#ifndef CTM_THERMAL_PROBLEM_HPP
#define CTM_THERMAL_PROBLEM_HPP

#include "CTM_Teuchos.hpp"

#include <Albany_ProblemUtils.hpp>
#include <Albany_AbstractProblem.hpp>
#include <PHAL_AlbanyTraits.hpp>
#include <MaterialDatabase.h>
#include <Phalanx.hpp>

namespace CTM {

class ThermalProblem : public Albany::AbstractProblem {

  public:

    ThermalProblem(
        const RCP<ParameterList>& params,
        RCP<ParamLib> const& param_lib,
        const int num_dims,
        RCP<const Teuchos::Comm<int> >& comm);

    ThermalProblem(const ThermalProblem&) = delete;
    ThermalProblem& operator=(const ThermalProblem&) = delete;

    ~ThermalProblem();

    int spatialDimension() const { return num_dims; }

    void buildProblem(
        ArrayRCP<RCP<Albany::MeshSpecsStruct> > mesh_specs,
        Albany::StateManager& state_mgr);

    Teuchos::Array<RCP<const PHX::FieldTag> > buildEvaluators(
        PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
        const Albany::MeshSpecsStruct& mesh_specs,
        Albany::StateManager& state_mgr,
        Albany::FieldManagerChoice fm_choice,
        const RCP<ParameterList>& response_list);

    template <typename EvalT>
    RCP<const PHX::FieldTag> constructEvaluators(
        PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
        const Albany::MeshSpecsStruct& mesh_specs,
        Albany::StateManager& state_mgr,
        Albany::FieldManagerChoice fm_choice,
        const RCP<ParameterList>& response_list);

    void constructDirichletEvaluators(
        const RCP<Albany::MeshSpecsStruct>& mesh_specs);

    void constructNeumannEvaluators(
        const RCP<Albany::MeshSpecsStruct>& mesh_specs);

    Teuchos::RCP<const Teuchos::ParameterList>
    getValidProblemParameters() const;

  private:

    int num_dims;
    RCP<Albany::Layouts> dl;
    RCP<LCM::MaterialDatabase> material_db_;
    std::string materialFileName_;
    RCP<const Teuchos::Comm<int>> comm_;

  protected:

    enum SOURCE_TYPE {
      SOURCE_TYPE_NONE, //! No source
      SOURCE_TYPE_INPUT, //! Source is specified in input file
      SOURCE_TYPE_MATERIAL //! Source is specified in material database
    };

    bool have_source_;
    SOURCE_TYPE thermal_source_;

    bool thermal_source_evaluated_;
    bool isTransient_;
};

} // namespace CTM

#include <Albany_EvaluatorUtils.hpp>
#include <Intrepid2_DefaultCubatureFactory.hpp>
#include <Shards_CellTopology.hpp>
#include <PHAL_Source.hpp>
#include <PHAL_SaveStateField.hpp>
#include <ConstitutiveModelInterface.hpp>
#include <ConstitutiveModelParameters.hpp>
#include <ThermoMechanicalCoefficients.hpp>
#include <TransportResidual.hpp>

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag> CTM::ThermalProblem::constructEvaluators(
    PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
    const Albany::MeshSpecsStruct& mesh_specs,
    Albany::StateManager& state_mgr,
    Albany::FieldManagerChoice fm_choice,
    const RCP<ParameterList>& response_list) {

  using Teuchos::RCP;
  using Teuchos::ParameterList;
  using Intrepid2Basis = RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> >;

  // get element block specific info
  auto eb_name = mesh_specs.ebName;
  std::string material_model_name =
    material_db_->getElementBlockSublist(
        eb_name, "Material Model").get<std::string>("Model Name");

  TEUCHOS_TEST_FOR_EXCEPTION(
      material_model_name.length() == 0,
      std::logic_error,
      "A material model must be defined for block: "
      + eb_name);

  // build element level information
  auto cell_type = rcp(new shards::CellTopology(&mesh_specs.ctd));
  auto intrepidBasis = Albany::getIntrepid2Basis(mesh_specs.ctd);
  Intrepid2::DefaultCubatureFactory cubFactory;
  auto cubature = cubFactory.create<PHX::Device, RealType, RealType>(
      *cell_type, mesh_specs.cubatureDegree);

  // construct a data layout
  const int num_nodes = intrepidBasis->getCardinality();
  const int ws_size = mesh_specs.worksetSize;
  const int num_qps = cubature->getNumPoints();
  const int num_vtx = cell_type->getNodeCount();
  dl = rcp(new Albany::Layouts(ws_size, num_vtx, num_nodes, num_qps, num_dims));

  // set up relevant names
  Teuchos::ArrayRCP<std::string> dof_names(1);
  Teuchos::ArrayRCP<std::string> dof_names_dot(1);
  Teuchos::ArrayRCP<std::string> resid_names(1);
  dof_names[0] = "Temperature";
  dof_names_dot[0] = "Temperature Dot";
  resid_names[0] = dof_names[0] + " Residual";

  // standard fem evaluator registration
  int offset = 0;
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);

  fm0.template registerEvaluator<EvalT>(
      evalUtils.constructGatherSolutionEvaluator(
        false, dof_names, dof_names_dot, offset));

  fm0.template registerEvaluator<EvalT>(
      evalUtils.constructDOFInterpolationEvaluator(
        dof_names_dot[0], offset));

  fm0.template registerEvaluator<EvalT>(
      evalUtils.constructGatherCoordinateVectorEvaluator());

  fm0.template registerEvaluator<EvalT>(
      evalUtils.constructDOFInterpolationEvaluator(dof_names[0], offset));

  fm0.template registerEvaluator<EvalT>(
      evalUtils.constructDOFGradInterpolationEvaluator(dof_names[0], offset));

  fm0.template registerEvaluator<EvalT>(
      evalUtils.constructMapToPhysicalFrameEvaluator(cell_type, cubature));

  fm0.template registerEvaluator<EvalT>(
      evalUtils.constructComputeBasisFunctionsEvaluator(
        cell_type, intrepidBasis, cubature));

  fm0.template registerEvaluator<EvalT>(
      evalUtils.constructScatterResidualEvaluator(
        false, resid_names, offset, "Scatter Temperature"));

  // dummy variable reused below
  Teuchos::RCP<PHX::Evaluator < PHAL::AlbanyTraits>> ev;

  // compute the thermal source if it exists
  if (thermal_source_ != SOURCE_TYPE_NONE) {

    // common parameters
    auto p = rcp(new ParameterList);
    p->set<std::string>("Source Name", "Heat Source");
    p->set<std::string>("Variable Name", "Temperature");
    p->set<RCP<PHX::DataLayout > >("QP Scalar Data Layout", dl->qp_scalar);
    p->set<Teuchos::RCP<ParamLib> >("Parameter Library", paramLib);

    // thermal source in the input file
    if (thermal_source_ == SOURCE_TYPE_INPUT) {
      auto paramList =
        params->sublist("Source Functions").sublist("Thermal Source");
      p->set<ParameterList*>("ParameterList", &paramList);
      ev = rcp(new PHAL::Source<EvalT, PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
      thermal_source_evaluated_ = true;
    }

    // block dependent sources
    else if (thermal_source_ == SOURCE_TYPE_MATERIAL) {

      // may not be a source in every block
      if (material_db_->isElementBlockSublist(eb_name, "Source Functions")) {
        auto srcParamList =
          material_db_->getElementBlockSublist(eb_name, "Source Functions");
        if (srcParamList.isSublist("Thermal Source")) {
          auto paramList = srcParamList.sublist("Thermal Source");
          p->set<ParameterList*>("Parameter List", &paramList);
          ev = rcp(new PHAL::Source<EvalT, PHAL::AlbanyTraits>(*p));
          fm0.template registerEvaluator<EvalT>(ev);
          thermal_source_evaluated_ = true;
        }
      }

      // otherwise do not evaluate the source
      else
        thermal_source_evaluated_ = false;
    }

    // unknown source function
    else
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
          "Unrecognized thermal source specified in input file");
  }
    
  { // save the temperature as a state.
    double temp = 0.0;
    if (material_db_->isElementBlockParam(eb_name, "Initial Temperature")) {
      temp = material_db_->
        getElementBlockParam<double>(eb_name, "Initial Temperature");
    }
    auto p = rcp(new ParameterList);
    p = state_mgr.registerStateVariable(
        "Temperature", dl->qp_scalar, dl->dummy, eb_name,
        "scalar", temp, true, false);
    ev = rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // Constitutive Model Parameters
    auto p = rcp(new ParameterList);
    auto matName = material_db_->getElementBlockParam<std::string>(
        eb_name, "material");
    auto param_list = material_db_->getElementBlockSublist(eb_name, matName);
    p->set<std::string>("Temperature Name", dof_names[0]);
    param_list.set<bool>("Have Temperature", true);
    p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");
    p->set<Teuchos::ParameterList*>("Material Parameters", &param_list);
    auto cmpEv =
      rcp(new LCM::ConstitutiveModelParameters<EvalT, PHAL::AlbanyTraits>(
            *p, dl));
    fm0.template registerEvaluator<EvalT>(cmpEv);
  }

  { // Thermomechanical Coefficients
    auto p = rcp(new ParameterList);
    auto matName = material_db_->getElementBlockParam<std::string>(
        eb_name, "material");
    auto param_list = material_db_->getElementBlockSublist(eb_name, matName);
    p->set<Teuchos::ParameterList*>("Material Parameters", &param_list);
    p->set<std::string>("Temperature Name", "Temperature");
    p->set<std::string>("Temperature Dot Name", "Temperature Dot");
    p->set<std::string>("Solution Method Type", "No Continuation");
    p->set<std::string>("Thermal Conductivity Name", "Thermal Conductivity");
    p->set<std::string>("Thermal Transient Coefficient Name",
        "Thermal Transient Coefficient");
    p->set<std::string>("Thermal Diffusivity Name", "Thermal Diffusivity");
    ev = rcp(new LCM::ThermoMechanicalCoefficients<EvalT, PHAL::AlbanyTraits>(
          *p, dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // Temperature Residual
    auto p = rcp(new ParameterList);
    p->set<std::string>("Scalar Variable Name", "Temperature");
    p->set<std::string>("Scalar Gradient Variable Name",
        "Temperature Gradient");
    p->set<std::string>("Weights Name", "Weights");
    p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
    p->set<std::string>("Weighted BF Name", "wBF");
    p->set<bool>("Have Transient", true);
    p->set<std::string>("Scalar Dot Name", "Temperature Dot");
    p->set<std::string>("Transient Coefficient Name",
        "Thermal Transient Coefficient");
    p->set<std::string>("Solution Method Type", "No Continuation");
    p->set<bool>("Have Diffusion", true);
    p->set<std::string>("Diffusivity Name", "Thermal Diffusivity");
    if (thermal_source_evaluated_) {
      p->set<bool>("Have Second Source", true);
      p->set<std::string>("Second Source Name", "Heat Source");
    }
    p->set<std::string>("Residual Name", "Temperature Residual");
    ev = rcp(new LCM::TransportResidual<EvalT, PHAL::AlbanyTraits>(*p, dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }
    
  if (fm_choice == Albany::BUILD_RESID_FM) {
    Teuchos::RCP<const PHX::FieldTag> ret_tag;
    PHX::Tag<typename EvalT::ScalarT > temperature_tag(
        "Scatter Temperature", dl->dummy);
    fm0.requireField<EvalT>(temperature_tag);
    ret_tag = temperature_tag.clone();
    return ret_tag;
  }

  return Teuchos::null;

}

#endif
