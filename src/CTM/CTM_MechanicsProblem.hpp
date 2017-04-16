#ifndef CTM_MECHANICS_PROBLEM_HPP
#define CTM_MECHANICS_PROBLEM_HPP

#include <Albany_ProblemUtils.hpp>
#include <Albany_AbstractProblem.hpp>
#include <PHAL_AlbanyTraits.hpp>
#include <Albany::MaterialDatabase.h>
#include <Phalanx.hpp>

namespace CTM {

using Teuchos::rcp;
using Teuchos::RCP;
using Teuchos::ArrayRCP;
using Teuchos::ParameterList;

class MechanicsProblem : public Albany::AbstractProblem {

  public:

    MechanicsProblem(
        const RCP<ParameterList>& params,
        RCP<ParamLib> const& param_lib,
        const int num_dims,
        RCP<const Teuchos::Comm<int> >& comm);

    MechanicsProblem(const MechanicsProblem&) = delete;
    MechanicsProblem& operator=(const MechanicsProblem&) = delete;

    ~MechanicsProblem();

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
    RCP<Albany::MaterialDatabase> material_db_;
    std::string materialFileName_;
    RCP<const Teuchos::Comm<int> > comm_;

};

} // namespace CTM

#include <Albany_EvaluatorUtils.hpp>
#include <Intrepid2_DefaultCubatureFactory.hpp>
#include <Shards_CellTopology.hpp>
#include <PHAL_Source.hpp>

#include <Albany_DiscretizationFactory.hpp>
#include <Albany_APFDiscretization.hpp>
#include <Albany_AbstractDiscretization.hpp>

#include <PHAL_Source.hpp>
#include <PHAL_SaveStateField.hpp>

#include <FieldNameMap.hpp>
#include <MechanicsResidual.hpp>
#include <CurrentCoords.hpp>
#include <MeshSizeField.hpp>
#include <Kinematics.hpp>
#include <ConstitutiveModelInterface.hpp>
#include <ConstitutiveModelParameters.hpp>
#include <FirstPK.hpp>

#include "CTM_TemperatureEvaluator.hpp"

template <typename EvalT>
Teuchos::RCP<const PHX::FieldTag> CTM::MechanicsProblem::constructEvaluators(
        PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
        const Albany::MeshSpecsStruct& mesh_specs,
        Albany::StateManager& state_mgr,
        Albany::FieldManagerChoice fm_choice,
        const RCP<ParameterList>& response_list) {

  using Teuchos::rcp;
  using Teuchos::RCP;
  using Teuchos::ParameterList;
  using Intrepid2Basis = RCP<Intrepid2::Basis<PHX::Device, RealType, RealType> >;

  // block specific info
  auto eb_name = mesh_specs.ebName;
  auto  material_model_name = material_db_->getElementBlockSublist(
      eb_name, "Material Model").get<std::string>("Model Name");
  TEUCHOS_TEST_FOR_EXCEPTION(
      material_model_name.length() == 0,
      std::logic_error,
      "A material model must be defined for block: "
      + eb_name);

  // element level specific info
  auto cell_type = rcp(new shards::CellTopology(&mesh_specs.ctd));
  Intrepid2Basis intrepidBasis = Albany::getIntrepid2Basis(mesh_specs.ctd);
  Intrepid2::DefaultCubatureFactory cubFactory;
  auto cubature = cubFactory.create<PHX::Device, RealType, RealType>(
      *cell_type, mesh_specs.cubatureDegree);

  // define the data layout
  const int num_nodes = intrepidBasis->getCardinality();
  const int ws_size = mesh_specs.worksetSize;
  const int num_qps = cubature->getNumPoints();
  const int num_vtx = cell_type->getNodeCount();
  dl = rcp(new Albany::Layouts(ws_size, num_vtx, num_nodes, num_qps, num_dims));

  // define field names
  LCM::FieldNameMap field_name_map(false);
  auto fnm = field_name_map.getMap();
  const std::string cauchy = (*fnm)["Cauchy_Stress"];
  const std::string firstPK = (*fnm)["FirstPK"];
  const std::string temperature = (*fnm)["Temperature"];
  const std::string mech_source = (*fnm)["Mechanical_Source"];
  const std::string defgrad = (*fnm)["F"];
  const std::string J = (*fnm)["J"];

  // evaluator utility
  Albany::EvaluatorUtils<EvalT, PHAL::AlbanyTraits> evalUtils(dl);
  Teuchos::RCP<PHX::Evaluator < PHAL::AlbanyTraits>> ev;

  // register variable names
  Teuchos::ArrayRCP<std::string> dof_names(1);
  Teuchos::ArrayRCP<std::string> resid_names(1);
  dof_names[0] = "Displacement";
  resid_names[0] = dof_names[0] + " Residual";

  // standard FEM evaluator registration

  int offset = 0;

  fm0.template registerEvaluator<EvalT>(
      evalUtils.constructGatherSolutionEvaluator_noTransient(true, dof_names));

  fm0.template registerEvaluator<EvalT>(
      evalUtils.constructGatherCoordinateVectorEvaluator());

  fm0.template registerEvaluator<EvalT>(
      evalUtils.constructDOFVecInterpolationEvaluator(dof_names[0]));

  fm0.template registerEvaluator<EvalT>(
      evalUtils.constructDOFVecGradInterpolationEvaluator(dof_names[0]));

  fm0.template registerEvaluator<EvalT>(
      evalUtils.constructMapToPhysicalFrameEvaluator(cell_type, cubature));

  fm0.template registerEvaluator<EvalT>(
      evalUtils.constructComputeBasisFunctionsEvaluator(
        cell_type, intrepidBasis, cubature));

  fm0.template registerEvaluator<EvalT>(
      evalUtils.constructScatterResidualEvaluator(true, resid_names));

  { // Current Coordinates
    auto p = rcp(new ParameterList);
    p->set<std::string>("Reference Coordinates Name", "Coord Vec");
    p->set<std::string>("Displacement Name", "Displacement");
    p->set<std::string>("Current Coordinates Name", "Current Coordinates");
    ev = rcp(new LCM::CurrentCoords<EvalT, PHAL::AlbanyTraits>(*p, dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  // set flag for small strain option
  bool small_strain(false);
  if (material_model_name == "Linear Elastic") {
    small_strain = true;
  }
  else {
    std::cout << "CTM only supports linear elastic model!!!" << std::endl;
    abort();
  }
   
  { // current temperature interpolated to qps, accessed through states
    auto p = rcp(new Teuchos::ParameterList);
    p->set<std::string>("Temperature Name", temperature);
    p->set<Teuchos::RCP < PHX::DataLayout >> ("Data Layout", dl->qp_scalar);
    ev = rcp(new CTM::Temperature<EvalT, PHAL::AlbanyTraits>(*p, dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }
    
  { // constitutive model parameters
    auto p = rcp(new ParameterList);
    auto matName = material_db_->getElementBlockParam<std::string>(
        eb_name, "material");
    auto param_list = material_db_->getElementBlockSublist(eb_name, matName);
    p->set<std::string>("Temperature Name", temperature);
    param_list.set<bool>("Have Temperature", true);
    p->set<std::string>("QP Coordinate Vector Name", "Coord Vec");
    p->set<Teuchos::ParameterList*>("Material Parameters", &param_list);
    auto cmpEv =
      rcp(new LCM::ConstitutiveModelParameters<EvalT, PHAL::AlbanyTraits>(
            *p, dl));
    fm0.template registerEvaluator<EvalT>(cmpEv);
  }

  { // constitutive model interface
    auto p = rcp(new ParameterList);
    std::string matName = material_db_->getElementBlockParam<std::string>(
        eb_name, "material");
    auto param_list = material_db_->getElementBlockSublist(eb_name, matName);
    p->set<std::string>("Temperature Name", temperature);
    param_list.set<bool>("Have Temperature", true);
    param_list.set<Teuchos::RCP<std::map < std::string, std::string> > >(
        "Name Map", fnm);
    p->set<Teuchos::ParameterList*>("Material Parameters", &param_list);
    auto cmiEv =
      rcp(new LCM::ConstitutiveModelInterface<EvalT, PHAL::AlbanyTraits>(
            *p, dl));
    fm0.template registerEvaluator<EvalT>(cmiEv);

    // register state variables
    for (int sv = 0; sv < cmiEv->getNumStateVars(); ++sv) {
      cmiEv->fillStateVariableStruct(sv);
      p = state_mgr.registerStateVariable(
          cmiEv->getName(), cmiEv->getLayout(), dl->dummy, eb_name,
          cmiEv->getInitType(), cmiEv->getInitValue(), cmiEv->getStateFlag(),
          cmiEv->getOutputFlag());
      ev = rcp(new PHAL::SaveStateField<EvalT, PHAL::AlbanyTraits>(*p));
      fm0.template registerEvaluator<EvalT>(ev);
    }
  }

  { // Kinematics quantities
    auto p = rcp(new ParameterList);
    if (small_strain) {
      p->set<std::string>("Strain Name", "Strain");
    }
    p->set<std::string>("Weights Name", "Weights");
    p->set<RCP<PHX::DataLayout> > ("QP Scalar Data Layout", dl->qp_scalar);
    p->set<std::string>("Gradient QP Variable Name", "Displacement Gradient");
    p->set<RCP<PHX::DataLayout> >("QP Tensor Data Layout", dl->qp_tensor);
    p->set<std::string>("DefGrad Name", defgrad);
    p->set<std::string>("DetDefGrad Name", J);
    p->set<std::string>("Strain Name", "Strain"); 
    ev = rcp(new LCM::Kinematics<EvalT, PHAL::AlbanyTraits>(*p, dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // First Piola-Kirchhoff evalautor
    auto p = rcp(new ParameterList);
    p->set<std::string>("Stress Name", cauchy);
    p->set<std::string>("DefGrad Name", defgrad);
    p->set<std::string>("First PK Stress Name", firstPK);
    if (small_strain) {
      p->set<bool>("Small Strain", true);
    }
    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    ev = rcp(new LCM::FirstPK<EvalT, PHAL::AlbanyTraits>(*p, dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  { // Momentum Residual
    auto p = rcp(new ParameterList);
    p->set<std::string>("Stress Name", firstPK);
    p->set<std::string>("Weighted Gradient BF Name", "wGrad BF");
    p->set<std::string>("Weighted BF Name", "wBF");
    p->set<std::string>("Acceleration Name", "Acceleration");
    p->set<bool>("Disable Dynamics", true);
    p->set<RCP<ParamLib> >("Parameter Library", paramLib);
    p->set<std::string>("Residual Name", "Displacement Residual");
    ev = rcp(new LCM::MechanicsResidual<EvalT, PHAL::AlbanyTraits>(*p, dl));
    fm0.template registerEvaluator<EvalT>(ev);
  }

  if (fm_choice == Albany::BUILD_RESID_FM) {
    Teuchos::RCP<const PHX::FieldTag> ret_tag;
    PHX::Tag<typename EvalT::ScalarT > res_tag("Scatter", dl->dummy);
    fm0.requireField<EvalT>(res_tag);
    ret_tag = res_tag.clone();
    return ret_tag;
  }

  return Teuchos::null;

}

#endif
