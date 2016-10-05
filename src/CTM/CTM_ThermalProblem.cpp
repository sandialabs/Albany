#include "CTM_ThermalProblem.hpp"

namespace CTM {
ThermalProblem::ThermalProblem(
            const RCP<ParameterList>& params,
            RCP<ParamLib> const& param_lib,
            const int n_dims,
            RCP<const Teuchos::Comm<int> >& comm) :
    Albany::AbstractProblem(params, param_lib),
    num_dims(n_dims),
    have_source_(false),
    thermal_source_(SOURCE_TYPE_NONE),
    thermal_source_evaluated_(false),
    isTransient_(false) {

        *out << "Problem name = Thermal Problem \n";
        this->setNumEquations(1);
        material_db_ = LCM::createMaterialDatabase(params, comm);

        // Are any source functions specified?
        have_source_ = params->isSublist("Source Functions");

        // Determine the Thermal source 
        //   - the "Source Functions" list must be present in the input file,
        //   - we must have temperature and have included a temperature equation

        if (have_source_) {
            // If a thermal source is specified
            if (params->sublist("Source Functions").isSublist("Thermal Source")) {

                Teuchos::ParameterList& thSrcPL = params->sublist("Source Functions")
                        .sublist("Thermal Source");

                if (thSrcPL.get<std::string>("Thermal Source Type", "None")
                        == "Block Dependent") {

                    if (Teuchos::nonnull(material_db_)) {
                        thermal_source_ = SOURCE_TYPE_MATERIAL;
                    }
                } else {
                    thermal_source_ = SOURCE_TYPE_INPUT;

                }
            }
        }
        
        // Is it a transient analysis?
        if (params->isParameter("Transient")){
            isTransient_ = params->get<bool>("Transient",true);
            *out << "Solving a transient analysis = " << isTransient_ << std::endl;
        }
    }

    ThermalProblem::~ThermalProblem() {
    }
void ThermalProblem::buildProblem(
            ArrayRCP<RCP<Albany::MeshSpecsStruct> > meshSpecs,
            Albany::StateManager& stateMgr) {
        // Construct All Phalanx Evaluators
        int physSets = meshSpecs.size();
        *out << "Num MeshSpecs: " << physSets << '\n';
        fm.resize(physSets);
        bool haveSidesets = false;

        *out << "Calling MechanicsProblem::buildEvaluators" << '\n';
        for (int ps = 0; ps < physSets; ++ps) {
            fm[ps] = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
            buildEvaluators(*fm[ps], *meshSpecs[ps], stateMgr, Albany::BUILD_RESID_FM,
                    Teuchos::null);
            if (meshSpecs[ps]->ssNames.size() > 0) haveSidesets = true;
        }
    }

    Teuchos::Array<RCP<const PHX::FieldTag> > ThermalProblem::buildEvaluators(
            PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
            const Albany::MeshSpecsStruct& meshSpecs,
            Albany::StateManager& stateMgr,
            Albany::FieldManagerChoice fmchoice,
            const RCP<ParameterList>& responseList) {
        // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
        // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
        Albany::ConstructEvaluatorsOp<ThermalProblem> op(*this,
                fm0,
                meshSpecs,
                stateMgr,
                fmchoice,
                responseList);
        Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);
        return *op.tags;
    }

    void ThermalProblem::constructDirichletEvaluators(
            const Albany::MeshSpecsStruct& mesh_specs) {
    }

    void ThermalProblem::constructNeumannEvaluators(
            const RCP<Albany::MeshSpecsStruct>& mesh_specs) {
    }

    void ThermalProblem::getAllocatedStates(
            ArrayRCP<ArrayRCP<RCP<FC> > > old_state,
            ArrayRCP<ArrayRCP<RCP<FC> > > new_state) const {
    }

}
