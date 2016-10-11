#include "CTM_ThermalProblem.hpp"
#include "Albany_Utils.hpp"
#include "Albany_BCUtils.hpp"

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
        if (params->isParameter("Transient")) {
            isTransient_ = params->get<bool>("Transient", true);
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

        *out << "Calling ThermalProblem::buildEvaluators" << '\n';
        for (int ps = 0; ps < physSets; ++ps) {
            fm[ps] = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
            buildEvaluators(*fm[ps], *meshSpecs[ps], stateMgr, Albany::BUILD_RESID_FM,
                    Teuchos::null);
            if (meshSpecs[ps]->ssNames.size() > 0) haveSidesets = true;
        }


        // Construct Dirichlet evaluators
        *out << "Calling ThermalProblem::constructDirichletEvaluators" << '\n';
        if (meshSpecs[0]->nsNames.size() > 0)
            constructDirichletEvaluators(meshSpecs[0]);
        //constructDirichletEvaluators(meshSpecs[0]->nsNames);
        // Construct Neumann evaluators
        *out << "Calling ThermalProblem::constructNeumannEvaluators" << '\n';
        if (meshSpecs[0]->ssNames.size() > 0)
            constructNeumannEvaluators(meshSpecs[0]);

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
            const RCP<Albany::MeshSpecsStruct>& mesh_specs) {
        std::vector<std::string> bcNames(neq);
        bcNames[0] = "T";
        Albany::BCUtils<Albany::DirichletTraits> bcUtils;
        std::vector<std::string>& nodeSetIDs = mesh_specs->nsNames;
        dfm = bcUtils.constructBCEvaluators(nodeSetIDs, bcNames,
                this->params, Teuchos::null);
    }

    void ThermalProblem::constructNeumannEvaluators(
            const RCP<Albany::MeshSpecsStruct>& mesh_specs) {
        
        Albany::BCUtils<Albany::NeumannTraits> bcUtils;

        if (!bcUtils.haveBCSpecified(this->params))
            return;

        std::vector<std::string> bcNames(neq);
        Teuchos::ArrayRCP<std::string> dof_names(neq);
        Teuchos::Array<Teuchos::Array<int> > offsets;
        offsets.resize(neq);

        bcNames[0] = "U";
        dof_names[0] = "u";
        offsets[0].resize(1);
        offsets[0][0] = 0;

        // Construct BC evaluators for all possible names of conditions
        // Should only specify flux vector components (dudx, dudy, dudz), or dudn, not both
        std::vector<std::string> condNames(4);
        //dudx, dudy, dudz, dudn, scaled jump (internal surface), or robin (like DBC plus scaled jump)

        // Note that sidesets are only supported for two and 3D currently
        if (num_dims == 2)
            condNames[0] = "(dudx, dudy)";
        else if (num_dims == 3)
            condNames[0] = "(dudx, dudy, dudz)";
        else
            TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                std::endl << "Error: Sidesets only supported in 2 and 3D." << std::endl);

        condNames[1] = "dudn";

        condNames[2] = "scaled jump";

        condNames[3] = "robin";

        nfm.resize(1); // Heat problem only has one physics set   
        nfm[0] = bcUtils.constructBCEvaluators(mesh_specs, bcNames, dof_names, false, 0,
                condNames, offsets, dl, this->params, Teuchos::null);
    }

    void ThermalProblem::getAllocatedStates(
            ArrayRCP<ArrayRCP<RCP<FC> > > old_state,
            ArrayRCP<ArrayRCP<RCP<FC> > > new_state) const {
    }

    //------------------------------------------------------------------------------

    Teuchos::RCP<const Teuchos::ParameterList>
    ThermalProblem::getValidProblemParameters() const {
        Teuchos::RCP<Teuchos::ParameterList> validPL =
                this->getGenericProblemParams("ValidThermalProblemParams");

        validPL->set<std::string>("MaterialDB Filename",
                "materials.xml",
                "Filename of material database xml file");

        return validPL;
    }

}
