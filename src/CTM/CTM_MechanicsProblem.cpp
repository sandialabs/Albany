#include "CTM_MechanicsProblem.hpp"
#include "Albany_Utils.hpp"
#include "Albany_BCUtils.hpp"

namespace CTM {

    MechanicsProblem::MechanicsProblem(
            const RCP<ParameterList>& params,
            RCP<ParamLib> const& param_lib,
            const int n_dims,
            RCP<const Teuchos::Comm<int> >& comm) :
    Albany::AbstractProblem(params, param_lib),
    num_dims(n_dims),
    comm_(comm),
    have_source_(false),
    isTransient_(false) {

        *out << "Problem name = Mechanics Problem \n";
        this->setNumEquations(num_dims);
        material_db_ = LCM::createMaterialDatabase(params, comm);

        /*This is needed because right now the function constructBCEvaluators
         require a QCAD::MaterialDataBase instead of LCM::MaterialDataBase if
         a robin boundary conditions is used. This need to be corrected.*/
        materialFileName_ = params->get<std::string>("MaterialDB Filename");

        // Are any source functions specified?
        have_source_ = params->isSublist("Source Functions");

        // Is it a transient analysis?
        if (params->isParameter("Transient")) {
            isTransient_ = params->get<bool>("Transient", true);
            TEUCHOS_TEST_FOR_EXCEPTION(isTransient_, std::out_of_range,
                    "We apologize but transient analysis is not implemented yet for CTM_MechanicsProblem");
        }
        if (isTransient_) {
            *out << "Solving a transient analysis" << std::endl;
        }
    }

    MechanicsProblem::~MechanicsProblem() {
    }

    void MechanicsProblem::buildProblem(
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


        if (meshSpecs[0]->nsNames.size() > 0) {
            // Construct Dirichlet evaluators
            *out << "Calling MechanicsProblem::constructDirichletEvaluators" << '\n';
            constructDirichletEvaluators(meshSpecs[0]);
        }

        if (meshSpecs[0]->ssNames.size() > 0) {
            // Construct Neumann evaluators
            *out << "Calling MechanicsProblem::constructNeumannEvaluators" << '\n';
            constructNeumannEvaluators(meshSpecs[0]);
        }
    }

    Teuchos::Array<RCP<const PHX::FieldTag> > MechanicsProblem::buildEvaluators(
            PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
            const Albany::MeshSpecsStruct& meshSpecs,
            Albany::StateManager& stateMgr,
            Albany::FieldManagerChoice fmchoice,
            const RCP<ParameterList>& responseList) {
        // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
        // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
        Albany::ConstructEvaluatorsOp<MechanicsProblem> op(*this,
                fm0,
                meshSpecs,
                stateMgr,
                fmchoice,
                responseList);
        Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);
        return *op.tags;
    }

    void MechanicsProblem::constructDirichletEvaluators(
            const RCP<Albany::MeshSpecsStruct>& mesh_specs) {
        // Construct Dirichlet evaluators for all nodesets and names
        std::vector<std::string> dirichletNames(neq);
        int index = 0;
        dirichletNames[index++] = "X";
        if (num_dims > 1) dirichletNames[index++] = "Y";
        if (num_dims > 2) dirichletNames[index++] = "Z";

        Albany::BCUtils<Albany::DirichletTraits> bcUtils;
        std::vector<std::string>& nodeSetIDs = mesh_specs->nsNames;
        dfm = bcUtils.constructBCEvaluators(nodeSetIDs, dirichletNames,
                this->params, Teuchos::null);
    }

    void MechanicsProblem::constructNeumannEvaluators(
            const RCP<Albany::MeshSpecsStruct>& mesh_specs) {

        // Note: we only enter this function if sidesets are defined in the mesh file
        // i.e. meshSpecs.ssNames.size() > 0

        Albany::BCUtils<Albany::NeumannTraits> neuUtils;

        // Check to make sure that Neumann BCs are given in the input file
        if (!neuUtils.haveBCSpecified(this->params)) {
            return;
        }

        // Construct BC evaluators for all side sets and names
        // Note that the string index sets up the equation offset,
        // so ordering is important

        std::vector<std::string> neumannNames(neq + 1);
        Teuchos::Array<Teuchos::Array<int>> offsets;
        offsets.resize(neq + 1);

        int index = 0;


        neumannNames[index] = "sig_x";
        offsets[index].resize(1);
        offsets[index++][0] = 0;

        // The Neumann BC code uses offsets[neq].size() as num dim, so use num_dims_
        // here rather than neq.
        offsets[neq].resize(num_dims);
        offsets[neq][0] = 0;

        if (num_dims > 1) {
            neumannNames[index] = "sig_y";
            offsets[index].resize(1);
            offsets[index++][0] = 1;
            offsets[neq][1] = 1;
        }

        if (num_dims > 2) {
            neumannNames[index] = "sig_z";
            offsets[index].resize(1);
            offsets[index++][0] = 2;
            offsets[neq][2] = 2;
        }


        neumannNames[neq] = "all";

        // Construct BC evaluators for all possible names of conditions
        // Should only specify flux vector components (dudx, dudy, dudz),
        // or dudn, not both
        std::vector<std::string> condNames(3); //dudx, dudy, dudz, dudn, P
        Teuchos::ArrayRCP<std::string> dof_names(1);
        dof_names[0] = "Displacement";

        // Note that sidesets are only supported for two and 3D currently
        if (num_dims == 2)
            condNames[0] = "(t_x, t_y)";
        else if (num_dims == 3)
            condNames[0] = "(t_x, t_y, t_z)";
        else
            TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                '\n' << "Error: Sidesets only supported in 2 and 3D." << '\n');

        condNames[1] = "dudn";
        condNames[2] = "P";

        // The resize below is not appropriate for MechanicsProblem
        // not sure what the right thing to do is
        nfm.resize(1); // Elasticity problem only has one element block

        nfm[0] = neuUtils.constructBCEvaluators(
                mesh_specs,
                neumannNames,
                dof_names,
                true, // isVectorField
                0, // offsetToFirstDOF
                condNames,
                offsets,
                dl,
                this->params,
                this->paramLib);
    }

    //    void MechanicsProblem::getAllocatedStates(
    //            ArrayRCP<ArrayRCP<RCP<FC> > > old_state,
    //            ArrayRCP<ArrayRCP<RCP<FC> > > new_state) const {
    //    }

    //------------------------------------------------------------------------------

    Teuchos::RCP<const Teuchos::ParameterList>
    MechanicsProblem::getValidProblemParameters() const {
        Teuchos::RCP<Teuchos::ParameterList> validPL =
                this->getGenericProblemParams("ValidMechanicsProblemParams");

        validPL->set<std::string>("MaterialDB Filename",
                "materials.xml",
                "Filename of material database xml file");

        validPL->set<bool>("Transient",
                true,
                "Specify if you want a transient analysis or not");
        
        validPL->sublist("Temperature", false, "");

        return validPL;
    }

}
