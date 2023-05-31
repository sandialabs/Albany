//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_ThermalProblem.hpp"

#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"
#include "PHAL_FactoryTraits.hpp"
#include "Albany_Utils.hpp"
#include "Albany_BCUtils.hpp"
#include "PHAL_SharedParameter.hpp"

template<typename ParamNameEnum>
struct ConstructSharedParameterOp
{
private:
  Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>> fm_;
  Teuchos::RCP<Teuchos::ParameterList> p_;
  Teuchos::RCP<Albany::Layouts> dl_;
  Albany::ThermalProblem& problem_;

public:
  ConstructSharedParameterOp (Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits>> fm, Teuchos::RCP<Teuchos::ParameterList> p, Teuchos::RCP<Albany::Layouts> dl, Albany::ThermalProblem& problem) :
      fm_(fm), p_(p), dl_(dl), problem_(problem) {}
  template<typename EvalT>
  void operator() (EvalT /*x*/) const {
    //access the accessors
    p_->set<Teuchos::RCP<Albany::ScalarParameterAccessors<EvalT>>>("Accessors", problem_.getAccessors()->template at<EvalT>());
    auto ev = Teuchos::rcp(new PHAL::SharedParameter<EvalT,PHAL::AlbanyTraits>(*p_,dl_));
    fm_->template registerEvaluator<EvalT>(ev);
  }
};

Albany::ThermalProblem::
ThermalProblem( const Teuchos::RCP<Teuchos::ParameterList>& params_,
             const Teuchos::RCP<ParamLib>& paramLib_,
             //const Teuchos::RCP<DistributedParameterLibrary>& distParamLib_,
             const int numDim_,
             const Teuchos::RCP<const Teuchos_Comm >& commT_) :
  Albany::AbstractProblem(params_, paramLib_/*, distParamLib_*/),
  numDim(numDim_),
  params(params_), 
  commT(commT_),
  use_sdbcs_(false)
{
  this->setNumEquations(1);
  //We just have 1 PDE per node
  neq = 1; 
  Teuchos::Array<double> defaultData;
  defaultData.resize(numDim, 1.0);
  kappa =
      params->get<Teuchos::Array<double>>("Thermal Conductivity", defaultData);
  if (kappa.size() != numDim) {
    ALBANY_ABORT("Thermal Conductivity array must have length = numDim!");
  }
  rho = params->get<double>("Density", 1.0);
  C = params->get<double>("Heat Capacity", 1.0);
  thermal_source = params->get<std::string>("Thermal Source", "None"); 

  conductivityIsDistParam = false;
  if(params->isSublist("Parameters")) {
    int total_num_param_vecs, num_param_vecs, numDistParams;
    Albany::getParameterSizes(params->sublist("Parameters"), total_num_param_vecs, num_param_vecs, numDistParams);
    for (int i=0; i<numDistParams; ++i) {
      Teuchos::ParameterList p = params->sublist("Parameters").sublist(util::strint("Parameter", 
                       i+num_param_vecs));
      if(p.get<std::string>("Name") == "thermal_conductivity" && p.get<std::string>("Type") == "Distributed")
        conductivityIsDistParam = true;
    }
  }
  // Set Parameters for passing coords/near null space to preconditioners
  const bool computeConstantModes = false;
  rigidBodyModes->setParameters(neq, computeConstantModes);
}

Albany::ThermalProblem::
~ThermalProblem()
{
}

void
Albany::ThermalProblem::
buildProblem(
  Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecs> >  meshSpecs,
  Albany::StateManager& stateMgr)
{
  /* Construct All Phalanx Evaluators */
  int physSets = meshSpecs.size();
  std::cout << "Thermal Problem Num MeshSpecs: " << physSets << std::endl;
  fm.resize(physSets);

  for (int ps=0; ps<physSets; ps++) {
    fm[ps]  = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
    buildEvaluators(*fm[ps], *meshSpecs[ps], stateMgr, BUILD_RESID_FM,
                    Teuchos::null);
  }

  if (meshSpecs[0]->nsNames.size() > 0) { // Build a nodeset evaluator if nodesets are present
    constructDirichletEvaluators(meshSpecs[0]->nsNames);
  }
  
  // Check if have Neumann sublist; throw error if attempting to specify
  // Neumann BCs, but there are no sidesets in the input mesh 
  bool isNeumannPL = params->isSublist("Neumann BCs");
  if (isNeumannPL && !(meshSpecs[0]->ssNames.size() > 0)) {
    ALBANY_ASSERT(false, "You are attempting to set Neumann BCs on a mesh with no sidesets!");
  }

  if (meshSpecs[0]->ssNames.size() > 0) { // Build a sideset evaluator if sidesets are present
    constructNeumannEvaluators(meshSpecs[0]);
  }

}

Teuchos::Array<Teuchos::RCP<const PHX::FieldTag> >
Albany::ThermalProblem::
buildEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecs& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fmchoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  ConstructEvaluatorsOp<ThermalProblem> op(
    *this, fm0, meshSpecs, stateMgr, fmchoice, responseList);
  Sacado::mpl::for_each_no_kokkos<PHAL::AlbanyTraits::BEvalTypes> fe(op);
  return *op.tags;
}

// Dirichlet BCs
void
Albany::ThermalProblem::constructDirichletEvaluators(const std::vector<std::string>& nodeSetIDs)
{
   // Construct BC evaluators for all node sets and names
   std::vector<std::string> bcNames(neq);
   bcNames[0] = "T";
   Albany::BCUtils<Albany::DirichletTraits> bcUtils;

   dfm = bcUtils.constructBCEvaluators(nodeSetIDs, bcNames,
                                          this->params, this->paramLib);

   {
    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(new Teuchos::ParameterList("Theta 0"));
    p->set< Teuchos::RCP<ParamLib> >("Parameter Library", this->paramLib);
    const std::string param_name = "Theta 0";
    p->set<std::string>("Parameter Name", param_name);
    p->set<const Teuchos::ParameterList*>("Parameters List", &params->sublist("Parameters"));
    p->set<double>("Default Nominal Value", 0.);
    ConstructSharedParameterOp<Albany::ParamEnum> constructor(dfm, p, dl, *this);
    Sacado::mpl::for_each_no_kokkos<PHAL::AlbanyTraits::BEvalTypes> fe(constructor);
   }
   {
    Teuchos::RCP<Teuchos::ParameterList> p = Teuchos::rcp(new Teuchos::ParameterList("Theta 1"));
    p->set< Teuchos::RCP<ParamLib> >("Parameter Library", this->paramLib);
    const std::string param_name = "Theta 1";
    p->set<std::string>("Parameter Name", param_name);
    p->set<const Teuchos::ParameterList*>("Parameters List", &params->sublist("Parameters"));
    p->set<double>("Default Nominal Value", 0.);
    ConstructSharedParameterOp<Albany::ParamEnum> constructor(dfm, p, dl, *this);
    Sacado::mpl::for_each_no_kokkos<PHAL::AlbanyTraits::BEvalTypes> fe(constructor);
   }

   use_sdbcs_ = bcUtils.useSDBCs(); 
   offsets_ = bcUtils.getOffsets(); 
   nodeSetIDs_ = bcUtils.getNodeSetIDs();
}

// Neumann BCs
void
Albany::ThermalProblem::constructNeumannEvaluators(const Teuchos::RCP<Albany::MeshSpecs>& meshSpecs)
{
   // Note: we only enter this function if sidesets are defined in the mesh file
   // i.e. meshSpecs.ssNames.size() > 0

   Albany::BCUtils<Albany::NeumannTraits> bcUtils;

   // Check to make sure that Neumann BCs are given in the input file

   if(!bcUtils.haveBCSpecified(this->params))

      return;

   // Construct BC evaluators for all side sets and names
   // Note that the string index sets up the equation offset, so ordering is important
   std::vector<std::string> bcNames(neq);
   Teuchos::ArrayRCP<std::string> dof_names(neq);
   Teuchos::Array<Teuchos::Array<int> > offsets;
   offsets.resize(neq);

   bcNames[0] = "T";
   dof_names[0] = "Temperature";
   offsets[0].resize(1);
   offsets[0][0] = 0;


   // Construct BC evaluators for all possible names of conditions
   // Should only specify flux vector components (dTdx, dTdy, dTdz), or dTdn, not both
   std::vector<std::string> condNames(5);
     //dTdx, dTdy, dTdz, dTdn, scaled jump (internal surface), or robin (like DBC plus scaled jump)

   // Note that sidesets are only supported for two and 3D currently
   if(numDim == 2)
    condNames[0] = "(dTdx, dTdy)";
   else if(numDim == 3)
    condNames[0] = "(dTdx, dTdy, dTdz)";
   else
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
       std::endl << "Error: Sidesets only supported in 2 and 3D." << std::endl);

   condNames[1] = "dTdn";
   condNames[2] = "scaled jump";
   condNames[3] = "robin";
   condNames[4] = "radiate";

   nfm.resize(1); // Thermal problem only has one physics set
   nfm[0] = bcUtils.constructBCEvaluators(meshSpecs, bcNames, dof_names, false, 0,
                                  condNames, offsets, dl, this->params, this->paramLib);

}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::ThermalProblem::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidThermalProblemParams");
  
  Teuchos::Array<double> defaultData;
  defaultData.resize(numDim, 1.0);
  validPL->set<Teuchos::Array<double>>(
      "Thermal Conductivity",
      defaultData,
      "Arrays of values of thermal conductivities in x, y, z [required]");
  validPL->set<double>(
      "Heat Capacity", 1.0, "Value of heat capacity [required]");
  validPL->set<double>(
      "Density", 1.0, "Value of density [required]");
  validPL->set<std::string>(
      "Thermal Source", "None", "Value of thermal source [required]");

  return validPL;
}
