//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "HMCProblem.hpp"
#include "Albany_Utils.hpp"
#include "Albany_BCUtils.hpp"
#include "Albany_ProblemUtils.hpp"
#ifdef ALBANY_ATO
#include "ATO_TopoTools.hpp"
#endif

Albany::HMCProblem::
HMCProblem(const Teuchos::RCP<Teuchos::ParameterList>& params_,
		  const Teuchos::RCP<ParamLib>& paramLib_,
		  const int numDim_,
                  Teuchos::RCP<const Teuchos::Comm<int>>& commT) :
#ifdef ALBANY_ATO
  ATO::OptimizationProblem(params_, paramLib_, numDim_+params_->get("Additional Scales",1)*numDim_*numDim_),
#endif
  Albany::AbstractProblem(params_, paramLib_, numDim_+params_->get("Additional Scales",1)*numDim_*numDim_),
  haveSource(false),
  use_sdbcs_(false),
  numDim(numDim_),
  numMicroScales(params_->get("Additional Scales",1))
{

  std::string& method = params->get("Name", "HMC ");
  *out << "Problem Name = " << method << std::endl;

  bool validMaterialDB(false);
  if(params->isType<std::string>("MaterialDB Filename")){
    validMaterialDB = true;
    std::string filename = params->get<std::string>("MaterialDB Filename");
    material_db_ = Teuchos::rcp(new Albany::MaterialDatabase(filename, commT));
  }
  TEUCHOS_TEST_FOR_EXCEPTION(!validMaterialDB,
                             std::logic_error,
                             "Mechanics Problem Requires a Material Database");


// the following function returns the problem information required for setting the rigid body modes (RBMs) for elasticity problems
//written by IK, Feb. 2012

  int numScalar = 0;
  int nullSpaceDim = 0;
  if (numDim == 1) {nullSpaceDim = 0; }
  else {
    if (numDim == 2) {nullSpaceDim = 3; }
    if (numDim == 3) {nullSpaceDim = 6; }
  }

  int numPDEs = numMicroScales*numDim*numDim;

  rigidBodyModes->setParameters(numPDEs, numDim, numScalar, nullSpaceDim);

}

Albany::HMCProblem::
~HMCProblem()
{
}

void
Albany::HMCProblem::
buildProblem(
  Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct>>  meshSpecs,
  Albany::StateManager& stateMgr)
{
  /* Construct All Phalanx Evaluators */
  TEUCHOS_TEST_FOR_EXCEPTION(meshSpecs.size()!=1,std::logic_error,"Problem supports one Material Block");

  fm.resize(1);

  fm[0]  = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
  buildEvaluators(*fm[0], *meshSpecs[0], stateMgr, BUILD_RESID_FM,
		  Teuchos::null);

  if(meshSpecs[0]->nsNames.size() > 0) // Build a nodeset evaluator if nodesets are present

    constructDirichletEvaluators(*meshSpecs[0]);

  if(meshSpecs[0]->ssNames.size() > 0) // Build a sideset evaluator if sidesets are present

    constructNeumannEvaluators(meshSpecs[0]);

#ifdef ALBANY_ATO
  if( params->isType<Teuchos::RCP<ATO::Topology>>("Topology") )
   setupTopOpt(meshSpecs,stateMgr);
#endif
}

Teuchos::Array<Teuchos::RCP<const PHX::FieldTag>>
Albany::HMCProblem::
buildEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fmchoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  ConstructEvaluatorsOp<HMCProblem> op(
    *this, fm0, meshSpecs, stateMgr, fmchoice, responseList);
  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);
  return *op.tags;
}

// Dirichlet BCs
void
Albany::HMCProblem::constructDirichletEvaluators(
        const Albany::MeshSpecsStruct& meshSpecs)
{
  // Construct Dirichlet evaluators for all nodesets and names
  std::vector<std::string> dirichletNames(neq);
  dirichletNames[0] = "X";
  if (neq>1) dirichletNames[1] = "Y";
  if (neq>2) dirichletNames[2] = "Z";
  Albany::BCUtils<Albany::DirichletTraits> dirUtils;
  dfm = dirUtils.constructBCEvaluators(meshSpecs.nsNames, dirichletNames,
                                       this->params, this->paramLib);
  use_sdbcs_ = dirUtils.useSDBCs();
  offsets_ = dirUtils.getOffsets();
  nodeSetIDs_ = dirUtils.getNodeSetIDs();
}

// Neumann BCs
void
Albany::HMCProblem::constructNeumannEvaluators(
        const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs)
{
   // Note: we only enter this function if sidesets are defined in the mesh file
   // i.e. meshSpecs.ssNames.size() > 0

   Albany::BCUtils<Albany::NeumannTraits> neuUtils;

   // Check to make sure that Neumann BCs are given in the input file

   if(!neuUtils.haveBCSpecified(this->params))

      return;

   // Construct BC evaluators for all side sets and names
   // Note that the string index sets up the equation offset, so ordering is important
   std::vector<std::string> neumannNames(neq + 1);
   Teuchos::Array<Teuchos::Array<int>> offsets;
   offsets.resize(neq + 1);

   neumannNames[0] = "sig_x";
   offsets[0].resize(1);
   offsets[0][0] = 0;
   offsets[neq].resize(neq);
   offsets[neq][0] = 0;

   if (neq>1){
      neumannNames[1] = "sig_y";
      offsets[1].resize(1);
      offsets[1][0] = 1;
      offsets[neq][1] = 1;
   }

   if (neq>2){
     neumannNames[2] = "sig_z";
      offsets[2].resize(1);
      offsets[2][0] = 2;
      offsets[neq][2] = 2;
   }

   neumannNames[neq] = "all";

   // Construct BC evaluators for all possible names of conditions
   // Should only specify flux vector components (dudx, dudy, dudz), or dudn, not both
   std::vector<std::string> condNames(3); //dudx, dudy, dudz, dudn, P
   Teuchos::ArrayRCP<std::string> dof_names(1);
     dof_names[0] = "Displacement";

   // Note that sidesets are only supported for two and 3D currently
   if(numDim == 2)
    condNames[0] = "(t_x, t_y)";
   else if(numDim == 3)
    condNames[0] = "(t_x, t_y, t_z)";
   else
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
       std::endl << "Error: Sidesets only supported in 2 and 3D." << std::endl);

   condNames[1] = "dudn";
   condNames[2] = "P";

   nfm.resize(1); // HMC problem only has one element block

   nfm[0] = neuUtils.constructBCEvaluators(meshSpecs, neumannNames, dof_names, true, 0,
                                          condNames, offsets, dl,
                                          this->params, this->paramLib);

}


Teuchos::RCP<const Teuchos::ParameterList>
Albany::HMCProblem::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidHMCProblemParams");

  validPL->set<int>("Additional Scales", false, "1");
  validPL->set<std::string>("MaterialDB Filename","materials.xml",
                            "Filename of material database xml file");
  validPL->sublist("Hierarchical Elasticity Model", false, "");

#ifdef ALBANY_ATO
  Teuchos::RCP<ATO::Topology> emptyTopo;
  emptyTopo = Teuchos::null;
  validPL->set<Teuchos::RCP<ATO::Topology>>("Topology", emptyTopo);
#endif
  validPL->sublist("Topology Parameters", false, "");
  validPL->sublist("Objective Aggregator", false, "");
  validPL->sublist("Apply Topology Weight Functions", false, "");

  return validPL;
}

void
Albany::HMCProblem::
parseMaterialModel(Teuchos::RCP<Teuchos::ParameterList>& p,
                   const Teuchos::RCP<Teuchos::ParameterList>& params) const
{
  Teuchos::ParameterList& modelList = params->sublist("Hierarchical Elasticity Model");
  p->set("C11", modelList.get("C11",0.0));
  p->set("C33", modelList.get("C33",0.0));
  p->set("C12", modelList.get("C12",0.0));
  p->set("C23", modelList.get("C23",0.0));
  p->set("C44", modelList.get("C44",0.0));
  p->set("C66", modelList.get("C66",0.0));

  for(int i=0; i<numMicroScales; i++){
    std::string scaleName = Albany::strint("Microscale",i+1);
    const Teuchos::ParameterList& scaleList = modelList.sublist(scaleName);
    p->sublist(scaleName);
    p->set(scaleName,scaleList);
  }

}

void
Albany::HMCProblem::getAllocatedStates(
   Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Kokkos::DynRankView<RealType, PHX::Device>>>> oldState_,
   Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Kokkos::DynRankView<RealType, PHX::Device>>>> newState_
   ) const
{
  oldState_ = oldState;
  newState_ = newState;
}
