//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_HeatProblem.hpp"

#include "PHAL_FactoryTraits.hpp"
#include "Albany_Utils.hpp"
#include "Albany_BCUtils.hpp"
#include "Albany_StringUtils.hpp"

#include "Intrepid2_DefaultCubatureFactory.hpp"
#include "Shards_CellTopology.hpp"

namespace Albany {

HeatProblem::
HeatProblem( const Teuchos::RCP<Teuchos::ParameterList>& params_,
             const Teuchos::RCP<ParamLib>& paramLib_,
             const int numDim_,
             const Teuchos::RCP<const Teuchos::Comm<int> >& comm_)
 : AbstractProblem(params_, paramLib_)
 , numDim(numDim_)
 , params(params_)
 , comm(comm_)
{
  this->setNumEquations(1);

  if (numDim==1) periodic = params->get("Periodic BC", false);
  else           periodic = false;

  if (periodic) *out <<" Periodic Boundary Conditions being used." <<std::endl;

  haveAbsorption =  params->isSublist("Absorption");

  if (params->isType<std::string>("MaterialDB Filename")){

    std::string mtrlDbFilename = params->get<std::string>("MaterialDB Filename");
    // Create Material Database
    materialDB = Teuchos::rcp(new MaterialDatabase(mtrlDbFilename, comm));
  }

  if(params->isSublist("Parameters")) {
    int total_num_param_vecs, num_param_vecs, numDistParams;
    getParameterSizes(params->sublist("Parameters"), total_num_param_vecs, num_param_vecs, numDistParams);
    for (int i=0; i<numDistParams; ++i) {
      Teuchos::ParameterList p = params->sublist("Parameters").sublist(util::strint("Parameter", i+num_param_vecs));
      if(p.get<std::string>("Name") == "thermal_conductivity" && p.get<std::string>("Type") == "Distributed")
        conductivityIsDistParam = true;
      if(p.get<std::string>("Name") == "dirichlet_field" && p.get<std::string>("Type") == "Distributed") {
        dirichletIsDistParam = true;
        meshPartDirichlet = p.get<std::string>("Mesh Part", "");
      }
    }
  }
  // Set parameters for passing coords/near null space to preconditioners
  const bool computeConstantModes = false;
  rigidBodyModes->setParameters(neq, computeConstantModes);
}

void
HeatProblem::
buildProblem(Teuchos::RCP<MeshSpecs>  meshSpecs,
    StateManager& stateMgr)
{
  /* Construct All Phalanx Evaluators */
  fm  = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
  buildEvaluators(*fm, *meshSpecs, stateMgr, BUILD_RESID_FM, Teuchos::null);

  if (meshSpecs->nsNames.size() > 0) { // Build a nodeset evaluator if nodesets are present
    constructDirichletEvaluators(meshSpecs->nsNames);
  }
  
  // Check if have Neumann sublist; throw error if attempting to specify
  // Neumann BCs, but there are no sidesets in the input mesh 
  bool isNeumannPL = params->isSublist("Neumann BCs");
  if (isNeumannPL && !(meshSpecs->ssNames.size() > 0)) {
    ALBANY_ASSERT(false, "You are attempting to set Neumann BCs on a mesh with no sidesets!");
  }

  if (meshSpecs->ssNames.size() > 0) { // Build a sideset evaluator if sidesets are present
    constructNeumannEvaluators(meshSpecs);
  }
}

Teuchos::Array<Teuchos::RCP<const PHX::FieldTag> >
HeatProblem::
buildEvaluators (PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
                 const MeshSpecs& meshSpecs,
                 StateManager& stateMgr,
                 FieldManagerChoice fmchoice,
                 const Teuchos::RCP<Teuchos::ParameterList>& responseList)
{
  // Call constructEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  ConstructEvaluatorsOp<HeatProblem> op(
    *this, fm0, meshSpecs, stateMgr, fmchoice, responseList);
  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes> fe(op);
  return *op.tags;
}

// Dirichlet BCs
void
HeatProblem::
constructDirichletEvaluators(const std::vector<std::string>& nodeSetIDs)
{
   // Construct BC evaluators for all node sets and names
   std::vector<std::string> bcNames(neq);
   bcNames[0] = "T";
   BCUtils<DirichletTraits> bcUtils;
   dfm = bcUtils.constructBCEvaluators(nodeSetIDs, bcNames,
                                       this->params, this->paramLib);
   use_sdbcs_ = bcUtils.useSDBCs(); 
   offsets_ = bcUtils.getOffsets(); 
   nodeSetIDs_ = bcUtils.getNodeSetIDs();
}

// Neumann BCs
void
HeatProblem::
constructNeumannEvaluators(const Teuchos::RCP<MeshSpecs>& meshSpecs)
{
  // Note: we only enter this function if sidesets are defined in the mesh file
  // i.e. meshSpecs.ssNames.size() > 0
  BCUtils<NeumannTraits> bcUtils;

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
  // Should only specify flux vector components (dudx, dudy, dudz), or dudn, not both
  std::vector<std::string> condNames(5);
  //dudx, dudy, dudz, dudn, scaled jump (internal surface), or robin (like DBC plus scaled jump)

  // Note that sidesets are only supported for two and 3D currently
  if(numDim == 2)
   condNames[0] = "(dudx, dudy)";
  else if(numDim == 3)
   condNames[0] = "(dudx, dudy, dudz)";
  else
   TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
      std::endl << "Error: Sidesets only supported in 2 and 3D." << std::endl);

  condNames[1] = "dudn";
  condNames[2] = "scaled jump";
  condNames[3] = "robin";
  condNames[4] = "radiate";

  nfm = bcUtils.constructBCEvaluators(meshSpecs, bcNames, dof_names, false, 0,
                                      condNames, offsets, dl, this->params, this->paramLib, materialDB);
}

Teuchos::RCP<const Teuchos::ParameterList>
HeatProblem::getValidProblemParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidHeatProblemParams");

  if (numDim==1)
    validPL->set<bool>("Periodic BC", false, "Flag to indicate periodic BC for 1D problems");
  validPL->sublist("ThermalConductivity", false, "");
  validPL->set("Convection Velocity", "{0,0,0}", "");
  validPL->set<bool>("Have Rho Cp", false, "Flag to indicate if rhoCp is used");
  validPL->set<std::string>("MaterialDB Filename","materials.xml","Filename of material database xml file");

  return validPL;
}

} // namespace Albany
