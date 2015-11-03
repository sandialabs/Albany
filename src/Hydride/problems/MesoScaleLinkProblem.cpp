//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

/*
 * Note 2: This code is not compiled unless we have MPI available (see ifdef in CMakeLists.txt)
 */

#include "MesoScaleLinkProblem.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ProblemUtils.hpp"

#if 0
#include "Teuchos_TypeNameTraits.hpp"

namespace Teuchos {
// Provide an explicit template specialization for the opaque type MPI_Comm
// so that the instantiation of Teuchos::RCP<MPI_Comm> objects compiles correctly in debug mode
// without relying on the implementation details of the MPI library.
  TEUCHOS_TYPE_NAME_TRAITS_BUILTIN_TYPE_SPECIALIZATION(MPI_Comm);
} // namespace Teuchos
#endif

Albany::MesoScaleLinkProblem::
MesoScaleLinkProblem(const Teuchos::RCP<Teuchos::ParameterList>& params_,
                     const Teuchos::RCP<ParamLib>& paramLib_,
                     const int numDim_,
                     Teuchos::RCP<const Teuchos::Comm<int> >& commT_):
  Albany::AbstractProblem(params_, paramLib_, numDim_),
  haveSource(false),
  numDim(numDim_),
  commT(commT_),
  mpi_comm(Albany::getMpiCommFromTeuchosComm(commT_)) {

  TEUCHOS_TEST_FOR_EXCEPTION(commT->getSize() != 1, std::logic_error,
                             "MesoScale bridge only supports 1 master processor currently:\n\tRun with \"mpirun -np 1 Albany\"");

  std::string& method = params->get("Name", "MesoScaleLink ");
  *out << "Problem Name = " << method << std::endl;

  haveSource =  params->isSublist("Source Functions");

  matModel = params->sublist("Material Model").get("Model Name", "LinearMesoScaleLink");

  TEUCHOS_TEST_FOR_EXCEPTION(matModel != "Bridge", std::logic_error,
                             "Must specify \"Bridge\" for the material model in the input file.");

  exeName = params->sublist("Material Model").get("Executable", "zzz");

  *out << "Establishing MPI bridging link to: " << exeName << std::endl;


  numMesoPEs = params->sublist("Material Model").get("Num Meso PEs", 1);

  interCommunicator = Teuchos::rcp(new MPI_Comm());

  // Fire off the remote processes

  MPI_Comm_spawn(&exeName[0], MPI_ARGV_NULL, numMesoPEs,
                 MPI_INFO_NULL, 0, mpi_comm, interCommunicator.get(), MPI_ERRCODES_IGNORE);


}

Albany::MesoScaleLinkProblem::
~MesoScaleLinkProblem() {

  // Tell the remote MPALE processes to kill themselves

  for(int i = 0; i < numMesoPEs; i++)

    MPI_Send(0, 0, MPI_INT, i, LCM::DIE, *interCommunicator.get());


}

//the following function returns the problem information required for setting the rigid body modes
// (RBMs) for elasticity problems (in src/Albany_SolverFactory.cpp)
//written by IK, Feb. 2012

void Albany::MesoScaleLinkProblem::getRBMInfoForML(
  int& numPDEs, int& numMesoScaleLinkDim, int& numScalar,  int& nullSpaceDim) {
  numPDEs = numDim;
  numMesoScaleLinkDim = numDim;
  numScalar = 0;

  if(numDim == 1) {
    nullSpaceDim = 0;
  }

  else {
    if(numDim == 2) {
      nullSpaceDim = 3;
    }

    if(numDim == 3) {
      nullSpaceDim = 6;
    }
  }
}


void
Albany::MesoScaleLinkProblem::
buildProblem(
  Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  meshSpecs,
  Albany::StateManager& stateMgr) {
  /* Construct All Phalanx Evaluators */
  TEUCHOS_TEST_FOR_EXCEPTION(meshSpecs.size() != 1, std::logic_error, "Problem supports one Material Block");

  fm.resize(1);

  fm[0]  = Teuchos::rcp(new PHX::FieldManager<PHAL::AlbanyTraits>);
  buildEvaluators(*fm[0], *meshSpecs[0], stateMgr, BUILD_RESID_FM,
                  Teuchos::null);

  if(meshSpecs[0]->nsNames.size() > 0) // Build a nodeset evaluator if nodesets are present

    constructDirichletEvaluators(*meshSpecs[0]);

  if(meshSpecs[0]->ssNames.size() > 0) // Build a sideset evaluator if sidesets are present

    constructNeumannEvaluators(meshSpecs[0]);

}

Teuchos::Array<Teuchos::RCP<const PHX::FieldTag> >
Albany::MesoScaleLinkProblem::
buildEvaluators(
  PHX::FieldManager<PHAL::AlbanyTraits>& fm0,
  const Albany::MeshSpecsStruct& meshSpecs,
  Albany::StateManager& stateMgr,
  Albany::FieldManagerChoice fmchoice,
  const Teuchos::RCP<Teuchos::ParameterList>& responseList) {
  // Call constructeEvaluators<EvalT>(*rfm[0], *meshSpecs[0], stateMgr);
  // for each EvalT in PHAL::AlbanyTraits::BEvalTypes
  ConstructEvaluatorsOp<MesoScaleLinkProblem> op(
    *this, fm0, meshSpecs, stateMgr, fmchoice, responseList);
  Sacado::mpl::for_each<PHAL::AlbanyTraits::BEvalTypes>(op);
  return *op.tags;
}

// Dirichlet BCs
void
Albany::MesoScaleLinkProblem::constructDirichletEvaluators(
  const Albany::MeshSpecsStruct& meshSpecs) {
  // Construct Dirichlet evaluators for all nodesets and names
  std::vector<std::string> dirichletNames(neq);
  dirichletNames[0] = "X";

  if(neq > 1) dirichletNames[1] = "Y";

  if(neq > 2) dirichletNames[2] = "Z";

  Albany::BCUtils<Albany::DirichletTraits> dirUtils;
  dfm = dirUtils.constructBCEvaluators(meshSpecs.nsNames, dirichletNames,
                                       this->params, this->paramLib);
}

// Neumann BCs
void
Albany::MesoScaleLinkProblem::constructNeumannEvaluators(
  const Teuchos::RCP<Albany::MeshSpecsStruct>& meshSpecs) {
  // Note: we only enter this function if sidesets are defined in the mesh file
  // i.e. meshSpecs.ssNames.size() > 0

  Albany::BCUtils<Albany::NeumannTraits> neuUtils;

  // Check to make sure that Neumann BCs are given in the input file

  if(!neuUtils.haveBCSpecified(this->params))

    return;

  // Construct BC evaluators for all side sets and names
  // Note that the string index sets up the equation offset, so ordering is important
  std::vector<std::string> neumannNames(neq + 1);
  Teuchos::Array<Teuchos::Array<int> > offsets;
  offsets.resize(neq + 1);

  neumannNames[0] = "Tx";
  offsets[0].resize(1);
  offsets[0][0] = 0;
  offsets[neq].resize(neq);
  offsets[neq][0] = 0;

  if(neq > 1) {
    neumannNames[1] = "Ty";
    offsets[1].resize(1);
    offsets[1][0] = 1;
    offsets[neq][1] = 1;
  }

  if(neq > 2) {
    neumannNames[2] = "Tz";
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
    condNames[0] = "(dudx, dudy)";

  else if(numDim == 3)
    condNames[0] = "(dudx, dudy, dudz)";

  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
                               std::endl << "Error: Sidesets only supported in 2 and 3D." << std::endl);

  condNames[1] = "dudn";
  condNames[2] = "P";

  nfm.resize(1); // MesoScaleLink problem only has one element block

  nfm[0] = neuUtils.constructBCEvaluators(meshSpecs, neumannNames, dof_names, true, 0,
                                          condNames, offsets, dl,
                                          this->params, this->paramLib);

}


Teuchos::RCP<const Teuchos::ParameterList>
Albany::MesoScaleLinkProblem::getValidProblemParameters() const {
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericProblemParams("ValidMesoScaleLinkProblemParams");

  validPL->sublist("Elastic Modulus", false, "");
  validPL->sublist("Poissons Ratio", false, "");
  validPL->sublist("Material Model", false, "");

  if(matModel == "Bridge") {
    validPL->set<std::string>("Executable", "zzz", "Name of mesoscale code executable file");
    validPL->set<int>("Num Meso PEs", false, "");
  }

  return validPL;
}

void
Albany::MesoScaleLinkProblem::getAllocatedStates(
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType> > > > oldState_,
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::RCP<Intrepid::FieldContainer<RealType> > > > newState_
) const {
  oldState_ = oldState;
  newState_ = newState;
}
