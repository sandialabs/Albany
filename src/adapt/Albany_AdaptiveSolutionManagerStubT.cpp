//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_AdaptiveSolutionManagerStubT.hpp"

Albany::AdaptiveSolutionManagerStubT::AdaptiveSolutionManagerStubT(
    const Teuchos::RCP<Teuchos::ParameterList>& appParams,
    const Teuchos::RCP<Albany::AbstractDiscretization>& disc,
    const Teuchos::RCP<const Tpetra_Vector>& initial_guessT)
{
  {
    const Teuchos::RCP<const Tpetra_Map> mapT = disc->getMapT();
    const Teuchos::RCP<const Tpetra_Map> overlapMapT = disc->getOverlapMapT();
    const Teuchos::RCP<const Tpetra_CrsGraph> overlapGraphT = disc->getOverlapJacobianGraphT();

    importerT = Teuchos::rcp(new Tpetra_Import(mapT, overlapMapT));
    exporterT = Teuchos::rcp(new Tpetra_Export(overlapMapT, mapT));

    overlapped_xT = Teuchos::rcp(new Tpetra_Vector(overlapMapT));
    overlapped_xdotT = Teuchos::rcp(new Tpetra_Vector(overlapMapT));
    overlapped_fT = Teuchos::rcp(new Tpetra_Vector(overlapMapT));
    overlapped_jacT = Teuchos::rcp(new Tpetra_CrsMatrix(overlapGraphT));

    tmp_ovlp_solT = Teuchos::rcp(new Tpetra_Vector(overlapMapT));

    initial_xT = disc->getSolutionFieldT();
    initial_xdotT = Teuchos::rcp(new Tpetra_Vector(mapT));
  }

  {
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > > > wsElNodeEqID = disc->getWsElNodeEqID();
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > > coords = disc->getCoords();
    Teuchos::ArrayRCP<std::string> wsEBNames = disc->getWsEBNames();
    const int numDim = disc->getNumDim();
    const int neq = disc->getNumEq();

    Teuchos::RCP<Teuchos::ParameterList> problemParams = Teuchos::sublist(appParams, "Problem", true);
    if (Teuchos::nonnull(initial_guessT)) {
      initial_xT = Teuchos::rcp(new Tpetra_Vector(*initial_guessT));
    } else {
      overlapped_xT->doImport(*initial_xT, *importerT, Tpetra::INSERT);

      Albany::InitialConditionsT(
          overlapped_xT, wsElNodeEqID, wsEBNames, coords, neq, numDim,
          problemParams->sublist("Initial Condition"),
          disc->hasRestartSolution());
      Albany::InitialConditionsT(
          overlapped_xdotT,  wsElNodeEqID, wsEBNames, coords, neq, numDim,
          problemParams->sublist("Initial Condition Dot"));

      initial_xT->doExport(*overlapped_xT, *exporterT, Tpetra::INSERT);
      initial_xdotT->doExport(*overlapped_xdotT, *exporterT, Tpetra::INSERT);
    }
  }
}

Teuchos::RCP<Tpetra_Vector>
Albany::AdaptiveSolutionManagerStubT::getOverlapSolutionT(const Tpetra_Vector& solutionT)
{
  tmp_ovlp_solT->doImport(solutionT, *importerT, Tpetra::INSERT);
  return tmp_ovlp_solT;
}

void
Albany::AdaptiveSolutionManagerStubT::scatterXT(
    const Tpetra_Vector& xT,
    const Tpetra_Vector* x_dotT)
{
  overlapped_xT->doImport(xT, *importerT, Tpetra::INSERT);

  if (x_dotT) {
    overlapped_xdotT->doImport(*x_dotT, *importerT, Tpetra::INSERT);
  }
}

