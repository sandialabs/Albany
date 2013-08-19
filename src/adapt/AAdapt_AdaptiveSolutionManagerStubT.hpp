//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef AADAPT_ADAPTIVESOLUTIONMANAGERSTUBT
#define AADAPT_ADAPTIVESOLUTIONMANAGERSTUBT

#include "Albany_DataTypes.hpp"
#include "Albany_AbstractDiscretization.hpp"
#include "AAdapt_InitialCondition.hpp"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

namespace AAdapt {

class AdaptiveSolutionManagerStubT {
public:
    AdaptiveSolutionManagerStubT(
        const Teuchos::RCP<Teuchos::ParameterList>& appParams,
        const Teuchos::RCP<Albany::AbstractDiscretization>& disc,
        const Teuchos::RCP<const Tpetra_Vector>& initial_guessT);

   Teuchos::RCP<const Tpetra_Vector> getInitialSolutionT() const { return initial_xT; }
   Teuchos::RCP<const Tpetra_Vector> getInitialSolutionDotT() const { return initial_xdotT; }
   Teuchos::RCP<Tpetra_Vector> getOverlapSolutionT(const Tpetra_Vector& solutionT);

   Teuchos::RCP<Tpetra_Vector>& get_overlapped_xT() {return overlapped_xT;}
   Teuchos::RCP<Tpetra_Vector>& get_overlapped_xdotT() {return overlapped_xdotT;}
   Teuchos::RCP<Tpetra_Vector>& get_overlapped_fT() {return overlapped_fT;}
   Teuchos::RCP<Tpetra_CrsMatrix>& get_overlapped_jacT() {return overlapped_jacT;}

   Teuchos::RCP<Tpetra_Import>& get_importerT() {return importerT;}
   Teuchos::RCP<Tpetra_Export>& get_exporterT() {return exporterT;}

   void scatterXT(
       const Tpetra_Vector& xT,
       const Tpetra_Vector* x_dotT);

private:
    Teuchos::RCP<Tpetra_Import> importerT;
    Teuchos::RCP<Tpetra_Export> exporterT;

    Teuchos::RCP<Tpetra_Vector> overlapped_xT;
    Teuchos::RCP<Tpetra_Vector> overlapped_xdotT;
    Teuchos::RCP<Tpetra_Vector> overlapped_fT;
    Teuchos::RCP<Tpetra_CrsMatrix> overlapped_jacT;

    Teuchos::RCP<Tpetra_Vector> tmp_ovlp_solT;

    Teuchos::RCP<Tpetra_Vector> initial_xT;
    Teuchos::RCP<Tpetra_Vector> initial_xdotT;
};

} // namespace AAdapt

#endif /*AADAPT_ADAPTIVESOLUTIONMANAGERSTUBT*/
