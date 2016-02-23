//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifndef AADAPT_ADAPTIVESOLUTIONMANAGERT
#define AADAPT_ADAPTIVESOLUTIONMANAGERT

#include "Albany_DataTypes.hpp"
#include "Albany_AbstractDiscretization.hpp"
#include "Albany_StateManager.hpp"
#include "AAdapt_InitialCondition.hpp"
#include "AAdapt_AbstractAdapterT.hpp"

#include "Thyra_AdaptiveSolutionManager.hpp"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

namespace AAdapt {

namespace rc { class Manager; }

class AdaptiveSolutionManagerT : public Thyra::AdaptiveSolutionManager {
public:
    AdaptiveSolutionManagerT(
        const Teuchos::RCP<Teuchos::ParameterList>& appParams,
        const Teuchos::RCP<const Tpetra_Vector>& initial_guessT,
        const Teuchos::RCP<ParamLib>& param_lib,
        const Albany::StateManager& StateMgr,
        const Teuchos::RCP<rc::Manager>& rc_mgr,
        const Teuchos::RCP<const Teuchos_Comm>& commT);

   //! Method called by the solver implementation to determine if the mesh needs adapting
   // A return type of true means that the mesh should be adapted
   virtual bool queryAdaptationCriteria(){ return adapter_->queryAdaptationCriteria(iter_); }

   //! Method called by solver implementation to actually adapt the mesh
   //! Apply adaptation method to mesh and problem. Returns true if adaptation is performed successfully.
   virtual bool adaptProblem();

   //! Remap "old" solution into new data structures
   virtual void projectCurrentSolution();

   Teuchos::RCP<const Tpetra_MultiVector> getInitialSolution() const { return current_soln; }

   Teuchos::RCP<Tpetra_MultiVector> getOverlappedSolution() { return overlapped_soln; }

   Teuchos::RCP<const Tpetra_MultiVector> getOverlappedSolution() const { return overlapped_soln; }

   Teuchos::RCP<Tpetra_Vector> updateAndReturnOverlapSolutionT(const Tpetra_Vector& solutionT /*not overlapped*/);
   Teuchos::RCP<const Tpetra_MultiVector> updateAndReturnOverlapSolutionMV(const Tpetra_MultiVector& solutionT /*not overlapped*/);

   Teuchos::RCP<Tpetra_Vector> get_overlapped_fT() {return overlapped_fT;}
   Teuchos::RCP<Tpetra_CrsMatrix> get_overlapped_jacT() {return overlapped_jacT;}

   Teuchos::RCP<Tpetra_Import> get_importerT() {return importerT;}
   Teuchos::RCP<Tpetra_Export> get_exporterT() {return exporterT;}

   Teuchos::RCP<Thyra::MultiVectorBase<double> > getCurrentSolution();

   void scatterXT(
       const Tpetra_Vector& xT,
       const Tpetra_Vector* x_dotT,
       const Tpetra_Vector* x_dotdotT);

   void scatterXT(
       const Tpetra_MultiVector& soln);

private:

    Teuchos::RCP<Tpetra_Import> importerT;
    Teuchos::RCP<Tpetra_Export> exporterT;

    Teuchos::RCP<Tpetra_Vector> overlapped_fT;
    Teuchos::RCP<Tpetra_CrsMatrix> overlapped_jacT;

    // The solution directly from the discretization class
    Teuchos::RCP<Tpetra_MultiVector> current_soln;
    Teuchos::RCP<Tpetra_MultiVector> overlapped_soln;

    // Number of time derivative vectors that we need to support
    const int num_time_deriv;

    const Teuchos::RCP<Teuchos::ParameterList> appParams_;
    const Teuchos::RCP<Albany::AbstractDiscretization> disc_;
    const Teuchos::RCP<ParamLib>& paramLib_;
    const Albany::StateManager& stateMgr_;
    const Teuchos::RCP<const Teuchos_Comm> commT_;

    //! Output stream, defaults to printing just Proc 0
    Teuchos::RCP<Teuchos::FancyOStream> out;

    Teuchos::RCP<AAdapt::AbstractAdapterT> adapter_;

    void buildAdapter(const Teuchos::RCP<rc::Manager>& rc_mgr);

    void resizeMeshDataArrays(
           const Teuchos::RCP<const Tpetra_Map> &mapT,
           const Teuchos::RCP<const Tpetra_Map> &overlapMapT,
           const Teuchos::RCP<const Tpetra_CrsGraph> &overlapJacGraphT);

};

} // namespace AAdapt

#endif /*AADAPT_ADAPTIVESOLUTIONMANAGERT*/
