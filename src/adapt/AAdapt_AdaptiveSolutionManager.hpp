//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AADAPT_ADAPTIVE_SOLUTION_MANAGER_HPP
#define AADAPT_ADAPTIVE_SOLUTION_MANAGER_HPP

#include "Albany_DataTypes.hpp"
#include "Albany_AbstractDiscretization.hpp"
#include "Albany_CombineAndScatterManager.hpp"

#include "AAdapt_AbstractAdapter.hpp"

#include "Thyra_AdaptiveSolutionManager.hpp"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

namespace Albany {

namespace AAdapt {

class AdaptiveSolutionManager : public Thyra::AdaptiveSolutionManager {
public:
    AdaptiveSolutionManager(
        const Teuchos::RCP<Teuchos::ParameterList>& appParams,
        const Teuchos::RCP<const Thyra_Vector>& initial_guess,
        const Teuchos::RCP<ParamLib>& param_lib,
        const Teuchos::RCP<AbstractDiscretization>& disc,
        const Teuchos::RCP<const Teuchos_Comm>& comm);

   //! Method called by the solver implementation to determine if the mesh needs adapting
   // A return type of true means that the mesh should be adapted
   bool queryAdaptationCriteria(){ return adapter_->queryAdaptationCriteria(iter_); }

   //! Method called by solver implementation to actually adapt the mesh
   //! Apply adaptation method to mesh and problem. Returns true if adaptation is performed successfully.
   virtual bool adaptProblem();

   //! Remap "old" solution into new data structures
   void projectCurrentSolution();

   Teuchos::RCP<const Thyra_MultiVector> getInitialSolution() const { return current_soln; }

   Teuchos::RCP<Thyra_MultiVector> getOverlappedSolution() { return overlapped_soln; }
   Teuchos::RCP<const Thyra_MultiVector> getOverlappedSolution() const { return overlapped_soln; }

   Teuchos::RCP<const Thyra_Vector> updateAndReturnOverlapSolution(const Thyra_Vector& solution /*not overlapped*/);
   Teuchos::RCP<const Thyra_Vector> updateAndReturnOverlapSolutionDot(const Thyra_Vector& solution_dot /*not overlapped*/);
   Teuchos::RCP<const Thyra_Vector> updateAndReturnOverlapSolutionDotDot(const Thyra_Vector& solution_dotdot /*not overlapped*/);
   Teuchos::RCP<const Thyra_MultiVector> updateAndReturnOverlapSolutionMV(const Thyra_MultiVector& solution /*not overlapped*/);
   Teuchos::RCP<Thyra_MultiVector> updateAndReturnOverlapSolutionDxDp(const Thyra_MultiVector& solution_dxdp /*not overlapped*/);

   Teuchos::RCP<Thyra_Vector>   get_overlapped_f()   const {return overlapped_f;}
   Teuchos::RCP<Thyra_LinearOp> get_overlapped_jac() const {return overlapped_jac;}

   Teuchos::RCP<const CombineAndScatterManager> get_cas_manager() const { return cas_manager; }

   Teuchos::RCP<Thyra_MultiVector> getCurrentSolution() { return current_soln; }

   void scatterX (const Thyra_Vector& x,
                  const Teuchos::Ptr<const Thyra_Vector> x_dot,
                  const Teuchos::Ptr<const Thyra_Vector> x_dotdot,
                  const Teuchos::Ptr<const Thyra_MultiVector> dxdp = Teuchos::null);

   void scatterX (const Thyra_MultiVector& soln,
                  const Teuchos::Ptr<const Thyra_MultiVector> dxdp = Teuchos::null);

private:

    void resizeMeshDataArrays(const Teuchos::RCP<const AbstractDiscretization>& disc);

    Teuchos::RCP<const CombineAndScatterManager> cas_manager;

    Teuchos::RCP<Thyra_Vector>   overlapped_f;
    Teuchos::RCP<Thyra_LinearOp> overlapped_jac;

    // The solution directly from the discretization class
    Teuchos::RCP<Thyra_MultiVector> current_soln;
    Teuchos::RCP<Thyra_MultiVector> overlapped_soln;
    Teuchos::RCP<Thyra_MultiVector> overlapped_soln_dxdp;

    // Number of time derivative vectors that we need to support
    const int num_time_deriv;

    const Teuchos::RCP<Teuchos::ParameterList> appParams_;
    const Teuchos::RCP<AbstractDiscretization> disc_;
    const Teuchos::RCP<ParamLib>& paramLib_;
    const Teuchos::RCP<const Teuchos_Comm> comm_;

    //! Output stream, defaults to printing just Proc 0
    Teuchos::RCP<Teuchos::FancyOStream> out;

    Teuchos::RCP<AbstractAdapter> adapter_;
  
    //! Number parameters
    int num_params_{0};
};

} // namespace AAdapt

} // namespace Albany

#endif // AADAPT_ADAPTIVE_SOLUTION_MANAGER_HPP
