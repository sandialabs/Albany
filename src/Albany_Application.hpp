//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_APPLICATION_HPP
#define ALBANY_APPLICATION_HPP

#include <vector>

#include "Teuchos_RCP.hpp"
#include "Teuchos_ArrayRCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_SerialDenseMatrix.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_TimeMonitor.hpp"

#include "Epetra_Map.h"
#include "Epetra_Vector.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_Import.h"
#include "Epetra_Export.h"

#include "Albany_AbstractDiscretization.hpp"
#include "Albany_AbstractProblem.hpp"
#include "Albany_AbstractResponseFunction.hpp"
#include "Albany_StateManager.hpp"
#include "AAdapt_AdaptiveSolutionManager.hpp"

#ifdef ALBANY_CUTR
  #include "CUTR_CubitMeshMover.hpp"
  #include "STKMeshData.hpp"
#endif

#include "Sacado_ScalarParameterLibrary.hpp"
#include "Sacado_ScalarParameterVector.hpp"
#include "Sacado_ParameterAccessor.hpp"
#include "Sacado_ParameterRegistration.hpp"

#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Workset.hpp"
#include "Phalanx.hpp"

#include "Stokhos_OrthogPolyExpansion.hpp"
#include "Stokhos_Quadrature.hpp"
#include "Stokhos_EpetraVectorOrthogPoly.hpp"
#include "Stokhos_EpetraMultiVectorOrthogPoly.hpp"
#include "EpetraExt_MultiComm.h"

#include "LOCA_Epetra_Group.H"

#include "Teko_InverseLibrary.hpp"

#ifdef ALBANY_MOR
  #include "MOR/Albany_MORFacade.hpp"
#endif

namespace Albany {

  class Application :
     public Sacado::ParameterAccessor<PHAL::AlbanyTraits::Residual, SPL_Traits> {
  public:

    enum SolutionMethod {Steady, Transient, Continuation, Eigensolve};

    //! Constructor
    Application(const Teuchos::RCP<const Epetra_Comm>& comm,
                const Teuchos::RCP<Teuchos::ParameterList>& params,
                const Teuchos::RCP<const Epetra_Vector>& initial_guess =
                Teuchos::null);

    //! Destructor
    ~Application();

    //! Get underlying abstract discretization
    Teuchos::RCP<Albany::AbstractDiscretization> getDiscretization() const;

    //! Get problem object
    Teuchos::RCP<Albany::AbstractProblem> getProblem() const;

    //! Get communicator
    Teuchos::RCP<const Epetra_Comm> getComm() const;

    //! Get DOF map
    Teuchos::RCP<const Epetra_Map> getMap() const;

    //! Get Jacobian graph
    Teuchos::RCP<const Epetra_CrsGraph> getJacobianGraph() const;

    //! Get Preconditioner Operator
    Teuchos::RCP<Epetra_Operator> getPreconditioner();

    //! Get the solution memory manager
    Teuchos::RCP<AAdapt::AdaptiveSolutionManager> getAdaptSolMgr(){ return solMgr;}

    //! Get parameter library
    Teuchos::RCP<ParamLib> getParamLib();

    //! Get distributed parameter library
    Teuchos::RCP<DistParamLib> getDistParamLib();

    //! Get solution method
    SolutionMethod getSolutionMethod() const {return solMethod; }

    //! Get number of responses
    int getNumResponses() const;

    //! Get response function
    Teuchos::RCP<AbstractResponseFunction> getResponse(int i) const;

    //! Return whether problem wants to use its own preconditioner
    bool suppliesPreconditioner() const;

    //! Get stochastic expansion
    Teuchos::RCP<Stokhos::OrthogPolyExpansion<int,double> >
    getStochasticExpansion();

    //! Intialize stochastic Galerkin method
#ifdef ALBANY_SG_MP
    void init_sg(
      const Teuchos::RCP<const Stokhos::OrthogPolyBasis<int,double> >& basis,
      const Teuchos::RCP<const Stokhos::Quadrature<int,double> >& quad,
      const Teuchos::RCP<Stokhos::OrthogPolyExpansion<int,double> >& expansion,
      const Teuchos::RCP<const EpetraExt::MultiComm>& multiComm);
#endif //ALBANY_SG_MP

    //! Compute global residual
    /*!
     * Set xdot to NULL for steady-state problems
     */
    void computeGlobalResidual(const double current_time,
                               const Epetra_Vector* xdot,
                               const Epetra_Vector* xdotdot,
                               const Epetra_Vector& x,
                               const Teuchos::Array<ParamVec>& p,
                               Epetra_Vector& f);

    //! Compute global Jacobian
    /*!
     * Set xdot to NULL for steady-state problems
     */
    void computeGlobalJacobian(const double alpha,
                               const double beta,
                               const double omega,
                               const double current_time,
                               const Epetra_Vector* xdot,
                               const Epetra_Vector* xdotdot,
                               const Epetra_Vector& x,
                               const Teuchos::Array<ParamVec>& p,
                               Epetra_Vector* f,
                               Epetra_CrsMatrix& jac);

    //! Compute global Preconditioner
    /*!
     * Set xdot to NULL for steady-state problems
     */
    void computeGlobalPreconditioner(const Teuchos::RCP<Epetra_CrsMatrix>& jac,
                                     const Teuchos::RCP<Epetra_Operator>& prec);

    //! Compute global Tangent
    /*!
     * Set xdot to NULL for steady-state problems
     */
    void computeGlobalTangent(const double alpha,
                              const double beta,
                               const double omega,
                              const double current_time,
                              bool sum_derivs,
                              const Epetra_Vector* xdot,
                              const Epetra_Vector* xdotdot,
                              const Epetra_Vector& x,
                              const Teuchos::Array<ParamVec>& p,
                              ParamVec* deriv_p,
                              const Epetra_MultiVector* Vx,
                              const Epetra_MultiVector* Vxdot,
                              const Epetra_MultiVector* Vxdotdot,
                              const Epetra_MultiVector* Vp,
                              Epetra_Vector* f,
                              Epetra_MultiVector* JV,
                              Epetra_MultiVector* fp);

    //! Compute df/dp*V or (df/dp)^T*V for distributed parameter p
    /*!
     * Set xdot to NULL for steady-state problems
     */
    void applyGlobalDistParamDeriv(const double current_time,
                                   const Epetra_Vector* xdot,
                                   const Epetra_Vector* xdotdot,
                                   const Epetra_Vector& x,
                                   const Teuchos::Array<ParamVec>& p,
                                   const std::string& dist_param_name,
                                   const bool trans,
                                   const Epetra_MultiVector& V,
                                   Epetra_MultiVector& fpV);

    //! Evaluate response functions
    /*!
     * Set xdot to NULL for steady-state problems
     */
    void evaluateResponse(
      int response_index,
      const double current_time,
      const Epetra_Vector* xdot,
      const Epetra_Vector* xdotdot,
      const Epetra_Vector& x,
      const Teuchos::Array<ParamVec>& p,
      Epetra_Vector& g);

    //! Evaluate tangent = alpha*dg/dx*Vx + beta*dg/dxdot*Vxdot + dg/dp*Vp
    /*!
     * Set xdot, dxdot_dp to NULL for steady-state problems
     */
    void evaluateResponseTangent(
      int response_index,
      const double alpha,
      const double beta,
      const double omega,
      const double current_time,
      bool sum_derivs,
      const Epetra_Vector* xdot,
      const Epetra_Vector* xdotdot,
      const Epetra_Vector& x,
      const Teuchos::Array<ParamVec>& p,
      ParamVec* deriv_p,
      const Epetra_MultiVector* Vxdot,
      const Epetra_MultiVector* Vxdotdot,
      const Epetra_MultiVector* Vx,
      const Epetra_MultiVector* Vp,
      Epetra_Vector* g,
      Epetra_MultiVector* gx,
      Epetra_MultiVector* gp);

    //! Evaluate gradient = dg/dx, dg/dxdot, dg/dp
    /*!
     * Set xdot, dg_dxdot to NULL for steady-state problems
     */
    void evaluateResponseDerivative(
      int response_index,
      const double current_time,
      const Epetra_Vector* xdot,
      const Epetra_Vector* xdotdot,
      const Epetra_Vector& x,
      const Teuchos::Array<ParamVec>& p,
      ParamVec* deriv_p,
      Epetra_Vector* g,
      const EpetraExt::ModelEvaluator::Derivative& dg_dx,
      const EpetraExt::ModelEvaluator::Derivative& dg_dxdot,
      const EpetraExt::ModelEvaluator::Derivative& dg_dxdotdot,
      const EpetraExt::ModelEvaluator::Derivative& dg_dp);

#ifdef ALBANY_SG_MP
    //! Compute global residual for stochastic Galerkin problem
    /*!
     * Set xdot to NULL for steady-state problems
     */
    void computeGlobalSGResidual(
      const double current_time,
      const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
      const Stokhos::EpetraVectorOrthogPoly* sg_xdotdot,
      const Stokhos::EpetraVectorOrthogPoly& sg_x,
      const Teuchos::Array<ParamVec>& p,
      const Teuchos::Array<int>& sg_p_index,
      const Teuchos::Array< Teuchos::Array<SGType> >& sg_p_vals,
      Stokhos::EpetraVectorOrthogPoly& sg_f);

    //! Compute global Jacobian for stochastic Galerkin problem
    /*!
     * Set xdot to NULL for steady-state problems
     */
    void computeGlobalSGJacobian(
      const double alpha,
      const double beta,
      const double omega,
      const double current_time,
      const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
      const Stokhos::EpetraVectorOrthogPoly* sg_xdotdot,
      const Stokhos::EpetraVectorOrthogPoly& sg_x,
      const Teuchos::Array<ParamVec>& p,
      const Teuchos::Array<int>& sg_p_index,
      const Teuchos::Array< Teuchos::Array<SGType> >& sg_p_vals,
      Stokhos::EpetraVectorOrthogPoly* sg_f,
      Stokhos::VectorOrthogPoly<Epetra_CrsMatrix>& sg_jac);

    //! Compute global Tangent for stochastic Galerkin problem
    /*!
     * Set xdot to NULL for steady-state problems
     */
    void computeGlobalSGTangent(
      const double alpha,
      const double beta,
      const double omega,
      const double current_time,
      bool sum_derivs,
      const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
      const Stokhos::EpetraVectorOrthogPoly* sg_xdotdot,
      const Stokhos::EpetraVectorOrthogPoly& sg_x,
      const Teuchos::Array<ParamVec>& p,
      const Teuchos::Array<int>& sg_p_index,
      const Teuchos::Array< Teuchos::Array<SGType> >& sg_p_vals,
      ParamVec* deriv_p,
      const Epetra_MultiVector* Vx,
      const Epetra_MultiVector* Vxdot,
      const Epetra_MultiVector* Vxdotdot,
      const Epetra_MultiVector* Vp,
      Stokhos::EpetraVectorOrthogPoly* sg_f,
      Stokhos::EpetraMultiVectorOrthogPoly* sg_JVx,
      Stokhos::EpetraMultiVectorOrthogPoly* sg_fVp);

    //! Evaluate stochastic Galerkin response functions
    /*!
     * Set xdot to NULL for steady-state problems
     */
    void evaluateSGResponse(
      int response_index,
      const double curr_time,
      const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
      const Stokhos::EpetraVectorOrthogPoly* sg_xdotdot,
      const Stokhos::EpetraVectorOrthogPoly& sg_x,
      const Teuchos::Array<ParamVec>& p,
      const Teuchos::Array<int>& sg_p_index,
      const Teuchos::Array< Teuchos::Array<SGType> >& sg_p_vals,
      Stokhos::EpetraVectorOrthogPoly& sg_g);

    //! Evaluate tangent = dg/dx*dx/dp + dg/dxdot*dxdot/dp + dg/dp
    /*!
     * Set xdot, dxdot_dp to NULL for steady-state problems
     */
    void
    evaluateSGResponseTangent(
      int response_index,
      const double alpha,
      const double beta,
      const double omega,
      const double current_time,
      bool sum_derivs,
      const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
      const Stokhos::EpetraVectorOrthogPoly* sg_xdotdot,
      const Stokhos::EpetraVectorOrthogPoly& sg_x,
      const Teuchos::Array<ParamVec>& p,
      const Teuchos::Array<int>& sg_p_index,
      const Teuchos::Array< Teuchos::Array<SGType> >& sg_p_vals,
      ParamVec* deriv_p,
      const Epetra_MultiVector* Vx,
      const Epetra_MultiVector* Vxdot,
      const Epetra_MultiVector* Vxdotdot,
      const Epetra_MultiVector* Vp,
      Stokhos::EpetraVectorOrthogPoly* sg_g,
      Stokhos::EpetraMultiVectorOrthogPoly* sg_JV,
      Stokhos::EpetraMultiVectorOrthogPoly* sg_gp);

    //! Evaluate gradient = dg/dx, dg/dxdot, dg/dp
    /*!
     * Set xdot, dg_dxdot to NULL for steady-state problems
     */
    void
    evaluateSGResponseDerivative(
      int response_index,
      const double current_time,
      const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
      const Stokhos::EpetraVectorOrthogPoly* sg_xdotdot,
      const Stokhos::EpetraVectorOrthogPoly& sg_x,
      const Teuchos::Array<ParamVec>& p,
      const Teuchos::Array<int>& sg_p_index,
      const Teuchos::Array< Teuchos::Array<SGType> >& sg_p_vals,
      ParamVec* deriv_p,
      Stokhos::EpetraVectorOrthogPoly* sg_g,
      const EpetraExt::ModelEvaluator::SGDerivative& sg_dg_dx,
      const EpetraExt::ModelEvaluator::SGDerivative& sg_dg_dxdot,
      const EpetraExt::ModelEvaluator::SGDerivative& sg_dg_dxdotdot,
      const EpetraExt::ModelEvaluator::SGDerivative& sg_dg_dp);

    //! Compute global residual for stochastic Galerkin problem
    /*!
     * Set xdot to NULL for steady-state problems
     */
    void computeGlobalMPResidual(
      const double current_time,
      const Stokhos::ProductEpetraVector* mp_xdot,
      const Stokhos::ProductEpetraVector* mp_xdotdot,
      const Stokhos::ProductEpetraVector& mp_x,
      const Teuchos::Array<ParamVec>& p,
      const Teuchos::Array<int>& mp_p_index,
      const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals,
      Stokhos::ProductEpetraVector& mp_f);

    //! Compute global Jacobian for stochastic Galerkin problem
    /*!
     * Set xdot to NULL for steady-state problems
     */
    void computeGlobalMPJacobian(
      const double alpha,
      const double beta,
      const double omega,
      const double current_time,
      const Stokhos::ProductEpetraVector* mp_xdot,
      const Stokhos::ProductEpetraVector* mp_xdotdot,
      const Stokhos::ProductEpetraVector& mp_x,
      const Teuchos::Array<ParamVec>& p,
      const Teuchos::Array<int>& mp_p_index,
      const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals,
      Stokhos::ProductEpetraVector* mp_f,
      Stokhos::ProductContainer<Epetra_CrsMatrix>& mp_jac);

    //! Compute global Tangent for multi-point problem
    /*!
     * Set xdot to NULL for steady-state problems
     */
    void computeGlobalMPTangent(
      const double alpha,
      const double beta,
      const double omega,
      const double current_time,
      bool sum_derivs,
      const Stokhos::ProductEpetraVector* mp_xdot,
      const Stokhos::ProductEpetraVector* mp_xdotdot,
      const Stokhos::ProductEpetraVector& mp_x,
      const Teuchos::Array<ParamVec>& p,
      const Teuchos::Array<int>& mp_p_index,
      const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals,
      ParamVec* deriv_p,
      const Epetra_MultiVector* Vx,
      const Epetra_MultiVector* Vxdot,
      const Epetra_MultiVector* Vxdotdot,
      const Epetra_MultiVector* Vp,
      Stokhos::ProductEpetraVector* mp_f,
      Stokhos::ProductEpetraMultiVector* mp_JVx,
      Stokhos::ProductEpetraMultiVector* mp_fVp);

    //! Evaluate stochastic Galerkin response functions
    /*!
     * Set xdot to NULL for steady-state problems
     */
    void evaluateMPResponse(
      int response_index,
      const double curr_time,
      const Stokhos::ProductEpetraVector* mp_xdot,
      const Stokhos::ProductEpetraVector* mp_xdotdot,
      const Stokhos::ProductEpetraVector& mp_x,
      const Teuchos::Array<ParamVec>& p,
      const Teuchos::Array<int>& mp_p_index,
      const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals,
      Stokhos::ProductEpetraVector& mp_g);

    //! Evaluate tangent = dg/dx*dx/dp + dg/dxdot*dxdot/dp + dg/dp
    /*!
     * Set xdot, dxdot_dp to NULL for steady-state problems
     */
    void
    evaluateMPResponseTangent(
      int response_index,
      const double alpha,
      const double beta,
      const double omega,
      const double current_time,
      bool sum_derivs,
      const Stokhos::ProductEpetraVector* mp_xdot,
      const Stokhos::ProductEpetraVector* mp_xdotdot,
      const Stokhos::ProductEpetraVector& mp_x,
      const Teuchos::Array<ParamVec>& p,
      const Teuchos::Array<int>& mp_p_index,
      const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals,
      ParamVec* deriv_p,
      const Epetra_MultiVector* Vx,
      const Epetra_MultiVector* Vxdot,
      const Epetra_MultiVector* Vxdotdot,
      const Epetra_MultiVector* Vp,
      Stokhos::ProductEpetraVector* mp_g,
      Stokhos::ProductEpetraMultiVector* mp_JV,
      Stokhos::ProductEpetraMultiVector* mp_gp);

    //! Evaluate gradient = dg/dx, dg/dxdot, dg/dp
    /*!
     * Set xdot, dg_dxdot to NULL for steady-state problems
     */
    void
    evaluateMPResponseDerivative(
      int response_index,
      const double current_time,
      const Stokhos::ProductEpetraVector* mp_xdot,
      const Stokhos::ProductEpetraVector* mp_xdotdot,
      const Stokhos::ProductEpetraVector& mp_x,
      const Teuchos::Array<ParamVec>& p,
      const Teuchos::Array<int>& mp_p_index,
      const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals,
      ParamVec* deriv_p,
      Stokhos::ProductEpetraVector* mp_g,
      const EpetraExt::ModelEvaluator::MPDerivative& mp_dg_dx,
      const EpetraExt::ModelEvaluator::MPDerivative& mp_dg_dxdot,
      const EpetraExt::ModelEvaluator::MPDerivative& mp_dg_dxdotdot,
      const EpetraExt::ModelEvaluator::MPDerivative& mp_dg_dp);
#endif //ALBANY_SG_MP

    //! Provide access to shapeParameters -- no AD
    PHAL::AlbanyTraits::Residual::ScalarT& getValue(const std::string &n);

    //! Class to manage state variables (a.k.a. history)
    StateManager& getStateMgr() {return stateMgr; }

    //! Evaluate state field manager
    void evaluateStateFieldManager(const double current_time,
                                   const Epetra_Vector* xdot,
                                   const Epetra_Vector* xdotdot,
                                   const Epetra_Vector& x);

    //! Access to number of worksets - needed for working with StateManager
    int getNumWorksets() {
        return disc->getWsElNodeEqID().size();
    }

    bool is_adjoint;

  private:

    //! Private to prohibit copying
    Application(const Application&);

    //! Private to prohibit copying
    Application& operator=(const Application&);

    //! Call to Teko to build strided block operator
    Teuchos::RCP<Epetra_Operator> buildWrappedOperator(
                           const Teuchos::RCP<Epetra_Operator>& Jac,
                           const Teuchos::RCP<Epetra_Operator>& wrapInput,
                           bool reorder=false) const;

    //! Utility function to set up ShapeParameters through Sacado
    void registerShapeParameters();

    void defineTimers();

  public:

    //! Routine to get workset (bucket) size info needed by all Evaluation types
    template <typename EvalT>
    void loadWorksetBucketInfo(PHAL::Workset& workset, const int& ws);

    //! Routine to load some basic workset info needed by many Evaluation types
    void loadBasicWorksetInfo(
            PHAL::Workset& workset,
            double current_time);

    void loadWorksetJacobianInfo(PHAL::Workset& workset,
                const double& alpha, const double& beta, const double& omega);

    //! Routine to load common nodeset info into workset
    void loadWorksetNodesetInfo(PHAL::Workset& workset);

    //! Routine to load common sideset info into workset
    void loadWorksetSidesetInfo(PHAL::Workset& workset, const int ws);

    void setupBasicWorksetInfo(
      PHAL::Workset& workset,
      double current_time,
      const Epetra_Vector* xdot,
      const Epetra_Vector* xdotdot,
      const Epetra_Vector* x,
      const Teuchos::Array<ParamVec>& p);

#ifdef ALBANY_SG_MP
    void setupBasicWorksetInfo(
      PHAL::Workset& workset,
      double current_time,
      const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
      const Stokhos::EpetraVectorOrthogPoly* sg_xdotdot,
      const Stokhos::EpetraVectorOrthogPoly* sg_x,
      const Teuchos::Array<ParamVec>& p,
      const Teuchos::Array<int>& sg_p_index,
      const Teuchos::Array< Teuchos::Array<SGType> >& sg_p_vals);

    void setupBasicWorksetInfo(
      PHAL::Workset& workset,
      double current_time,
      const Stokhos::ProductEpetraVector* mp_xdot,
      const Stokhos::ProductEpetraVector* mp_xdotdot,
      const Stokhos::ProductEpetraVector* mp_x,
      const Teuchos::Array<ParamVec>& p,
      const Teuchos::Array<int>& mp_p_index,
      const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals);
#endif //ALBANY_SG_MP

    void setupTangentWorksetInfo(
      PHAL::Workset& workset,
      double current_time,
      bool sum_derivs,
      const Epetra_Vector* xdot,
      const Epetra_Vector* xdotdot,
      const Epetra_Vector* x,
      const Teuchos::Array<ParamVec>& p,
      ParamVec* deriv_p,
      const Epetra_MultiVector* Vxdot,
      const Epetra_MultiVector* Vxdotdot,
      const Epetra_MultiVector* Vx,
      const Epetra_MultiVector* Vp);

#ifdef ALBANY_SG_MP
    void setupTangentWorksetInfo(
      PHAL::Workset& workset,
      double current_time,
      bool sum_derivs,
      const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
      const Stokhos::EpetraVectorOrthogPoly* sg_xdotdot,
      const Stokhos::EpetraVectorOrthogPoly* sg_x,
      const Teuchos::Array<ParamVec>& p,
      ParamVec* deriv_p,
      const Teuchos::Array<int>& sg_p_index,
      const Teuchos::Array< Teuchos::Array<SGType> >& sg_p_vals,
      const Epetra_MultiVector* Vxdot,
      const Epetra_MultiVector* Vxdotdot,
      const Epetra_MultiVector* Vx,
      const Epetra_MultiVector* Vp);

    void setupTangentWorksetInfo(
      PHAL::Workset& workset,
      double current_time,
      bool sum_derivs,
      const Stokhos::ProductEpetraVector* mp_xdot,
      const Stokhos::ProductEpetraVector* mp_xdotdot,
      const Stokhos::ProductEpetraVector* mp_x,
      const Teuchos::Array<ParamVec>& p,
      ParamVec* deriv_p,
      const Teuchos::Array<int>& mp_p_index,
      const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals,
      const Epetra_MultiVector* Vxdot,
      const Epetra_MultiVector* Vxdotdot,
      const Epetra_MultiVector* Vx,
      const Epetra_MultiVector* Vp);
#endif //ALBANY_SG_MP

    void postRegSetup(std::string eval);

#ifdef ALBANY_MOR
    Teuchos::RCP<MORFacade> getMorFacade();
#endif

  protected:

    //! Communicator
    Teuchos::RCP<const Epetra_Comm> comm;

    //! Output stream, defaults to pronting just Proc 0
    Teuchos::RCP<Teuchos::FancyOStream> out;

    //! Element discretization
    Teuchos::RCP<Albany::AbstractDiscretization> disc;

    //! Problem class
    Teuchos::RCP<Albany::AbstractProblem> problem;

    //! Parameter library
    Teuchos::RCP<ParamLib> paramLib;

    //! Distributed parameter library
    Teuchos::RCP<DistParamLib> distParamLib;

    //! Solution memory manager
    Teuchos::RCP<AAdapt::AdaptiveSolutionManager> solMgr;

    //! Response functions
    Teuchos::Array< Teuchos::RCP<Albany::AbstractResponseFunction> > responses;

    //! Phalanx Field Manager for volumetric fills
    Teuchos::ArrayRCP<Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> > > fm;

    //! Phalanx Field Manager for Dirichlet Conditions
    Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> > dfm;

    //! Phalanx Field Manager for Neumann Conditions
    Teuchos::ArrayRCP<Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> > > nfm;

    //! Phalanx Field Manager for states
    Teuchos::Array< Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> > > sfm;

    //! Stochastic Galerkin basis
    Teuchos::RCP<const Stokhos::OrthogPolyBasis<int,double> > sg_basis;

    //! Stochastic Galerkin quadrature
    Teuchos::RCP<const Stokhos::Quadrature<int,double> > sg_quad;

    //! Stochastic Galerkin expansion
    Teuchos::RCP<Stokhos::OrthogPolyExpansion<int,double> > sg_expansion;

    //! Product multi-comm
    Teuchos::RCP<const EpetraExt::MultiComm> product_comm;

    //! Overlap stochastic map
    Teuchos::RCP<const Epetra_BlockMap> sg_overlap_map;

    //! SG overlapped solution vectors
    Teuchos::RCP< Stokhos::EpetraVectorOrthogPoly >  sg_overlapped_x;

    //! SG overlapped time derivative vectors
    Teuchos::RCP< Stokhos::EpetraVectorOrthogPoly > sg_overlapped_xdot;
    Teuchos::RCP< Stokhos::EpetraVectorOrthogPoly > sg_overlapped_xdotdot;

    //! SG overlapped residual vectors
    Teuchos::RCP< Stokhos::EpetraVectorOrthogPoly > sg_overlapped_f;

    //! Overlapped Jacobian matrixs
    Teuchos::RCP< Stokhos::VectorOrthogPoly<Epetra_CrsMatrix> > sg_overlapped_jac;

    //! MP overlapped solution vectors
    Teuchos::RCP< Stokhos::ProductEpetraVector >  mp_overlapped_x;

    //! MP overlapped time derivative vectors
    Teuchos::RCP< Stokhos::ProductEpetraVector > mp_overlapped_xdot;
    Teuchos::RCP< Stokhos::ProductEpetraVector > mp_overlapped_xdotdot;

    //! MP overlapped residual vectors
    Teuchos::RCP< Stokhos::ProductEpetraVector > mp_overlapped_f;

    //! Overlapped Jacobian matrixs
    Teuchos::RCP< Stokhos::ProductContainer<Epetra_CrsMatrix> > mp_overlapped_jac;

    //! Data for Physics-Based Preconditioners
    bool physicsBasedPreconditioner;
    Teuchos::RCP<Teuchos::ParameterList> tekoParams;

    //! Type of solution method
    SolutionMethod solMethod;

    //! Integer specifying whether user wants to write Jacobian to MatrixMarket file
    // writeToMatrixMarketJac = 0: no writing to MatrixMarket (default)
    // writeToMatrixMarketJac =-1: write to MatrixMarket every time a Jacobian arises
    // writeToMatrixMarketJac = N: write N^th Jacobian to MatrixMarket
    // ...and similarly for writeToMatrixMarketRes (integer specifying whether user wants to write
    // residual to MatrixMarket file)
    int writeToMatrixMarketJac;
    int writeToMatrixMarketRes;
    //! Integer specifying whether user wants to write Jacobian and residual to Standard output (cout)
    int writeToCoutJac;
    int writeToCoutRes;

    //! Shape Optimization data
    bool shapeParamsHaveBeenReset;
    std::vector<RealType> shapeParams;
    std::vector<std::string> shapeParamNames;
#ifdef ALBANY_CUTR
    Teuchos::RCP<CUTR::CubitMeshMover> meshMover;
#endif

    unsigned int neq;

    //! Teko stuff
    Teuchos::RCP<Teko::InverseLibrary> inverseLib;
    Teuchos::RCP<Teko::InverseFactory> inverseFac;
    Teuchos::RCP<Epetra_Operator> wrappedJac;
    std::vector<int> blockDecomp;

    std::set<std::string> setupSet;
    mutable int phxGraphVisDetail;
    mutable int stateGraphVisDetail;

    StateManager stateMgr;

    bool morphFromInit;
    bool ignore_residual_in_jacobian;

    //! To prevent a singular mass matrix associated with Dirichlet
    //  conditions, optionally add a small perturbation to the diag
    double perturbBetaForDirichlets;

    void determinePiroSolver(const Teuchos::RCP<Teuchos::ParameterList>& topLevelParams);

#ifdef ALBANY_MOR
    Teuchos::RCP<MORFacade> morFacade;
#endif
  };
}

template <typename EvalT>
void Albany::Application::loadWorksetBucketInfo(PHAL::Workset& workset,
                                                const int& ws)
{

  const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > > >::type&
        wsElNodeEqID = disc->getWsElNodeEqID();
  const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > >::type&
        wsElNodeID = disc->getWsElNodeID();
  const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type&
        coords = disc->getCoords();
  const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> > >::type&
        sHeight = disc->getSurfaceHeight();
  const WorksetArray<Teuchos::ArrayRCP<double> >::type&
        temperature  = disc->getTemperature();
  const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> > >::type&
        basalFriction  = disc->getBasalFriction();
  const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> > >::type&
        thickness = disc->getThickness();
  const WorksetArray<Teuchos::ArrayRCP<double> >::type&
        flowFactor  = disc->getFlowFactor();
  const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type&
        surfaceVelocity = disc->getSurfaceVelocity();
  const WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type&
        velocityRMS = disc->getVelocityRMS();
  const WorksetArray<std::string>::type& wsEBNames = disc->getWsEBNames();
  const WorksetArray<Teuchos::ArrayRCP<double> >::type&
        sphereVolume = disc->getSphereVolume();

  workset.numCells = wsElNodeEqID[ws].size();
  workset.wsElNodeEqID = wsElNodeEqID[ws];
  workset.wsElNodeID = wsElNodeID[ws];
  workset.wsCoords = coords[ws];
  workset.wsSHeight = sHeight[ws];
  workset.wsSphereVolume = sphereVolume[ws];
  workset.wsTemperature = temperature[ws];
  workset.wsBasalFriction = basalFriction[ws];
  workset.wsThickness = thickness[ws];
  workset.wsFlowFactor = flowFactor[ws];
  workset.wsSurfaceVelocity = surfaceVelocity[ws];
  workset.wsVelocityRMS = velocityRMS[ws];
  workset.EBName = wsEBNames[ws];
  workset.wsIndex = ws;

  workset.local_Vp.resize(workset.numCells);
  workset.dist_param_index.resize(workset.numCells);

//  workset.print(*out);

  // Sidesets are integrated within the Cells
  loadWorksetSidesetInfo(workset, ws);

  workset.stateArrayPtr = &stateMgr.getStateArray(Albany::StateManager::ELEM, ws);
  workset.eigenDataPtr = stateMgr.getEigenData();
  workset.auxDataPtr = stateMgr.getAuxData();

  PHAL::BuildSerializer<EvalT> bs(workset);
}

#endif // ALBANY_APPLICATION_HPP
