//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef GOAL_ADJOINTRESPONSEFUNCTION_HPP
#define GOAL_ADJOINTRESPONSEFUNCTION_HPP

#include "Albany_ScalarResponseFunction.hpp"
#include "Albany_Application.hpp"
#include "Albany_AbstractProblem.hpp"
#include "Albany_StateManager.hpp"
#include "Albany_StateInfoStruct.hpp"
#include "Phalanx.hpp"
#include "PHAL_AlbanyTraits.hpp"

namespace GOAL {

class AdjointResponseFunction :
  public Albany::ScalarResponseFunction
{

  public:

    AdjointResponseFunction(
        const Teuchos::RCP<Albany::Application>& application_,
        const Teuchos::RCP<Albany::AbstractProblem>& problem_,
        const Teuchos::RCP<Albany::MeshSpecsStruct>&  meshSpecs_,
        const Teuchos::RCP<Albany::StateManager>& stateMgr_,
        Teuchos::ParameterList& responseParams);

    ~AdjointResponseFunction();

    void setupT();
    
    Teuchos::RCP<const Tpetra_Map> responseMapT() const;
    
    bool isScalarResponse() const;

    unsigned int numResponses() const;
    
    Teuchos::RCP<Tpetra_Operator> createGradientOpT() const;

    void evaluateResponseT(
        const double current_time,
        const Tpetra_Vector* xdotT,
        const Tpetra_Vector* xdotdotT,
        const Tpetra_Vector& xT,
        const Teuchos::Array<ParamVec>& p,
        Tpetra_Vector& gT);

    void evaluateTangentT(
        const double alpha, 
        const double beta,
        const double omega,
        const double current_time,
        bool sum_derivs,
        const Tpetra_Vector* xdot,
        const Tpetra_Vector* xdotdot,
        const Tpetra_Vector& x,
        const Teuchos::Array<ParamVec>& p,
        ParamVec* deriv_p,
        const Tpetra_MultiVector* Vxdot,
        const Tpetra_MultiVector* Vxdotdot,
        const Tpetra_MultiVector* Vx,
        const Tpetra_MultiVector* Vp,
        Tpetra_Vector* g,
        Tpetra_MultiVector* gx,
        Tpetra_MultiVector* gp);

    void evaluateDerivativeT(
        const double current_time,
        const Tpetra_Vector* xdotT,
        const Tpetra_Vector* xdotdotT,
        const Tpetra_Vector& xT,
        const Teuchos::Array<ParamVec>& p,
        ParamVec* deriv_p,
        Tpetra_Vector* gT,
        const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dx,
        const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dxdot,
        const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dxdotdot,
        const Thyra::ModelEvaluatorBase::Derivative<ST>& dg_dp);

    void evaluateGradientT(
        const double current_time,
        const Tpetra_Vector* xdotT,
        const Tpetra_Vector* xdotdotT,
        const Tpetra_Vector& xT,
        const Teuchos::Array<ParamVec>& p,
        ParamVec* deriv_p,
        Tpetra_Vector* gT,
        Tpetra_MultiVector* dg_dxT,
        Tpetra_MultiVector* dg_dxdotT,
        Tpetra_MultiVector* dg_dxdotdotT,
        Tpetra_MultiVector* dg_dpT);

#ifdef ALBANY_EPETRA
    void setup() {}

    Teuchos::RCP<const Epetra_Map> responseMap() const {}

    Teuchos::RCP<Epetra_Operator> createGradientOp() const {}

    void evaluateDerivative(
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
        const EpetraExt::ModelEvaluator::Derivative& dg_dp) {}

    void evaluateDistParamDeriv(
        const double current_time,
        const Epetra_Vector* xdot,
        const Epetra_Vector* xdotdot,
        const Epetra_Vector& x,
        const Teuchos::Array<ParamVec>& param_array,
        const std::string& dist_param_name,
        Epetra_MultiVector*  dg_dp) {}

    void evaluateGradient(
        const double current_time,
        const Epetra_Vector* xdot,
        const Epetra_Vector* xdotdot,
        const Epetra_Vector& x,
        const Teuchos::Array<ParamVec>& p,
        ParamVec* deriv_p,
        Epetra_Vector* g,
        Epetra_MultiVector* dg_dx,
        Epetra_MultiVector* dg_dxdot,
        Epetra_MultiVector* dg_dxdotdot,
        Epetra_MultiVector* dg_dp) {}
#endif

#ifdef ALBANY_SG_MP
    void init_sg(
        const Teuchos::RCP<const Stokhos::OrthogPolyBasis<int,double> >& basis,
        const Teuchos::RCP<const Stokhos::Quadrature<int,double> >& quad,
        const Teuchos::RCP<Stokhos::OrthogPolyExpansion<int,double> >& expansion,
        const Teuchos::RCP<const EpetraExt::MultiComm>& multiComm) {}

    void evaluateSGResponse(
        const double curr_time,
        const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
        const Stokhos::EpetraVectorOrthogPoly* sg_xdotdot,
        const Stokhos::EpetraVectorOrthogPoly& sg_x,
        const Teuchos::Array<ParamVec>& p,
        const Teuchos::Array<int>& sg_p_index,
        const Teuchos::Array< Teuchos::Array<SGType> >& sg_p_vals,
        Stokhos::EpetraVectorOrthogPoly& sg_g) {}

    void evaluateSGTangent(
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
        Stokhos::EpetraMultiVectorOrthogPoly* sg_gp) {}

    void evaluateSGDerivative(
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
        const EpetraExt::ModelEvaluator::SGDerivative& sg_dg_dp) {}

    void evaluateMPResponse(
        const double curr_time,
        const Stokhos::ProductEpetraVector* mp_xdot,
        const Stokhos::ProductEpetraVector* mp_xdotdot,
        const Stokhos::ProductEpetraVector& mp_x,
        const Teuchos::Array<ParamVec>& p,
        const Teuchos::Array<int>& mp_p_index,
        const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals,
        Stokhos::ProductEpetraVector& mp_g) {}

    void evaluateMPTangent(
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
        Stokhos::ProductEpetraMultiVector* mp_gp) {}

    void evaluateMPDerivative(
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
        const EpetraExt::ModelEvaluator::MPDerivative& mp_dg_dp) {}
    
    void evaluateSGGradient(
        const double current_time,
        const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
        const Stokhos::EpetraVectorOrthogPoly* sg_xdotdot,
        const Stokhos::EpetraVectorOrthogPoly& sg_x,
        const Teuchos::Array<ParamVec>& p,
        const Teuchos::Array<int>& sg_p_index,
        const Teuchos::Array< Teuchos::Array<SGType> >& sg_p_vals,
        ParamVec* deriv_p,
        Stokhos::EpetraVectorOrthogPoly* sg_g,
        Stokhos::EpetraMultiVectorOrthogPoly* sg_dg_dx,
        Stokhos::EpetraMultiVectorOrthogPoly* sg_dg_dxdot,
        Stokhos::EpetraMultiVectorOrthogPoly* sg_dg_dxdotdot,
        Stokhos::EpetraMultiVectorOrthogPoly* sg_dg_dp) {}

    void evaluateMPGradient(
        const double current_time,
        const Stokhos::ProductEpetraVector* mp_xdot,
        const Stokhos::ProductEpetraVector* mp_xdotdot,
        const Stokhos::ProductEpetraVector& mp_x,
        const Teuchos::Array<ParamVec>& p,
        const Teuchos::Array<int>& mp_p_index,
        const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals,
        ParamVec* deriv_p,
        Stokhos::ProductEpetraVector* mp_g,
        Stokhos::ProductEpetraMultiVector* mp_dg_dx,
        Stokhos::ProductEpetraMultiVector* mp_dg_dxdot,
        Stokhos::ProductEpetraMultiVector* mp_dg_dxdotdot,
        Stokhos::ProductEpetraMultiVector* mp_dg_dp) {}
#endif

  protected:

    Teuchos::RCP<Albany::Application> application;
    Teuchos::RCP<Albany::AbstractProblem> problem;
    Teuchos::RCP<Albany::MeshSpecsStruct> meshSpecs;
    Teuchos::RCP<Albany::StateManager> stateMgr;
    Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> > rfm;
    Teuchos::ParameterList responseParams;

  private:

    /* private to prohibit copying */
    AdjointResponseFunction(const AdjointResponseFunction&);
    AdjointResponseFunction& operator=(const AdjointResponseFunction&);

};

}

#endif
