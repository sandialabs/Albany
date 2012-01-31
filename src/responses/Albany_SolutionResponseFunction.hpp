/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


#ifndef ALBANY_SOLUTION_RESPONSE_FUNCTION_HPP
#define ALBANY_SOLUTION_RESPONSE_FUNCTION_HPP

#include "Albany_DistributedResponseFunction.hpp"
#include "Albany_Application.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_Array.hpp"

#include "Epetra_Map.h"
#include "Epetra_Import.h"
#include "Epetra_CrsGraph.h"

namespace Albany {

  /*!
   * \brief A response function given by (possibly a portion of) the solution
   */
  class SolutionResponseFunction : public DistributedResponseFunction {
  public:
  
    //! Default constructor
    SolutionResponseFunction(
      const Teuchos::RCP<Albany::Application>& application,
      Teuchos::ParameterList& responseParams);

    //! Destructor
    virtual ~SolutionResponseFunction();

    //! Get the map associate with this response
    virtual Teuchos::RCP<const Epetra_Map> responseMap() const;

    //! Create operator for gradient
    virtual Teuchos::RCP<Epetra_Operator> createGradientOp() const;

    //! \name Deterministic evaluation functions
    //@{

    //! Evaluate responses
    virtual void evaluateResponse(
      const double current_time,
      const Epetra_Vector* xdot,
      const Epetra_Vector& x,
      const Teuchos::Array<ParamVec>& p,
      Epetra_Vector& g);

    //! Evaluate tangent = dg/dx*dx/dp + dg/dxdot*dxdot/dp + dg/dp
    virtual void evaluateTangent(
      const double alpha, 
      const double beta,
      const double current_time,
      bool sum_derivs,
      const Epetra_Vector* xdot,
      const Epetra_Vector& x,
      const Teuchos::Array<ParamVec>& p,
      ParamVec* deriv_p,
      const Epetra_MultiVector* Vxdot,
      const Epetra_MultiVector* Vx,
      const Epetra_MultiVector* Vp,
      Epetra_Vector* g,
      Epetra_MultiVector* gx,
      Epetra_MultiVector* gp);

    //! Evaluate gradient = dg/dx, dg/dxdot, dg/dp
    virtual void evaluateGradient(
      const double current_time,
      const Epetra_Vector* xdot,
      const Epetra_Vector& x,
      const Teuchos::Array<ParamVec>& p,
      ParamVec* deriv_p,
      Epetra_Vector* g,
      Epetra_Operator* dg_dx,
      Epetra_Operator* dg_dxdot,
      Epetra_MultiVector* dg_dp);

    //@}

    //! \name Stochastic Galerkin evaluation functions
    //@{

    //! Intialize stochastic Galerkin method
    virtual void init_sg(
      const Teuchos::RCP<const Stokhos::OrthogPolyBasis<int,double> >& basis,
      const Teuchos::RCP<const Stokhos::Quadrature<int,double> >& quad,
      const Teuchos::RCP<Stokhos::OrthogPolyExpansion<int,double> >& expansion,
      const Teuchos::RCP<const EpetraExt::MultiComm>& multiComm);

    //! Evaluate stochastic Galerkin response
    virtual void evaluateSGResponse(
      const double curr_time,
      const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
      const Stokhos::EpetraVectorOrthogPoly& sg_x,
      const Teuchos::Array<ParamVec>& p,
      const Teuchos::Array<int>& sg_p_index,
      const Teuchos::Array< Teuchos::Array<SGType> >& sg_p_vals,
      Stokhos::EpetraVectorOrthogPoly& sg_g);

    //! Evaluate stochastic Galerkin tangent
    virtual void evaluateSGTangent(
      const double alpha, 
      const double beta, 
      const double current_time,
      bool sum_derivs,
      const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
      const Stokhos::EpetraVectorOrthogPoly& sg_x,
      const Teuchos::Array<ParamVec>& p,
      const Teuchos::Array<int>& sg_p_index,
      const Teuchos::Array< Teuchos::Array<SGType> >& sg_p_vals,
      ParamVec* deriv_p,
      const Epetra_MultiVector* Vx,
      const Epetra_MultiVector* Vxdot,
      const Epetra_MultiVector* Vp,
      Stokhos::EpetraVectorOrthogPoly* sg_g,
      Stokhos::EpetraMultiVectorOrthogPoly* sg_JV,
      Stokhos::EpetraMultiVectorOrthogPoly* sg_gp);

    //! Evaluate stochastic Galerkin derivative
    virtual void evaluateSGGradient(
      const double current_time,
      const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
      const Stokhos::EpetraVectorOrthogPoly& sg_x,
      const Teuchos::Array<ParamVec>& p,
      const Teuchos::Array<int>& sg_p_index,
      const Teuchos::Array< Teuchos::Array<SGType> >& sg_p_vals,
      ParamVec* deriv_p,
      Stokhos::EpetraVectorOrthogPoly* sg_g,
      Stokhos::EpetraOperatorOrthogPoly* sg_dg_dx,
      Stokhos::EpetraOperatorOrthogPoly* sg_dg_dxdot,
      Stokhos::EpetraMultiVectorOrthogPoly* sg_dg_dp);

    //@}

    //! \name Multi-point evaluation functions
    //@{

    //! Evaluate multi-point response functions
    virtual void evaluateMPResponse(
      const double curr_time,
      const Stokhos::ProductEpetraVector* mp_xdot,
      const Stokhos::ProductEpetraVector& mp_x,
      const Teuchos::Array<ParamVec>& p,
      const Teuchos::Array<int>& mp_p_index,
      const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals,
      Stokhos::ProductEpetraVector& mp_g);

    //! Evaluate multi-point tangent
    virtual void evaluateMPTangent(
      const double alpha, 
      const double beta, 
      const double current_time,
      bool sum_derivs,
      const Stokhos::ProductEpetraVector* mp_xdot,
      const Stokhos::ProductEpetraVector& mp_x,
      const Teuchos::Array<ParamVec>& p,
      const Teuchos::Array<int>& mp_p_index,
      const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals,
      ParamVec* deriv_p,
      const Epetra_MultiVector* Vx,
      const Epetra_MultiVector* Vxdot,
      const Epetra_MultiVector* Vp,
      Stokhos::ProductEpetraVector* mp_g,
      Stokhos::ProductEpetraMultiVector* mp_JV,
      Stokhos::ProductEpetraMultiVector* mp_gp);

    //! Evaluate stochastic Galerkin derivative
    virtual void evaluateMPGradient(
      const double current_time,
      const Stokhos::ProductEpetraVector* mp_xdot,
      const Stokhos::ProductEpetraVector& mp_x,
      const Teuchos::Array<ParamVec>& p,
      const Teuchos::Array<int>& mp_p_index,
      const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals,
      ParamVec* deriv_p,
      Stokhos::ProductEpetraVector* mp_g,
      Stokhos::ProductEpetraOperator* mp_dg_dx,
      Stokhos::ProductEpetraOperator* mp_dg_dxdot,
      Stokhos::ProductEpetraMultiVector* mp_dg_dp);

    //@}

  private:

    //! Private to prohibit copying
    SolutionResponseFunction(const SolutionResponseFunction&);
    
    //! Private to prohibit copying
    SolutionResponseFunction& operator=(const SolutionResponseFunction&);

  protected:

    Teuchos::RCP<Epetra_Map> 
    buildCulledMap(const Epetra_Map& x_map, 
		   const Teuchos::Array<int>& keepDOF) const;

    void cullSolution(const Epetra_MultiVector& x, 
		      Epetra_MultiVector& x_culled) const;

  protected:

    //! Epetra map for response
    Teuchos::RCP<const Epetra_Map> culled_map;

    //! Importer mapping between full and culled solution
    Teuchos::RCP<Epetra_Import> importer;

    //! Graph of gradient operator
    Teuchos::RCP<Epetra_CrsGraph> gradient_graph;

  };

}

#endif // ALBANY_SOLUTION_RESPONSE_FUNCTION_HPP
