//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_FIELD_MANAGER_SCALAR_RESPONSE_FUNCTION_HPP
#define ALBANY_FIELD_MANAGER_SCALAR_RESPONSE_FUNCTION_HPP

#include "Albany_ScalarResponseFunction.hpp"
#include "Albany_Application.hpp"
#include "Albany_AbstractProblem.hpp"
#include "Albany_StateManager.hpp"
#include "Albany_StateInfoStruct.hpp" // contains MeshSpecsStuct
#include "PHAL_AlbanyTraits.hpp"
#include "Phalanx.hpp"

namespace Albany {

  /*!
   * \brief Reponse function representing the average of the solution values
   */
  class FieldManagerScalarResponseFunction : 
    public ScalarResponseFunction {
  public:
  
    //! Constructor
    FieldManagerScalarResponseFunction(
      const Teuchos::RCP<Albany::Application>& application,
      const Teuchos::RCP<Albany::AbstractProblem>& problem,
      const Teuchos::RCP<Albany::MeshSpecsStruct>&  ms,
      const Teuchos::RCP<Albany::StateManager>& stateMgr,
      Teuchos::ParameterList& responseParams);

    //! Destructor
    virtual ~FieldManagerScalarResponseFunction();

    //! Get the number of responses
    virtual unsigned int numResponses() const;

    //! Evaluate responses
    virtual void 
    evaluateResponseT(const double current_time,
		     const Tpetra_Vector* xdotT,
		     const Tpetra_Vector* xdotdotT,
		     const Tpetra_Vector& xT,
		     const Teuchos::Array<ParamVec>& p,
		     Tpetra_Vector& gT);
   
    //! Evaluate tangent = dg/dx*dx/dp + dg/dxdot*dxdot/dp + dg/dp
    virtual void 
    evaluateTangentT(const double alpha, 
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

#if defined(ALBANY_EPETRA)
    //! Evaluate gradient = dg/dx, dg/dxdot, dg/dp
    virtual void 
    evaluateGradient(const double current_time,
		     const Epetra_Vector* xdot,
		     const Epetra_Vector* xdotdot,
		     const Epetra_Vector& x,
		     const Teuchos::Array<ParamVec>& p,
		     ParamVec* deriv_p,
		     Epetra_Vector* g,
		     Epetra_MultiVector* dg_dx,
		     Epetra_MultiVector* dg_dxdot,
		     Epetra_MultiVector* dg_dxdotdot,
		     Epetra_MultiVector* dg_dp);
#endif
    
    virtual void 
    evaluateGradientT(const double current_time,
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

#if defined(ALBANY_EPETRA)
    //! Evaluate distributed parameter derivative dg/dp, in MultiVector form
    virtual void
    evaluateDistParamDeriv(
          const double current_time,
          const Epetra_Vector* xdot,
          const Epetra_Vector* xdotdot,
          const Epetra_Vector& x,
          const Teuchos::Array<ParamVec>& param_array,
          const std::string& dist_param_name,
          Epetra_MultiVector* dg_dp);
#endif

    //! \name Stochastic Galerkin evaluation functions
    //@{

#ifdef ALBANY_SG
    //! Intialize stochastic Galerkin method
    virtual void init_sg(
      const Teuchos::RCP<const Stokhos::OrthogPolyBasis<int,double> >& basis,
      const Teuchos::RCP<const Stokhos::Quadrature<int,double> >& quad,
      const Teuchos::RCP<Stokhos::OrthogPolyExpansion<int,double> >& expansion,
      const Teuchos::RCP<const EpetraExt::MultiComm>& multiComm) {}

    //! Evaluate stochastic Galerkin response
    virtual void evaluateSGResponse(
      const double curr_time,
      const Stokhos::EpetraVectorOrthogPoly* sg_xdot,
      const Stokhos::EpetraVectorOrthogPoly* sg_xdotdot,
      const Stokhos::EpetraVectorOrthogPoly& sg_x,
      const Teuchos::Array<ParamVec>& p,
      const Teuchos::Array<int>& sg_p_index,
      const Teuchos::Array< Teuchos::Array<SGType> >& sg_p_vals,
      Stokhos::EpetraVectorOrthogPoly& sg_g);

    //! Evaluate stochastic Galerkin tangent
    virtual void evaluateSGTangent(
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

    //! Evaluate stochastic Galerkin derivative
    virtual void evaluateSGGradient(
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
      Stokhos::EpetraMultiVectorOrthogPoly* sg_dg_dp);
#endif

    //@}

    //! \name Multi-point evaluation functions
    //@{

#ifdef ALBANY_ENSEMBLE
    //! Evaluate multi-point response functions
    virtual void evaluateMPResponse(
      const double curr_time,
      const Stokhos::ProductEpetraVector* mp_xdot,
      const Stokhos::ProductEpetraVector* mp_xdotdot,
      const Stokhos::ProductEpetraVector& mp_x,
      const Teuchos::Array<ParamVec>& p,
      const Teuchos::Array<int>& mp_p_index,
      const Teuchos::Array< Teuchos::Array<MPType> >& mp_p_vals,
      Stokhos::ProductEpetraVector& mp_g);

    //! Evaluate multi-point tangent
    virtual void evaluateMPTangent(
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

    //! Evaluate multi-point derivative
    virtual void evaluateMPGradient(
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
      Stokhos::ProductEpetraMultiVector* mp_dg_dp);
#endif

    //@}

  private:

    //! Private to prohibit copying
    FieldManagerScalarResponseFunction(const FieldManagerScalarResponseFunction&);
    
    //! Private to prohibit copying
    FieldManagerScalarResponseFunction& operator=(const FieldManagerScalarResponseFunction&);

  protected:

    //! Constructor for derived classes
    /*!
     * Derived classes must call setup after using this constructor.
     */
    FieldManagerScalarResponseFunction(
      const Teuchos::RCP<Albany::Application>& application,
      const Teuchos::RCP<Albany::AbstractProblem>& problem,
      const Teuchos::RCP<Albany::MeshSpecsStruct>&  ms,
      const Teuchos::RCP<Albany::StateManager>& stateMgr);

    //! Setup method for derived classes
    void setup(Teuchos::ParameterList& responseParams);

    //! Helper function for visualizing response graph 
    template <typename EvalT> 
    void visResponseGraph(const std::string& res_type);

  protected:

    //! Application class
    Teuchos::RCP<Albany::Application> application;

    //! Problem class
    Teuchos::RCP<Albany::AbstractProblem> problem;

    //! Mesh specs
    Teuchos::RCP<Albany::MeshSpecsStruct> meshSpecs;

    //! State manager
    Teuchos::RCP<Albany::StateManager> stateMgr;

    //! Field manager for Responses 
    Teuchos::RCP<PHX::FieldManager<PHAL::AlbanyTraits> > rfm;

    //! Number of responses we compute
    unsigned int num_responses;

    //! Visualize response graph
    int vis_response_graph;

    //! Response name for visualization file
    std::string vis_response_name;

  private:

    template <typename EvalT> void evaluate(PHAL::Workset& workset);

    //! Restrict the field manager to an element block, as is done for fm and
    //! sfm in Albany::Application.
    int element_block_index;

  };

  template <typename EvalT> 
  void
  Albany::FieldManagerScalarResponseFunction::
  visResponseGraph(const std::string& res_type) {
    // Only write out the graph file first time function is called
    static bool first = true;
    if (first && vis_response_graph > 0) {
      bool detail = false; if (vis_response_graph > 1) detail=true;
      Teuchos::RCP<Teuchos::FancyOStream> out = 
	Teuchos::VerboseObjectBase::getDefaultOStream();
      *out << "Phalanx writing graphviz file for graph of Response fill "
	   << "(detail = "<< vis_response_graph << ")"<< std::endl;
      std::string detail_name = 
	"responses_graph_" + vis_response_name + res_type;
      *out << "Process using 'dot -Tpng -O ' " << detail_name << "\n" << std::endl;
      rfm->writeGraphvizFile<EvalT>(detail_name,detail,detail);
      first = false;
    }
  }

}

#endif // ALBANY_FIELD_MANAGER_SCALAR_RESPONSE_FUNCTION_HPP
