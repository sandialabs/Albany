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


#ifndef ALBANY_FIELD_MANAGER_SCALAR_RESPONSE_FUNCTION_HPP
#define ALBANY_FIELD_MANAGER_SCALAR_RESPONSE_FUNCTION_HPP

#include "Albany_SamplingBasedScalarResponseFunction.hpp"
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
    public SamplingBasedScalarResponseFunction {
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
    evaluateResponse(const double current_time,
		     const Epetra_Vector* xdot,
		     const Epetra_Vector& x,
		     const Teuchos::Array<ParamVec>& p,
		     Epetra_Vector& g);

    //! Evaluate tangent = dg/dx*dx/dp + dg/dxdot*dxdot/dp + dg/dp
    virtual void 
    evaluateTangent(const double alpha, 
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
    virtual void 
    evaluateGradient(const double current_time,
		     const Epetra_Vector* xdot,
		     const Epetra_Vector& x,
		     const Teuchos::Array<ParamVec>& p,
		     ParamVec* deriv_p,
		     Epetra_Vector* g,
		     Epetra_MultiVector* dg_dx,
		     Epetra_MultiVector* dg_dxdot,
		     Epetra_MultiVector* dg_dp);

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
      *out << "Process using 'dot -Tpng -O ' " << detail_name << "\n" << endl;
      rfm->writeGraphvizFile<EvalT>(detail_name,detail,detail);
      first = false;
    }
  }

}

#endif // ALBANY_FIELD_MANAGER_SCALAR_RESPONSE_FUNCTION_HPP
