//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
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

    //! Perform post registration setup
    void postRegSetup();

    //! Evaluate responses
    virtual void 
    evaluateResponse(const double current_time,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
		  const Teuchos::Array<ParamVec>& p,
      const Teuchos::RCP<Thyra_Vector>& g);
   
    //! Evaluate tangent = dg/dx*dx/dp + dg/dxdot*dxdot/dp + dg/dp
    virtual void 
    evaluateTangent(const double alpha, 
		  const double beta,
		  const double omega,
		  const double current_time,
		  bool sum_derivs,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
		  const Teuchos::Array<ParamVec>& p,
		  ParamVec* deriv_p,
      const Teuchos::RCP<const Thyra_MultiVector>& Vx,
      const Teuchos::RCP<const Thyra_MultiVector>& Vxdot,
      const Teuchos::RCP<const Thyra_MultiVector>& Vxdotdot,
      const Teuchos::RCP<const Thyra_MultiVector>& Vp,
      const Teuchos::RCP<Thyra_Vector>& g,
      const Teuchos::RCP<Thyra_MultiVector>& gx,
      const Teuchos::RCP<Thyra_MultiVector>& gp);
    
    virtual void 
    evaluateGradient(const double current_time,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
		  const Teuchos::Array<ParamVec>& p,
		  ParamVec* deriv_p,
      const Teuchos::RCP<Thyra_Vector>& g,
      const Teuchos::RCP<Thyra_MultiVector>& dg_dx,
      const Teuchos::RCP<Thyra_MultiVector>& dg_dxdot,
      const Teuchos::RCP<Thyra_MultiVector>& dg_dxdotdot,
      const Teuchos::RCP<Thyra_MultiVector>& dg_dp);

    //! Evaluate distributed parameter derivative dg/dp, in MultiVector form
    virtual void
    evaluateDistParamDeriv(
      const double current_time,
      const Teuchos::RCP<const Thyra_Vector>& x,
      const Teuchos::RCP<const Thyra_Vector>& xdot,
      const Teuchos::RCP<const Thyra_Vector>& xdotdot,
      const Teuchos::Array<ParamVec>& param_array,
      const std::string& dist_param_name,
      const Teuchos::RCP<Thyra_MultiVector>& dg_dp);

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

    bool performedPostRegSetup;
  };

  template <typename EvalT> 
  void
  FieldManagerScalarResponseFunction::
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

} // namespace Albany

#endif // ALBANY_FIELD_MANAGER_SCALAR_RESPONSE_FUNCTION_HPP
