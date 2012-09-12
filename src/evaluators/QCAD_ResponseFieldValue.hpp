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


#ifndef QCAD_RESPONSEFIELDVALUE_HPP
#define QCAD_RESPONSEFIELDVALUE_HPP

#include "QCAD_MaterialDatabase.hpp"
#include "PHAL_ScatterScalarResponse.hpp"

namespace QCAD {

  template<typename EvalT, typename Traits> 
  class FieldValueScatterScalarResponse : 
    public PHAL::ScatterScalarResponse<EvalT,Traits>  {
  
  public:
  
    FieldValueScatterScalarResponse(const Teuchos::ParameterList& p,
			      const Teuchos::RCP<Albany::Layouts>& dl) :
      PHAL::ScatterScalarResponse<EvalT,Traits>(p,dl) {}
  
  protected:

    // Default constructor for child classes
    FieldValueScatterScalarResponse() :
      PHAL::ScatterScalarResponse<EvalT,Traits>() {}

    // Child classes should call setup once p is filled out
    void setup(const Teuchos::ParameterList& p,
	       const Teuchos::RCP<Albany::Layouts>& dl) {
      PHAL::ScatterScalarResponse<EvalT,Traits>::setup(p,dl);
    }

    // Set NodeID structure for cell corrsponding to max/min
    void setNodeID(const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >&) {}

    Teuchos::Array<int> field_components;
  };

  template<typename Traits> 
  class FieldValueScatterScalarResponse<PHAL::AlbanyTraits::Jacobian,Traits> : 
    public PHAL::ScatterScalarResponseBase<PHAL::AlbanyTraits::Jacobian,Traits> {
  
  public:
    typedef PHAL::AlbanyTraits::Jacobian EvalT;
    typedef typename EvalT::ScalarT ScalarT;

    FieldValueScatterScalarResponse(const Teuchos::ParameterList& p,
			      const Teuchos::RCP<Albany::Layouts>& dl) :
      PHAL::ScatterScalarResponseBase<EvalT,Traits>(p,dl) {}

    void preEvaluate(typename Traits::PreEvalData d) {}
    void evaluateFields(typename Traits::EvalData d) {}
    void postEvaluate(typename Traits::PostEvalData d);
  
  protected:

    // Default constructor for child classes
    FieldValueScatterScalarResponse() :
      PHAL::ScatterScalarResponseBase<EvalT,Traits>() {}

    // Child classes should call setup once p is filled out
    void setup(const Teuchos::ParameterList& p,
	       const Teuchos::RCP<Albany::Layouts>& dl) {
      PHAL::ScatterScalarResponseBase<EvalT,Traits>::setup(p,dl);
      numNodes = dl->node_scalar->dimension(1);
    }

    // Set NodeID structure for cell corrsponding to max/min
    void setNodeID(const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID_) {
      nodeID = nodeID_;
    }

  private:

    Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > nodeID;
    int numNodes;
    Teuchos::Array<int> field_components;
  };

  template<typename Traits> 
  class FieldValueScatterScalarResponse<PHAL::AlbanyTraits::SGJacobian,Traits> : 
    public PHAL::ScatterScalarResponseBase<PHAL::AlbanyTraits::SGJacobian,Traits> {
  
  public:
    typedef PHAL::AlbanyTraits::SGJacobian EvalT;
    typedef typename EvalT::ScalarT ScalarT;

    FieldValueScatterScalarResponse(const Teuchos::ParameterList& p,
			      const Teuchos::RCP<Albany::Layouts>& dl) :
      PHAL::ScatterScalarResponseBase<EvalT,Traits>(p,dl) {}

    void preEvaluate(typename Traits::PreEvalData d) {}
    void evaluateFields(typename Traits::EvalData d) {}
    void postEvaluate(typename Traits::PostEvalData d);
  
  protected:

    // Default constructor for child classes
    FieldValueScatterScalarResponse() :
      PHAL::ScatterScalarResponseBase<EvalT,Traits>() {}

    // Child classes should call setup once p is filled out
    void setup(const Teuchos::ParameterList& p,
	       const Teuchos::RCP<Albany::Layouts>& dl) {
      PHAL::ScatterScalarResponseBase<EvalT,Traits>::setup(p,dl);
      numNodes = dl->node_scalar->dimension(1);
    }

    // Set NodeID structure for cell corrsponding to max/min
    void setNodeID(const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID_) {
      nodeID = nodeID_;
    }

  private:

    Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > nodeID;
    int numNodes;
    Teuchos::Array<int> field_components;
  };

  template<typename Traits> 
  class FieldValueScatterScalarResponse<PHAL::AlbanyTraits::MPJacobian,Traits> : 
    public PHAL::ScatterScalarResponseBase<PHAL::AlbanyTraits::MPJacobian,Traits> {
  
  public:
    typedef PHAL::AlbanyTraits::MPJacobian EvalT;
    typedef typename EvalT::ScalarT ScalarT;

    FieldValueScatterScalarResponse(const Teuchos::ParameterList& p,
			      const Teuchos::RCP<Albany::Layouts>& dl) :
      PHAL::ScatterScalarResponseBase<EvalT,Traits>(p,dl) {}

    void preEvaluate(typename Traits::PreEvalData d) {}
    void evaluateFields(typename Traits::EvalData d) {}
    void postEvaluate(typename Traits::PostEvalData d);
  
  protected:

    // Default constructor for child classes
    FieldValueScatterScalarResponse() :
      PHAL::ScatterScalarResponseBase<EvalT,Traits>() {}

    // Child classes should call setup once p is filled out
    void setup(const Teuchos::ParameterList& p,
	       const Teuchos::RCP<Albany::Layouts>& dl) {
      PHAL::ScatterScalarResponseBase<EvalT,Traits>::setup(p,dl);
      numNodes = dl->node_scalar->dimension(1);
    }

    // Set NodeID structure for cell corrsponding to max/min
    void setNodeID(const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID_) {
      nodeID = nodeID_;
    }

  private:

    Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > nodeID;
    int numNodes;
    Teuchos::Array<int> field_components;
  };

/** 
 * \brief Response Description
 */
  template<typename EvalT, typename Traits>
  class ResponseFieldValue : 
    public FieldValueScatterScalarResponse<EvalT, Traits>
  {
  public:
    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;
    
    ResponseFieldValue(Teuchos::ParameterList& p,
		       const Teuchos::RCP<Albany::Layouts>& dl);
  
    void postRegistrationSetup(typename Traits::SetupData d,
			       PHX::FieldManager<Traits>& vm);
  
    void preEvaluate(typename Traits::PreEvalData d);
  
    void evaluateFields(typename Traits::EvalData d);

    void postEvaluate(typename Traits::PostEvalData d);
	  
  private:
    Teuchos::RCP<const Teuchos::ParameterList> getValidResponseParameters() const;

    std::size_t numQPs;
    std::size_t numDims;
    
    PHX::MDField<ScalarT> opField;
    PHX::MDField<ScalarT> retField;
    PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim> coordVec;
    PHX::MDField<MeshScalarT,Cell,QuadPoint> weights;
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > max_nodeID;
    Teuchos::Array<int> field_components;
    
    bool bOpFieldIsVector, bRetFieldIsVector;

    std::string operation;
    std::string opFieldName;
    std::string retFieldName;
    std::string opDomain;
    std::vector<std::string> ebNames;
    bool bQuantumEBsOnly;

    bool bReturnOpField;
    bool opX, opY, opZ;
    bool limitX, limitY, limitZ;
    double xmin, xmax, ymin, ymax, zmin, zmax;

    Teuchos::Array<double> initVals;

    //! Material database
    Teuchos::RCP<QCAD::MaterialDatabase> materialDB;
  };
	
}

#endif
