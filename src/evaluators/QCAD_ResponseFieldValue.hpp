//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef QCAD_RESPONSEFIELDVALUE_HPP
#define QCAD_RESPONSEFIELDVALUE_HPP

#include "QCAD_MeshRegion.hpp"
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

    Teuchos::Array<int> field_components;

  private:

    Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > nodeID;
    int numNodes;
  };

  template<typename Traits>
  class FieldValueScatterScalarResponse<PHAL::AlbanyTraits::DistParamDeriv,Traits> :
    public PHAL::ScatterScalarResponseBase<PHAL::AlbanyTraits::DistParamDeriv,Traits> {

  public:
    typedef PHAL::AlbanyTraits::DistParamDeriv EvalT;
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

    Teuchos::Array<int> field_components;

  private:

    Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > nodeID;
    int numNodes;
  };

#ifdef ALBANY_SG_MP
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

    Teuchos::Array<int> field_components;

  private:

    Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > nodeID;
    int numNodes;
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

    Teuchos::Array<int> field_components;

  private:

    Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > nodeID;
    int numNodes;
  };
#endif //ALBANY_SG_MP

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
    //Teuchos::Array<int> field_components;

    bool bOpFieldIsVector, bRetFieldIsVector;

    std::string operation;
    std::string opFieldName;
    std::string retFieldName;

    bool bReturnOpField;
    bool opX, opY, opZ;

    Teuchos::RCP< MeshRegion<EvalT, Traits> > opRegion;

    Teuchos::Array<double> initVals;

    //! Material database
    Teuchos::RCP<QCAD::MaterialDatabase> materialDB;
  };

}

#endif
