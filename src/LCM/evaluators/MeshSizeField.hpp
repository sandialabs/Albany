//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_MeshSizeField_hpp)
#define LCM_MeshSizeField_hpp

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"
#include "Intrepid_Cubature.hpp"
#include "Intrepid_Basis.hpp"


namespace LCM {
  ///\brief MeshSizeField
  ///
  /// This evaluator computes the MeshSizeField of the current elements in the mesh
  /// 
  ///
  template<typename EvalT, typename Traits>
  class MeshSizeFieldBase : public PHX::EvaluatorWithBaseImpl<Traits>,
                 public PHX::EvaluatorDerived<EvalT, Traits>  {

  public:

    ///
    /// Constructor
    ///
    MeshSizeFieldBase(const Teuchos::RCP<Albany::Layouts>& dl){
       field_tag_ = Teuchos::rcp(
          new PHX::Tag<typename EvalT::ScalarT>("Mesh Nodal Size Field",
                                            dl->dummy));
       this->addEvaluatedField(*field_tag_);
    }

    Teuchos::RCP<const PHX::FieldTag> getEvaluatedFieldTag() const
    { return field_tag_; }
    Teuchos::RCP<const PHX::FieldTag> getResponseFieldTag() const
    { return field_tag_; }

  private:

    Teuchos::RCP< PHX::Tag<typename EvalT::ScalarT>> field_tag_; 

  };

  ///\brief Isotropic MeshSizeField
  ///
  /// This evaluator computes the MeshSizeField of the current elements in the mesh
  /// 
  ///

  // Generic template signature
  template<typename EvalT, typename Traits>
  class IsoMeshSizeField : public MeshSizeFieldBase<EvalT, Traits> {

  public:

    ///
    /// Constructor
    ///
    IsoMeshSizeField(const Teuchos::ParameterList& p,
           const Teuchos::RCP<Albany::Layouts>& dl)
           : MeshSizeFieldBase<EvalT, Traits> (dl) 
    {}

    ///
    /// Phalanx method to allocate space
    ///
    void postRegistrationSetup(typename Traits::SetupData d,
                               PHX::FieldManager<Traits>& vm){}

    ///
    /// Implementation of physics
    ///
    void evaluateFields(typename Traits::EvalData d){}

    ///
    /// Called after evaluation
    ///
    void postEvaluate(typename Traits::PostEvalData d){}
  };

  // The residual specialization of the MeshSizeField
  template<typename Traits>
  class IsoMeshSizeField<PHAL::AlbanyTraits::Residual, Traits>  : 
                 public MeshSizeFieldBase<PHAL::AlbanyTraits::Residual, Traits> {

  public:

    ///
    /// Constructor
    ///
    IsoMeshSizeField(const Teuchos::ParameterList& p,
           const Teuchos::RCP<Albany::Layouts>& dl);

    ///
    /// Phalanx method to allocate space
    ///
    void postRegistrationSetup(typename Traits::SetupData d,
                               PHX::FieldManager<Traits>& vm);

    ///
    /// Implementation of physics
    ///
    void evaluateFields(typename Traits::EvalData d);

    ///
    /// Called after evaluation
    ///
    void postEvaluate(typename Traits::PostEvalData d){
        if(adapt_PL->get<int>("LastIter", 0) == 3)
        adapt_PL->set<bool>("AdaptNow", true);
    }

  private:

    typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
    typedef typename PHAL::AlbanyTraits::Residual::MeshScalarT MeshScalarT;

    ///
    /// Input: current coordinates of the nodes
    ///
    PHX::MDField<ScalarT,Cell,Vertex,Dim> currentCoords;

    ///
    /// Output: MeshSizeField (isotropic scalar)
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint> isoMeshSizeField;

    // Temporary FieldContainers
    Intrepid::FieldContainer<RealType> grad_at_cub_points;
    Intrepid::FieldContainer<RealType> refPoints;
    Intrepid::FieldContainer<RealType> refWeights;
    Intrepid::FieldContainer<ScalarT> dxdxi;
    Intrepid::FieldContainer<ScalarT> dEDdxi;

    Teuchos::RCP<Intrepid::Cubature<RealType>> cubature;
    Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType>>> intrepidBasis;

    ///
    /// Number of integration points
    ///
    unsigned int numQPs;

    ///
    /// Number of problem dimensions
    ///
    unsigned int numDims;

    // Number of nodes in the element
    unsigned int numNodes;

    // pointer to the Adaptation PL
    //
    Teuchos::ParameterList* adapt_PL;

  };


  ///\brief Anisotropic MeshSizeField
  ///
  /// This evaluator computes the MeshSizeField of the current elements in the mesh
  /// 
  ///
  // Generic template instance
  template<typename EvalT, typename Traits>
  class AnisoMeshSizeField : public MeshSizeFieldBase<EvalT, Traits> {

  public:

    ///
    /// Constructor
    ///
    AnisoMeshSizeField(const Teuchos::ParameterList& p,
           const Teuchos::RCP<Albany::Layouts>& dl)
           : MeshSizeFieldBase<EvalT, Traits> (dl) 
    {}

    ///
    /// Phalanx method to allocate space
    ///
    void postRegistrationSetup(typename Traits::SetupData d,
                               PHX::FieldManager<Traits>& vm){}

    ///
    /// Implementation of physics
    ///
    void evaluateFields(typename Traits::EvalData d){}

    ///
    /// Called after evaluation
    ///
    void postEvaluate(typename Traits::PostEvalData d){}

  };

  // The residual specialization of the MeshSizeField
  template<typename Traits>
  class AnisoMeshSizeField<PHAL::AlbanyTraits::Residual, Traits>  : 
                 public MeshSizeFieldBase<PHAL::AlbanyTraits::Residual, Traits> {

  public:

    ///
    /// Constructor
    ///
    AnisoMeshSizeField(const Teuchos::ParameterList& p,
           const Teuchos::RCP<Albany::Layouts>& dl);

    ///
    /// Phalanx method to allocate space
    ///
    void postRegistrationSetup(typename Traits::SetupData d,
                               PHX::FieldManager<Traits>& vm);

    ///
    /// Implementation of physics
    ///
    void evaluateFields(typename Traits::EvalData d);

    ///
    /// Called after evaluation
    ///
    void postEvaluate(typename Traits::PostEvalData d){
        adapt_PL->set<bool>("AdaptNow", true);
    }

  private:

    typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
    typedef typename PHAL::AlbanyTraits::Residual::MeshScalarT MeshScalarT;

    ///
    /// Input: current coordinates of the nodes
    ///
    PHX::MDField<ScalarT,Cell,Vertex,Dim> currentCoords;

    ///
    /// Output: MeshSizeField (anisotropic scalar)
    ///
    PHX::MDField<ScalarT,Cell,QuadPoint,Dim> anisoMeshSizeField;

    // Temporary FieldContainers
    Intrepid::FieldContainer<RealType> grad_at_cub_points;
    Intrepid::FieldContainer<RealType> refPoints;
    Intrepid::FieldContainer<RealType> refWeights;
    Intrepid::FieldContainer<ScalarT> dxdxi;

    Teuchos::RCP<Intrepid::Cubature<RealType>> cubature;
    Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType>>> intrepidBasis;

    ///
    /// Number of integration points
    ///
    unsigned int numQPs;

    ///
    /// Number of problem dimensions
    ///
    unsigned int numDims;

    // Number of nodes in the element
    unsigned int numNodes;

    // pointer to the Adaptation PL
    //
    Teuchos::ParameterList* adapt_PL;

  };
}

#endif
