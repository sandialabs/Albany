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
  ///\brief Isotropic MeshSizeField
  ///
  /// This evaluator computes the MeshSizeField of the current elements in the mesh
  /// 
  ///
  template<typename EvalT, typename Traits>
  class IsoMeshSizeField : public PHX::EvaluatorWithBaseImpl<Traits>,
                 public PHX::EvaluatorDerived<EvalT, Traits>  {

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

  private:

    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;

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

    Teuchos::RCP<Intrepid::Cubature<RealType> > cubature;
    Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > intrepidBasis;

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

  };


  ///\brief Anisotropic MeshSizeField
  ///
  /// This evaluator computes the MeshSizeField of the current elements in the mesh
  /// 
  ///
  template<typename EvalT, typename Traits>
  class AnisoMeshSizeField : public PHX::EvaluatorWithBaseImpl<Traits>,
                 public PHX::EvaluatorDerived<EvalT, Traits>  {

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

  private:

    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;

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

    Teuchos::RCP<Intrepid::Cubature<RealType> > cubature;
    Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > intrepidBasis;

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

  };
}

#endif
