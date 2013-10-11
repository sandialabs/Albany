//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#if !defined(LCM_SurfaceScalarResidual_hpp)
#define LCM_SurfaceScalarResidual_hpp

#include <Phalanx_ConfigDefs.hpp>
#include <Phalanx_Evaluator_WithBaseImpl.hpp>
#include <Phalanx_Evaluator_Derived.hpp>
#include <Phalanx_MDField.hpp>
#include <Intrepid_CellTools.hpp>
#include <Intrepid_Cubature.hpp>

#include "Albany_Layouts.hpp"

namespace LCM
{
  /// \brief
  ///
  /// Compute the scalar residual on a surface
  ///
  template<typename EvalT, typename Traits>
  class SurfaceScalarResidual: public PHX::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalT, Traits>
  {

  public:

    ///
    /// Constructor
    ///
    SurfaceScalarResidual(Teuchos::ParameterList& p,
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
    /// Length scale parameter for localization zone
    ///
    ScalarT thickness_;
    
    ///
    /// Numerical integration rule
    ///
    Teuchos::RCP<Intrepid::Cubature<RealType> > cubature_;

    ///
    /// Finite element basis for the midplane
    ///
    Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > intrepid_basis_;

    ///
    /// Reference configuration dual basis
    ///
    PHX::MDField<ScalarT, Cell, QuadPoint, Dim, Dim> ref_dual_basis_;

    ///
    /// Reference configuration normal
    ///
    PHX::MDField<ScalarT, Cell, QuadPoint, Dim> ref_normal_;

    ///
    /// Reference configuration area
    ///
    PHX::MDField<ScalarT, Cell, QuadPoint, Dim> ref_area_;

    /// Reference Cell FieldContainers
    Intrepid::FieldContainer<RealType> ref_values_;
    Intrepid::FieldContainer<RealType> ref_grads_;
    Intrepid::FieldContainer<RealType> ref_points_;
    Intrepid::FieldContainer<RealType> ref_weights_;

    ///
    /// Output
    ///
    PHX::MDField<ScalarT, Cell, Node, Dim> residual_;

    unsigned int num_nodes_;
    unsigned int num_pts_;
    unsigned int num_dims_;
    unsigned int num_plane_nodes_;
    unsigned int num_plane_dims_;;
  };
}

#endif
