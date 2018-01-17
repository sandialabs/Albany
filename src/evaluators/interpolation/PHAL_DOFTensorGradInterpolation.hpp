//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_DOFTENSORGRAD_INTERPOLATION_HPP
#define PHAL_DOFTENSORGRAD_INTERPOLATION_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"

namespace PHAL {
/** \brief Finite Element InterpolationBase Evaluator

    This evaluator interpolates nodal DOFTensor values to their
    gradients at quad points.

*/

template<typename EvalT, typename Traits, typename ScalarT>
class DOFTensorGradInterpolationBase : public PHX::EvaluatorWithBaseImpl<Traits>,
                                       public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  DOFTensorGradInterpolationBase(const Teuchos::ParameterList& p,
                                 const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                             PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

protected:

  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  //! Values at nodes
  PHX::MDField<const ScalarT,Cell,Node,VecDim,VecDim> val_node;
  //! Basis Functions
  PHX::MDField<const MeshScalarT,Cell,Node,QuadPoint,Dim> GradBF;

  // Output:
  //! Values at quadrature points
  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim,VecDim,Dim> grad_val_qp;

  std::size_t numNodes;
  std::size_t numQPs;
  std::size_t numDims;
  std::size_t vecDim;

};

/** \brief Fast Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to their gradients at quad points.
    It is an optimized version of DOFTensorGradInterpolationBase that exploits the sparsity pattern of the derivatives in the Jacobian evaluation
    WARNING: it does not work for general fields: it works when the field to be interpolated is the solution
             or a part (a few contiguous components) of the solution
             It does not work when the mesh coordinates are of type ScalarT
*/

template<typename EvalT, typename Traits, typename ScalarT>
class FastSolutionTensorGradInterpolationBase : public DOFTensorGradInterpolationBase<EvalT, Traits, ScalarT>
{
public:

  FastSolutionTensorGradInterpolationBase(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl)
    : DOFTensorGradInterpolationBase<EvalT, Traits,  ScalarT>(p, dl) {
    this->setName("FastSolutionTensorGradInterpolationBase"+PHX::typeAsString<EvalT>());
  };

  void postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& vm) {
    DOFTensorGradInterpolationBase<EvalT, Traits,  ScalarT>::postRegistrationSetup(d, vm);
  }

  void evaluateFields(typename Traits::EvalData d) {
    DOFTensorGradInterpolationBase<EvalT, Traits,  ScalarT>::evaluateFields(d);
  }
};


//! Specialization for Jacobian evaluation taking advantage of known sparsity
#ifndef ALBANY_MESH_DEPENDS_ON_SOLUTION
template<typename Traits>
class FastSolutionTensorGradInterpolationBase<PHAL::AlbanyTraits::Jacobian, Traits, typename PHAL::AlbanyTraits::Jacobian::ScalarT>
  : public DOFTensorGradInterpolationBase<PHAL::AlbanyTraits::Jacobian, Traits, typename PHAL::AlbanyTraits::Jacobian::ScalarT>
{
public:

  FastSolutionTensorGradInterpolationBase(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
    : DOFTensorGradInterpolationBase<PHAL::AlbanyTraits::Jacobian, Traits,  typename PHAL::AlbanyTraits::Jacobian::ScalarT>(p, dl) {
    this->setName("FastSolutionTensorGradInterpolationBase"+PHX::typeAsString<PHAL::AlbanyTraits::Jacobian>());
    offset = p.get<int>("Offset of First DOF");
  };

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm) {
    DOFTensorGradInterpolationBase<PHAL::AlbanyTraits::Jacobian, Traits,  typename PHAL::AlbanyTraits::Jacobian::ScalarT>
      ::postRegistrationSetup(d, vm);
  }

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
  typedef PHAL::AlbanyTraits::Jacobian::MeshScalarT MeshScalarT;

  std::size_t offset;
};
#endif

// Some shortcut names
template<typename EvalT, typename Traits>
using DOFTensorGradInterpolation = DOFTensorGradInterpolationBase<EvalT,Traits,typename EvalT::ScalarT>;

template<typename EvalT, typename Traits>
using DOFTensorGradInterpolationMesh = DOFTensorGradInterpolationBase<EvalT,Traits,typename EvalT::MeshScalarT>;

template<typename EvalT, typename Traits>
using DOFTensorGradInterpolationParam = DOFTensorGradInterpolationBase<EvalT,Traits,typename EvalT::ParamScalarT>;

template<typename EvalT, typename Traits>
using FastSolutionTensorGradInterpolation = FastSolutionTensorGradInterpolationBase<EvalT,Traits,typename EvalT::ScalarT>;

} // Namespace PHAL

#endif // PHAL_DOFTENSORGRAD_INTERPOLATION_HPP
