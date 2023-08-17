//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_DOFVECGRAD_INTERPOLATION_HPP
#define PHAL_DOFVECGRAD_INTERPOLATION_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"
#include "Albany_SacadoTypes.hpp"
#include "PHAL_Utilities.hpp"

namespace PHAL {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOFVec values to their
    gradients at quad points.

*/

template<typename EvalT, typename Traits, typename ScalarT>
class DOFVecGradInterpolationBase : public PHX::EvaluatorWithBaseImpl<Traits>,
                                    public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  DOFVecGradInterpolationBase(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                             PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

protected:

  typedef typename EvalT::MeshScalarT MeshScalarT;
  typedef typename Albany::StrongestScalarType<ScalarT,MeshScalarT>::type OutputScalarT;

  // Input:
  //! Values at nodes
  PHX::MDField<const ScalarT,Cell,Node,VecDim> val_node;
  //! Basis Functions
  PHX::MDField<const MeshScalarT,Cell,Node,QuadPoint,Dim> GradBF;

  // Output:
  //! Values at quadrature points
  PHX::MDField<OutputScalarT,Cell,QuadPoint,VecDim,Dim> grad_val_qp;

  std::size_t numNodes;
  std::size_t numQPs;
  std::size_t numDims;
  std::size_t vecDim;

  MDFieldMemoizer<Traits> memoizer;

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
public:

  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;
  struct DOFVecGradInterpolationBase_Residual_Tag{};
  typedef Kokkos::RangePolicy<ExecutionSpace, DOFVecGradInterpolationBase_Residual_Tag> DOFVecGradInterpolationBase_Residual_Policy;

  KOKKOS_INLINE_FUNCTION
  void operator() (const DOFVecGradInterpolationBase_Residual_Tag& tag, const int& cell) const;

#endif

};

/** \brief Fast Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to their gradients at quad points.
    It is an optimized version of DOFVecGradInterpolationBase that exploits the sparsity pattern of the derivatives in the Jacobian evaluation
    WARNING: it does not work for general fields: it works when the field to be interpolated is the solution
             or a part (a few contiguous components) of the solution
             It does not work when the mesh coordinates are of type ScalarT
*/
template<typename EvalT, typename Traits, typename ScalarT>
class FastSolutionVecGradInterpolationBase : public DOFVecGradInterpolationBase<EvalT, Traits, ScalarT>
{
public:

  FastSolutionVecGradInterpolationBase(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl)
    : DOFVecGradInterpolationBase<EvalT, Traits,  ScalarT>(p, dl) {
    this->setName("FastSolutionVecGradInterpolationBase"+PHX::print<EvalT>());
  };

  void postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& vm) {
    DOFVecGradInterpolationBase<EvalT, Traits,  ScalarT>::postRegistrationSetup(d, vm);
  }

  void evaluateFields(typename Traits::EvalData d) {
    DOFVecGradInterpolationBase<EvalT, Traits,  ScalarT>::evaluateFields(d);
  }
};

//! Specialization for Jacobian evaluation taking advantage of known sparsity
//! This assumes that the Mesh coordinates are not a FAD type

template<typename Traits>
class FastSolutionVecGradInterpolationBase<PHAL::AlbanyTraits::Jacobian, Traits, typename PHAL::AlbanyTraits::Jacobian::ScalarT>
  : public DOFVecGradInterpolationBase<PHAL::AlbanyTraits::Jacobian, Traits, typename PHAL::AlbanyTraits::Jacobian::ScalarT>
{
public:


  FastSolutionVecGradInterpolationBase(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
    : DOFVecGradInterpolationBase<PHAL::AlbanyTraits::Jacobian, Traits,  typename PHAL::AlbanyTraits::Jacobian::ScalarT>(p, dl) {
    this->setName("FastSolutionVecGradInterpolationBase"+PHX::print<PHAL::AlbanyTraits::Jacobian>());
    offset = p.get<int>("Offset of First DOF");
  };

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm) {
    DOFVecGradInterpolationBase<PHAL::AlbanyTraits::Jacobian, Traits,  typename PHAL::AlbanyTraits::Jacobian::ScalarT>
      ::postRegistrationSetup(d, vm);
  }

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
  typedef PHAL::AlbanyTraits::Jacobian::MeshScalarT MeshScalarT;

  std::size_t offset;

//KOKKOS:
 #ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
public:

  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;
  struct FastSolutionVecGradInterpolationBase_Jacobian_Tag{};
  typedef Kokkos::RangePolicy<ExecutionSpace, FastSolutionVecGradInterpolationBase_Jacobian_Tag> FastSolutionVecGradInterpolationBase_Jacobian_Policy;

  KOKKOS_INLINE_FUNCTION
  void operator() (const FastSolutionVecGradInterpolationBase_Jacobian_Tag& tag, const int& cell) const;

  int neq;

#endif
};

// Some shortcut names
template<typename EvalT, typename Traits>
using DOFVecGradInterpolation = DOFVecGradInterpolationBase<EvalT,Traits,typename EvalT::ScalarT>;

template<typename EvalT, typename Traits>
using DOFVecGradInterpolationMesh = DOFVecGradInterpolationBase<EvalT,Traits,typename EvalT::MeshScalarT>;

template<typename EvalT, typename Traits>
using DOFVecGradInterpolationParam = DOFVecGradInterpolationBase<EvalT,Traits,typename EvalT::ParamScalarT>;

template<typename EvalT, typename Traits>
using FastSolutionVecGradInterpolation = FastSolutionVecGradInterpolationBase<EvalT,Traits,typename EvalT::ScalarT>;
} // Namespace PHAL

#endif // PHAL_DOFVECGRAD_INTERPOLATION_HPP
