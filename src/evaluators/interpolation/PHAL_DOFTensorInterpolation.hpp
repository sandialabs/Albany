//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_DOFTENSOR_INTERPOLATION_HPP
#define PHAL_DOFTENSOR_INTERPOLATION_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"

namespace PHAL {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOFTensor values to quad points.

*/

template<typename EvalT, typename Traits, typename ScalarT>
class DOFTensorInterpolationBase : public PHX::EvaluatorWithBaseImpl<Traits>,
                                   public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  DOFTensorInterpolationBase(const Teuchos::ParameterList& p,
                             const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                             PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

protected:

  // Input:
  //! Values at nodes
  PHX::MDField<const ScalarT,Cell,Node,VecDim,VecDim> val_node;

  //! Basis Functions
  typedef typename EvalT::MeshScalarT MeshScalarT;
  PHX::MDField<const MeshScalarT,Cell,Node,QuadPoint> BF;

  // Output:
  //! Values at quadrature points
  typedef typename Albany::StrongestScalarType<ScalarT,MeshScalarT>::type OutputScalarT;
  PHX::MDField<OutputScalarT,Cell,QuadPoint,VecDim,VecDim> val_qp;

  std::size_t numNodes;
  std::size_t numQPs;
  std::size_t vecDim;
};

/** \brief Fast Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values at quad points.
    It is an optimized version of DOFTensorInterpolationBase that exploits the sparsity pattern of the derivatives in the Jacobian evaluation
    WARNING: it does not work for general fields: it works when the field to be interpolated is the solution
             or a part (a few contiguous components) of the solution
             It does not work when the mesh coordinates are of type ScalarT
*/
template<typename EvalT, typename Traits, typename ScalarT>
class FastSolutionTensorInterpolationBase : public DOFTensorInterpolationBase<EvalT, Traits, ScalarT>
{
public:

  FastSolutionTensorInterpolationBase(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl)
    : DOFTensorInterpolationBase<EvalT, Traits,  ScalarT>(p, dl) {
    this->setName("FastSolutionTensorInterpolationBase"+PHX::print<EvalT>());
  };

  void postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& vm) {
    DOFTensorInterpolationBase<EvalT, Traits,  ScalarT>::postRegistrationSetup(d, vm);
  }

  void evaluateFields(typename Traits::EvalData d) {
    DOFTensorInterpolationBase<EvalT, Traits,  ScalarT>::evaluateFields(d);
  }
};


//! Specialization for Jacobian evaluation taking advantage of known sparsity
//! This assumes that the mesh coordinates are not a FAD type
template<typename Traits>
class FastSolutionTensorInterpolationBase<PHAL::AlbanyTraits::Jacobian, Traits, typename PHAL::AlbanyTraits::Jacobian::ScalarT>
  : public DOFTensorInterpolationBase<PHAL::AlbanyTraits::Jacobian, Traits, typename PHAL::AlbanyTraits::Jacobian::ScalarT>
{
public:

  FastSolutionTensorInterpolationBase(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
    : DOFTensorInterpolationBase<PHAL::AlbanyTraits::Jacobian, Traits,  typename PHAL::AlbanyTraits::Jacobian::ScalarT>(p, dl) {
    this->setName("FastSolutionTensorInterpolationBase"+PHX::print<PHAL::AlbanyTraits::Jacobian>());
    offset = p.get<int>("Offset of First DOF");
  };

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm) {
    DOFTensorInterpolationBase<PHAL::AlbanyTraits::Jacobian, Traits,  typename PHAL::AlbanyTraits::Jacobian::ScalarT>
      ::postRegistrationSetup(d, vm);
  }

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
  typedef PHAL::AlbanyTraits::Jacobian::MeshScalarT MeshScalarT;
  typedef typename Albany::StrongestScalarType<ScalarT,MeshScalarT>::type OutputScalarT;

  std::size_t offset;
};


// Some shortcut names
template<typename EvalT, typename Traits>
using DOFTensorInterpolation = DOFTensorInterpolationBase<EvalT,Traits,typename EvalT::ScalarT>;

template<typename EvalT, typename Traits>
using DOFTensorInterpolationMesh = DOFTensorInterpolationBase<EvalT,Traits,typename EvalT::MeshScalarT>;

template<typename EvalT, typename Traits>
using DOFTensorInterpolationParam = DOFTensorInterpolationBase<EvalT,Traits,typename EvalT::ParamScalarT>;

template<typename EvalT, typename Traits>
using FastSolutionTensorInterpolation = FastSolutionTensorInterpolationBase<EvalT,Traits,typename EvalT::ScalarT>;

} // Namespace PHAL

#endif // PHAL_DOFTENSOR_INTERPOLATION_HPP
