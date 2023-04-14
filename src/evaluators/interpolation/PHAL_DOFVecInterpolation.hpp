//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_DOFVEC_INTERPOLATION_HPP
#define PHAL_DOFVEC_INTERPOLATION_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"
#include "PHAL_Utilities.hpp"

namespace PHAL {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOFVec values to quad points.

*/

template<typename EvalT, typename Traits, typename ScalarT>
class DOFVecInterpolationBase : public PHX::EvaluatorWithBaseImpl<Traits>,
                                public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  DOFVecInterpolationBase (const Teuchos::ParameterList& p,
                           const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

protected:

  // Input:
  //! Values at nodes
  PHX::MDField<const ScalarT,Cell,Node,VecDim> val_node;
  //! Basis Functions

  typedef typename EvalT::MeshScalarT MeshScalarT;
  PHX::MDField<const MeshScalarT,Cell,Node,QuadPoint> BF;

  // Output:
  //! Values at quadrature points
  typedef typename Albany::StrongestScalarType<ScalarT,MeshScalarT>::type OutputScalarT;
  PHX::MDField<OutputScalarT,Cell,QuadPoint,VecDim> val_qp;

  std::size_t numNodes;
  std::size_t numQPs;
  std::size_t vecDim;

  MDFieldMemoizer<Traits> memoizer;
};

/** \brief Fast Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values at quad points.
    It is an optimized version of DOFVecInterpolationBase that exploits the sparsity pattern of the derivatives in the Jacobian evaluation
    WARNING: it does not work for general fields: it works when the field to be interpolated is the solution
             or a part (a few contiguous components) of the solution
             It does not work when the mesh coordinates are of type ScalarT
*/
template<typename EvalT, typename Traits, typename ScalarT>
class FastSolutionVecInterpolationBase
      : public DOFVecInterpolationBase<EvalT, Traits, ScalarT>  {
public:

  FastSolutionVecInterpolationBase(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl)
    : DOFVecInterpolationBase<EvalT, Traits,  ScalarT>(p, dl) {
    this->setName("FastSolutionVecInterpolationBase"+PHX::print<EvalT>());
  };

  void postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& vm) {
    DOFVecInterpolationBase<EvalT, Traits,  ScalarT>::postRegistrationSetup(d, vm);
  }

  void evaluateFields(typename Traits::EvalData d) {
    DOFVecInterpolationBase<EvalT, Traits,  ScalarT>::evaluateFields(d);
  }
};

//! Specialization for Jacobian evaluation taking advantage of known sparsity
// This assumes that the Mesh coordinates are not of FAD type
template<typename Traits>
class FastSolutionVecInterpolationBase<PHAL::AlbanyTraits::Jacobian, Traits, typename PHAL::AlbanyTraits::Jacobian::ScalarT>
  : public DOFVecInterpolationBase<PHAL::AlbanyTraits::Jacobian, Traits, typename PHAL::AlbanyTraits::Jacobian::ScalarT>
{
public:
  FastSolutionVecInterpolationBase(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
    : DOFVecInterpolationBase<PHAL::AlbanyTraits::Jacobian, Traits,  typename PHAL::AlbanyTraits::Jacobian::ScalarT>(p, dl) {
    this->setName("FastSolutionVecInterpolationBase"+PHX::print<PHAL::AlbanyTraits::Jacobian>());
    offset = p.get<int>("Offset of First DOF");
  };

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm) {
    DOFVecInterpolationBase<PHAL::AlbanyTraits::Jacobian, Traits,  typename PHAL::AlbanyTraits::Jacobian::ScalarT>
      ::postRegistrationSetup(d, vm);
  }

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
  typedef PHAL::AlbanyTraits::Jacobian::MeshScalarT MeshScalarT;

  std::size_t offset;
};

// Some shortcut names
template<typename EvalT, typename Traits>
using DOFVecInterpolation = DOFVecInterpolationBase<EvalT,Traits,typename EvalT::ScalarT>;

template<typename EvalT, typename Traits>
using DOFVecInterpolationMesh = DOFVecInterpolationBase<EvalT,Traits,typename EvalT::MeshScalarT>;

template<typename EvalT, typename Traits>
using DOFVecInterpolationParam = DOFVecInterpolationBase<EvalT,Traits,typename EvalT::ParamScalarT>;

template<typename EvalT, typename Traits>
using FastSolutionVecInterpolation = FastSolutionVecInterpolationBase<EvalT,Traits,typename EvalT::ScalarT>;

} // Namespace PHAL

#endif // PHAL_DOFVEC_INTERPOLATION_HPP
