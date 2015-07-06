//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


//IK, 9/13/14: only Epetra is SG and MP 

#ifndef PHAL_GATHER_SOLUTION_HPP
#define PHAL_GATHER_SOLUTION_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"

#include "Teuchos_ParameterList.hpp"
#if defined(ALBANY_EPETRA)
#include "Epetra_Vector.h"
#endif

#include "Kokkos_Vector.hpp"

namespace PHAL {
/** \brief Gathers solution values from the Newton solution vector into
    the nodal fields of the field manager

    Currently makes an assumption that the stride is constant for dofs
    and that the nmber of dofs is equal to the size of the solution
    names vector.

*/
// **************************************************************
// Base Class with Generic Implementations: Specializations for
// Automatic Differentiation Below
// **************************************************************

template<typename EvalT, typename Traits>
class GatherSolutionBase
  : public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  GatherSolutionBase(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  // This function requires template specialization, in derived class below
  virtual void evaluateFields(typename Traits::EvalData d) = 0;

  Kokkos::View<int***, PHX::Device> Index;

protected:

  typedef typename EvalT::ScalarT ScalarT;
  std::vector< PHX::MDField<ScalarT,Cell,Node> > val;
  std::vector< PHX::MDField<ScalarT,Cell,Node> > val_dot;
  std::vector< PHX::MDField<ScalarT,Cell,Node> > val_dotdot;
  PHX::MDField<ScalarT,Cell,Node,VecDim>  valVec;
  PHX::MDField<ScalarT,Cell,Node,VecDim>  valVec_dot;
  PHX::MDField<ScalarT,Cell,Node,VecDim>  valVec_dotdot;
  PHX::MDField<ScalarT,Cell,Node,VecDim,VecDim> valTensor;
  PHX::MDField<ScalarT,Cell,Node,VecDim,VecDim> valTensor_dot;
  PHX::MDField<ScalarT,Cell,Node,VecDim,VecDim> valTensor_dotdot;
  std::size_t numNodes;
  std::size_t numFieldsBase; // Number of fields gathered in this call
  std::size_t offset; // Offset of first DOF being gathered when numFields<neq
  unsigned short int tensorRank;
  bool enableTransient;
  bool enableAcceleration;
/*#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT 
 typedef typename Kokkos::View<double*,PHX::Device>::execution_space executionSpace;
 Kokkos::vector< PHX::MDField<ScalarT, Cell, Node>, PHX::Device > val_kokkos;
 typename Kokkos::vector< PHX::MDField<ScalarT, Cell, Node>, PHX::Device >::t_dev d_val;

#endif*/
};

template<typename EvalT, typename Traits> class GatherSolution;

// **************************************************************
// **************************************************************
// * Specializations
// **************************************************************
// **************************************************************


// **************************************************************
// Residual
// **************************************************************
template<typename Traits>
class GatherSolution<PHAL::AlbanyTraits::Residual,Traits>
   : public GatherSolutionBase<PHAL::AlbanyTraits::Residual, Traits>  {

public:
  GatherSolution(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  // Old constructor, still needed by BCs that use PHX Factory
  GatherSolution(const Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);

 //Kokkos:
 struct tensorRank_2Tag{};
 struct tensorRank_2_enableTransientTag{};
 struct tensorRank_2_enableAccelerationTag{};

 struct tensorRank_1Tag{};
 struct tensorRank_1_enableTransientTag{};
 struct tensorRank_1_enableAccelerationTag{};

 struct tensorRank_0Tag{};
 struct tensorRank_0_enableTransientTag{};
 struct tensorRank_0_enableAccelerationTag{};

 typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

 typedef Kokkos::RangePolicy<ExecutionSpace,tensorRank_2Tag> tensorRank_2Policy;
 typedef Kokkos::RangePolicy<ExecutionSpace,tensorRank_2_enableTransientTag> tensorRank_2_enableTransientPolicy;
 typedef Kokkos::RangePolicy<ExecutionSpace,tensorRank_2_enableAccelerationTag> tensorRank_2_enableAccelerationPolicy;
 typedef Kokkos::RangePolicy<ExecutionSpace,tensorRank_1Tag> tensorRank_1Policy;
 typedef Kokkos::RangePolicy<ExecutionSpace,tensorRank_1_enableTransientTag> tensorRank_1_enableTransientPolicy;
 typedef Kokkos::RangePolicy<ExecutionSpace,tensorRank_1_enableAccelerationTag> tensorRank_1_enableAccelerationPolicy;
 typedef Kokkos::RangePolicy<ExecutionSpace,tensorRank_0Tag> tensorRank_0Policy;
 typedef Kokkos::RangePolicy<ExecutionSpace,tensorRank_0_enableTransientTag> tensorRank_0_enableTransientPolicy;
 typedef Kokkos::RangePolicy<ExecutionSpace,tensorRank_0_enableAccelerationTag> tensorRank_0_enableAccelerationPolicy;

  KOKKOS_INLINE_FUNCTION
  void operator() (const tensorRank_2Tag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const tensorRank_2_enableTransientTag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const tensorRank_2_enableAccelerationTag& tag, const int& i) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (const tensorRank_1Tag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const tensorRank_1_enableTransientTag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const tensorRank_1_enableAccelerationTag& tag, const int& i) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (const tensorRank_0Tag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const tensorRank_0_enableTransientTag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const tensorRank_0_enableAccelerationTag& tag, const int& i) const;




private:
  typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
  const int numFields;
  int numDim; 

  Teuchos::ArrayRCP<const ST> xT_constView;
  Teuchos::ArrayRCP<const ST> xdotT_constView;
  Teuchos::ArrayRCP<const ST> xdotdotT_constView;
  Kokkos::View<int***, PHX::Device> wsID_kokkos;

  typedef typename Kokkos::View<double*,PHX::Device>::execution_space executionSpace;
  Kokkos::vector< Kokkos::View< ScalarT**, PHX::Device> , PHX::Device > val_kokkos;
//  Kokkos::vector< PHX::MDField<ScalarT,Cell,Node> , PHX::Device > val_kokkos;
  Kokkos::vector< Kokkos::View< ScalarT**, PHX::Device> , PHX::Device > val_dot_kokkos; 
  Kokkos::vector< Kokkos::View< ScalarT**, PHX::Device> , PHX::Device > val_dotdot_kokkos;

//  typename Kokkos::vector< PHX::MDField<ScalarT,Cell,Node> , PHX::Device >::t_dev d_val;
  typename Kokkos::vector< Kokkos::View< ScalarT**, PHX::Device> , PHX::Device >::t_dev d_val;
  typename Kokkos::vector< Kokkos::View< ScalarT**, PHX::Device> , PHX::Device >::t_dev d_val_dot;
  typename Kokkos::vector< Kokkos::View< ScalarT**, PHX::Device> , PHX::Device >::t_dev d_val_dotdot;
};

// **************************************************************
// Jacobian
// **************************************************************
template<typename Traits>
class GatherSolution<PHAL::AlbanyTraits::Jacobian,Traits>
   : public GatherSolutionBase<PHAL::AlbanyTraits::Jacobian, Traits>  {

public:
  GatherSolution(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  GatherSolution(const Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
 
 //Kokkos
 struct tensorRank_2Tag{};
 struct tensorRank_2_enableTransientTag{};
 struct tensorRank_2_enableAccelerationTag{};

 struct tensorRank_1Tag{};
 struct tensorRank_1_enableTransientTag{};
 struct tensorRank_1_enableAccelerationTag{};

 struct tensorRank_0Tag{};
 struct tensorRank_0_enableTransientTag{};
 struct tensorRank_0_enableAccelerationTag{};

 typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

 typedef Kokkos::RangePolicy<ExecutionSpace,tensorRank_2Tag> tensorRank_2Policy;
 typedef Kokkos::RangePolicy<ExecutionSpace,tensorRank_2_enableTransientTag> tensorRank_2_enableTransientPolicy;
 typedef Kokkos::RangePolicy<ExecutionSpace,tensorRank_2_enableAccelerationTag> tensorRank_2_enableAccelerationPolicy;
 typedef Kokkos::RangePolicy<ExecutionSpace,tensorRank_1Tag> tensorRank_1Policy;
 typedef Kokkos::RangePolicy<ExecutionSpace,tensorRank_1_enableTransientTag> tensorRank_1_enableTransientPolicy;
 typedef Kokkos::RangePolicy<ExecutionSpace,tensorRank_1_enableAccelerationTag> tensorRank_1_enableAccelerationPolicy;
 typedef Kokkos::RangePolicy<ExecutionSpace,tensorRank_0Tag> tensorRank_0Policy;
 typedef Kokkos::RangePolicy<ExecutionSpace,tensorRank_0_enableTransientTag> tensorRank_0_enableTransientPolicy;
 typedef Kokkos::RangePolicy<ExecutionSpace,tensorRank_0_enableAccelerationTag> tensorRank_0_enableAccelerationPolicy;

  void operator() (const tensorRank_2Tag& tag, const int& i) const;
  void operator() (const tensorRank_2_enableTransientTag& tag, const int& i) const;
  void operator() (const tensorRank_2_enableAccelerationTag& tag, const int& i) const;

  void operator() (const tensorRank_1Tag& tag, const int& i) const;
  void operator() (const tensorRank_1_enableTransientTag& tag, const int& i) const;
  void operator() (const tensorRank_1_enableAccelerationTag& tag, const int& i) const;

  void operator() (const tensorRank_0Tag& tag, const int& i) const;
  void operator() (const tensorRank_0_enableTransientTag& tag, const int& i) const;
  void operator() (const tensorRank_0_enableAccelerationTag& tag, const int& i) const;
 
private:
  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
  const int numFields;

  Teuchos::ArrayRCP<const ST> xT_constView;
  Teuchos::ArrayRCP<const ST> xdotT_constView;
  Teuchos::ArrayRCP<const ST> xdotdotT_constView;
  int numDim;
  Kokkos::View<int***, PHX::Device> wsID_kokkos;
  double j_coeff;
  double n_coeff;
  double m_coeff;
  bool ignore_residual;

  typedef typename Kokkos::View<double*,PHX::Device>::execution_space executionSpace;
  Kokkos::vector< Kokkos::View< ScalarT**, PHX::Device> , PHX::Device > val_kokkosjac;
  Kokkos::vector< Kokkos::View< ScalarT**, PHX::Device> , PHX::Device > val_dot_kokkosjac;
  Kokkos::vector< Kokkos::View< ScalarT**, PHX::Device> , PHX::Device > val_dotdot_kokkosjac;

  typename Kokkos::vector< Kokkos::View< ScalarT**, PHX::Device> , PHX::Device >::t_dev d_val;
  typename Kokkos::vector< Kokkos::View< ScalarT**, PHX::Device> , PHX::Device >::t_dev d_val_dot;
  typename Kokkos::vector< Kokkos::View< ScalarT**, PHX::Device> , PHX::Device >::t_dev d_val_dotdot;
 
};


// **************************************************************
// Tangent (Jacobian mat-vec + parameter derivatives)
// **************************************************************
template<typename Traits>
class GatherSolution<PHAL::AlbanyTraits::Tangent,Traits>
   : public GatherSolutionBase<PHAL::AlbanyTraits::Tangent, Traits>  {

public:
  GatherSolution(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  GatherSolution(const Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::Tangent::ScalarT ScalarT;
  typedef typename Kokkos::View<ScalarT*, PHX::Device>::reference_type reference_type;
  const std::size_t numFields;
};

// **************************************************************
// Distributed Parameter Derivative
// **************************************************************
template<typename Traits>
class GatherSolution<PHAL::AlbanyTraits::DistParamDeriv,Traits>
   : public GatherSolutionBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>  {

public:
  GatherSolution(const Teuchos::ParameterList& p,
                 const Teuchos::RCP<Albany::Layouts>& dl);
  GatherSolution(const Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT ScalarT;
  const std::size_t numFields;
};

// **************************************************************
// Stochastic Galerkin Residual
// **************************************************************

#ifdef ALBANY_SG
template<typename Traits>
class GatherSolution<PHAL::AlbanyTraits::SGResidual,Traits>
   : public GatherSolutionBase<PHAL::AlbanyTraits::SGResidual, Traits>  {

public:
  GatherSolution(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  GatherSolution(const Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::SGResidual::ScalarT ScalarT;
  const std::size_t numFields;
};


// **************************************************************
// Stochastic Galerkin Jacobian
// **************************************************************
template<typename Traits>
class GatherSolution<PHAL::AlbanyTraits::SGJacobian,Traits>
   : public GatherSolutionBase<PHAL::AlbanyTraits::SGJacobian, Traits>  {

public:
  GatherSolution(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  GatherSolution(const Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::SGJacobian::ScalarT ScalarT;
  const std::size_t numFields;
};

// **************************************************************
// Stochastic Galerkin Tangent (Jacobian mat-vec + parameter derivatives)
// **************************************************************
template<typename Traits>
class GatherSolution<PHAL::AlbanyTraits::SGTangent,Traits>
   : public GatherSolutionBase<PHAL::AlbanyTraits::SGTangent, Traits>  {

public:
  GatherSolution(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  GatherSolution(const Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::SGTangent::ScalarT ScalarT;
  typedef typename Kokkos::View<ScalarT*, PHX::Device>::reference_type reference_type;
  const std::size_t numFields;
};
#endif 
#ifdef ALBANY_ENSEMBLE 

// **************************************************************
// Multi-point Residual
// **************************************************************

template<typename Traits>
class GatherSolution<PHAL::AlbanyTraits::MPResidual,Traits>
   : public GatherSolutionBase<PHAL::AlbanyTraits::MPResidual, Traits>  {

public:
  GatherSolution(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  GatherSolution(const Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::MPResidual::ScalarT ScalarT;
  const std::size_t numFields;
};


// **************************************************************
// Multi-point Jacobian
// **************************************************************
template<typename Traits>
class GatherSolution<PHAL::AlbanyTraits::MPJacobian,Traits>
   : public GatherSolutionBase<PHAL::AlbanyTraits::MPJacobian, Traits>  {

public:
  GatherSolution(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  GatherSolution(const Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::MPJacobian::ScalarT ScalarT;
  const std::size_t numFields;
};

// **************************************************************
// Multi-point Tangent (Jacobian mat-vec + parameter derivatives)
// **************************************************************
template<typename Traits>
class GatherSolution<PHAL::AlbanyTraits::MPTangent,Traits>
   : public GatherSolutionBase<PHAL::AlbanyTraits::MPTangent, Traits>  {

public:
  GatherSolution(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);
  GatherSolution(const Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::MPTangent::ScalarT ScalarT;
  const std::size_t numFields;
};
#endif

// **************************************************************
}

#endif
