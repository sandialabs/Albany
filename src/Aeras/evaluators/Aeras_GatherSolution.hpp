//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AERAS_GATHER_SOLUTION_HPP
#define AERAS_GATHER_SOLUTION_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Aeras_Layouts.hpp"

#include "Teuchos_ParameterList.hpp"

#include "Kokkos_Vector.hpp"

namespace Aeras {
/** \brief Gathers Coordinates values from the Newton coordinates vector into 
    the nodal fields of the field manager

    Currently makes an assumption that the stride is constant for dofs
    and that the nmber of dofs is equal to the size of the coordinates
    names vector.

*/

template<typename EvalT, typename Traits>
class GatherSolutionBase
  : public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits>  {
  
public:
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  
  GatherSolutionBase(const Teuchos::ParameterList& p,
                     const Teuchos::RCP<Aeras::Layouts>& dl);
  
  void postRegistrationSetup(typename Traits::SetupData d,
                             PHX::FieldManager<Traits>& vm);
  
  // This function requires template specialization, in derived class below
  virtual void evaluateFields(typename Traits::EvalData d) = 0;
  
protected:
#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  std::vector< PHX::MDField<ScalarT> > val;
  std::vector< PHX::MDField<ScalarT> > val_dot;
#else
  std::vector< PHX::MDField<ScalarT> > val;
  std::vector< PHX::MDField<ScalarT> > val_dot;
//  Kokkos::vector< PHX::MDField<ScalarT>, PHX::Device > val;
//  Kokkos::vector< PHX::MDField<ScalarT>, PHX::Device > val_dot;
#endif
  const int numNodes;
  const int numDims;
  const int numLevels;
  const int worksetSize;
  int numFields; 
  int numNodeVar; 
  int numVectorLevelVar;
  int numScalarLevelVar;
  int numTracerVar;
};

//template<typename EvalT, typename Traits> class GatherSolution;

// **************************************************************
// GENERIC: SG and MP specialization not implemented   
// **************************************************************
template<typename EvalT, typename Traits>
class GatherSolution
   : public GatherSolutionBase<EvalT, Traits>  {
  
public:
  typedef typename EvalT::ScalarT ScalarT;
  GatherSolution(const Teuchos::ParameterList& p,
                 const Teuchos::RCP<Aeras::Layouts>& dl);
  void evaluateFields(typename Traits::EvalData d);
};


// **************************************************************
// Residual 
// **************************************************************
template<typename Traits>
class GatherSolution<PHAL::AlbanyTraits::Residual,Traits>
   : public GatherSolutionBase<PHAL::AlbanyTraits::Residual, Traits>  {
  
public:
  typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
  GatherSolution(const Teuchos::ParameterList& p,
                 const Teuchos::RCP<Aeras::Layouts>& dl);
  void evaluateFields(typename Traits::EvalData d); 

  Teuchos::ArrayRCP<const ST> xT_constView;
  Teuchos::ArrayRCP<const ST> xdotT_constView;
/*#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  typedef typename PHX::Device execution_space;
  Kokkos::View<int***, PHX::Device> wsID_kokkos;

  KOKKOS_INLINE_FUNCTION
  void operator() (const int &cell) const;
#endif
*/
};

// **************************************************************
// Jacobian
// **************************************************************
template<typename Traits>
class GatherSolution<PHAL::AlbanyTraits::Jacobian,Traits>
   : public GatherSolutionBase<PHAL::AlbanyTraits::Jacobian, Traits>  {
  
public:
  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
  GatherSolution(const Teuchos::ParameterList& p,
                 const Teuchos::RCP<Aeras::Layouts>& dl);
  void evaluateFields(typename Traits::EvalData d); 

/*#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  Teuchos::ArrayRCP<const ST> xT_constView;
  Teuchos::ArrayRCP<const ST> xdotT_constView;

  bool ignore_residual;
  double j_coeff, m_coeff;

  struct GatherSolution_Tag{};
  struct GatherSolution_transientTerms_Tag{};
  
  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;
  typedef typename PHX::Device execution_space;
  Kokkos::View<int***, PHX::Device> wsID_kokkos;

  typedef Kokkos::RangePolicy<ExecutionSpace,GatherSolution_Tag> GatherSolution_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,GatherSolution_transientTerms_Tag> GatherSolution_transientTerms_Policy; 

  KOKKOS_INLINE_FUNCTION
  void operator() (const GatherSolution_Tag &tag, const int &cell) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const GatherSolution_transientTerms_Tag &tag, const int &cell) const;

  KOKKOS_INLINE_FUNCTION
  void gather_solution(const int &cell, const int &node, const int &neq, const int &num_dof, const int &firstunk) const;
  KOKKOS_INLINE_FUNCTION
  void gather_solution_transientTerms(const int &cell, const int &node, const int &neq, const int &num_dof, const int &firstunk) const;

#endif
*/
};


// **************************************************************
// Tangent (Jacobian mat-vec + parameter derivatives)
// **************************************************************
template<typename Traits>
class GatherSolution<PHAL::AlbanyTraits::Tangent,Traits>
   : public GatherSolutionBase<PHAL::AlbanyTraits::Tangent, Traits>  {
  
public:
  typedef typename PHAL::AlbanyTraits::Tangent::ScalarT ScalarT;
  GatherSolution(const Teuchos::ParameterList& p,
                 const Teuchos::RCP<Aeras::Layouts>& dl);
  void evaluateFields(typename Traits::EvalData d); 
};

// **************************************************************
// ???????
// **************************************************************
template<typename Traits>
class GatherSolution<PHAL::AlbanyTraits::DistParamDeriv,Traits>
   : public GatherSolutionBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>  {
  
public:
  GatherSolution(const Teuchos::ParameterList& p,
                 const Teuchos::RCP<Aeras::Layouts>& dl);
  void evaluateFields(typename Traits::EvalData d); 
};

#ifdef ALBANY_SG_MP
// **************************************************************
// Multi-point Residual 
// **************************************************************
template<typename Traits>
class GatherSolution<PHAL::AlbanyTraits::MPResidual,Traits>
   : public GatherSolutionBase<PHAL::AlbanyTraits::MPResidual, Traits>  {
  
public:
  typedef typename PHAL::AlbanyTraits::MPResidual::ScalarT ScalarT;
  GatherSolution(const Teuchos::ParameterList& p,
                 const Teuchos::RCP<Aeras::Layouts>& dl);
  void evaluateFields(typename Traits::EvalData d); 
};

// **************************************************************
// Multi-point Jacobian
// **************************************************************
template<typename Traits>
class GatherSolution<PHAL::AlbanyTraits::MPJacobian,Traits>
   : public GatherSolutionBase<PHAL::AlbanyTraits::MPJacobian, Traits>  {
  
public:
  typedef typename PHAL::AlbanyTraits::MPJacobian::ScalarT ScalarT;
  GatherSolution(const Teuchos::ParameterList& p,
                 const Teuchos::RCP<Aeras::Layouts>& dl);
  void evaluateFields(typename Traits::EvalData d); 
};
#endif //ALBANY_SG_MP

}

#endif
