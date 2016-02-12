//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AERAS_SCATTER_RESIDUAL_HPP
#define AERAS_SCATTER_RESIDUAL_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Aeras_Layouts.hpp"

#include "Teuchos_ParameterList.hpp"

namespace Aeras {
/** \brief Gathers Coordinates values from the Newton coordinates vector into 
    the nodal fields of the field manager

    Currently makes an assumption that the stride is constant for dofs
    and that the nmber of dofs is equal to the size of the coordinates
    names vector.

*/

template<typename EvalT, typename Traits> 
class ScatterResidualBase
  : public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits>  {
  
public:
  typedef typename EvalT::ScalarT ScalarT;
  
  ScatterResidualBase(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Aeras::Layouts>& dl);
  
  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);
  
  virtual void evaluateFields(typename Traits::EvalData d)=0;
  
protected:
  Teuchos::RCP<PHX::FieldTag> scatter_operation;
  std::vector< PHX::MDField<ScalarT> > val;
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

template<typename EvalT, typename Traits> class ScatterResidual;

// **************************************************************
// **************************************************************
// * Specializations
// **************************************************************
// **************************************************************


// **************************************************************
// Residual 
// **************************************************************
template<typename Traits>
class ScatterResidual<PHAL::AlbanyTraits::Residual,Traits>
  : public ScatterResidualBase<PHAL::AlbanyTraits::Residual, Traits>  {
public:
  typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
  ScatterResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Aeras::Layouts>& dl);
  void evaluateFields(typename Traits::EvalData d); 

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
public:

Teuchos::RCP<Tpetra_Vector> fT;
Teuchos::ArrayRCP<ST> fT_nonconstView;

Kokkos::View<int***, PHX::Device> Index;

struct ScatterResid_Tag{};

typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

typedef Kokkos::RangePolicy<ExecutionSpace, ScatterResid_Tag> ScatterResid_Policy;

KOKKOS_INLINE_FUNCTION
  void operator() (const ScatterResid_Tag& tag, const int& i) const;

#endif

};
// **************************************************************
// Jacobian
// **************************************************************
template<typename Traits>
class ScatterResidual<PHAL::AlbanyTraits::Jacobian,Traits>
  : public ScatterResidualBase<PHAL::AlbanyTraits::Jacobian, Traits>  {
public:
  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
  ScatterResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Aeras::Layouts>& dl);
  void evaluateFields(typename Traits::EvalData d); 

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
public:

Teuchos::RCP<Tpetra_Vector> fT;
Teuchos::RCP<Tpetra_CrsMatrix> JacT;
typedef typename Tpetra_CrsMatrix::k_local_matrix_type  LocalMatrixType;
LocalMatrixType jacobian;
bool loadResid;
int neq, nunk;

Kokkos::View<int***, PHX::Device> Index;

struct ScatterResid_noFastAccess_Tag{};
struct ScatterResid_hasFastAccess_is_adjoint_Tag{};
struct ScatterResid_hasFastAccess_no_adjoint_Tag{};

typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

typedef Kokkos::RangePolicy<ExecutionSpace, ScatterResid_noFastAccess_Tag> ScatterResid_noFastAccess_Policy;
typedef Kokkos::RangePolicy<ExecutionSpace, ScatterResid_hasFastAccess_is_adjoint_Tag> ScatterResid_hasFastAccess_is_adjoint_Policy;
typedef Kokkos::RangePolicy<ExecutionSpace, ScatterResid_hasFastAccess_no_adjoint_Tag> ScatterResid_hasFastAccess_no_adjoint_Policy;

KOKKOS_INLINE_FUNCTION
  void operator() (const ScatterResid_noFastAccess_Tag& tag, const int& i) const;

KOKKOS_INLINE_FUNCTION
  void operator() (const ScatterResid_hasFastAccess_is_adjoint_Tag& tag, const int& i) const;

KOKKOS_INLINE_FUNCTION
  void operator() (const ScatterResid_hasFastAccess_no_adjoint_Tag& tag, const int& i) const;
#endif

};

// **************************************************************
// Tangent
// **************************************************************
template<typename Traits>
class ScatterResidual<PHAL::AlbanyTraits::Tangent,Traits>
  : public ScatterResidualBase<PHAL::AlbanyTraits::Tangent, Traits>  {
public:
  typedef typename PHAL::AlbanyTraits::Tangent::ScalarT ScalarT;
  ScatterResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Aeras::Layouts>& dl);
  void evaluateFields(typename Traits::EvalData d); 
};


// **************************************************************
// GENERIC: Specializations for SG and MP not yet implemented
// **************************************************************
template<typename Traits>
class ScatterResidual<PHAL::AlbanyTraits::DistParamDeriv,Traits>
  : public ScatterResidualBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>  {
public:
  typedef typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT ScalarT;
  ScatterResidual(const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Aeras::Layouts>& dl) : 
    ScatterResidualBase<PHAL::AlbanyTraits::DistParamDeriv,Traits>(p,dl){}
  void evaluateFields(typename Traits::EvalData d)
    {throw "Aeras::GatherSolution not implemented for all tempate specializations";};
};


// **************************************************************
// GENERIC: Specializations for SG and MP not yet implemented
// **************************************************************
template<typename EvalT, typename Traits>
class ScatterResidual
  : public ScatterResidualBase<EvalT, Traits>  {
public:
  typedef typename EvalT::ScalarT ScalarT;
  ScatterResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Aeras::Layouts>& dl)
      : ScatterResidualBase<EvalT,Traits>(p,dl)
    {
    };
  void evaluateFields(typename Traits::EvalData d)
    {throw "Aeras::GatherSolution not implemented for all tempate specializations";};
};

#ifdef ALBANY_ENSEMBLE 
// **************************************************************
// Multi-point Residual 
// **************************************************************
template<typename Traits>
class ScatterResidual<PHAL::AlbanyTraits::MPResidual,Traits>
  : public ScatterResidualBase<PHAL::AlbanyTraits::MPResidual, Traits>  {
public:
  typedef typename PHAL::AlbanyTraits::MPResidual::ScalarT ScalarT;
  ScatterResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Aeras::Layouts>& dl);
  void evaluateFields(typename Traits::EvalData d); 
};

// **************************************************************
// Multi-point Jacobian
// **************************************************************
template<typename Traits>
class ScatterResidual<PHAL::AlbanyTraits::MPJacobian,Traits>
  : public ScatterResidualBase<PHAL::AlbanyTraits::MPJacobian, Traits>  {
public:
  typedef typename PHAL::AlbanyTraits::MPJacobian::ScalarT ScalarT;
  ScatterResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Aeras::Layouts>& dl);
  void evaluateFields(typename Traits::EvalData d); 
};
#endif

}

#endif
