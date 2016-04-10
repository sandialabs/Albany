//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AERAS_COMPUTE_AND_SCATTER_JAC_HPP
#define AERAS_COMPUTE_AND_SCATTER_JAC_HPP

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
class ComputeAndScatterJacBase
  : public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits>  {
  
public:
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;
  
  ComputeAndScatterJacBase(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Aeras::Layouts>& dl);
  
  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);
  
  virtual void evaluateFields(typename Traits::EvalData d)=0;
  
protected:

  //OG not used anymore
  Teuchos::RCP<PHX::FieldTag> scatter_operation;
  //OG not used anymore
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

protected:

  PHX::MDField<RealType,Cell,Node,QuadPoint> BF;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint> wBF;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> GradBF;
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> wGradBF;

protected:
  double sqrtHVcoef;

};

template<typename EvalT, typename Traits> class ComputeAndScatterJac;

// **************************************************************
// **************************************************************
// * Specializations
// **************************************************************
// **************************************************************


// **************************************************************
// Jacobian
// **************************************************************
template<typename Traits>
class ComputeAndScatterJac<PHAL::AlbanyTraits::Jacobian,Traits>
  : public ComputeAndScatterJacBase<PHAL::AlbanyTraits::Jacobian, Traits>  {
public:
  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
  ComputeAndScatterJac(const Teuchos::ParameterList& p,
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
// GENERIC: Specializations for SG and MP not yet implemented
// **************************************************************
template<typename Traits>
class ComputeAndScatterJac<PHAL::AlbanyTraits::DistParamDeriv,Traits>
  : public ComputeAndScatterJacBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>  {
public:
  typedef typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT ScalarT;
  ComputeAndScatterJac(const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Aeras::Layouts>& dl) : 
    ComputeAndScatterJacBase<PHAL::AlbanyTraits::DistParamDeriv,Traits>(p,dl){}
  void evaluateFields(typename Traits::EvalData d)
    {throw "Aeras::GatherSolution not implemented for all tempate specializations";};
};


// **************************************************************
// GENERIC: Specializations for SG and MP not yet implemented
// **************************************************************
template<typename EvalT, typename Traits>
class ComputeAndScatterJac
  : public ComputeAndScatterJacBase<EvalT, Traits>  {
public:
  typedef typename EvalT::ScalarT ScalarT;
  ComputeAndScatterJac(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Aeras::Layouts>& dl)
      : ComputeAndScatterJacBase<EvalT,Traits>(p,dl)
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
class ComputeAndScatterJac<PHAL::AlbanyTraits::MPResidual,Traits>
  : public ComputeAndScatterJacBase<PHAL::AlbanyTraits::MPResidual, Traits>  {
public:
  typedef typename PHAL::AlbanyTraits::MPResidual::ScalarT ScalarT;
  ComputeAndScatterJac(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Aeras::Layouts>& dl);
  void evaluateFields(typename Traits::EvalData d); 
};

// **************************************************************
// Multi-point Jacobian
// **************************************************************
template<typename Traits>
class ComputeAndScatterJac<PHAL::AlbanyTraits::MPJacobian,Traits>
  : public ComputeAndScatterJacBase<PHAL::AlbanyTraits::MPJacobian, Traits>  {
public:
  typedef typename PHAL::AlbanyTraits::MPJacobian::ScalarT ScalarT;
  ComputeAndScatterJac(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Aeras::Layouts>& dl);
  void evaluateFields(typename Traits::EvalData d); 
};
#endif

}

#endif
