//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_GATHER_VERTICALLY_CONTRACTED_SOLUTION_HPP
#define LANDICE_GATHER_VERTICALLY_CONTRACTED_SOLUTION_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

#include "Albany_ThyraUtils.hpp"
#include "Albany_GlobalLocalIndexer.hpp"
#include "Albany_AbstractDiscretization.hpp"

#include "PHAL_AlbanyTraits.hpp"

namespace LandIce {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits>
class GatherVerticallyContractedSolutionBase : public PHX::EvaluatorWithBaseImpl<Traits>,
		    public PHX::EvaluatorDerived<EvalT, Traits> {

public:

  GatherVerticallyContractedSolutionBase(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl);

  virtual ~GatherVerticallyContractedSolutionBase(){};

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  virtual void evaluateFields(typename Traits::EvalData d)=0;

  enum ContractionOperator {VerticalAverage, VerticalSum};

  Teuchos::ArrayRCP<const ST> x_constView;

protected:

  void computeQuadWeights(const Albany::LayeredMeshNumbering<GO>& layeredMeshNumbering);

  typedef typename EvalT::ScalarT ScalarT;

  // Output:
  PHX::MDField<ScalarT>  contractedSol;

  bool isVector;

  int offset;

  std::size_t vecDim;
  std::size_t numNodes;

  std::string meshPart;

  Teuchos::RCP<const CellTopologyData> cell_topo;

  ContractionOperator op;

  Albany::LocalSideSetInfo sideSet;

  Kokkos::View<double*, PHX::Device> quadWeights;
  Kokkos::View<int*, PHX::Device> numSideNodes;

  int numLayers;

};


template<typename EvalT, typename Traits> class GatherVerticallyContractedSolution;



template<typename Traits>
class GatherVerticallyContractedSolution<PHAL::AlbanyTraits::Residual,Traits>
    : public GatherVerticallyContractedSolutionBase<PHAL::AlbanyTraits::Residual,Traits> {

public:

  GatherVerticallyContractedSolution(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl);

  void evaluateFields(typename Traits::EvalData d);

  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

  struct ResidualScalar_Tag{};
  struct ResidualVector_Tag{};

  typedef Kokkos::RangePolicy<ExecutionSpace,ResidualScalar_Tag> ResidualScalar_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,ResidualVector_Tag> ResidualVector_Policy;

  KOKKOS_INLINE_FUNCTION
  void operator () (const ResidualScalar_Tag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void operator () (const ResidualVector_Tag& tag, const int& i) const;

  Kokkos::View<LO****, PHX::Device> localDOFView;

  Albany::DeviceView1d<const ST> x_constView_device;

private:
  typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
};

template<typename Traits>
class GatherVerticallyContractedSolution<PHAL::AlbanyTraits::Jacobian,Traits>
    : public GatherVerticallyContractedSolutionBase<PHAL::AlbanyTraits::Jacobian,Traits> {

public:

  GatherVerticallyContractedSolution(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl);

  void evaluateFields(typename Traits::EvalData d);

  KOKKOS_INLINE_FUNCTION
  void operator () (const int i) const;

private:
  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
};

template<typename Traits>
class GatherVerticallyContractedSolution<PHAL::AlbanyTraits::Tangent,Traits>
    : public GatherVerticallyContractedSolutionBase<PHAL::AlbanyTraits::Tangent,Traits> {

public:

  GatherVerticallyContractedSolution(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl);

  void evaluateFields(typename Traits::EvalData d);

  KOKKOS_INLINE_FUNCTION
  void operator () (const int i) const;

private:
  typedef typename PHAL::AlbanyTraits::Tangent::ScalarT ScalarT;
};

template<typename Traits>
class GatherVerticallyContractedSolution<PHAL::AlbanyTraits::DistParamDeriv,Traits>
    : public GatherVerticallyContractedSolutionBase<PHAL::AlbanyTraits::DistParamDeriv,Traits> {

public:

  GatherVerticallyContractedSolution(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl);

  void evaluateFields(typename Traits::EvalData d);

  KOKKOS_INLINE_FUNCTION
  void operator () (const int i) const;

private:
  typedef typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT ScalarT;
};


template<typename Traits>
class GatherVerticallyContractedSolution<PHAL::AlbanyTraits::HessianVec,Traits>
    : public GatherVerticallyContractedSolutionBase<PHAL::AlbanyTraits::HessianVec,Traits> {

public:

  GatherVerticallyContractedSolution(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl);

  void evaluateFields(typename Traits::EvalData d);

  KOKKOS_INLINE_FUNCTION
  void operator () (const int i) const;

private:
  typedef typename PHAL::AlbanyTraits::HessianVec::ScalarT ScalarT;
};

}

#endif
