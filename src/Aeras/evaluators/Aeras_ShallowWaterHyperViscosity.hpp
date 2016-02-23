//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AERAS_SHALLOWWATERHYPERVISCOSITY_HPP
#define AERAS_SHALLOWWATERHYPERVISCOSITY_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Sacado_ParameterAccessor.hpp" 
#include "Albany_Layouts.hpp"
#include "Aeras_EvaluatorUtilities.hpp"

namespace Aeras {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits>
class ShallowWaterHyperViscosity : public PHX::EvaluatorWithBaseImpl<Traits>,
		    public PHX::EvaluatorDerived<EvalT, Traits>,
		    public Sacado::ParameterAccessor<EvalT, SPL_Traits> {

public:

  typedef typename EvalT::ScalarT ScalarT;

  ShallowWaterHyperViscosity(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);
  
  ScalarT& getValue(const std::string &n); 
 
  typedef typename EvalT::MeshScalarT MeshScalarT;

private:

          
  // Input:
  PHX::MDField<MeshScalarT,Cell,Node,QuadPoint,Dim> wGradBF;
  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim,Dim> Ugrad;

  enum HVTYPE {CONSTANT};
  HVTYPE hvType;

  bool useHyperviscosity;
  double hvTau; 
  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim> hyperviscosity;

  std::size_t numQPs, numDims, numNodes, vecDim, spatialDim;

  MDFieldMemoizer<Traits> memoizer_;
          
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
public:

 typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

 struct ShallowWaterHyperViscosity_Tag{};
 typedef Kokkos::RangePolicy<ExecutionSpace, ShallowWaterHyperViscosity_Tag> ShallowWaterHyperViscosity_Policy;

 KOKKOS_INLINE_FUNCTION
 void operator() (const ShallowWaterHyperViscosity_Tag& tag, const int& cell) const;

#endif
          
};
  
  

  

}

#endif
