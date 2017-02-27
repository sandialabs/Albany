//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AERAS_SURFACEHEIGHT_HPP
#define AERAS_SURFACEHEIGHT_HPP

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
class SurfaceHeight : public PHX::EvaluatorWithBaseImpl<Traits>,
		    public PHX::EvaluatorDerived<EvalT, Traits>,
		    public Sacado::ParameterAccessor<EvalT, SPL_Traits> {

public:

  typedef typename EvalT::ScalarT ScalarT;

  SurfaceHeight(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);
  
  ScalarT& getValue(const std::string &n); 


private:
 
  const double pi;

  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<const MeshScalarT,Cell,QuadPoint, Dim> sphere_coord;

  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint> hs;

  unsigned int numQPs, numDims, numNodes;
  
  ScalarT hs0; // Mountain Height

  enum SURFHEIGHTTYPE {NONE, MOUNTAIN};
  SURFHEIGHTTYPE hs_type;
  MDFieldMemoizer<Traits> memoizer_;

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
public:
  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

  struct SurfaceHeight_Tag{};
  struct SurfaceHeight_MOUNTAIN_Tag{};

  typedef Kokkos::RangePolicy<ExecutionSpace, SurfaceHeight_Tag> SurfaceHeight_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace, SurfaceHeight_MOUNTAIN_Tag> SurfaceHeight_MOUNTAIN_Policy;
  
  KOKKOS_INLINE_FUNCTION
  void operator() (const SurfaceHeight_Tag& tag, const int& i) const;

  KOKKOS_INLINE_FUNCTION
  void operator() (const SurfaceHeight_MOUNTAIN_Tag& tag, const int& i) const;


#endif
 
};

}

#endif
