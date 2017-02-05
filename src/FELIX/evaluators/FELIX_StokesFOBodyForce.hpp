//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_STOKESFOBODYFORCE_HPP
#define FELIX_STOKESFOBODYFORCE_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

namespace FELIX {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits>
class StokesFOBodyForce : public PHX::EvaluatorWithBaseImpl<Traits>,
		    public PHX::EvaluatorDerived<EvalT, Traits> {

public:

  typedef typename EvalT::ScalarT ScalarT;

  StokesFOBodyForce(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);
  

private:
 
  typedef typename EvalT::MeshScalarT MeshScalarT;
  typedef typename EvalT::ParamScalarT ParamScalarT;

  // Input:  
  PHX::MDField<const ScalarT,Cell,QuadPoint> muFELIX;
  PHX::MDField<const MeshScalarT,Cell,QuadPoint, Dim> coordVec;
  PHX::MDField<const ParamScalarT,Cell,QuadPoint, Dim> surfaceGrad;
  PHX::MDField<const ParamScalarT,Cell,QuadPoint> surface;
  Teuchos::Array<double> gravity;

  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim> force;

   //Random field types
  enum BFTYPE {NONE, FO_INTERP_SURF_GRAD, FO_SURF_GRAD_PROVIDED, POISSON, FO_SINCOS2D, FO_COSEXP2D, FO_COSEXP2DFLIP, FO_COSEXP2DALL,
	  FO_SINCOSZ, FO_SINEXP2D, FO_DOME, FO_XZMMS};
  BFTYPE bf_type;

  std::size_t numQPs;
  std::size_t numDims;
  std::size_t vecDimFO;
  std::size_t numNodes;

  //Glen's law parameters
  double n; 
  double A;
  //ISMIP-HOM parameter
  double alpha;
  //physical parameters
  double g; //gravity
  double rho; //ice density  

  Teuchos::ParameterList* stereographicMapList;
  bool useStereographicMap;

  #ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  public:

  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

  struct FO_INTERP_SURF_GRAD_Tag{}; 
  struct FO_SURF_GRAD_PROVIDED_Tag{};
  struct POISSON_Tag{};
  struct FO_SINCOS2D_Tag{};
  struct FO_COSEXP2D_Tag{};
  struct FO_COSEXP2DFLIP_Tag{};
  struct FO_COSEXP2DALL_Tag{};
  struct FO_SINCOSZ_Tag{};
  struct FO_SINEXP2D_Tag{};
  struct FO_DOME_Tag{};
  struct FO_XZMMS_Tag{};

  typedef Kokkos::RangePolicy<ExecutionSpace,FO_INTERP_SURF_GRAD_Tag> FO_INTERP_SURF_GRAD_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,FO_SURF_GRAD_PROVIDED_Tag> FO_SURF_GRAD_PROVIDED_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,POISSON_Tag> POISSON_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,FO_SINCOS2D_Tag> FO_SINCOS2D_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,FO_COSEXP2D_Tag> FO_COSEXP2D_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,FO_COSEXP2DFLIP_Tag> FO_COSEXP2DFLIP_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,FO_COSEXP2DALL_Tag> FO_COSEXP2DALL_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,FO_SINCOSZ_Tag> FO_SINCOSZ_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,FO_SINEXP2D_Tag> FO_SINEXP2D_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,FO_DOME_Tag> FO_DOME_Policy;
  typedef Kokkos::RangePolicy<ExecutionSpace,FO_XZMMS_Tag> FO_XZMMS_Policy;

  KOKKOS_INLINE_FUNCTION
  void operator() (const FO_INTERP_SURF_GRAD_Tag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const FO_SURF_GRAD_PROVIDED_Tag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const POISSON_Tag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const FO_SINCOS2D_Tag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const FO_COSEXP2D_Tag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const FO_COSEXP2DFLIP_Tag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const FO_COSEXP2DALL_Tag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const FO_SINCOSZ_Tag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const FO_SINEXP2D_Tag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const FO_DOME_Tag& tag, const int& i) const;
  KOKKOS_INLINE_FUNCTION
  void operator() (const FO_XZMMS_Tag& tag, const int& i) const;
  
  double R, x_0, y_0, R2;

  double rho_g_kernel;

  #endif

};
}

#endif
