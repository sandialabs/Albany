//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_STOKES_FO_BASAL_RESID_HPP
#define LANDICE_STOKES_FO_BASAL_RESID_HPP 1

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"
#include "Albany_ScalarOrdinalTypes.hpp"
#include "PHAL_Dimension.hpp"

namespace LandIce
{

/** \brief Hydrology Residual Evaluator

    This evaluator evaluates the residual of the Hydrology model
*/

template<typename EvalT, typename Traits, typename BetaScalarT>
class StokesFOBasalResid : public PHX::EvaluatorWithBaseImpl<Traits>,
                           public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  StokesFOBasalResid (const Teuchos::ParameterList& p,
                      const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:

  // Input:
  PHX::MDField<const BetaScalarT,Side,QuadPoint>     beta;
  PHX::MDField<const ScalarT,Side,QuadPoint,VecDim>  u;
  PHX::MDField<const RealType,Side,Node,QuadPoint>   BF;
  PHX::MDField<const MeshScalarT,Side,QuadPoint>     w_measure;
  PHX::MDField<const MeshScalarT,Side,QuadPoint,Dim> normals;
  
  PHX::MDField<const ScalarT,Dim> homotopyParam;
  PHX::MDField<const ScalarT,Dim> homotopy;

  // Output:
  PHX::MDField<ScalarT,Cell,Node,VecDim> residual;

  Kokkos::View<int**, PHX::Device> sideNodes;
  std::string                     basalSideName;

  unsigned int numSideNodes;
  unsigned int numSideQPs;
  unsigned int vecDim;
  unsigned int vecDimFO;

  Albany::LocalSideSetInfo sideSet;

public:

  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;
  struct StokesFOBasalResid_Tag{};

  typedef Kokkos::RangePolicy<ExecutionSpace, StokesFOBasalResid_Tag> StokesFOBasalResid_Policy;

  KOKKOS_INLINE_FUNCTION
  void operator() (const StokesFOBasalResid_Tag& tag, const int& sideSet_idx) const;

};

} // Namespace LandIce

#endif // LANDICE_STOKES_FO_BASAL_RESID_HPP
