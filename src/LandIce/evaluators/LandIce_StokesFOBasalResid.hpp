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

  ScalarT printedFF;

  // Input:
  // TODO: restore layout template arguments when removing old sideset layout
  PHX::MDField<const BetaScalarT> beta;       // Side, QuadPoint
  PHX::MDField<const ScalarT>     u;          // Side, QuadPoint, VecDim
  PHX::MDField<const RealType>    BF;         // Side, Node, QuadPoint
  PHX::MDField<const MeshScalarT> w_measure;  // Side, QuadPoint
  PHX::MDField<const ScalarT,Dim> homotopyParam;

  PHX::MDField<const ScalarT,Dim> homotopy;

  // Output:
  PHX::MDField<ScalarT,Cell,Node,VecDim> residual;

  Kokkos::View<int**, PHX::Device> sideNodes;
  std::string                     basalSideName;

  unsigned int numSideNodes;
  unsigned int numSideQPs;
  unsigned int sideDim;
  unsigned int vecDim;
  unsigned int vecDimFO;

  bool regularized;

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
