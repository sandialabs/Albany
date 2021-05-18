//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_ICE_OVERBURDEN_HPP
#define LANDICE_ICE_OVERBURDEN_HPP 1

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"
#include "Albany_ScalarOrdinalTypes.hpp"
#include "Albany_DiscretizationUtils.hpp"

namespace LandIce
{

/** \brief Ice overburden

    This evaluator evaluates the ice overburden P_o = rho_i*g*H,
    with H being the ice thickness
*/

template<typename EvalT, typename Traits>
class IceOverburden : public PHX::EvaluatorWithBaseImpl<Traits>,
                      public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  typedef typename EvalT::ParamScalarT ParamScalarT;

  IceOverburden (const Teuchos::ParameterList& p,
                 const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData,
                              PHX::FieldManager<Traits>&) {}

  void evaluateFields(typename Traits::EvalData d);

private:

  // Input:
  PHX::MDField<const RealType>  H;

  // Output:
  PHX::MDField<RealType>        P_o;

  bool eval_on_side;
  std::string sideSetName;  // Only used if eval_on_side=true

  unsigned int numPts;

  double rho_i;
  double g;
  
  Albany::LocalSideSetInfo sideSet;

public:

  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

  struct IceOverburden_Tag{};

  typedef Kokkos::RangePolicy<ExecutionSpace, IceOverburden_Tag> IceOverburden_Policy;

  KOKKOS_INLINE_FUNCTION
  void operator() (const IceOverburden_Tag& tag, const int& side_or_cell_idx) const;

};

} // Namespace LandIce

#endif // LANDICE_ICE_OVERBURDEN_HPP
