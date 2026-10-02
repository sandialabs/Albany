/*
 * LandIce_EnthalpyBasalResid.hpp
 *
 *  Created on: May 31, 2016
 *      Author: abarone
 */

#ifndef LANDICE_ENTHALPY_BASAL_RESID_HPP
#define LANDICE_ENTHALPY_BASAL_RESID_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "PHAL_Dimension.hpp"
#include "Albany_Layouts.hpp"
#include "Albany_ScalarOrdinalTypes.hpp"
#include "Albany_DiscretizationUtils.hpp"

namespace LandIce
{

/** \brief Enthalpy Basal Residual

  This evaluator integrates the basal enthalpy flux (Stefan condition) against the basis
  functions of the basal side and scatters it to the cell nodes
 */

template<typename EvalT, typename Traits, typename Type>
class EnthalpyBasalResid : public PHX::EvaluatorWithBaseImpl<Traits>,
                           public PHX::EvaluatorDerived<EvalT, Traits>
{
public:
  EnthalpyBasalResid(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d, PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;
  typedef typename EvalT::ParamScalarT ParamScalarT;

  // Input:
  PHX::MDField<const RealType>         BF;          // []
  PHX::MDField<const MeshScalarT>           w_measure;   // [km^2]
  PHX::MDField<const ScalarT>               basalMeltRateQP;      // [MW] = [m/yr]

  // Output:
  PHX::MDField<ScalarT> enthalpyBasalResid;      // [MW] = [k^{-2} kPa s^{-1} km^3]
  
  Albany::LocalSideSetInfo sideSet;

  Kokkos::DualView<int**, PHX::Device> sideNodes;
  std::string                     basalSideName;

  unsigned int numSideNodes;
  unsigned int numSideQPs;
  unsigned int sideDim;

  public:

  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

  struct Enthalpy_Basal_Residual_Tag{};

  typedef Kokkos::RangePolicy<ExecutionSpace,Enthalpy_Basal_Residual_Tag> Enthalpy_Basal_Residual_Policy;

  KOKKOS_INLINE_FUNCTION
  void operator() (const Enthalpy_Basal_Residual_Tag& tag, const int& i) const;

};

} // namespace LandIce

#endif // LANDICE_ENTHALPY_BASAL_RESID_HPP
