/*
 * LandIce_HydrostaticPressure.hpp
 *
 *  Created on: Jun 6, 2016
 *      Author: abarone
 */

#ifndef LANDICE_HYDROSTATIC_PRESSURE_HPP
#define LANDICE_HYDROSTATIC_PRESSURE_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "PHAL_Dimension.hpp"
#include "Albany_Layouts.hpp"
#include "Albany_SacadoTypes.hpp"
#include "PHAL_Utilities.hpp"

namespace LandIce
{

/** \brief Hydrostatic pressure

    This evaluator evaluates the hydrostatic approximation of the pressure to compute the pressure-melting point Tm(p)
*/

template<typename EvalT, typename Traits, typename SurfHeightST>
class HydrostaticPressure : public PHX::EvaluatorWithBaseImpl<Traits>,
                            public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  //typedef typename EvalT::ParamScalarT ParamScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  HydrostaticPressure (const Teuchos::ParameterList& p,
                       const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup (typename Traits::SetupData d,
                              PHX::FieldManager<Traits>& fm);

  void evaluateFields(typename Traits::EvalData d);

private:
  // This is just to allow ETI machinery to work. In a real setting, ScalarT should always be constructible from TemperatureST.
  typedef typename Albany::StrongestScalarType<MeshScalarT,SurfHeightST>::type OutputST;

  // Input:
  PHX::MDField<const MeshScalarT,Cell,Node,Dim> z;
  PHX::MDField<const SurfHeightST,Cell,Node>     s; //surface height

  // Output:
  PHX::MDField<OutputST,Cell,Node> pressure; // [Pa], [kg m^{-1} s^{-2}]

  int numNodes;

  double rho_i;  // [kg m^{-3}]
  double g;      // [m s^{-2}]
  double pressure_scaling; // [kg m^{-2} s^{-2}]

  PHAL::MDFieldMemoizer<Traits> memoizer;

  typedef Kokkos::View<int***, PHX::Device>::execution_space ExecutionSpace;

  typedef Kokkos::RangePolicy< ExecutionSpace > Hydrostatic_Pressure_Policy;

  KOKKOS_INLINE_FUNCTION
  void operator() (const int& i) const;

};

} // namespace LandIce

#endif // LANDICE_HYDROSTATIC_PRESSURE_HPP
