/*
 * LandIce_Integral1Dw_Z.hpp
 *
 *  Created on: Jun 15, 2016
 *      Author: abarone
 */

#ifndef LANDICE_INTEGRAL_1D_W_Z_HPP
#define LANDICE_INTEGRAL_1D_W_Z_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

#include "PHAL_AlbanyTraits.hpp"
#include "Albany_SacadoTypes.hpp"

namespace LandIce {
/** \brief Integral 1D w_Z

    This evaluator computes the integral int1d_b^z of w_z
*/

template<typename EvalT, typename Traits, typename ThicknessScalarT>
class Integral1Dw_ZBase : public PHX::EvaluatorWithBaseImpl<Traits>,
                  		    public PHX::EvaluatorDerived<EvalT, Traits> {
public:

	Integral1Dw_ZBase(const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Albany::Layouts>& dl);

	virtual ~Integral1Dw_ZBase(){};

	void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

	virtual void evaluateFields(typename Traits::EvalData /* d */) {}

protected:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::ParamScalarT ParamScalarT;

  // This only helps to let ETI machinery work. In a real problem,
  // ScalarT should always be constructible from a ThicknessScalarT
  typedef typename Albany::StrongestScalarType<ThicknessScalarT,ScalarT>::type OutputScalarT;

  // Input
  PHX::MDField<const ScalarT,Cell,Node>           basal_velocity;
  PHX::MDField<const ThicknessScalarT,Cell,Node>  thickness;

  // Output:
  PHX::MDField<OutputScalarT,Cell,Node>  int1Dw_z;

  std::size_t numNodes;

  Teuchos::RCP<const CellTopologyData> cell_topo;

  bool StokesThermoCoupled;

  int offset, neq;
};

template<typename EvalT, typename Traits, typename ThicknessScalarT>
class Integral1Dw_Z : public Integral1Dw_ZBase<EvalT,Traits,ThicknessScalarT> {
public:
	  Integral1Dw_Z(const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Albany::Layouts>& dl) :
      Integral1Dw_ZBase<EvalT,Traits,ThicknessScalarT>(p,dl) {}
};


template<typename Traits, typename ThicknessScalarT>
class Integral1Dw_Z<PHAL::AlbanyTraits::Residual,Traits,ThicknessScalarT>
    : public Integral1Dw_ZBase<PHAL::AlbanyTraits::Residual,Traits,ThicknessScalarT> {

public:

	  Integral1Dw_Z(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl);

	  void evaluateFields(typename Traits::EvalData d);

private:
  typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
};

template<typename Traits, typename ThicknessScalarT>
class Integral1Dw_Z<PHAL::AlbanyTraits::Jacobian,Traits,ThicknessScalarT>
    : public Integral1Dw_ZBase<PHAL::AlbanyTraits::Jacobian,Traits,ThicknessScalarT> {

public:

	Integral1Dw_Z(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl);

  void evaluateFields(typename Traits::EvalData d);

private:
  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
};

}

#endif // LANDICE_INTEGRAL_1D_W_Z_HPP
