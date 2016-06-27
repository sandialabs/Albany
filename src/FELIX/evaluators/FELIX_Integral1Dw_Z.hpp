/*
 * FELIX_Integral1Dw_Z.hpp
 *
 *  Created on: Jun 15, 2016
 *      Author: abarone
 */

#ifndef FELIX_INTEGRAL1DW_Z_HPP_
#define FELIX_INTEGRAL1DW_Z_HPP_

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"

#include "PHAL_AlbanyTraits.hpp"

namespace FELIX {
/** \brief Integral 1D w_Z

    This evaluator computes the integral int1d_b^z of w_z
*/

template<typename EvalT, typename Traits>
class Integral1Dw_ZBase : public PHX::EvaluatorWithBaseImpl<Traits>,
		    public PHX::EvaluatorDerived<EvalT, Traits> {

public:

	Integral1Dw_ZBase(const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Albany::Layouts>& dl);

	virtual ~Integral1Dw_ZBase(){};

	void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

	virtual void evaluateFields(typename Traits::EvalData d) = 0;

protected:

  typedef typename EvalT::ScalarT ScalarT;

  // Output:
  PHX::MDField<ScalarT,Cell,Node>  int1Dw_z;

  std::size_t numNodes;

  Teuchos::RCP<const CellTopologyData> cell_topo;

  bool StokesThermoCoupled;

  int offset, neq;
};

template<typename EvalT, typename Traits> class Integral1Dw_Z;


template<typename Traits>
class Integral1Dw_Z<PHAL::AlbanyTraits::Residual,Traits>
    : public Integral1Dw_ZBase<PHAL::AlbanyTraits::Residual,Traits> {

public:

	  Integral1Dw_Z(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl);

	  void evaluateFields(typename Traits::EvalData d);

	  KOKKOS_INLINE_FUNCTION
	  void operator () (const int i) const;

private:
  typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
};

template<typename Traits>
class Integral1Dw_Z<PHAL::AlbanyTraits::Jacobian,Traits>
    : public Integral1Dw_ZBase<PHAL::AlbanyTraits::Jacobian,Traits> {

public:

	Integral1Dw_Z(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl);

  void evaluateFields(typename Traits::EvalData d);

  KOKKOS_INLINE_FUNCTION
  void operator () (const int i) const;

private:
  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
};

template<typename Traits>
class Integral1Dw_Z<PHAL::AlbanyTraits::Tangent,Traits>
    : public Integral1Dw_ZBase<PHAL::AlbanyTraits::Tangent,Traits> {

public:

	Integral1Dw_Z(const Teuchos::ParameterList& p,
                  const Teuchos::RCP<Albany::Layouts>& dl);

    void evaluateFields(typename Traits::EvalData d);

    KOKKOS_INLINE_FUNCTION
    void operator () (const int i) const;

private:
  typedef typename PHAL::AlbanyTraits::Tangent::ScalarT ScalarT;
};

template<typename Traits>
class Integral1Dw_Z<PHAL::AlbanyTraits::DistParamDeriv,Traits>
    : public Integral1Dw_ZBase<PHAL::AlbanyTraits::DistParamDeriv,Traits> {

public:

	Integral1Dw_Z(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl);

    void evaluateFields(typename Traits::EvalData d);

    KOKKOS_INLINE_FUNCTION
    void operator () (const int i) const;

private:
  typedef typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT ScalarT;
};


#ifdef ALBANY_ENSEMBLE
template<typename Traits>
class Integral1Dw_Z<PHAL::AlbanyTraits::MPResidual,Traits>
    : public Integral1Dw_ZBase<PHAL::AlbanyTraits::MPResidual,Traits> {

public:

	Integral1Dw_Z(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl);

  void evaluateFields(typename Traits::EvalData d);

  KOKKOS_INLINE_FUNCTION
  void operator () (const int i) const;

private:
  typedef typename PHAL::AlbanyTraits::MPResidual::ScalarT ScalarT;
};

template<typename Traits>
class Integral1Dw_Z<PHAL::AlbanyTraits::MPJacobian,Traits>
    : public Integral1Dw_ZBase<PHAL::AlbanyTraits::MPJacobian,Traits> {

public:

	Integral1Dw_Z(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl);

  void evaluateFields(typename Traits::EvalData d);

  KOKKOS_INLINE_FUNCTION
  void operator () (const int i) const;

private:
  typedef typename PHAL::AlbanyTraits::MPJacobian::ScalarT ScalarT;
};

template<typename Traits>
class Integral1Dw_Z<PHAL::AlbanyTraits::MPTangent,Traits>
    : public Integral1Dw_ZBase<PHAL::AlbanyTraits::MPTangent,Traits> {

public:

	Integral1Dw_Z(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl);

  void evaluateFields(typename Traits::EvalData d);

  KOKKOS_INLINE_FUNCTION
  void operator () (const int i) const;

private:
  typedef typename PHAL::AlbanyTraits::MPTangent::ScalarT ScalarT;
};
#endif

}

#endif /* FELIX_INTEGRAL1DW_Z_HPP_ */
