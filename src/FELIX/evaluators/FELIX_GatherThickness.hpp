//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef FELIX_GATHERTHICKNESS_HPP
#define FELIX_GATHERTHICKNESS_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"


#include "PHAL_AlbanyTraits.hpp"


namespace FELIX {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits>
class GatherThicknessBase : public PHX::EvaluatorWithBaseImpl<Traits>,
		    public PHX::EvaluatorDerived<EvalT, Traits> {

public:

  GatherThicknessBase(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl);

  virtual ~GatherThicknessBase(){};

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  virtual void evaluateFields(typename Traits::EvalData d)=0;

protected:


  typedef typename EvalT::ScalarT ScalarT;

  // Output:
  PHX::MDField<ScalarT,Cell,Node>  thickness;

  std::size_t vecDimFO;
  std::size_t vecDim;
  std::size_t numNodes;
  std::size_t offset; // Offset of first DOF being gathered

  int HLevel;
  std::string meshPart;

  Teuchos::RCP<const CellTopologyData> cell_topo;
};


template<typename EvalT, typename Traits> class GatherThickness;



template<typename Traits>
class GatherThickness<PHAL::AlbanyTraits::Residual,Traits>
    : public GatherThicknessBase<PHAL::AlbanyTraits::Residual,Traits> {

public:

  GatherThickness(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl);

  void evaluateFields(typename Traits::EvalData d);

  KOKKOS_INLINE_FUNCTION
  void operator () (const int i) const;

private:
  typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
};

template<typename Traits>
class GatherThickness<PHAL::AlbanyTraits::Jacobian,Traits>
    : public GatherThicknessBase<PHAL::AlbanyTraits::Jacobian,Traits> {

public:

  GatherThickness(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl);

  void evaluateFields(typename Traits::EvalData d);

  KOKKOS_INLINE_FUNCTION
  void operator () (const int i) const;

private:
  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
};

template<typename Traits>
class GatherThickness<PHAL::AlbanyTraits::Tangent,Traits>
    : public GatherThicknessBase<PHAL::AlbanyTraits::Tangent,Traits> {

public:

  GatherThickness(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl);

  void evaluateFields(typename Traits::EvalData d);

  KOKKOS_INLINE_FUNCTION
  void operator () (const int i) const;

private:
  typedef typename PHAL::AlbanyTraits::Tangent::ScalarT ScalarT;
};

template<typename Traits>
class GatherThickness<PHAL::AlbanyTraits::DistParamDeriv,Traits>
    : public GatherThicknessBase<PHAL::AlbanyTraits::DistParamDeriv,Traits> {

public:

  GatherThickness(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl);

  void evaluateFields(typename Traits::EvalData d);

  KOKKOS_INLINE_FUNCTION
  void operator () (const int i) const;

private:
  typedef typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT ScalarT;
};


#ifdef ALBANY_ENSEMBLE
template<typename Traits>
class GatherThickness<PHAL::AlbanyTraits::MPResidual,Traits>
    : public GatherThicknessBase<PHAL::AlbanyTraits::MPResidual,Traits> {

public:

  GatherThickness(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl);

  void evaluateFields(typename Traits::EvalData d);

  KOKKOS_INLINE_FUNCTION
  void operator () (const int i) const;

private:
  typedef typename PHAL::AlbanyTraits::MPResidual::ScalarT ScalarT;
};

template<typename Traits>
class GatherThickness<PHAL::AlbanyTraits::MPJacobian,Traits>
    : public GatherThicknessBase<PHAL::AlbanyTraits::MPJacobian,Traits> {

public:

  GatherThickness(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl);

  void evaluateFields(typename Traits::EvalData d);

  KOKKOS_INLINE_FUNCTION
  void operator () (const int i) const;

private:
  typedef typename PHAL::AlbanyTraits::MPJacobian::ScalarT ScalarT;
};

template<typename Traits>
class GatherThickness<PHAL::AlbanyTraits::MPTangent,Traits>
    : public GatherThicknessBase<PHAL::AlbanyTraits::MPTangent,Traits> {

public:

  GatherThickness(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl);

  void evaluateFields(typename Traits::EvalData d);

  KOKKOS_INLINE_FUNCTION
  void operator () (const int i) const;

private:
  typedef typename PHAL::AlbanyTraits::MPTangent::ScalarT ScalarT;
};
#endif


template<typename EvalT, typename Traits> class GatherThickness3D;



template<typename Traits>
class GatherThickness3D<PHAL::AlbanyTraits::Residual,Traits>
    : public GatherThicknessBase<PHAL::AlbanyTraits::Residual,Traits> {

public:

  GatherThickness3D(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl);

  void evaluateFields(typename Traits::EvalData d);

  KOKKOS_INLINE_FUNCTION
  void operator () (const int i) const;

private:
  typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
};

template<typename Traits>
class GatherThickness3D<PHAL::AlbanyTraits::Jacobian,Traits>
    : public GatherThicknessBase<PHAL::AlbanyTraits::Jacobian,Traits> {

public:

  GatherThickness3D(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl);

  void evaluateFields(typename Traits::EvalData d);

  KOKKOS_INLINE_FUNCTION
  void operator () (const int i) const;

private:
  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
};

template<typename Traits>
class GatherThickness3D<PHAL::AlbanyTraits::Tangent,Traits>
    : public GatherThicknessBase<PHAL::AlbanyTraits::Tangent,Traits> {

public:

  GatherThickness3D(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl);

  void evaluateFields(typename Traits::EvalData d);

  KOKKOS_INLINE_FUNCTION
  void operator () (const int i) const;

private:
  typedef typename PHAL::AlbanyTraits::Tangent::ScalarT ScalarT;
};

template<typename Traits>
class GatherThickness3D<PHAL::AlbanyTraits::DistParamDeriv,Traits>
    : public GatherThicknessBase<PHAL::AlbanyTraits::DistParamDeriv,Traits> {

public:

  GatherThickness3D(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl);

  void evaluateFields(typename Traits::EvalData d);

  KOKKOS_INLINE_FUNCTION
  void operator () (const int i) const;

private:
  typedef typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT ScalarT;
};

#ifdef ALBANY_ENSEMBLE
template<typename Traits>
class GatherThickness3D<PHAL::AlbanyTraits::MPResidual,Traits>
    : public GatherThicknessBase<PHAL::AlbanyTraits::MPResidual,Traits> {

public:

  GatherThickness3D(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl);

  void evaluateFields(typename Traits::EvalData d);

  KOKKOS_INLINE_FUNCTION
  void operator () (const int i) const;

private:
  typedef typename PHAL::AlbanyTraits::MPResidual::ScalarT ScalarT;
};

template<typename Traits>
class GatherThickness3D<PHAL::AlbanyTraits::MPJacobian,Traits>
    : public GatherThicknessBase<PHAL::AlbanyTraits::MPJacobian,Traits> {

public:

  GatherThickness3D(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl);

  void evaluateFields(typename Traits::EvalData d);

  KOKKOS_INLINE_FUNCTION
  void operator () (const int i) const;

private:
  typedef typename PHAL::AlbanyTraits::MPJacobian::ScalarT ScalarT;
};

template<typename Traits>
class GatherThickness3D<PHAL::AlbanyTraits::MPTangent,Traits>
    : public GatherThicknessBase<PHAL::AlbanyTraits::MPTangent,Traits> {

public:

  GatherThickness3D(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl);

  void evaluateFields(typename Traits::EvalData d);

  KOKKOS_INLINE_FUNCTION
  void operator () (const int i) const;

private:
  typedef typename PHAL::AlbanyTraits::MPTangent::ScalarT ScalarT;
};
#endif


}

#endif
