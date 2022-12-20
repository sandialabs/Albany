//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_GATHER2DFIELD_HPP
#define LANDICE_GATHER2DFIELD_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Albany_Layouts.hpp"


#include "PHAL_AlbanyTraits.hpp"


namespace LandIce {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits>
class Gather2DFieldBase : public PHX::EvaluatorWithBaseImpl<Traits>,
		                      public PHX::EvaluatorDerived<EvalT, Traits>
{
public:

  Gather2DFieldBase(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl);

  virtual ~Gather2DFieldBase () = default;

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  virtual void evaluateFields(typename Traits::EvalData d) = 0;

protected:

  typedef typename EvalT::ScalarT ScalarT;

  // Output:
  PHX::MDField<ScalarT,Cell,Node>  field2D;

  std::size_t vecDim;
  std::size_t numNodes;
  std::size_t offset; // Offset of first DOF being gathered

  int fieldLevel;
  std::string meshPart;

  Teuchos::RCP<const CellTopologyData> cell_topo;
};

// ================ Gather2DField =============== //

template<typename EvalT, typename Traits> class Gather2DField;

template<typename Traits>
class Gather2DField<PHAL::AlbanyTraits::Residual,Traits>
    : public Gather2DFieldBase<PHAL::AlbanyTraits::Residual,Traits> {

public:
  Gather2DField(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl);

  void evaluateFields(typename Traits::EvalData d);

private:
  typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
};

template<typename Traits>
class Gather2DField<PHAL::AlbanyTraits::Jacobian,Traits>
    : public Gather2DFieldBase<PHAL::AlbanyTraits::Jacobian,Traits> {

public:

  Gather2DField(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl);

  void evaluateFields(typename Traits::EvalData d);

private:
  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
};

template<typename Traits>
class Gather2DField<PHAL::AlbanyTraits::Tangent,Traits>
    : public Gather2DFieldBase<PHAL::AlbanyTraits::Tangent,Traits> {

public:

  Gather2DField(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl);

  void evaluateFields(typename Traits::EvalData /* d */) {}

private:
  typedef typename PHAL::AlbanyTraits::Tangent::ScalarT ScalarT;
};

template<typename Traits>
class Gather2DField<PHAL::AlbanyTraits::DistParamDeriv,Traits>
    : public Gather2DFieldBase<PHAL::AlbanyTraits::DistParamDeriv,Traits> {

public:

  Gather2DField(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl);

  void evaluateFields(typename Traits::EvalData /* d */) {};

private:
  typedef typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT ScalarT;
};

template<typename Traits>
class Gather2DField<PHAL::AlbanyTraits::HessianVec,Traits>
    : public Gather2DFieldBase<PHAL::AlbanyTraits::HessianVec,Traits> {

public:

  Gather2DField(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl);

  void evaluateFields(typename Traits::EvalData d);

private:
  typedef typename PHAL::AlbanyTraits::HessianVec::ScalarT ScalarT;
};

// ================ GatherExtruded2DField =============== //

template<typename EvalT, typename Traits> class GatherExtruded2DField;

template<typename Traits>
class GatherExtruded2DField<PHAL::AlbanyTraits::Residual,Traits>
    : public Gather2DFieldBase<PHAL::AlbanyTraits::Residual,Traits> {

public:

  GatherExtruded2DField(const Teuchos::ParameterList& p,
                        const Teuchos::RCP<Albany::Layouts>& dl);

  void evaluateFields(typename Traits::EvalData d);

private:
  typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
};

template<typename Traits>
class GatherExtruded2DField<PHAL::AlbanyTraits::Jacobian,Traits>
    : public Gather2DFieldBase<PHAL::AlbanyTraits::Jacobian,Traits> {

public:

  GatherExtruded2DField(const Teuchos::ParameterList& p,
                        const Teuchos::RCP<Albany::Layouts>& dl);

  void evaluateFields(typename Traits::EvalData d);

private:
  typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
};

template<typename Traits>
class GatherExtruded2DField<PHAL::AlbanyTraits::Tangent,Traits>
    : public Gather2DFieldBase<PHAL::AlbanyTraits::Tangent,Traits> {

public:

  GatherExtruded2DField(const Teuchos::ParameterList& p,
                        const Teuchos::RCP<Albany::Layouts>& dl);

  void evaluateFields(typename Traits::EvalData /* d */) {}

private:
  typedef typename PHAL::AlbanyTraits::Tangent::ScalarT ScalarT;
};

template<typename Traits>
class GatherExtruded2DField<PHAL::AlbanyTraits::DistParamDeriv,Traits>
    : public Gather2DFieldBase<PHAL::AlbanyTraits::DistParamDeriv,Traits> {

public:

  GatherExtruded2DField(const Teuchos::ParameterList& p,
                        const Teuchos::RCP<Albany::Layouts>& dl);

  void evaluateFields(typename Traits::EvalData /* d */) {}

private:
  typedef typename PHAL::AlbanyTraits::DistParamDeriv::ScalarT ScalarT;
};

template<typename Traits>
class GatherExtruded2DField<PHAL::AlbanyTraits::HessianVec,Traits>
    : public Gather2DFieldBase<PHAL::AlbanyTraits::HessianVec,Traits> {

public:

  GatherExtruded2DField(const Teuchos::ParameterList& p,
                        const Teuchos::RCP<Albany::Layouts>& dl);

  void evaluateFields(typename Traits::EvalData d);

private:
  typedef typename PHAL::AlbanyTraits::HessianVec::ScalarT ScalarT;
};

}

#endif
