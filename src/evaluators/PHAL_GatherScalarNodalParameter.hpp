//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_GATHER_SCALAR_NODAL_PARAMETER_HPP
#define PHAL_GATHER_SCALAR_NODAL_PARAMETER_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"

#include "Teuchos_ParameterList.hpp"

namespace PHAL {
/** \brief Gathers parameter values from distributed vectors into
    scalar nodal fields of the field manager

    Currently makes an assumption that the stride is constant for dofs
    and that the nmber of dofs is equal to the size of the solution
    names vector.

*/
// **************************************************************
// Base Class with Generic Implementations: Specializations for
// Automatic Differentiation Below
// **************************************************************

template<typename EvalT, typename Traits>
class GatherScalarNodalParameterBase
  : public PHX::EvaluatorWithBaseImpl<Traits>,
    public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  GatherScalarNodalParameterBase(const Teuchos::ParameterList& p,
                                 const Teuchos::RCP<Albany::Layouts>& dl);

  void postRegistrationSetup(typename Traits::SetupData d,
                             PHX::FieldManager<Traits>& vm);

  // This function requires template specialization, in derived class below
  virtual void evaluateFields(typename Traits::EvalData d) = 0;
  virtual ~GatherScalarNodalParameterBase(){};

protected:

  typedef typename EvalT::ParamScalarT ParamScalarT;
  PHX::MDField<ParamScalarT,Cell,Node> val;
  std::string param_name;
  std::size_t numNodes;
};

// General version for most evaluation types
template<typename EvalT, typename Traits>
class GatherScalarNodalParameter :
    public GatherScalarNodalParameterBase<EvalT, Traits>  {

public:
  GatherScalarNodalParameter(const Teuchos::ParameterList& p,
                       const Teuchos::RCP<Albany::Layouts>& dl);
  // Old constructor, still needed by BCs that use PHX Factory
  GatherScalarNodalParameter(const Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename EvalT::ParamScalarT ParamScalarT;
};

// General version for most evaluation types
template<typename EvalT, typename Traits>
class GatherScalarExtruded2DNodalParameter :
    public GatherScalarNodalParameterBase<EvalT, Traits>  {

public:
  GatherScalarExtruded2DNodalParameter(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl) :
    GatherScalarNodalParameterBase<EvalT, Traits>(p, dl) {
    fieldLevel = p.get<int>("Field Level");
  }

  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename EvalT::ParamScalarT ParamScalarT;
  int fieldLevel;
};


// **************************************************************
// **************************************************************
// * Specializations
// **************************************************************
// **************************************************************


// **************************************************************
// DistParamDeriv
// **************************************************************
template<typename Traits>
class GatherScalarNodalParameter<PHAL::AlbanyTraits::DistParamDeriv,Traits> :
    public GatherScalarNodalParameterBase<PHAL::AlbanyTraits::DistParamDeriv,
                                          Traits>  {

public:
  GatherScalarNodalParameter(const Teuchos::ParameterList& p,
                             const Teuchos::RCP<Albany::Layouts>& dl);
  // Old constructor, still needed by BCs that use PHX Factory
  GatherScalarNodalParameter(const Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::DistParamDeriv::ParamScalarT ParamScalarT;
};


template<typename Traits>
class GatherScalarExtruded2DNodalParameter<PHAL::AlbanyTraits::DistParamDeriv,Traits> :
    public GatherScalarNodalParameterBase<PHAL::AlbanyTraits::DistParamDeriv,
                                          Traits>  {

public:
  GatherScalarExtruded2DNodalParameter(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl) :
    GatherScalarNodalParameterBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p, dl) {
    fieldLevel = p.get<int>("Field Level");
  }

  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename PHAL::AlbanyTraits::DistParamDeriv::ParamScalarT ParamScalarT;
  int fieldLevel;
};

}

#endif
