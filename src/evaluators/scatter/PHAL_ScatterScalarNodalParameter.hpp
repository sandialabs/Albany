//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_SCATTER_SCALAR_NODAL_PARAMETER_HPP
#define PHAL_SCATTER_SCALAR_NODAL_PARAMETER_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Teuchos_ParameterList.hpp"

#include "PHAL_Dimension.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "Albany_Layouts.hpp"
#include "PHAL_Utilities.hpp"

namespace PHAL {

/** \brief Scatters parameter values from distributed vectors into
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
class ScatterScalarNodalParameterBase
    : public PHX::EvaluatorWithBaseImpl<Traits>,
      public PHX::EvaluatorDerived<EvalT, Traits>
{
public:
  ScatterScalarNodalParameterBase(const Teuchos::ParameterList& p,
                                  const Teuchos::RCP<Albany::Layouts>& dl);

  virtual ~ScatterScalarNodalParameterBase() = default;

  void postRegistrationSetup(typename Traits::SetupData d,
                             PHX::FieldManager<Traits>& vm);

protected:

  using ParamScalarT = typename EvalT::ParamScalarT;

  PHX::MDField<const ParamScalarT,Cell,Node> val;

  std::string param_name;
  int numNodes;

  MDFieldMemoizer<Traits> memoizer;
};

// General version for most evaluation types.
template<typename EvalT, typename Traits>
class ScatterScalarNodalParameter :
  public ScatterScalarNodalParameterBase<EvalT, Traits>
{
public:
  ScatterScalarNodalParameter(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
   : ScatterScalarNodalParameterBase<EvalT, Traits>(p, dl)
  {
    // Nothing to do here
  }

  void evaluateFields (typename Traits::EvalData d);
};

// General version for most evaluation types
template<typename EvalT, typename Traits>
class ScatterScalarExtruded2DNodalParameter :
    public ScatterScalarNodalParameterBase<EvalT, Traits>
{
public:
  ScatterScalarExtruded2DNodalParameter(const Teuchos::ParameterList& p,
                                        const Teuchos::RCP<Albany::Layouts>& dl)
   : ScatterScalarNodalParameterBase<EvalT, Traits>(p, dl)
  {
    // Nothing to do here
  }

  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename EvalT::ParamScalarT ParamScalarT;
};


// **************************************************************
// **************************************************************
// * Specializations
// **************************************************************
// **************************************************************

// **************************************************************
// Residual
// **************************************************************
template<typename Traits>
class ScatterScalarNodalParameter<AlbanyTraits::Residual,Traits> :
  public ScatterScalarNodalParameterBase<AlbanyTraits::Residual,Traits>
{
public:
  ScatterScalarNodalParameter(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl);

  // Old constructor, still needed by BCs that use PHX Factory
  ScatterScalarNodalParameter(const Teuchos::ParameterList& p);
  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename AlbanyTraits::Residual::ParamScalarT ParamScalarT;
  Teuchos::RCP< PHX::Tag<ParamScalarT> > nodal_field_tag;
  static const std::string className;
};

template<typename Traits>
class ScatterScalarExtruded2DNodalParameter<AlbanyTraits::Residual,Traits> :
  public ScatterScalarNodalParameterBase<AlbanyTraits::Residual,Traits>
{
public:
  ScatterScalarExtruded2DNodalParameter(const Teuchos::ParameterList& p,
                                        const Teuchos::RCP<Albany::Layouts>& dl);

  void evaluateFields(typename Traits::EvalData d);
private:
  typedef typename AlbanyTraits::Residual::ParamScalarT ParamScalarT;
  Teuchos::RCP< PHX::Tag<ParamScalarT> > nodal_field_tag;
  static const std::string className;
  int fieldLevel;
};

} // namespace PHAL

#endif // PHAL_SCATTER_SCALAR_NODAL_PARAMETER_HPP
