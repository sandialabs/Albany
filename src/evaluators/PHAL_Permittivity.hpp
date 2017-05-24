//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_PERMITTIVITY_HPP
#define PHAL_PERMITTIVITY_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Sacado_ParameterAccessor.hpp"
#ifdef ALBANY_STOKHOS
#include "Stokhos_KL_ExponentialRandomField.hpp"
#endif
#include "Teuchos_Array.hpp"

#include "Albany_MaterialDatabase.hpp"

namespace PHAL {
/** 
 * \brief Evaluates permittivity, either as a constant or a truncated
 * KL expansion.

This class may be used in two ways. 

1. The simplest is to use a constant permittivity across the entire domain (one element block,
one material), say with a value of 5.0. In this case, one would declare at the "Problem" level, that a 
constant permittivity was being used, and its value was 5.0:

<ParameterList name="Problem">
   ...
    <ParameterList name="Permittivity">
       <Parameter name="Permittivity Type" type="string" value="Constant"/>
       <Parameter name="Value" type="double" value="5.0"/>
    </ParameterList>
</ParameterList>

An example of this is test problem is PNP

2. The other extreme is to have a multiple element block problem, say 3, with each element block corresponding
to a material. Each element block has its own field manager, and different evaluators are used in each element
block.

 */

template<typename EvalT, typename Traits>
class Permittivity : 
  public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits>,
  public Sacado::ParameterAccessor<EvalT, SPL_Traits> {
  
public:
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  enum SG_RF {CONSTANT, UNIFORM, LOGNORMAL};

  Permittivity(Teuchos::ParameterList& p);
  
  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);
  
  void evaluateFields(typename Traits::EvalData d);
  
  ScalarT& getValue(const std::string &n);

private:       

//! Validate the name strings under "Permittivity" section in xml input file, 
  Teuchos::RCP<const Teuchos::ParameterList>
               getValidPermittivityParameters() const;

  bool is_constant;

  std::size_t numQPs;
  std::size_t numDims;
  //Input:
  PHX::MDField<const MeshScalarT,Cell,QuadPoint,Dim> coordVec;

  //Output:
  PHX::MDField<ScalarT,Cell,QuadPoint> permittivity;

  //! Permittivity type
  std::string type; 

  //! Constant value
  ScalarT constant_value;

#ifdef ALBANY_STOKHOS
  //! Exponential random field
  Teuchos::RCP< Stokhos::KL::ExponentialRandomField<RealType> > exp_rf_kl;
#endif

  //! Values of the random variables
  Teuchos::Array<ScalarT> rv;

  //! Material database - holds permittivity among other quantities
  Teuchos::RCP<Albany::MaterialDatabase> materialDB;

  //! Convenience function to initialize constant permittivity
  void init_constant(ScalarT value, Teuchos::ParameterList& p);

#ifdef ALBANY_STOKHOS
  //! Convenience function to initialize permittivity based on 
  //  Truncated KL Expansion || Log Normal RF
  void init_KL_RF(std::string &type, Teuchos::ParameterList& subList, Teuchos::ParameterList& p);
#endif

  SG_RF randField;

};
}

#endif
