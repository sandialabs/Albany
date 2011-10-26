/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


#ifndef PHAL_NSMATERIAL_PROPERTY_HPP
#define PHAL_NSMATERIAL_PROPERTY_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Epetra_Vector.h"
#include "Sacado_ParameterAccessor.hpp"
#include "Stokhos_KL_ExponentialRandomField.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_TwoDArray.hpp"

namespace PHAL {
/** 
 * \brief Evaluates thermal conductivity, either as a constant or a truncated
 * KL expansion.
 */

template<typename EvalT, typename Traits>
class NSMaterialProperty : 
  public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits>,
  public Sacado::ParameterAccessor<EvalT, SPL_Traits> {
  
public:
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  NSMaterialProperty(Teuchos::ParameterList& p);
  
  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);
  
  void evaluateFields(typename Traits::EvalData d);
  
  ScalarT& getValue(const std::string &n);

private:
  std::string name_mp;
  Teuchos::RCP<PHX::DataLayout> layout;
  PHX::MDField<MeshScalarT> coordVec;
  PHX::MDField<ScalarT> matprop;
  PHX::MDField<ScalarT> T;
  PHX::MDField<ScalarT> sigma_a;
  PHX::MDField<ScalarT> sigma_s;
  PHX::MDField<ScalarT> mu;
  PHX::DataLayout::size_type rank;
  std::vector<PHX::DataLayout::size_type> dims;

  // material property types
  enum MAT_PROP_TYPE {
    SCALAR_CONSTANT,
    VECTOR_CONSTANT,
    TENSOR_CONSTANT,
    KL_RAND_FIELD, 
    EXP_KL_RAND_FIELD,
    SQRT_TEMP,
    INV_SQRT_TEMP,
    NEUTRON_DIFFUSION
  };
  MAT_PROP_TYPE matPropType;

  //! Constant value
  ScalarT scalar_constant_value;
  Teuchos::Array<ScalarT> vector_constant_value;
  Teuchos::TwoDArray<ScalarT> tensor_constant_value;
  ScalarT ref_temp;

  //! Exponential random field
  Teuchos::RCP< Stokhos::KL::ExponentialRandomField<MeshScalarT> > exp_rf_kl;

  //! Values of the random variables
  Teuchos::Array<ScalarT> rv;
  Teuchos::Array<MeshScalarT> point;
};
}

#endif
