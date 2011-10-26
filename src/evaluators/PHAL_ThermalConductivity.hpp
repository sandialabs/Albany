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


#ifndef PHAL_THERMAL_CONDUCTIVITY_HPP
#define PHAL_THERMAL_CONDUCTIVITY_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Epetra_Vector.h"
#include "Sacado_ParameterAccessor.hpp"
#include "Stokhos_KL_ExponentialRandomField.hpp"
#include "Teuchos_Array.hpp"

#include "QCAD_MaterialDatabase.hpp"

namespace PHAL {
/** 
 * \brief Evaluates thermal conductivity, either as a constant or a truncated
 * KL expansion.

This class may be used in several ways. 

1. The simplest is to use a constant thermal conductivity across the entire domain (one element block,
one material), say with a value of 5.0. In this case, one would declare at the "Problem" level, that a 
constant thermal conductivity was being used, and its value was 5.0:

<ParameterList name="Problem">
   ...
    <ParameterList name="Thermal Conductivity">
       <Parameter name="Thermal Conductivity Type" type="string" value="Constant"/>
       <Parameter name="Value" type="double" value="5.0"/>
    </ParameterList>
</ParameterList>

An example of this is test problem ...

2. The other extreme is to have a multiple element block problem, say 3, with each element block corresponding
to a material. See the test problem ... for an example of this use case.

 */

template<typename EvalT, typename Traits>
class ThermalConductivity : 
  public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits>,
  public Sacado::ParameterAccessor<EvalT, SPL_Traits> {
  
public:
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  ThermalConductivity(Teuchos::ParameterList& p);
//  ThermalConductivity(){ cout << "In default const" << endl;}
//  ~ThermalConductivity(){ cout << "In destructor" << endl;}
  
  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);
  
  void evaluateFields(typename Traits::EvalData d);
  
  ScalarT& getValue(const std::string &n);

private:       

//! Validate the name strings under "Thermal Conductivity" section in xml input file, 
  Teuchos::RCP<const Teuchos::ParameterList>
               getValidThermalCondParameters() const;

  bool haveMatDB; // is a material database being used for thermal conductivity?

  bool is_constant;

  std::size_t numQPs;
  std::size_t numDims;
  PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim> coordVec;
  PHX::MDField<ScalarT,Cell,QuadPoint> thermalCond;

  //! Conductivity type
  std::string type; 

  //! Constant value
  ScalarT constant_value;

  //! Exponential random field
  Teuchos::RCP< Stokhos::KL::ExponentialRandomField<MeshScalarT> > exp_rf_kl;

  //! Values of the random variables
  Teuchos::Array<ScalarT> rv;

  //! Material database - holds thermal conductivity among other quantities
  Teuchos::RCP<QCAD::MaterialDatabase> materialDB;

  //! Convenience function to initialize constant thermal conductivity
  void init_constant(ScalarT value, Teuchos::ParameterList& p);

  //! Convenience function to initialize thermal conductivity based on 
  //  Truncated KL Expansion || Log Normal RF
  void init_KL_RF(std::string &type, Teuchos::ParameterList& subList, Teuchos::ParameterList& p);

};
}

#endif
