//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef QCAD_PERMITTIVITY_HPP
#define QCAD_PERMITTIVITY_HPP

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

#include "Albany_Layouts.hpp"

#include "Albany_MaterialDatabase.hpp"

namespace QCAD {
/** 
 * \brief Evaluates thermal conductivity, either as a constant or a truncated
 * KL expansion.
 */
	template<typename EvalT, typename Traits>
	class Permittivity : 
  	public PHX::EvaluatorWithBaseImpl<Traits>,
  	public PHX::EvaluatorDerived<EvalT, Traits>,
  	public Sacado::ParameterAccessor<EvalT, SPL_Traits> 
	{
	public:
  	typedef typename EvalT::ScalarT ScalarT;
  	typedef typename EvalT::MeshScalarT MeshScalarT;

 	 	Permittivity(Teuchos::ParameterList& p,
                 const Teuchos::RCP<Albany::Layouts>& dl);
  
  	void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);
  
  	void evaluateFields(typename Traits::EvalData d);
  
  	ScalarT& getValue(const std::string &n);

	private:

  	//! Validate the name strings under "Permittivity" section in xml input file, 
  	Teuchos::RCP<const Teuchos::ParameterList>
    		getValidPermittivityParameters() const;

  	std::size_t numQPs;
  	std::size_t numDims;
  	PHX::MDField<ScalarT,Cell,QuadPoint> permittivity;
  	PHX::MDField<const ScalarT,Cell,QuadPoint> Temp;
  	PHX::MDField<const MeshScalarT,Cell,QuadPoint,Dim> coordVec;

  	//! Define boolean variables to categorize 
	  std::string typ;		// permittivity type
  	bool temp_dependent;		// is permittivity temperature dependent ?
  	bool position_dependent;	// is permittivity position dependent ?

  	//! Constant value
  	ScalarT constant_value;		// for constant permittivity
  	ScalarT factor;			// for temperature-dependent permittivity

	  //! Material database - holds permittivity among other quantities
    Teuchos::RCP<Albany::MaterialDatabase> materialDB;
    
    //! specific parameters for 1D MOSCapacitor
    double oxideWidth;
    double siliconWidth;
    
  };
	
}

#endif
