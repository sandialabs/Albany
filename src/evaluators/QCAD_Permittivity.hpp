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


#ifndef QCAD_PERMITTIVITY_HPP
#define QCAD_PERMITTIVITY_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Epetra_Vector.h"
#include "Sacado_ParameterAccessor.hpp"
#include "Stokhos_KL_ExponentialRandomField.hpp"
#include "Teuchos_Array.hpp"

/** 
 * \brief Evaluates thermal conductivity, either as a constant or a truncated
 * KL expansion.
 */
namespace QCAD 
{
	template<typename EvalT, typename Traits>
	class Permittivity : 
  	public PHX::EvaluatorWithBaseImpl<Traits>,
  	public PHX::EvaluatorDerived<EvalT, Traits>,
  	public Sacado::ParameterAccessor<EvalT, SPL_Traits> 
	{
	public:
  	typedef typename EvalT::ScalarT ScalarT;
  	typedef typename EvalT::MeshScalarT MeshScalarT;

 	 	Permittivity(Teuchos::ParameterList& p);
  
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
  	PHX::MDField<ScalarT,Cell,QuadPoint> Temp;
  	PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim> coordVec;

  	//! Define boolean variables to categorize 
	std::string typ;		// permittivity type
  	bool temp_dependent;		// is permittivity temperature dependent ?
  	bool position_dependent;	// is permittivity position dependent ?

  	//! Constant value
  	ScalarT constant_value;		// for constant permittivity
  	ScalarT factor;			// for temperature-dependent permittivity
  
  	//! Variables to hold Silicon and SiO2 relative permittivity
  	ScalarT silicon_value;
  	ScalarT oxide_value;
  	ScalarT poly_value;
	};
	
}

#endif
