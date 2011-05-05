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


#ifndef QCAD_SCHRODINGERPOTENTIAL_HPP
#define QCAD_SCHRODINGERPOTENTIAL_HPP

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
 * \brief Evaluates Poisson Source Term 
 */
namespace QCAD 
{
	template<typename EvalT, typename Traits>
	class SchrodingerPotential : 
  public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits>,
  public Sacado::ParameterAccessor<EvalT, SPL_Traits> 
  {
	public:
  	typedef typename EvalT::ScalarT ScalarT;
  	typedef typename EvalT::MeshScalarT MeshScalarT;

  	SchrodingerPotential(Teuchos::ParameterList& p);
  
  	void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);
  
  	void evaluateFields(typename Traits::EvalData d);
  
  	//! Function to allow parameters to be exposed for embedded analysis
  	ScalarT& getValue(const std::string &n);

	private:

  	//! Reference parameter list generator to check xml input file
  	Teuchos::RCP<const Teuchos::ParameterList>
    		getValidSchrodingerPotentialParameters() const;

  	// Suzey: need to assign values to private variables, so remove "const"
  	ScalarT potentialValue( const int numDim, const MeshScalarT* coord);

  	//! input
  	std::size_t numQPs;
  	std::size_t numDims;
  	PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim> coordVec;
        PHX::MDField<ScalarT,Cell,QuadPoint> psi;  //wavefunction

  	//! output
        PHX::MDField<ScalarT,Cell,QuadPoint> V; //potential 

  	//! energy parameter of potential, precise meaning dependent on type of potential:
        //   Parabolic case -> confinement energy
  	ScalarT E0;

	//! constant scaling of potential
  	ScalarT scalingFactor;
  	
  	//! string variable to differ the various devices implementation
  	std::string potentialType;

        //! units
        double energy_unit_in_eV, length_unit_in_m;
	};
}


#endif
