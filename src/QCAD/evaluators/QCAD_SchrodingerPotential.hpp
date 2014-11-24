//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef QCAD_SCHRODINGERPOTENTIAL_HPP
#define QCAD_SCHRODINGERPOTENTIAL_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Sacado_ParameterAccessor.hpp"
#include "Stokhos_KL_ExponentialRandomField.hpp"
#include "Teuchos_Array.hpp"

namespace QCAD {
/** 
 * \brief Evaluates Poisson Source Term 
 */
	template<typename EvalT, typename Traits>
	class SchrodingerPotential : 
  public PHX::EvaluatorWithBaseImpl<Traits>,
  public PHX::EvaluatorDerived<EvalT, Traits>,
  public Sacado::ParameterAccessor<EvalT, SPL_Traits> 
  {
	public:
  	typedef typename EvalT::ScalarT ScalarT;
  	typedef typename EvalT::MeshScalarT MeshScalarT;

  	SchrodingerPotential(Teuchos::ParameterList& p,
                 const Teuchos::RCP<Albany::Layouts>& dl);
  
  	void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);
  
  	void evaluateFields(typename Traits::EvalData d);
  
  	//! Function to allow parameters to be exposed for embedded analysis
  	ScalarT& getValue(const std::string &n);

	private:

  	//! Reference parameter list generator to check xml input file
  	Teuchos::RCP<const Teuchos::ParameterList>
    		getValidSchrodingerPotentialParameters() const;

  	ScalarT parabolicPotentialValue( const int numDim, const MeshScalarT* coord);
  	ScalarT finiteWallPotential( const int numDim, const MeshScalarT* coord);
  	ScalarT stringFormulaPotential( const int numDim, const MeshScalarT* coord);

  	//! input
  	std::size_t numQPs;
  	std::size_t numDims;
  	PHX::MDField<MeshScalarT,Cell,QuadPoint,Dim> coordVec;
    PHX::MDField<ScalarT,Cell,QuadPoint> psi;  //wavefunction

  	//! output
    PHX::MDField<ScalarT,Cell,QuadPoint> V; //potential 

  	//! energy parameter of potential, precise meaning dependent on type of potential:
    //  Parabolic case -> confinement energy
    //  Finite Wall case -> barrier height 
  	ScalarT E0;

    //!! specific parameters for string formula
    std::string stringFormula;
    
    //! specific parameters for Finite Wall 
    double barrEffMass; // in [m0]
    double barrWidth;   // in length_unit_in_m
    double wellEffMass;
    double wellWidth; 
    
	  //! constant scaling of potential
  	ScalarT scalingFactor;
  	
  	std::string potentialType;
    std::string potentialStateName;

    //! units
    double energy_unit_in_eV, length_unit_in_m;
	};
}


#endif
