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


#ifndef TVERGAARDHUTCHINSON_HPP
#define TVERGAARDHUTCHINSON_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"
#include "Intrepid_CellTools.hpp"
#include "Intrepid_Cubature.hpp"

namespace LCM {
/** \brief

    This evaluator surface traction based on
    Tvergaard-Hutchinson 1992 traction-separation law

*/
template<typename EvalT, typename Traits>
class TvergaardHutchinson : public PHX::EvaluatorWithBaseImpl<Traits>,
                            public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

	TvergaardHutchinson(const Teuchos::ParameterList& p,
	                        const Teuchos::RCP<Albany::Layouts>& dl);

	void postRegistrationSetup(typename Traits::SetupData d,
				     PHX::FieldManager<Traits>& vm);

	void evaluateFields(typename Traits::EvalData d);

private:

	typedef typename EvalT::ScalarT ScalarT;
	typedef typename EvalT::MeshScalarT MeshScalarT;

	// Input
	//! Numerical integration rule
	Teuchos::RCP<Intrepid::Cubature<RealType> > cubature;
	//! Finite element basis for the midplane
	Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > intrepidBasis;
	// current basis vector of the surface element
	PHX::MDField<ScalarT,Cell,QuadPoint,Dim, Dim> currentBasis;
	// displacement jump
	PHX::MDField<ScalarT,Cell,QuadPoint,Dim> jump;

	// material constants defining the traction-separation law
	RealType delta_1;
	RealType delta_2;
	RealType delta_c;
	RealType sigma_c;

	// material constants controlling the relative effect of shear and normal opening
	RealType beta_0;
	RealType beta_1;
	RealType beta_2;

	// Output
	PHX::MDField<ScalarT,Cell,QuadPoint,Dim> cohesiveTraction;

	unsigned int worksetSize;
	unsigned int numQPs;
	unsigned int numDims;

};
} // end namespace LCM

#endif
