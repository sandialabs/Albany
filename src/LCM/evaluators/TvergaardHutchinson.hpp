//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef TVERGAARDHUTCHINSON_HPP
#define TVERGAARDHUTCHINSON_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Albany_Layouts.hpp"
#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_Cubature.hpp"

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
	Teuchos::RCP<Intrepid2::Cubature<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout,PHX::Device> >> cubature;
	//! Finite element basis for the midplane
	Teuchos::RCP<Intrepid2::Basis<RealType, Intrepid2::FieldContainer_Kokkos<RealType, PHX::Layout, PHX::Device>>> intrepidBasis;
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
