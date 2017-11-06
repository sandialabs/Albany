//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


// Generic implementations that can be used for more sophisticated functions

namespace PHAL {

//----------------------------------------------------------------------------
template <typename EvalT>
IdentityCoordFunctionTraits<EvalT>::
IdentityCoordFunctionTraits(Teuchos::ParameterList& p) {

  numEqn = p.get<int>("Number of Equations");
  eqnOffset = p.get<int>("Equation Offset");

}

// **********************************************************************
template<typename EvalT>
void
IdentityCoordFunctionTraits<EvalT>::
computeBCs(double* coord, std::vector<ScalarT>& BCVals, const RealType time) {

  // Apply the desired function to the coordinate values here
  for(int i = 0; i < numEqn; i++)

    BCVals[i] = coord[i];

}

}
