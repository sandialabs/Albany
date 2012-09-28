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


#ifndef FELIX_STOKESFOBODYFORCE_HPP
#define FELIX_STOKESFOBODYFORCE_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

namespace FELIX {
/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/

template<typename EvalT, typename Traits>
class StokesFOBodyForce : public PHX::EvaluatorWithBaseImpl<Traits>,
		    public PHX::EvaluatorDerived<EvalT, Traits> {

public:

  typedef typename EvalT::ScalarT ScalarT;

  StokesFOBodyForce(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);


private:
 
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:  
  PHX::MDField<ScalarT,Cell,QuadPoint> muFELIX;
  PHX::MDField<MeshScalarT,Cell,QuadPoint, Dim> coordVec;
  Teuchos::Array<double> gravity;
  
  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint,VecDim> force;

   //Radom field types
  enum BFTYPE {NONE, FO_SINCOS2D, FO_SINCOSZ, FO_ISMIPHOM_TESTA};
  BFTYPE bf_type;

  std::size_t numQPs;
  std::size_t numDims;
  std::size_t vecDim;

  //Glen's law parameters
  double n; 
  double A;
  //ISMIP-HOM parameter
  double alpha; 
};
}

#endif
