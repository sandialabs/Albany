/********************************************************************\
*            Albany, Copyright (2012) Sandia Corporation             *
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
*    Questions to Glen Hansen, gahanse@sandia.gov                    *
\********************************************************************/


#ifndef PHAL_LANGEVINNOISETERM_HPP
#define PHAL_LANGEVINNOISETERM_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "QCAD_EvaluatorTools.hpp"

// Random and Gaussian number distribution
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>

/** \brief Finite Element Interpolation Evaluator

    This evaluator interpolates nodal DOF values to quad points.

*/
namespace PHAL {

template<typename EvalT, typename Traits>
class LangevinNoiseTerm : public PHX::EvaluatorWithBaseImpl<Traits>,
	     public PHX::EvaluatorDerived<EvalT, Traits>,
       public QCAD::EvaluatorTools<EvalT, Traits>
{

public:

  typedef typename EvalT::ScalarT ScalarT;

  LangevinNoiseTerm(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

  ScalarT& getValue(const std::string &n);


private:
 
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Input:
  PHX::MDField<ScalarT,Cell,QuadPoint> rho;
  
  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint> noiseTerm;

  unsigned int numQPs, numDims, numNodes;

  ScalarT sd;
  Teuchos::Array<int> duration;

  boost::mt19937 rng;
  Teuchos::RCP<boost::normal_distribution<double> > nd;
  Teuchos::RCP<boost::variate_generator<boost::mt19937&, boost::normal_distribution<double> > > var_nor;
//  Teuchos::RCP<boost::normal_distribution<ScalarT> > nd;
//  Teuchos::RCP<boost::variate_generator<boost::mt19937&, boost::normal_distribution<ScalarT> > > var_nor;

  // generate seed convenience function
  long seedgen();

};
}

#endif
