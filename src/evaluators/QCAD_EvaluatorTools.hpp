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


#ifndef QCAD_EVALUATORTOOLS_HPP
#define QCAD_EVALUATORTOOLS_HPP

/** 
 * \brief Provides general-purpose template-specialized functions
 *  for use in other evaluator classes.
 */
namespace QCAD 
{
  template<typename EvalT, typename Traits> class EvaluatorTools;

  //! Specializations

  // Residual
  template<typename Traits>
  class EvaluatorTools<PHAL::AlbanyTraits::Residual, Traits>
  {
  public:
    typedef typename PHAL::AlbanyTraits::Residual::ScalarT ScalarT;
    typedef typename PHAL::AlbanyTraits::Residual::MeshScalarT MeshScalarT;
    
    EvaluatorTools();  
    double getDoubleValue(const ScalarT& t);	  
  };


  // Jacobian
  template<typename Traits>
  class EvaluatorTools<PHAL::AlbanyTraits::Jacobian, Traits>
  {
  public:
    typedef typename PHAL::AlbanyTraits::Jacobian::ScalarT ScalarT;
    typedef typename PHAL::AlbanyTraits::Jacobian::MeshScalarT MeshScalarT;
    
    EvaluatorTools();  
    double getDoubleValue(const ScalarT& t);	  
  };

  
  // Tangent
  template<typename Traits>
  class EvaluatorTools<PHAL::AlbanyTraits::Tangent, Traits>
  {
  public:
    typedef typename PHAL::AlbanyTraits::Tangent::ScalarT ScalarT;
    typedef typename PHAL::AlbanyTraits::Tangent::MeshScalarT MeshScalarT;
    
    EvaluatorTools();  
    double getDoubleValue(const ScalarT& t);	  
  };


  // Stochastic Galerkin Residual
  template<typename Traits>
  class EvaluatorTools<PHAL::AlbanyTraits::SGResidual, Traits>
  {
  public:
    typedef typename PHAL::AlbanyTraits::SGResidual::ScalarT ScalarT;
    typedef typename PHAL::AlbanyTraits::SGResidual::MeshScalarT MeshScalarT;
    
    EvaluatorTools();  
    double getDoubleValue(const ScalarT& t);	  
  };

  
  // Stochastic Galerkin Jacobian
  template<typename Traits>
  class EvaluatorTools<PHAL::AlbanyTraits::SGJacobian, Traits>
  {
  public:
    typedef typename PHAL::AlbanyTraits::SGJacobian::ScalarT ScalarT;
    typedef typename PHAL::AlbanyTraits::SGJacobian::MeshScalarT MeshScalarT;
    
    EvaluatorTools();  
    double getDoubleValue(const ScalarT& t);	  
  };

  
  // Multi-point residual
  template<typename Traits>
  class EvaluatorTools<PHAL::AlbanyTraits::MPResidual, Traits>
  {
  public:
    typedef typename PHAL::AlbanyTraits::MPResidual::ScalarT ScalarT;
    typedef typename PHAL::AlbanyTraits::MPResidual::MeshScalarT MeshScalarT;
    
    EvaluatorTools();  
    double getDoubleValue(const ScalarT& t);	  
  };


  // Multi-point Jacobian
  template<typename Traits>
  class EvaluatorTools<PHAL::AlbanyTraits::MPJacobian, Traits>
  {
  public:
    typedef typename PHAL::AlbanyTraits::MPJacobian::ScalarT ScalarT;
    typedef typename PHAL::AlbanyTraits::MPJacobian::MeshScalarT MeshScalarT;
    
    EvaluatorTools();  
    double getDoubleValue(const ScalarT& t);	  
  };

	
}

#endif
