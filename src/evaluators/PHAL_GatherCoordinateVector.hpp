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


#ifndef PHAL_GATHER_COORDINATEVECTOR_HPP
#define PHAL_GATHER_COORDINATEVECTOR_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Epetra_Vector.h"

/** \brief Gathers Coordinates values from the Newton coordinates vector into 
    the nodal fields of the field manager

    Currently makes an assumption that the stride is constant for dofs
    and that the nmber of dofs is equal to the size of the coordinates
    names vector.

*/

template<typename EvalT, typename Traits> 
class GatherCoordinateVector : public PHX::EvaluatorWithBaseImpl<Traits>,
                          public PHX::EvaluatorDerived<EvalT, Traits>  {
  
public:
  
  GatherCoordinateVector(const Teuchos::ParameterList& p);
  
  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);
  
  void evaluateFields(typename Traits::EvalData d);
  
private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  PHX::MDField<MeshScalarT,Cell,Vertex,Dim> coordVec;
 
  bool  periodic;
  std::size_t numVertices;
  std::size_t numDim;
};


template<typename Traits> 
class GatherCoordinateVector<PHAL::AlbanyTraits::Tangent, Traits> 
                    : public PHX::EvaluatorWithBaseImpl<Traits>,
                      public PHX::EvaluatorDerived<PHAL::AlbanyTraits::Tangent, Traits>  {
  
public:
  
  GatherCoordinateVector(const Teuchos::ParameterList& p);
  
  void postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& vm);
  
  void evaluateFields(typename Traits::EvalData d);
  
private:

  typedef typename PHAL::AlbanyTraits::Tangent::ScalarT ScalarT;
  typedef typename PHAL::AlbanyTraits::Tangent::MeshScalarT MeshScalarT;

  PHX::MDField<MeshScalarT,Cell,Vertex,Dim> coordVec;
 
  bool  periodic;
  std::size_t numVertices;
  std::size_t numDim;
};
#ifndef PHAL_ETI
#include "PHAL_GatherCoordinateVector_Def.hpp"
#endif

#endif
