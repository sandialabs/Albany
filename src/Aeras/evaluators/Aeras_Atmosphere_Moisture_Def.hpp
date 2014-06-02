//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <vector>
#include <string>

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

namespace Aeras {
using Teuchos::rcp;
using PHX::MDALayout;


template<typename EvalT, typename Traits>
Atmosphere<EvalT, Traits>::
Atmosphere(Teuchos::ParameterList& p,
           const Teuchos::RCP<Albany::Layouts>& dl) :
  coordVec  (p.get<std::string> ("Coordinate Vector Name"), dl->node_3vector ),
  ResidualIn (p.get<std::string> ("Residual Name In"), dl->node_vector),
  ResidualOut(p.get<std::string> ("Residual Name"), dl->node_vector),
  tracersOld(p.get<std::string> ("Tracer Vector Old Name"),     
      rcp(new MDALayout<Cell,Node,Dim,Dim>(
             dl->node_vector->dimension(0),
             dl->node_vector->dimension(1),
             p.get<int>("Number of Tracers",0),
             p.get<int>("Number of Levels",0)))),
  tracersNew(p.get<std::string> ("Tracer Vector New Name"),     
      rcp(new MDALayout<Cell,Node,Dim,Dim>(
             dl->node_vector->dimension(0),
             dl->node_vector->dimension(1),
             p.get<int>("Number of Tracers",0),
             p.get<int>("Number of Levels",0)))),
  U         (p.get<std::string> ("QP Variable Name"),       dl->qp_vector),
  numNodes(0), numCoords(0), 
  numTracers(p.get<int>("Number of Tracers")),
  numLevels (p.get<int>("Number of Levels")),
  worksetSize(0)
{  
  this->addDependentField(U);
  this->addDependentField(coordVec);
  this->addDependentField(ResidualIn);

  this->addEvaluatedField(ResidualOut);
  this->addEvaluatedField(tracersOld);
  this->addEvaluatedField(tracersNew);
  this->setName("Aeras::Atmosphere"+PHX::TypeString<EvalT>::value);
}

// **********************************************************************
template<typename EvalT, typename Traits> 
void Atmosphere<EvalT, Traits>::postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(U,fm);
  this->utils.setFieldData(coordVec,fm);
  this->utils.setFieldData(tracersOld,fm);
  this->utils.setFieldData(tracersNew,fm);
  this->utils.setFieldData(ResidualIn,fm);
  this->utils.setFieldData(ResidualOut,fm);

  typename std::vector< typename PHX::template MDField<MeshScalarT,Cell,Vertex,Dim>::size_type > dims;
  coordVec.dimensions(dims); //get dimensions

  worksetSize = dims[0];
  numNodes = dims[1];
  numCoords = dims[2];
}

// **********************************************************************
template<typename EvalT, typename Traits>
void Atmosphere<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{ 
  unsigned int numCells = workset.numCells;

  for (std::size_t cell=0; cell < numCells; ++cell) {
    for (std::size_t node=0; node < numNodes; ++node) {
      ResidualOut(cell,node,0) = ResidualIn(cell,node,0);
      ResidualOut(cell,node,1) = ResidualIn(cell,node,1);
      ResidualOut(cell,node,2) = ResidualIn(cell,node,2);
    }
  }

  Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > wsCoords = workset.wsCoords;

  for (std::size_t cell=0; cell < numCells; ++cell) {
    for (std::size_t node = 0; node < numNodes; ++node) {
      for (std::size_t t=0; t < numTracers; ++t) { 
        for (std::size_t l=1; l < numLevels-1; ++l) { 
          tracersNew(cell,node,t,l) = tracersOld(cell,node,t,l);
        }
      }
    }
  }
  // Copy New to Old for next iteration.
  tracersOld = tracersNew;
}
}
