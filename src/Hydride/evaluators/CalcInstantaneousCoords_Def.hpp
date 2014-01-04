//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits>
CalcInstantaneousCoords<EvalT, Traits>::
CalcInstantaneousCoords(const Teuchos::ParameterList& p, const Teuchos::RCP<Albany::Layouts>& dl) :

  coordVec      (p.get<std::string> ("Coordinate Vector Name"), dl->vertices_vector),
  dispVec       (p.get<std::string> ("Solution Vector Name"), dl->node_vector),
  instCoords    (p.get<std::string> ("Instantaneous Coordinates Name"), dl->node_vector)

{

  this->addDependentField(dispVec);
  this->addDependentField(coordVec);
  this->addEvaluatedField(instCoords);

  // Get Dimensions
  std::vector<PHX::DataLayout::size_type> dim;
  dl->node_qp_vector->dimensions(dim);
  int containerSize = dim[0];
  numNodes = dim[1];
  numQPs = dim[2];
  numDims = dim[3];

  this->setName("CalcInstantaneousCoords"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void CalcInstantaneousCoords<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(dispVec, fm);
  this->utils.setFieldData(coordVec, fm);
  this->utils.setFieldData(instCoords, fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void CalcInstantaneousCoords<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

  for(std::size_t cell = 0; cell < workset.numCells; ++cell) {
    for(std::size_t node = 0; node < numNodes; ++node) {
      for(std::size_t eq = 0; eq < numDims; eq++)  {

          instCoords(cell, node, eq) = coordVec(cell, node, eq) + dispVec(cell, node, eq);

      }
    }
  }
}

//**********************************************************************
}
