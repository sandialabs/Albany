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


#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace LCM {

//**********************************************************************
template<typename EvalT, typename Traits>
CurrentCoords<EvalT, Traits>::
CurrentCoords(const Teuchos::ParameterList& p) :
  refCoords     (p.get<std::string>("Reference Coordinates Name"),
                 p.get<Teuchos::RCP<PHX::DataLayout> >("Coordinate Data Layout") ),
  displacement  (p.get<std::string>("Displacement Name"),
                 p.get<Teuchos::RCP<PHX::DataLayout> >("Coordinate Data Layout") ),
  currentCoords (p.get<std::string>("Current Coordinates Name"),
                 p.get<Teuchos::RCP<PHX::DataLayout> >("Coordinate Data Layout") )
{
  this->addDependentField(refCoords);
  this->addDependentField(displacement);

  this->addEvaluatedField(currentCoords);

  this->setName("Current Coordinates"+PHX::TypeString<EvalT>::value);

  Teuchos::RCP<PHX::DataLayout> dl =
     p.get< Teuchos::RCP<PHX::DataLayout> >("Coordinate Data Layout");
     std::vector<PHX::DataLayout::size_type> dims;
     dl->dimensions(dims);
     worksetSize = dims[0];
     numNodes = dims[1];
     numDims = dims[2];


}

//**********************************************************************
template<typename EvalT, typename Traits>
void CurrentCoords<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(refCoords,fm);
  this->utils.setFieldData(displacement,fm);
  this->utils.setFieldData(currentCoords,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void CurrentCoords<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  for (std::size_t cell=0; cell < workset.numCells; ++cell)
    for (std::size_t node=0; node < numNodes; ++node)
      for (std::size_t dim=0; dim < numDims; ++dim)
        currentCoords(cell,node,dim) = refCoords(cell,node,dim) + displacement(cell,node,dim);
}

//**********************************************************************
}

