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
QptLocation<EvalT, Traits>::
QptLocation(const Teuchos::ParameterList& p) :
  BF          (p.get<std::string>                   ("BF Name"),
	 p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Scalar Data Layout") ),
  GradBF      (p.get<std::string>                   ("Gradient BF Name"),
	 p.get<Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout") ),
  coordVec    (p.get<std::string>                   ("Coordinate Vector Name"),
	 p.get<Teuchos::RCP<PHX::DataLayout> >("Coordinate Vector Data Layout") ),
  gptLocation  (p.get<std::string>                 ("Integration Point Location Name"),
	 p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") )
{
  this->addDependentField(coordVec);
  this->addDependentField(GradBF);
  this->addDependentField(BF);

  this->addEvaluatedField(gptLocation);

  this->setName("Integration Point Location"+PHX::TypeString<EvalT>::value);

  Teuchos::RCP<PHX::DataLayout> node_dl =
     p.get< Teuchos::RCP<PHX::DataLayout> >("Node QP Vector Data Layout");
     std::vector<PHX::DataLayout::size_type> dims;
     node_dl->dimensions(dims);
     worksetSize = dims[0];
     numNodes = dims[1];
     numQPs  = dims[2];
     numDims = dims[3];


}

//**********************************************************************
template<typename EvalT, typename Traits>
void QptLocation<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(GradBF,fm);
  this->utils.setFieldData(coordVec,fm);
  this->utils.setFieldData(gptLocation,fm);

}

//**********************************************************************
template<typename EvalT, typename Traits>
void QptLocation<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{


 for (std::size_t cell=0; cell < workset.numCells; ++cell) {
	  for (std::size_t qp=0; qp < numQPs; ++qp) {
		for (std::size_t i=0; i<numDims; i++) {
			gptLocation(cell,qp,i) = 0.0;
			for (std::size_t node=0; node < numNodes; ++node) {
				gptLocation(cell,qp,i) += BF(cell,node, qp)*coordVec(cell,node,i);
			}
		}
	  }
 }

/*
  for (int i=0; i < gptLocation.size() ; i++) gptLocation[i] = 0.0;

  Intrepid::FunctionSpaceTools::
      evaluate<ScalarT>(gptLocation, coordVec, BF);
*/

}

//**********************************************************************
}

