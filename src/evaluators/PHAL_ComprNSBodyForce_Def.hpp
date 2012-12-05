//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace PHAL {
const double pi = 3.1415926535897932385;
//**********************************************************************

template<typename EvalT, typename Traits>
ComprNSBodyForce<EvalT, Traits>::
ComprNSBodyForce(const Teuchos::ParameterList& p) :
  force(p.get<std::string>("Body Force Name"),
 	p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ) 
{
  cout << "Compr NS body force constructor!" << endl; 
  Teuchos::ParameterList* bf_list = 
    p.get<Teuchos::ParameterList*>("Parameter List");

  std::string type = bf_list->get("Type", "None");
  if (type == "None") {
    bf_type = NONE;
  }

  this->addEvaluatedField(force);

  Teuchos::RCP<PHX::DataLayout> gradient_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Gradient Data Layout");
  std::vector<PHX::DataLayout::size_type> dims;
  gradient_dl->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];
  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout");
  vector_dl->dimensions(dims);
  vecDim  = dims[2];

cout << " in Compr NS Stokes source! " << endl;
cout << " vecDim = " << vecDim << endl;
cout << " numDims = " << numDims << endl;
cout << " numQPs = " << numQPs << endl; 


  this->setName("ComprNSBodyForce"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ComprNSBodyForce<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(force,fm); 
}

//**********************************************************************
template<typename EvalT, typename Traits>
void ComprNSBodyForce<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
 if (bf_type == NONE) {
   for (std::size_t cell=0; cell < workset.numCells; ++cell) 
     for (std::size_t qp=0; qp < numQPs; ++qp)       
       for (std::size_t i=0; i < vecDim; ++i) 
  	 force(cell,qp,i) = 0.0;
 }
}

}

