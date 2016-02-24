//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/13/14: does not get compiled if ALBANY_EPETRA_EXE is off.  Has epetra.

#include <vector>
#include <string>

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

namespace PHAL {


template<typename EvalT, typename Traits>
GatherEigenvectors<EvalT,Traits>::
GatherEigenvectors(const Teuchos::ParameterList& p,
                   const Teuchos::RCP<Albany::Layouts>& dl)
{ 
  char buf[200];
  
  std::string eigenvector_name_root = p.get<std::string>("Eigenvector field name root"); 
  nEigenvectors = p.get<int>("Number of eigenvectors");

  eigenvector_Re.resize(nEigenvectors);
  eigenvector_Im.resize(nEigenvectors);
  for (std::size_t k = 0; k < nEigenvectors; ++k) {
    sprintf(buf, "%s_Re%d", eigenvector_name_root.c_str(), (int)k);
    PHX::MDField<ScalarT,Cell,Node> fr(buf,dl->node_scalar);
    eigenvector_Re[k] = fr;
    this->addEvaluatedField(eigenvector_Re[k]);

    sprintf(buf, "%s_Im%d", eigenvector_name_root.c_str(), (int)k);
    PHX::MDField<ScalarT,Cell,Node> fi(buf,dl->node_scalar);
    eigenvector_Im[k] = fi;
    this->addEvaluatedField(eigenvector_Im[k]);
  }
  
  this->setName("Gather Eigenvectors" );
}

// **********************************************************************
template<typename EvalT, typename Traits>
void GatherEigenvectors<EvalT,Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  for (std::size_t k = 0; k < nEigenvectors; ++k) {
    this->utils.setFieldData(eigenvector_Re[k],fm);
    this->utils.setFieldData(eigenvector_Im[k],fm);
  }

  numNodes = (nEigenvectors > 0) ? eigenvector_Re[0].dimension(1) : 0;
}

// **********************************************************************

template<typename EvalT, typename Traits>
void GatherEigenvectors<EvalT,Traits>::
evaluateFields(typename Traits::EvalData workset)
{ 
  if(nEigenvectors == 0) return;

  if(workset.eigenDataPtr->eigenvectorRe != Teuchos::null) {
    if(workset.eigenDataPtr->eigenvectorIm != Teuchos::null) {
  
      //Gather real and imaginary parts from workset Eigendata info structure
      const Epetra_MultiVector& e_r = *(workset.eigenDataPtr->eigenvectorRe);
      const Epetra_MultiVector& e_i = *(workset.eigenDataPtr->eigenvectorIm);
      int numVecsInWorkset = std::min(e_r.NumVectors(),e_i.NumVectors());
      int numVecsToGather  = std::min(numVecsInWorkset, (int)nEigenvectors);

      for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
	const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID = workset.wsElNodeEqID[cell];
    
	for(std::size_t node =0; node < this->numNodes; ++node) {
	  int offsetIntoVec = nodeID[node][0]; // neq==1 hardwired

	  for (std::size_t k = 0; k < numVecsToGather; ++k) {
	    (this->eigenvector_Re[k])(cell,node) = (*(e_r(k)))[offsetIntoVec];
	    (this->eigenvector_Im[k])(cell,node) = (*(e_i(k)))[offsetIntoVec];
	  }
	}
      }
    }
    else { // Only real parts of eigenvectors is given -- "gather" zeros into imaginary fields

      //Gather real and imaginary parts from workset Eigendata info structure
      const Epetra_MultiVector& e_r = *(workset.eigenDataPtr->eigenvectorRe);
      int numVecsInWorkset = e_r.NumVectors();
      int numVecsToGather  = std::min(numVecsInWorkset, (int)nEigenvectors);

      for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
	const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID = workset.wsElNodeEqID[cell];
    
	for(std::size_t node =0; node < this->numNodes; ++node) {
	  int offsetIntoVec = nodeID[node][0]; // neq==1 hardwired

	  for (std::size_t k = 0; k < numVecsToGather; ++k) {
	    (this->eigenvector_Re[k])(cell,node) = (*(e_r(k)))[offsetIntoVec];
	    (this->eigenvector_Im[k])(cell,node) = 0.0;
	  }
	}
      }
    }
  }
  // else (if both Re and Im are null) gather zeros into both??
}

// **********************************************************************

}
