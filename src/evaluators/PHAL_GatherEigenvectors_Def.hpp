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


#include <vector>
#include <string>

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

namespace PHAL {


template<typename EvalT, typename Traits>
GatherEigenvectors<EvalT,Traits>::
GatherEigenvectors(const Teuchos::ParameterList& p)
{ 
  char buf[200];
  
  std::string eigenvector_name_root = p.get<string>("Eigenvector field name root"); 
  nEigenvectors = p.get<int>("Number of eigenvectors");

  Teuchos::RCP<PHX::DataLayout> dl = 
    p.get< Teuchos::RCP<PHX::DataLayout> >("Data Layout");

  eigenvector_Re.resize(nEigenvectors);
  eigenvector_Im.resize(nEigenvectors);
  for (std::size_t k = 0; k < nEigenvectors; ++k) {
    sprintf(buf, "%s_Re%d", eigenvector_name_root.c_str(), (int)k);
    PHX::MDField<ScalarT,Cell,Node> fr(buf,dl);
    eigenvector_Re[k] = fr;
    this->addEvaluatedField(eigenvector_Re[k]);

    sprintf(buf, "%s_Im%d", eigenvector_name_root.c_str(), (int)k);
    PHX::MDField<ScalarT,Cell,Node> fi(buf,dl);
    eigenvector_Im[k] = fi;
    this->addEvaluatedField(eigenvector_Im[k]);
  }
  
  this->setName("Gather Eigenvectors"+PHX::TypeString<EvalT>::value);
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

  Epetra_MultiVector& e_r = workset.eigenDataPtr->eigenvectorRe->getEpetraMultiVector();
  Epetra_MultiVector& e_i = workset.eigenDataPtr->eigenvectorIm->getEpetraMultiVector();

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID = workset.wsElNodeEqID[cell];
    
    for(std::size_t node =0; node < this->numNodes; ++node) {
      int offsetIntoVec = nodeID[node][0]; // neq==1 hardwired

      for (std::size_t k = 0; k < nEigenvectors; ++k) {
	(this->eigenvector_Re[k])(cell,node) = (*(e_r(k)))[offsetIntoVec];
	(this->eigenvector_Im[k])(cell,node) = (*(e_i(k)))[offsetIntoVec];
      }
    }
  }

}

// **********************************************************************

}
