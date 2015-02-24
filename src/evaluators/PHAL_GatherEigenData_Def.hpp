//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
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
GatherEigenDataBase<EvalT,Traits>::
GatherEigenDataBase(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Albany::Layouts>& dl)
{ 
  char buf[200];
  
  std::string eigenvector_name_root = p.get<std::string>("Eigenvector name root"); 
  std::string eigenvalue_name_root = p.get<std::string>("Eigenvalue name root");

  nEigenvectors = p.get<int>("Number of eigenvectors");

  eigenvector_Re.resize(nEigenvectors);
  eigenvector_Im.resize(nEigenvectors);
  eigenvalue_Re.resize(nEigenvectors);
  eigenvalue_Im.resize(nEigenvectors);
  for (std::size_t k = 0; k < nEigenvectors; ++k) {

    // add real eigenvectors to evaluated fields
    sprintf(buf, "%s_Re%d", eigenvector_name_root.c_str(), (int)k);
    PHX::MDField<ScalarT,Cell,Node,Dim> fr(buf,dl->node_vector);
    eigenvector_Re[k] = fr;
    this->addEvaluatedField(eigenvector_Re[k]);

    // add imaginary eigenvectors to evaluated fields
    sprintf(buf, "%s_Im%d", eigenvector_name_root.c_str(), (int)k);
    PHX::MDField<ScalarT,Cell,Node,Dim> fi(buf,dl->node_vector);
    eigenvector_Im[k] = fi;
    this->addEvaluatedField(eigenvector_Im[k]);

    // add real eigenvalues to evaluated fields
    sprintf(buf, "%s_Re%d", eigenvalue_name_root.c_str(), (int)k);
    PHX::MDField<ScalarT> vr(buf,dl->workset_scalar);
    eigenvalue_Re[k] = vr;
    this->addEvaluatedField(eigenvalue_Re[k]);

    // add imaginary eigenvalues to evaluated fields
    sprintf(buf, "%s_Im%d", eigenvalue_name_root.c_str(), (int)k);
    PHX::MDField<ScalarT> vi(buf,dl->workset_scalar);
    eigenvalue_Im[k] = vi;
    this->addEvaluatedField(eigenvalue_Im[k]);

  }
  
  this->setName("Gather Eigenvectors 2D"+PHX::TypeString<EvalT>::value);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void GatherEigenDataBase<EvalT,Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  for (std::size_t k = 0; k < nEigenvectors; ++k) {
    this->utils.setFieldData(eigenvector_Re[k],fm);
    this->utils.setFieldData(eigenvector_Im[k],fm);
    this->utils.setFieldData(eigenvalue_Re[k],fm);
    this->utils.setFieldData(eigenvalue_Im[k],fm);
  }
  numNodes = (nEigenvectors > 0) ? eigenvector_Re[0].dimension(1) : 0;
}


template<typename EvalT, typename Traits>
void GatherEigenDataBase<EvalT,Traits>::
evaluateFields(typename Traits::EvalData workset)
{ 
  if(nEigenvectors == 0) return;

  if(workset.eigenDataPtr != Teuchos::null) {
  if(workset.eigenDataPtr->eigenvectorRe != Teuchos::null) {
    if(workset.eigenDataPtr->eigenvectorIm != Teuchos::null) {
  
      //Gather real and imaginary eigenvalues from workset Eigendata info structure
      const Epetra_MultiVector& e_r = *(workset.eigenDataPtr->eigenvectorRe);
      const Epetra_MultiVector& e_i = *(workset.eigenDataPtr->eigenvectorIm);
      const std::vector<double> &v_r = *(workset.eigenDataPtr->eigenvalueRe);
      const std::vector<double> &v_i = *(workset.eigenDataPtr->eigenvalueIm);
      int numVecsInWorkset = std::min(e_r.NumVectors(),e_i.NumVectors());
      int numVecsToGather  = std::min(numVecsInWorkset, (int)nEigenvectors);
      for (std::size_t k = 0; k < numVecsToGather; ++k) {
        (this->eigenvalue_Re[k])(0) = v_r[k];
        (this->eigenvalue_Im[k])(0) = v_i[k];
      }

      //Gather real and imaginary eigenvectors from workset Eigendata info structure
      for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
	const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID = workset.wsElNodeEqID[cell];
    
	for(std::size_t node =0; node < this->numNodes; ++node) {
	  int offset_eq1 = nodeID[node][0];
	  int offset_eq2 = nodeID[node][1];

	  for (std::size_t k = 0; k < numVecsToGather; ++k) {
	    (this->eigenvector_Re[k])(cell,node,0) = (*(e_r(k)))[offset_eq1];
	    (this->eigenvector_Im[k])(cell,node,0) = (*(e_i(k)))[offset_eq1];

	    (this->eigenvector_Re[k])(cell,node,1) = (*(e_r(k)))[offset_eq2];
	    (this->eigenvector_Im[k])(cell,node,1) = (*(e_i(k)))[offset_eq2];
	  }
	}
      }
    }

    else { // Only real parts of eigenvectors is given -- "gather" zeros into imaginary fields

      //Gather real and imaginary eigenvalues from workset Eigendata info structure
      const Epetra_MultiVector& e_r = *(workset.eigenDataPtr->eigenvectorRe);
      const std::vector<double> &v_r = *(workset.eigenDataPtr->eigenvalueRe);
      int numVecsInWorkset = e_r.NumVectors();
      int numVecsToGather  = std::min(numVecsInWorkset, (int)nEigenvectors);
      for (std::size_t k = 0; k < numVecsToGather; ++k) {
        (this->eigenvalue_Re[k])(0) = v_r[k];
        (this->eigenvalue_Im[k])(0) = 0.0;
      }

      //Gather real and imaginary eigenvectors from workset Eigendata info structure
      for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
	const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID = workset.wsElNodeEqID[cell];
    
	for(std::size_t node =0; node < this->numNodes; ++node) {
	  int offset_eq1 = nodeID[node][0];
	  int offset_eq2 = nodeID[node][1];

	  for (std::size_t k = 0; k < numVecsToGather; ++k) {
	    (this->eigenvector_Re[k])(cell,node,0) = (*(e_r(k)))[offset_eq1];
	    (this->eigenvector_Im[k])(cell,node,0) = 0.0;

	    (this->eigenvector_Re[k])(cell,node,1) = (*(e_r(k)))[offset_eq2];
	    (this->eigenvector_Im[k])(cell,node,1) = 0.0;
	  }
	}
      }
    }
  }
  // else (if both Re and Im are null) gather zeros into both??
  }
}

// **********************************************************************
//
//

template<typename EvalT, typename Traits>
GatherEigenData<EvalT, Traits>::
GatherEigenData(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl) :
  GatherEigenDataBase<EvalT, Traits>(p,dl)
{
}

// **********************************************************************
// Specialization: Residual
// **********************************************************************
//

template<typename Traits>
GatherEigenData<PHAL::AlbanyTraits::Residual, Traits>::
GatherEigenData(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl) :
  GatherEigenDataBase<PHAL::AlbanyTraits::Residual, Traits>(p,dl)
{
}


//
// **********************************************************************
// Specialization: Jacobian
// **********************************************************************
//

template<typename Traits>
GatherEigenData<PHAL::AlbanyTraits::Jacobian, Traits>::
GatherEigenData(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl) :
  GatherEigenDataBase<PHAL::AlbanyTraits::Jacobian, Traits>(p,dl)
{
}

template< typename Traits>
void GatherEigenData<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{ 

  if(nEigenvectors == 0) return;

  ScalarT* valptr;

  if(workset.eigenDataPtr != Teuchos::null) {
    if(workset.eigenDataPtr->eigenvectorRe != Teuchos::null) {
      if(workset.eigenDataPtr->eigenvectorIm != Teuchos::null) {

        //Gather real and imaginary eigenvalues
        const Epetra_MultiVector& e_r = *(workset.eigenDataPtr->eigenvectorRe);
        const Epetra_MultiVector& e_i = *(workset.eigenDataPtr->eigenvectorIm);
        const std::vector<double> &v_r = *(workset.eigenDataPtr->eigenvalueRe);
        const std::vector<double> &v_i = *(workset.eigenDataPtr->eigenvalueIm);
        int numVecsInWorkset = std::min(e_r.NumVectors(),e_i.NumVectors());
        int numVecsToGather  = std::min(numVecsInWorkset, (int)nEigenvectors);
        for (std::size_t k = 0; k < numVecsToGather; ++k) {
          (this->eigenvalue_Re[k])(0) = v_r[k];
          (this->eigenvalue_Im[k])(0) = v_i[k];
        }
  
        //Gather real and imaginary eigenvectors from workset Eigendata info structure
        for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
  	  const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID = workset.wsElNodeEqID[cell];
	  for(std::size_t node =0; node < this->numNodes; ++node) {
	    int offset_eq1 = nodeID[node][0];
	    int offset_eq2 = nodeID[node][1];

	    for (std::size_t k = 0; k < numVecsToGather; ++k) {
              valptr = &(this->eigenvector_Re[k](cell,node,0));
              *valptr = FadType(2, (*(e_r(k)))[offset_eq1]);
              valptr = &(this->eigenvector_Im[k](cell,node,0));
              *valptr = FadType(2, (*(e_r(k)))[offset_eq1]);

              valptr = &(this->eigenvector_Re[k](cell,node,1));
              *valptr = FadType(2, (*(e_r(k)))[offset_eq2]);
              valptr = &(this->eigenvector_Im[k](cell,node,1));
              *valptr = FadType(2, (*(e_r(k)))[offset_eq2]);
            }
          }
        }
      }
    }
  }
}

}
