//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/12/14: only Epetra is SG and MP 

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "Tpetra_CrsMatrix.hpp"

// **********************************************************************
// Genereric Template Code for Constructor and PostRegistrationSetup
// **********************************************************************

namespace PHAL {

template <typename EvalT, typename Traits>
DirichletField_Base<EvalT, Traits>::
DirichletField_Base(Teuchos::ParameterList& p) :
  PHAL::DirichletBase<EvalT, Traits>(p) {

  // Get field type and corresponding layouts
  field_name = p.get<std::string>("Field Name");
}

// **********************************************************************
// Specialization: Residual
// **********************************************************************
template<typename Traits>
DirichletField<PHAL::AlbanyTraits::Residual, Traits>::
DirichletField(Teuchos::ParameterList& p) :
  DirichletField_Base<PHAL::AlbanyTraits::Residual, Traits>(p) {
}

// **********************************************************************
template<typename Traits>
void
DirichletField<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset) {


  const Albany::NodalDOFManager& fieldDofManager = dirichletWorkset.disc->getDOFManager(this->field_name);
  Teuchos::RCP<const Tpetra_Map> fiedNodeMap = dirichletWorkset.disc->getNodeMapT(this->field_name);
  const std::vector<GO>& nsNodesGIDs = dirichletWorkset.disc->getNodeSetGIDs().find(this->nodeSetID)->second;

  Teuchos::RCP<Tpetra_Vector> pvecT = dirichletWorkset.distParamLib->get(this->field_name)->vector();
  Teuchos::ArrayRCP<const ST> pT = pvecT->get1dView();
  Teuchos::RCP<Tpetra_Vector> fT = dirichletWorkset.fT;
  Teuchos::RCP<const Tpetra_Vector> xT = dirichletWorkset.xT;
  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();
  Teuchos::ArrayRCP<ST> fT_nonconstView = fT->get1dViewNonConst();

  const std::vector<std::vector<int> >& nsNodes = dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
      int lunk = nsNodes[inode][this->offset];
      GO node_gid = nsNodesGIDs[inode];
      int lfield = fieldDofManager.getLocalDOF(fiedNodeMap->getLocalElement(node_gid),this->offset);
      fT_nonconstView[lunk] = xT_constView[lunk] - pT[lfield];
  }
}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************
template<typename Traits>
DirichletField<PHAL::AlbanyTraits::Jacobian, Traits>::
DirichletField(Teuchos::ParameterList& p) :
  DirichletField_Base<PHAL::AlbanyTraits::Jacobian, Traits>(p) {
}

// **********************************************************************
template<typename Traits>
void DirichletField<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset) {

  const Albany::NodalDOFManager& fieldDofManager = dirichletWorkset.disc->getDOFManager(this->field_name);
  Teuchos::RCP<const Tpetra_Map> fiedNodeMap = dirichletWorkset.disc->getNodeMapT(this->field_name);
  const std::vector<GO>& nsNodesGIDs = dirichletWorkset.disc->getNodeSetGIDs().find(this->nodeSetID)->second;
  Teuchos::RCP<Tpetra_Vector> pvecT = dirichletWorkset.distParamLib->get(this->field_name)->vector();
  Teuchos::ArrayRCP<const ST> pT = pvecT->get1dView();
  Teuchos::RCP<Tpetra_Vector> fT = dirichletWorkset.fT;
  Teuchos::RCP<const Tpetra_Vector> xT = dirichletWorkset.xT;
  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();
  Teuchos::RCP<Tpetra_CrsMatrix> jacT = dirichletWorkset.JacT;

  const RealType j_coeff = dirichletWorkset.j_coeff;
  const std::vector<std::vector<int> >& nsNodes = dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

  bool fillResid = (fT != Teuchos::null);
  Teuchos::ArrayRCP<ST> fT_nonconstView;                                         
  if (fillResid) fT_nonconstView = fT->get1dViewNonConst();

  Teuchos::Array<LO> index(1);
  Teuchos::Array<ST> value(1); 
  size_t numEntriesT;  
  value[0] = j_coeff; 
  Teuchos::Array<ST> matrixEntriesT; 
  Teuchos::Array<LO> matrixIndicesT; 

  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
      int lunk = nsNodes[inode][this->offset];
      index[0] = lunk; 
      numEntriesT = jacT->getNumEntriesInLocalRow(lunk);
      matrixEntriesT.resize(numEntriesT); 
      matrixIndicesT.resize(numEntriesT); 

      jacT->getLocalRowCopy(lunk, matrixIndicesT(), matrixEntriesT(), numEntriesT); 
      for (int i=0; i<numEntriesT; i++) matrixEntriesT[i]=0;
      jacT->replaceLocalValues(lunk, matrixIndicesT(), matrixEntriesT()); 

      jacT->replaceLocalValues(lunk, index(), value()); 
      
      if (fillResid) {
        GO node_gid = nsNodesGIDs[inode];
        int lfield = fieldDofManager.getLocalDOF(fiedNodeMap->getLocalElement(node_gid),this->offset);
        fT_nonconstView[lunk] = xT_constView[lunk] - pT[lfield];
      }
  }
}

// **********************************************************************
// Specialization: Tangent
// **********************************************************************
template<typename Traits>
DirichletField<PHAL::AlbanyTraits::Tangent, Traits>::
DirichletField(Teuchos::ParameterList& p) :
  DirichletField_Base<PHAL::AlbanyTraits::Tangent, Traits>(p) {
}

// **********************************************************************
template<typename Traits>
void DirichletField<PHAL::AlbanyTraits::Tangent, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset) {

  const Albany::NodalDOFManager& fieldDofManager = dirichletWorkset.disc->getDOFManager(this->field_name);
  Teuchos::RCP<const Tpetra_Map> fiedNodeMap = dirichletWorkset.disc->getNodeMapT(this->field_name);
  const std::vector<GO>& nsNodesGIDs = dirichletWorkset.disc->getNodeSetGIDs().find(this->nodeSetID)->second;

  Teuchos::RCP<Tpetra_Vector> pvecT = dirichletWorkset.distParamLib->get(this->field_name)->vector();
  Teuchos::ArrayRCP<const ST> pT = pvecT->get1dView();
  Teuchos::RCP<Tpetra_Vector> fT = dirichletWorkset.fT;
  Teuchos::RCP<Tpetra_MultiVector> fpT = dirichletWorkset.fpT;
  Teuchos::RCP<Tpetra_MultiVector> JVT = dirichletWorkset.JVT;
  Teuchos::RCP<const Tpetra_Vector> xT = dirichletWorkset.xT;
  Teuchos::RCP<const Tpetra_MultiVector> VxT = dirichletWorkset.VxT;
  
  Teuchos::ArrayRCP<const ST> VxT_constView; 
  Teuchos::ArrayRCP<ST> fT_nonconstView;                                         
  if (fT != Teuchos::null) fT_nonconstView = fT->get1dViewNonConst();
  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();                                       

  const RealType j_coeff = dirichletWorkset.j_coeff;
  const std::vector<std::vector<int> >& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    int lunk = nsNodes[inode][this->offset];

    if (fT != Teuchos::null) { 
      GO node_gid = nsNodesGIDs[inode];
      int lfield = fieldDofManager.getLocalDOF(fiedNodeMap->getLocalElement(node_gid),this->offset);
      fT_nonconstView[lunk] = xT_constView[lunk] - pT[lfield];
    }
    
    if (JVT != Teuchos::null) {
      Teuchos::ArrayRCP<ST> JVT_nonconstView; 
      for (int i=0; i<dirichletWorkset.num_cols_x; i++) {
        JVT_nonconstView = JVT->getDataNonConst(i); 
        VxT_constView = VxT->getData(i); 
	      JVT_nonconstView[lunk] = j_coeff*VxT_constView[lunk];
      }
    }
    
    if (fpT != Teuchos::null) {
      Teuchos::ArrayRCP<ST> fpT_nonconstView;                                         
      for (int i=0; i<dirichletWorkset.num_cols_p; i++) {
        fpT_nonconstView = fpT->getDataNonConst(i); 
	      fpT_nonconstView[lunk] = 0;
      }
    }
  }
}

// **********************************************************************
// Specialization: DistParamDeriv
// **********************************************************************
template<typename Traits>
DirichletField<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
DirichletField(Teuchos::ParameterList& p) :
  DirichletField_Base<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p) {
}

// **********************************************************************
template<typename Traits>
void DirichletField<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset) {

  bool isFieldParameter =  dirichletWorkset.dist_param_deriv_name == this->field_name;
  Teuchos::RCP<Tpetra_MultiVector> fpVT = dirichletWorkset.fpVT;
  //non-const view of fpVT
  Teuchos::ArrayRCP<ST> fpVT_nonconstView;
  bool trans = dirichletWorkset.transpose_dist_param_deriv;
  int num_cols = fpVT->getNumVectors();

  const std::vector<std::vector<int> >& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

  // For (df/dp)^T*V we zero out corresponding entries in V
  if (trans) {
    Teuchos::RCP<Tpetra_MultiVector> VpT = dirichletWorkset.Vp_bcT;
    //non-const view of VpT
    Teuchos::ArrayRCP<ST> VpT_nonconstView;
    if(isFieldParameter) {
      const Albany::NodalDOFManager& fieldDofManager = dirichletWorkset.disc->getDOFManager(this->field_name);
      Teuchos::RCP<const Tpetra_Map> fiedNodeMap = dirichletWorkset.disc->getNodeMapT(this->field_name);
      const std::vector<GO>& nsNodesGIDs = dirichletWorkset.disc->getNodeSetGIDs().find(this->nodeSetID)->second;
      for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
        int lunk = nsNodes[inode][this->offset];
        GO node_gid = nsNodesGIDs[inode];
        int lfield = fieldDofManager.getLocalDOF(fiedNodeMap->getLocalElement(node_gid),this->offset);
        for (int col=0; col<num_cols; ++col) {
          VpT_nonconstView = VpT->getDataNonConst(col);
          fpVT_nonconstView = fpVT->getDataNonConst(col);
          fpVT_nonconstView[lfield] -= VpT_nonconstView[lunk];
          VpT_nonconstView[lunk] = 0.0;
         }
      }
    }
    else {
      for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
        int lunk = nsNodes[inode][this->offset];

        for (int col=0; col<num_cols; ++col) {
          VpT_nonconstView = VpT->getDataNonConst(col);
          VpT_nonconstView[lunk] = 0.0;
         }
      }
    }
  }

  // for (df/dp)*V we zero out corresponding entries in df/dp
  else {
    if(isFieldParameter) {
      const Albany::NodalDOFManager& fieldDofManager = dirichletWorkset.disc->getDOFManager(this->field_name);
      Teuchos::RCP<const Tpetra_Map> fiedNodeMap = dirichletWorkset.disc->getNodeMapT(this->field_name);
      const std::vector<GO>& nsNodesGIDs = dirichletWorkset.disc->getNodeSetGIDs().find(this->nodeSetID)->second;
      for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
        int lunk = nsNodes[inode][this->offset];
        GO node_gid = nsNodesGIDs[inode];
        int lfield = fieldDofManager.getLocalDOF(fiedNodeMap->getLocalElement(node_gid),this->offset);
        for (int col=0; col<num_cols; ++col) {
          //(*fpV)[col][lunk] = 0.0;
          fpVT_nonconstView = fpVT->getDataNonConst(col);
          fpVT_nonconstView[lunk] = -double(col == lfield);
        }
      }
    }
    else {
      for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
        int lunk = nsNodes[inode][this->offset];

        for (int col=0; col<num_cols; ++col) {
          fpVT_nonconstView = fpVT->getDataNonConst(col);
          fpVT_nonconstView[lunk] = 0.0;
        }
      }
    }
  }
}

// **********************************************************************
// Specialization: Stochastic Galerkin Residual
// **********************************************************************

#ifdef ALBANY_SG
template<typename Traits>
DirichletField<PHAL::AlbanyTraits::SGResidual, Traits>::
DirichletField(Teuchos::ParameterList& p) :
  DirichletField_Base<PHAL::AlbanyTraits::SGResidual, Traits>(p) {
}
// **********************************************************************
template<typename Traits>
void DirichletField<PHAL::AlbanyTraits::SGResidual, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset) {
  Teuchos::RCP<Tpetra_Vector> pvecT =
    dirichletWorkset.distParamLib->get(this->field_name)->vector();
  Teuchos::ArrayRCP<const ST> pT = pvecT->get1dView();
  Teuchos::RCP<Stokhos::EpetraVectorOrthogPoly> f =
    dirichletWorkset.sg_f;
  Teuchos::RCP<const Stokhos::EpetraVectorOrthogPoly> x =
    dirichletWorkset.sg_x;
  const std::vector<std::vector<int> >& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

  int nblock = x->size();
  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
      int lunk = nsNodes[inode][this->offset];
      for (int block=0; block<nblock; block++)
        (*f)[block][lunk] = (*x)[block][lunk];
      if(nblock>0) (*f)[0][lunk] -= pT[lunk];
  }
}

// **********************************************************************
// Specialization: Stochastic Galerkin Jacobian
// **********************************************************************
template<typename Traits>
DirichletField<PHAL::AlbanyTraits::SGJacobian, Traits>::
DirichletField(Teuchos::ParameterList& p) :
  DirichletField_Base<PHAL::AlbanyTraits::SGJacobian, Traits>(p) {
}

// **********************************************************************
template<typename Traits>
void DirichletField<PHAL::AlbanyTraits::SGJacobian, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset) {
  Teuchos::RCP<Tpetra_Vector> pvecT =
    dirichletWorkset.distParamLib->get(this->field_name)->vector();
  Teuchos::ArrayRCP<const ST> pT = pvecT->get1dView();
  Teuchos::RCP< Stokhos::EpetraVectorOrthogPoly> f =
    dirichletWorkset.sg_f;
  Teuchos::RCP< Stokhos::VectorOrthogPoly<Epetra_CrsMatrix> > jac =
    dirichletWorkset.sg_Jac;
  Teuchos::RCP<const Stokhos::EpetraVectorOrthogPoly> x =
    dirichletWorkset.sg_x;
  const RealType j_coeff = dirichletWorkset.j_coeff;
  const std::vector<std::vector<int> >& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

  RealType* matrixEntries;
  int*    matrixIndices;
  int     numEntries;
  int nblock = 0;
  if (f != Teuchos::null)
    nblock = f->size();
  int nblock_jac = jac->size();
  RealType diag=j_coeff;
  bool fillResid = (f != Teuchos::null);

  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
      int lunk = nsNodes[inode][this->offset];
      for (int block=0; block<nblock_jac; block++) {
        (*jac)[block].ExtractMyRowView(lunk, numEntries, matrixEntries,
                                       matrixIndices);
        for (int i=0; i<numEntries; i++)
          matrixEntries[i]=0;
      }
      (*jac)[0].ReplaceMyValues(lunk, 1, &diag, &lunk);
      if (fillResid) {
        for (int block=0; block<nblock; block++)
          (*f)[block][lunk] = (*x)[block][lunk];
        if(nblock>0) (*f)[0][lunk] -= pT[lunk];
      }
  }
}

// **********************************************************************
// Specialization: Stochastic Galerkin Tangent
// **********************************************************************
template<typename Traits>
DirichletField<PHAL::AlbanyTraits::SGTangent, Traits>::
DirichletField(Teuchos::ParameterList& p) :
  DirichletField_Base<PHAL::AlbanyTraits::SGTangent, Traits>(p) {
}

// **********************************************************************
template<typename Traits>
void DirichletField<PHAL::AlbanyTraits::SGTangent, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset) {
  Teuchos::RCP<Tpetra_Vector> pvecT =
    dirichletWorkset.distParamLib->get(this->field_name)->vector();
  Teuchos::ArrayRCP<const ST> pT = pvecT->get1dView();
  Teuchos::RCP<Stokhos::EpetraVectorOrthogPoly> f =
    dirichletWorkset.sg_f;
  Teuchos::RCP<Stokhos::EpetraMultiVectorOrthogPoly> fp =
    dirichletWorkset.sg_fp;
  Teuchos::RCP<Stokhos::EpetraMultiVectorOrthogPoly> JV =
    dirichletWorkset.sg_JV;
  Teuchos::RCP<const Stokhos::EpetraVectorOrthogPoly> x =
    dirichletWorkset.sg_x;
  Teuchos::RCP<const Epetra_MultiVector> Vx = dirichletWorkset.Vx;
  const RealType j_coeff = dirichletWorkset.j_coeff;
  const std::vector<std::vector<int> >& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

  int nblock = x->size();

  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    int lunk = nsNodes[inode][this->offset];

    if (f != Teuchos::null) {
      for (int block=0; block<nblock; block++)
        (*f)[block][lunk] = (*x)[block][lunk];
      if(nblock>0) (*f)[0][lunk] -= pT[lunk];
    }

    if (JV != Teuchos::null)
      for (int i=0; i<dirichletWorkset.num_cols_x; i++)
        (*JV)[0][i][lunk] = j_coeff*(*Vx)[i][lunk];

    if (fp != Teuchos::null)
      for (int i=0; i<dirichletWorkset.num_cols_p; i++)
        for (int block=0; block<nblock; block++)
          (*fp)[block][i][lunk] = 0;
  }
}
#endif 
#ifdef ALBANY_ENSEMBLE 

// **********************************************************************
// Specialization: Multi-point Residual
// **********************************************************************
template<typename Traits>
DirichletField<PHAL::AlbanyTraits::MPResidual, Traits>::
DirichletField(Teuchos::ParameterList& p) :
  DirichletField_Base<PHAL::AlbanyTraits::MPResidual, Traits>(p) {
}
// **********************************************************************
template<typename Traits>
void DirichletField<PHAL::AlbanyTraits::MPResidual, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset) {
  Teuchos::RCP<Tpetra_Vector> pvecT =
    dirichletWorkset.distParamLib->get(this->field_name)->vector();
  Teuchos::ArrayRCP<const ST> pT = pvecT->get1dView();
  Teuchos::RCP<Stokhos::ProductEpetraVector> f =
    dirichletWorkset.mp_f;
  Teuchos::RCP<const Stokhos::ProductEpetraVector> x =
    dirichletWorkset.mp_x;
  const std::vector<std::vector<int> >& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

  int nblock = x->size();
  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
      int lunk = nsNodes[inode][this->offset];
      for (int block=0; block<nblock; block++)
        (*f)[block][lunk] = (*x)[block][lunk];
      if(nblock>0) (*f)[0][lunk] -= pT[lunk];
  }
}

// **********************************************************************
// Specialization: Multi-point Jacobian
// **********************************************************************
template<typename Traits>
DirichletField<PHAL::AlbanyTraits::MPJacobian, Traits>::
DirichletField(Teuchos::ParameterList& p) :
  DirichletField_Base<PHAL::AlbanyTraits::MPJacobian, Traits>(p) {
}

// **********************************************************************
template<typename Traits>
void DirichletField<PHAL::AlbanyTraits::MPJacobian, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{

  Teuchos::RCP<Tpetra_Vector> pvecT =
    dirichletWorkset.distParamLib->get(this->field_name)->vector();
  Teuchos::ArrayRCP<const ST> pT = pvecT->get1dView();
  Teuchos::RCP<Stokhos::ProductEpetraVector> f =
    dirichletWorkset.mp_f;
  Teuchos::RCP< Stokhos::ProductContainer<Epetra_CrsMatrix> > jac =
    dirichletWorkset.mp_Jac;
  Teuchos::RCP<const Stokhos::ProductEpetraVector> x =
    dirichletWorkset.mp_x;
  const RealType j_coeff = dirichletWorkset.j_coeff;
  const std::vector<std::vector<int> >& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

  RealType* matrixEntries;
  int*    matrixIndices;
  int     numEntries;
  int nblock = 0;
  if (f != Teuchos::null)
    nblock = f->size();

  int nblock_jac = jac->size();
  RealType diag=j_coeff;
  bool fillResid = (f != Teuchos::null);

  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
      int lunk = nsNodes[inode][this->offset];
      for (int block=0; block<nblock_jac; block++) {
        (*jac)[block].ExtractMyRowView(lunk, numEntries, matrixEntries,
                                       matrixIndices);
        for (int i=0; i<numEntries; i++)
          matrixEntries[i]=0;
        (*jac)[block].ReplaceMyValues(lunk, 1, &diag, &lunk);
      }
      if (fillResid) {
        for (int block=0; block<nblock; block++)
          (*f)[block][lunk] = (*x)[block][lunk];
        if(nblock>0) (*f)[0][lunk] -= pT[lunk];
      }
  }
}

// **********************************************************************
// Specialization: Multi-point Tangent
// **********************************************************************
template<typename Traits>
DirichletField<PHAL::AlbanyTraits::MPTangent, Traits>::
DirichletField(Teuchos::ParameterList& p) :
  DirichletField_Base<PHAL::AlbanyTraits::MPTangent, Traits>(p) {
}

// **********************************************************************
template<typename Traits>
void DirichletField<PHAL::AlbanyTraits::MPTangent, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{

  Teuchos::RCP<Tpetra_Vector> pvecT =
    dirichletWorkset.distParamLib->get(this->field_name)->vector();
  Teuchos::ArrayRCP<const ST> pT = pvecT->get1dView();
  Teuchos::RCP<Stokhos::ProductEpetraVector> f =
    dirichletWorkset.mp_f;
  Teuchos::RCP<Stokhos::ProductEpetraMultiVector> fp =
    dirichletWorkset.mp_fp;
  Teuchos::RCP<Stokhos::ProductEpetraMultiVector> JV =
    dirichletWorkset.mp_JV;
  Teuchos::RCP<const Stokhos::ProductEpetraVector> x =
    dirichletWorkset.mp_x;
  Teuchos::RCP<const Epetra_MultiVector> Vx = dirichletWorkset.Vx;
  const RealType j_coeff = dirichletWorkset.j_coeff;
  const std::vector<std::vector<int> >& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

  int nblock = x->size();

  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    int lunk = nsNodes[inode][this->offset];

    if (f != Teuchos::null) {
      for (int block=0; block<nblock; block++)
        (*f)[block][lunk] = (*x)[block][lunk];
      if(nblock>0) (*f)[0][lunk] -= pT[lunk];
    }

    if (JV != Teuchos::null)
      for (int i=0; i<dirichletWorkset.num_cols_x; i++)
        for (int block=0; block<nblock; block++)
          (*JV)[block][i][lunk] = j_coeff*(*Vx)[i][lunk];

    if (fp != Teuchos::null)
      for (int i=0; i<dirichletWorkset.num_cols_p; i++)
        for (int block=0; block<nblock; block++)
          (*fp)[block][i][lunk] = 0;
  }
}
#endif

// **********************************************************************
}

