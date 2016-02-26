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

template <typename EvalT, typename Traits/*, typename cfunc_traits*/>
DirichletCoordFunction_Base<EvalT, Traits/*, cfunc_traits*/>::
DirichletCoordFunction_Base(Teuchos::ParameterList& p) :
  PHAL::DirichletBase<EvalT, Traits>(p),
  func(p) {
}

// **********************************************************************
// Specialization: Residual
// **********************************************************************
template<typename Traits/*, typename cfunc_traits*/>
DirichletCoordFunction<PHAL::AlbanyTraits::Residual, Traits/*, cfunc_traits*/>::
DirichletCoordFunction(Teuchos::ParameterList& p) :
  DirichletCoordFunction_Base<PHAL::AlbanyTraits::Residual, Traits/*, cfunc_traits*/>(p) {
}

// **********************************************************************
template<typename Traits/*, typename cfunc_traits*/>
void
DirichletCoordFunction<PHAL::AlbanyTraits::Residual, Traits/*, cfunc_traits*/>::
evaluateFields(typename Traits::EvalData dirichletWorkset) {

  Teuchos::RCP<Tpetra_Vector> fT = dirichletWorkset.fT;
  Teuchos::RCP<const Tpetra_Vector> xT = dirichletWorkset.xT;
  Teuchos::ArrayRCP<ST> fT_nonconstView = fT->get1dViewNonConst();
  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();


  // Grab the vector off node GIDs for this Node Set ID from the std::map
  const std::vector<std::vector<int> >& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
  const std::vector<double*>& nsNodeCoords =
    dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;

  RealType time = dirichletWorkset.current_time;
  int number_of_components = this->func.getNumComponents();

  double* coord;
  std::vector<ScalarT> BCVals(number_of_components);

  for(unsigned int inode = 0; inode < nsNodes.size(); inode++) {

    coord = nsNodeCoords[inode];

    this->func.computeBCs(coord, BCVals, time);

    for(unsigned int j = 0; j < number_of_components; j++) {
      int offset = nsNodes[inode][j];
      fT_nonconstView[offset] = (xT_constView[offset] - BCVals[j]);
    }
  }
}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************
template<typename Traits/*, typename cfunc_traits*/>
DirichletCoordFunction<PHAL::AlbanyTraits::Jacobian, Traits/*, cfunc_traits*/>::
DirichletCoordFunction(Teuchos::ParameterList& p) :
  DirichletCoordFunction_Base<PHAL::AlbanyTraits::Jacobian, Traits/*, cfunc_traits*/>(p) {
}
// **********************************************************************
template<typename Traits/*, typename cfunc_traits*/>
void DirichletCoordFunction<PHAL::AlbanyTraits::Jacobian, Traits/*, cfunc_traits*/>::
evaluateFields(typename Traits::EvalData dirichletWorkset) {

  Teuchos::RCP<Tpetra_Vector> fT = dirichletWorkset.fT;
  Teuchos::ArrayRCP<ST> fT_nonconstView;
  Teuchos::RCP<const Tpetra_Vector> xT = dirichletWorkset.xT;
  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();
  Teuchos::RCP<Tpetra_CrsMatrix> jacT = dirichletWorkset.JacT;

  const RealType j_coeff = dirichletWorkset.j_coeff;
  const std::vector<std::vector<int> >& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
  const std::vector<double*>& nsNodeCoords =
    dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;

  Teuchos::Array<LO> index(1);
  Teuchos::Array<ST> value(1); 
  size_t numEntriesT;  
  value[0] = j_coeff; 
  Teuchos::Array<ST> matrixEntriesT; 
  Teuchos::Array<LO> matrixIndicesT; 

  bool fillResid = (fT != Teuchos::null);
  if (fillResid) fT_nonconstView = fT->get1dViewNonConst();

  RealType time = dirichletWorkset.current_time;
  int number_of_components = this->func.getNumComponents();

  double* coord;
  std::vector<ScalarT> BCVals(number_of_components);

  for(unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    coord = nsNodeCoords[inode];

    this->func.computeBCs(coord, BCVals, time);

    for(unsigned int j = 0; j < number_of_components; j++) {

      int offset = nsNodes[inode][j];
      index[0] = offset; 
      numEntriesT = jacT->getNumEntriesInLocalRow(offset);
      matrixEntriesT.resize(numEntriesT); 
      matrixIndicesT.resize(numEntriesT); 

      jacT->getLocalRowCopy(offset, matrixIndicesT(), matrixEntriesT(), numEntriesT); 
      for (int i=0; i<numEntriesT; i++) matrixEntriesT[i]=0;
      jacT->replaceLocalValues(offset, matrixIndicesT(), matrixEntriesT()); 

      jacT->replaceLocalValues(offset, index(), value()); 

      if(fillResid) {
        fT_nonconstView[offset] = (xT_constView[offset] - BCVals[j].val());
      }
    }
  }
}

// **********************************************************************
// Specialization: Tangent
// **********************************************************************
template<typename Traits/*, typename cfunc_traits*/>
DirichletCoordFunction<PHAL::AlbanyTraits::Tangent, Traits/*, cfunc_traits*/>::
DirichletCoordFunction(Teuchos::ParameterList& p) :
  DirichletCoordFunction_Base<PHAL::AlbanyTraits::Tangent, Traits/*, cfunc_traits*/>(p) {
}
// **********************************************************************
template<typename Traits/*, typename cfunc_traits*/>
void DirichletCoordFunction<PHAL::AlbanyTraits::Tangent, Traits/*, cfunc_traits*/>::
evaluateFields(typename Traits::EvalData dirichletWorkset) {

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
  const std::vector<double*>& nsNodeCoords =
    dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;

  RealType time = dirichletWorkset.current_time;
  int number_of_components = this->func.getNumComponents();

  double* coord;
  std::vector<ScalarT> BCVals(number_of_components);

  for(unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    coord = nsNodeCoords[inode];

    this->func.computeBCs(coord, BCVals, time);

    for(unsigned int j = 0; j < number_of_components; j++) {

      int offset = nsNodes[inode][j];

      if(fT != Teuchos::null)
        fT_nonconstView[offset] = (xT_constView[offset] - BCVals[j].val());

      if(JVT != Teuchos::null){
        Teuchos::ArrayRCP<ST> JVT_nonconstView; 
        for(int i = 0; i < dirichletWorkset.num_cols_x; i++){
          JVT_nonconstView = JVT->getDataNonConst(i); 
          VxT_constView = VxT->getData(i); 
          JVT_nonconstView[offset] = j_coeff * VxT_constView[offset];
        }
      }

      if(fpT != Teuchos::null){
        Teuchos::ArrayRCP<ST> fpT_nonconstView; 
        for(int i = 0; i < dirichletWorkset.num_cols_p; i++){
          fpT_nonconstView = fpT->getDataNonConst(i); 
          fpT_nonconstView[offset] = -BCVals[j].dx(dirichletWorkset.param_offset + i);
        }
      }
    }
  }
}

// **********************************************************************
// Specialization: DistParamDeriv
// **********************************************************************
template<typename Traits/*, typename cfunc_traits*/>
DirichletCoordFunction<PHAL::AlbanyTraits::DistParamDeriv, Traits/*, cfunc_traits*/>::
DirichletCoordFunction(Teuchos::ParameterList& p) :
  DirichletCoordFunction_Base<PHAL::AlbanyTraits::DistParamDeriv, Traits/*, cfunc_traits*/>(p) {
}
// **********************************************************************
template<typename Traits/*, typename cfunc_traits*/>
void DirichletCoordFunction<PHAL::AlbanyTraits::DistParamDeriv, Traits/*, cfunc_traits*/>::
evaluateFields(typename Traits::EvalData dirichletWorkset) {

  Teuchos::RCP<Tpetra_MultiVector> fpVT = dirichletWorkset.fpVT;
  //non-const view of VpT 
  Teuchos::ArrayRCP<ST> fpVT_nonconstView; 
  bool trans = dirichletWorkset.transpose_dist_param_deriv;
  int num_cols = fpVT->getNumVectors();

  //
  // We're currently assuming Dirichlet BC's can't be distributed parameters.
  // Thus we don't need to actually evaluate the BC's here.  The code to do
  // so is still here, just commented out for future reference.
  //

  const std::vector<std::vector<int> >& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
  const std::vector<double*>& nsNodeCoords =
    dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;

  // RealType time = dirichletWorkset.current_time;
  int number_of_components = this->func.getNumComponents();

  // double* coord;
  // std::vector<ScalarT> BCVals(number_of_components);

  // For (df/dp)^T*V we zero out corresponding entries in V
  if (trans) {
    Teuchos::RCP<Tpetra_MultiVector> VpT = dirichletWorkset.Vp_bcT;
    //non-const view of VpT 
    Teuchos::ArrayRCP<ST> VpT_nonconstView; 
    for(unsigned int inode = 0; inode < nsNodes.size(); inode++) {
      // coord = nsNodeCoords[inode];
      // this->func.computeBCs(coord, BCVals, time);

      for(unsigned int j = 0; j < number_of_components; j++) {
        int offset = nsNodes[inode][j];
        for (int col=0; col<num_cols; ++col) {
          //(*Vp)[col][offset] = 0.0;
          VpT_nonconstView = VpT->getDataNonConst(col); 
          VpT_nonconstView[offset] = 0.0; 
        }
      }
    }
  }

  // for (df/dp)*V we zero out corresponding entries in df/dp
  else {
    for(unsigned int inode = 0; inode < nsNodes.size(); inode++) {
      // coord = nsNodeCoords[inode];
      // this->func.computeBCs(coord, BCVals, time);

      for(unsigned int j = 0; j < number_of_components; j++) {
        int offset = nsNodes[inode][j];
        for (int col=0; col<num_cols; ++col) {
          //(*fpV)[col][offset] = 0.0;
          fpVT_nonconstView = fpVT->getDataNonConst(col); 
          fpVT_nonconstView[offset] = 0.0; 
        }
      }
    }
  }

}

// **********************************************************************
// Specialization: Stochastic Galerkin Residual
// **********************************************************************
#ifdef ALBANY_SG
template<typename Traits/*, typename cfunc_traits*/>
DirichletCoordFunction<PHAL::AlbanyTraits::SGResidual, Traits/*, cfunc_traits*/>::
DirichletCoordFunction(Teuchos::ParameterList& p) :
  DirichletCoordFunction_Base<PHAL::AlbanyTraits::SGResidual, Traits/*, cfunc_traits*/>(p) {
}
// **********************************************************************
template<typename Traits/*, typename cfunc_traits*/>
void DirichletCoordFunction<PHAL::AlbanyTraits::SGResidual, Traits/*, cfunc_traits*/>::
evaluateFields(typename Traits::EvalData dirichletWorkset) {
  Teuchos::RCP<Stokhos::EpetraVectorOrthogPoly> f =
    dirichletWorkset.sg_f;
  Teuchos::RCP<const Stokhos::EpetraVectorOrthogPoly> x =
    dirichletWorkset.sg_x;

  const std::vector<std::vector<int> >& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
  const std::vector<double*>& nsNodeCoords =
    dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;

  RealType time = dirichletWorkset.current_time;
  int number_of_components = this->func.getNumComponents();

  double* coord;
  std::vector<ScalarT> BCVals(number_of_components);

  int nblock = x->size();

  for(unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    coord = nsNodeCoords[inode];

    this->func.computeBCs(coord, BCVals, time);

    for(unsigned int j = 0; j < number_of_components; j++) {
      int offset = nsNodes[inode][j];

      for(int block = 0; block < nblock; block++)
        (*f)[block][offset] = ((*x)[block][offset] - BCVals[j].coeff(block));

    }
  }
}

// **********************************************************************
// Specialization: Stochastic Galerkin Jacobian
// **********************************************************************
template<typename Traits/*, typename cfunc_traits*/>
DirichletCoordFunction<PHAL::AlbanyTraits::SGJacobian, Traits/*, cfunc_traits*/>::
DirichletCoordFunction(Teuchos::ParameterList& p) :
  DirichletCoordFunction_Base<PHAL::AlbanyTraits::SGJacobian, Traits/*, cfunc_traits*/>(p) {
}
// **********************************************************************
template<typename Traits/*, typename cfunc_traits*/>
void DirichletCoordFunction<PHAL::AlbanyTraits::SGJacobian, Traits/*, cfunc_traits*/>::
evaluateFields(typename Traits::EvalData dirichletWorkset) {
  Teuchos::RCP< Stokhos::EpetraVectorOrthogPoly> f =
    dirichletWorkset.sg_f;
  Teuchos::RCP< Stokhos::VectorOrthogPoly<Epetra_CrsMatrix> > jac =
    dirichletWorkset.sg_Jac;
  Teuchos::RCP<const Stokhos::EpetraVectorOrthogPoly> x =
    dirichletWorkset.sg_x;

  const RealType j_coeff = dirichletWorkset.j_coeff;
  const std::vector<std::vector<int> >& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
  const std::vector<double*>& nsNodeCoords =
    dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;

  RealType* matrixEntries;
  int*    matrixIndices;
  int     numEntries;
  RealType diag = j_coeff;
  bool fillResid = (f != Teuchos::null);

  int nblock = 0;

  if(f != Teuchos::null)
    nblock = f->size();

  int nblock_jac = jac->size();

  RealType time = dirichletWorkset.current_time;
  int number_of_components = this->func.getNumComponents();

  double* coord;
  std::vector<ScalarT> BCVals(number_of_components);

  for(unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    coord = nsNodeCoords[inode];

    this->func.computeBCs(coord, BCVals, time);

    for(unsigned int j = 0; j < number_of_components; j++) {
      int offset = nsNodes[inode][j];

      // replace jac values for the dof
      for(int block = 0; block < nblock_jac; block++) {

        (*jac)[block].ExtractMyRowView(offset, numEntries, matrixEntries,
                                       matrixIndices);

        for(int i = 0; i < numEntries; i++) matrixEntries[i] = 0;

      }

      (*jac)[0].ReplaceMyValues(offset, 1, &diag, &offset);

      if(fillResid)
        for(int block = 0; block < nblock; block++)
          (*f)[block][offset] = ((*x)[block][offset] - BCVals[j].val().coeff(block));

    }
  }
}

// **********************************************************************
// Specialization: Stochastic Galerkin Tangent
// **********************************************************************
template<typename Traits/*, typename cfunc_traits*/>
DirichletCoordFunction<PHAL::AlbanyTraits::SGTangent, Traits/*, cfunc_traits*/>::
DirichletCoordFunction(Teuchos::ParameterList& p) :
  DirichletCoordFunction_Base<PHAL::AlbanyTraits::SGTangent, Traits/*, cfunc_traits*/>(p) {
}
// **********************************************************************
template<typename Traits/*, typename cfunc_traits*/>
void DirichletCoordFunction<PHAL::AlbanyTraits::SGTangent, Traits/*, cfunc_traits*/>::
evaluateFields(typename Traits::EvalData dirichletWorkset) {
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
  const std::vector<double*>& nsNodeCoords =
    dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;

  int nblock = x->size();

  RealType time = dirichletWorkset.current_time;
  int number_of_components = this->func.getNumComponents();

  double* coord;
  std::vector<ScalarT> BCVals(number_of_components);

  for(unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    coord = nsNodeCoords[inode];

    this->func.computeBCs(coord, BCVals, time);

    for(unsigned int j = 0; j < number_of_components; j++) {
      int offset = nsNodes[inode][j];

      if(f != Teuchos::null)
        for(int block = 0; block < nblock; block++)
          (*f)[block][offset] = (*x)[block][offset] - BCVals[j].val().coeff(block);

      if(JV != Teuchos::null)
        for(int i = 0; i < dirichletWorkset.num_cols_x; i++)
          (*JV)[0][i][offset] = j_coeff * (*Vx)[i][offset];

      if(fp != Teuchos::null)
        for(int i = 0; i < dirichletWorkset.num_cols_p; i++)
          for(int block = 0; block < nblock; block++)
            (*fp)[block][i][offset] = -BCVals[j].dx(dirichletWorkset.param_offset + i).coeff(block);

    }
  }
}

#endif 
#ifdef ALBANY_ENSEMBLE 

// **********************************************************************
// Specialization: Multi-point Residual
// **********************************************************************
template<typename Traits/*, typename cfunc_traits*/>
DirichletCoordFunction<PHAL::AlbanyTraits::MPResidual, Traits/*, cfunc_traits*/>::
DirichletCoordFunction(Teuchos::ParameterList& p) :
  DirichletCoordFunction_Base<PHAL::AlbanyTraits::MPResidual, Traits/*, cfunc_traits*/>(p) {
}
// **********************************************************************
template<typename Traits/*, typename cfunc_traits*/>
void DirichletCoordFunction<PHAL::AlbanyTraits::MPResidual, Traits/*, cfunc_traits*/>::
evaluateFields(typename Traits::EvalData dirichletWorkset) {
  Teuchos::RCP<Stokhos::ProductEpetraVector> f =
    dirichletWorkset.mp_f;
  Teuchos::RCP<const Stokhos::ProductEpetraVector> x =
    dirichletWorkset.mp_x;

  const std::vector<std::vector<int> >& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
  const std::vector<double*>& nsNodeCoords =
    dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;

  RealType time = dirichletWorkset.current_time;
  int number_of_components = this->func.getNumComponents();

  double* coord;
  std::vector<ScalarT> BCVals(number_of_components);

  int nblock = x->size();

  for(unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    coord = nsNodeCoords[inode];

    this->func.computeBCs(coord, BCVals, time);

    for(unsigned int j = 0; j < number_of_components; j++) {
      int offset = nsNodes[inode][j];

      for(int block = 0; block < nblock; block++)
        (*f)[block][offset] = ((*x)[block][offset] - BCVals[j].coeff(block));

    }
  }
}

// **********************************************************************
// Specialization: Multi-point Jacobian
// **********************************************************************
template<typename Traits/*, typename cfunc_traits*/>
DirichletCoordFunction<PHAL::AlbanyTraits::MPJacobian, Traits/*, cfunc_traits*/>::
DirichletCoordFunction(Teuchos::ParameterList& p) :
  DirichletCoordFunction_Base<PHAL::AlbanyTraits::MPJacobian, Traits/*, cfunc_traits*/>(p) {
}
// **********************************************************************
template<typename Traits/*, typename cfunc_traits*/>
void DirichletCoordFunction<PHAL::AlbanyTraits::MPJacobian, Traits/*, cfunc_traits*/>::
evaluateFields(typename Traits::EvalData dirichletWorkset) {
  Teuchos::RCP<Stokhos::ProductEpetraVector> f =
    dirichletWorkset.mp_f;
  Teuchos::RCP< Stokhos::ProductContainer<Epetra_CrsMatrix> > jac =
    dirichletWorkset.mp_Jac;
  Teuchos::RCP<const Stokhos::ProductEpetraVector> x =
    dirichletWorkset.mp_x;

  const RealType j_coeff = dirichletWorkset.j_coeff;
  const std::vector<std::vector<int> >& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
  const std::vector<double*>& nsNodeCoords =
    dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;

  RealType* matrixEntries;
  int*    matrixIndices;
  int     numEntries;
  RealType diag = j_coeff;
  bool fillResid = (f != Teuchos::null);

  int nblock = 0;

  if(f != Teuchos::null)
    nblock = f->size();

  int nblock_jac = jac->size();

  RealType time = dirichletWorkset.current_time;
  int number_of_components = this->func.getNumComponents();

  double* coord;
  std::vector<ScalarT> BCVals(number_of_components);

  for(unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    coord = nsNodeCoords[inode];

    this->func.computeBCs(coord, BCVals, time);

    for(unsigned int j = 0; j < number_of_components; j++) {
      int offset = nsNodes[inode][j];

      // replace jac values for the dof
      for(int block = 0; block < nblock_jac; block++) {
        (*jac)[block].ExtractMyRowView(offset, numEntries, matrixEntries,
                                       matrixIndices);

        for(int i = 0; i < numEntries; i++) matrixEntries[i] = 0;

        (*jac)[block].ReplaceMyValues(offset, 1, &diag, &offset);
      }

      if(fillResid)
        for(int block = 0; block < nblock; block++)
          (*f)[block][offset] = ((*x)[block][offset] - BCVals[j].val().coeff(block));

    }
  }
}

// **********************************************************************
// Specialization: Multi-point Tangent
// **********************************************************************
template<typename Traits/*, typename cfunc_traits*/>
DirichletCoordFunction<PHAL::AlbanyTraits::MPTangent, Traits/*, cfunc_traits*/>::
DirichletCoordFunction(Teuchos::ParameterList& p) :
  DirichletCoordFunction_Base<PHAL::AlbanyTraits::MPTangent, Traits/*, cfunc_traits*/>(p) {
}
// **********************************************************************
template<typename Traits/*, typename cfunc_traits*/>
void DirichletCoordFunction<PHAL::AlbanyTraits::MPTangent, Traits/*, cfunc_traits*/>::
evaluateFields(typename Traits::EvalData dirichletWorkset) {
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
  const std::vector<double*>& nsNodeCoords =
    dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;

  int nblock = x->size();

  RealType time = dirichletWorkset.current_time;
  int number_of_components = this->func.getNumComponents();

  double* coord;
  std::vector<ScalarT> BCVals(number_of_components);

  for(unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    coord = nsNodeCoords[inode];

    this->func.computeBCs(coord, BCVals, time);

    for(unsigned int j = 0; j < number_of_components; j++) {
      int offset = nsNodes[inode][j];

      if(f != Teuchos::null)
        for(int block = 0; block < nblock; block++)
          (*f)[block][offset] = (*x)[block][offset] - BCVals[j].val().coeff(block);

      if(JV != Teuchos::null)
        for(int i = 0; i < dirichletWorkset.num_cols_x; i++)
          for(int block = 0; block < nblock; block++)
            (*JV)[block][i][offset] = j_coeff * (*Vx)[i][offset];

      if(fp != Teuchos::null)
        for(int i = 0; i < dirichletWorkset.num_cols_p; i++)
          for(int block = 0; block < nblock; block++)
            (*fp)[block][i][offset] = -BCVals[j].dx(dirichletWorkset.param_offset + i).coeff(block);

    }
  }
}
#endif

} // namespace LCM

