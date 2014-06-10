//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"

// **********************************************************************
// Genereric Template Code for Constructor and PostRegistrationSetup
// **********************************************************************

namespace PHAL {

template <typename EvalT, typename Traits, typename cfunc_traits>
DirichletCoordFunction_Base<EvalT, Traits, cfunc_traits>::
DirichletCoordFunction_Base(Teuchos::ParameterList& p) :
  PHAL::DirichletBase<EvalT, Traits>(p),
  func(p) {
}

// **********************************************************************
// Specialization: Residual
// **********************************************************************
template<typename Traits, typename cfunc_traits>
DirichletCoordFunction<PHAL::AlbanyTraits::Residual, Traits,  cfunc_traits>::
DirichletCoordFunction(Teuchos::ParameterList& p) :
  DirichletCoordFunction_Base<PHAL::AlbanyTraits::Residual, Traits, cfunc_traits>(p) {
}

// **********************************************************************
template<typename Traits, typename cfunc_traits>
void
DirichletCoordFunction<PHAL::AlbanyTraits::Residual, Traits, cfunc_traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset) {
  Teuchos::RCP<Epetra_Vector> f = dirichletWorkset.f;
  Teuchos::RCP<const Epetra_Vector> x = dirichletWorkset.x;

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
      (*f)[offset] = ((*x)[offset] - BCVals[j]);
    }
  }
}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************
template<typename Traits, typename cfunc_traits>
DirichletCoordFunction<PHAL::AlbanyTraits::Jacobian, Traits, cfunc_traits>::
DirichletCoordFunction(Teuchos::ParameterList& p) :
  DirichletCoordFunction_Base<PHAL::AlbanyTraits::Jacobian, Traits, cfunc_traits>(p) {
}
// **********************************************************************
template<typename Traits, typename cfunc_traits>
void DirichletCoordFunction<PHAL::AlbanyTraits::Jacobian, Traits, cfunc_traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset) {

  Teuchos::RCP<Epetra_Vector> f = dirichletWorkset.f;
  Teuchos::RCP<Epetra_CrsMatrix> jac = dirichletWorkset.Jac;
  Teuchos::RCP<const Epetra_Vector> x = dirichletWorkset.x;

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
      jac->ExtractMyRowView(offset, numEntries, matrixEntries, matrixIndices);

      for(int i = 0; i < numEntries; i++) matrixEntries[i] = 0;

      jac->ReplaceMyValues(offset, 1, &diag, &offset);

      if(fillResid) {
        (*f)[offset] = ((*x)[offset] - BCVals[j].val());
      }
    }
  }
}

// **********************************************************************
// Specialization: Tangent
// **********************************************************************
template<typename Traits, typename cfunc_traits>
DirichletCoordFunction<PHAL::AlbanyTraits::Tangent, Traits, cfunc_traits>::
DirichletCoordFunction(Teuchos::ParameterList& p) :
  DirichletCoordFunction_Base<PHAL::AlbanyTraits::Tangent, Traits, cfunc_traits>(p) {
}
// **********************************************************************
template<typename Traits, typename cfunc_traits>
void DirichletCoordFunction<PHAL::AlbanyTraits::Tangent, Traits, cfunc_traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset) {
  Teuchos::RCP<Epetra_Vector> f = dirichletWorkset.f;
  Teuchos::RCP<Epetra_MultiVector> fp = dirichletWorkset.fp;
  Teuchos::RCP<Epetra_MultiVector> JV = dirichletWorkset.JV;
  Teuchos::RCP<const Epetra_Vector> x = dirichletWorkset.x;
  Teuchos::RCP<const Epetra_MultiVector> Vx = dirichletWorkset.Vx;

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

      if(f != Teuchos::null)
        (*f)[offset] = ((*x)[offset] - BCVals[j].val());

      if(JV != Teuchos::null)
        for(int i = 0; i < dirichletWorkset.num_cols_x; i++)
          (*JV)[i][offset] = j_coeff * (*Vx)[i][offset];

      if(fp != Teuchos::null)
        for(int i = 0; i < dirichletWorkset.num_cols_p; i++)
          (*fp)[i][offset] = -BCVals[j].dx(dirichletWorkset.param_offset + i);

    }
  }
}

// **********************************************************************
// Specialization: DistParamDeriv
// **********************************************************************
template<typename Traits, typename cfunc_traits>
DirichletCoordFunction<PHAL::AlbanyTraits::DistParamDeriv, Traits, cfunc_traits>::
DirichletCoordFunction(Teuchos::ParameterList& p) :
  DirichletCoordFunction_Base<PHAL::AlbanyTraits::DistParamDeriv, Traits, cfunc_traits>(p) {
}
// **********************************************************************
template<typename Traits, typename cfunc_traits>
void DirichletCoordFunction<PHAL::AlbanyTraits::DistParamDeriv, Traits, cfunc_traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset) {
  Teuchos::RCP<Epetra_MultiVector> fpV = dirichletWorkset.fpV;
  bool trans = dirichletWorkset.transpose_dist_param_deriv;
  int num_cols = fpV->NumVectors();

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
    Teuchos::RCP<Epetra_MultiVector> Vp = dirichletWorkset.Vp_bc;
    for(unsigned int inode = 0; inode < nsNodes.size(); inode++) {
      // coord = nsNodeCoords[inode];
      // this->func.computeBCs(coord, BCVals, time);

      for(unsigned int j = 0; j < number_of_components; j++) {
        int offset = nsNodes[inode][j];
        for (int col=0; col<num_cols; ++col)
          (*Vp)[col][offset] = 0.0;
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
        for (int col=0; col<num_cols; ++col)
          (*fpV)[col][offset] = 0.0;
      }
    }
  }

}

// **********************************************************************
// Specialization: Stochastic Galerkin Residual
// **********************************************************************
#ifdef ALBANY_SG_MP
template<typename Traits, typename cfunc_traits>
DirichletCoordFunction<PHAL::AlbanyTraits::SGResidual, Traits, cfunc_traits>::
DirichletCoordFunction(Teuchos::ParameterList& p) :
  DirichletCoordFunction_Base<PHAL::AlbanyTraits::SGResidual, Traits, cfunc_traits>(p) {
}
// **********************************************************************
template<typename Traits, typename cfunc_traits>
void DirichletCoordFunction<PHAL::AlbanyTraits::SGResidual, Traits, cfunc_traits>::
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
template<typename Traits, typename cfunc_traits>
DirichletCoordFunction<PHAL::AlbanyTraits::SGJacobian, Traits, cfunc_traits>::
DirichletCoordFunction(Teuchos::ParameterList& p) :
  DirichletCoordFunction_Base<PHAL::AlbanyTraits::SGJacobian, Traits, cfunc_traits>(p) {
}
// **********************************************************************
template<typename Traits, typename cfunc_traits>
void DirichletCoordFunction<PHAL::AlbanyTraits::SGJacobian, Traits, cfunc_traits>::
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
template<typename Traits, typename cfunc_traits>
DirichletCoordFunction<PHAL::AlbanyTraits::SGTangent, Traits, cfunc_traits>::
DirichletCoordFunction(Teuchos::ParameterList& p) :
  DirichletCoordFunction_Base<PHAL::AlbanyTraits::SGTangent, Traits, cfunc_traits>(p) {
}
// **********************************************************************
template<typename Traits, typename cfunc_traits>
void DirichletCoordFunction<PHAL::AlbanyTraits::SGTangent, Traits, cfunc_traits>::
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


// **********************************************************************
// Specialization: Multi-point Residual
// **********************************************************************
template<typename Traits, typename cfunc_traits>
DirichletCoordFunction<PHAL::AlbanyTraits::MPResidual, Traits, cfunc_traits>::
DirichletCoordFunction(Teuchos::ParameterList& p) :
  DirichletCoordFunction_Base<PHAL::AlbanyTraits::MPResidual, Traits, cfunc_traits>(p) {
}
// **********************************************************************
template<typename Traits, typename cfunc_traits>
void DirichletCoordFunction<PHAL::AlbanyTraits::MPResidual, Traits, cfunc_traits>::
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
template<typename Traits, typename cfunc_traits>
DirichletCoordFunction<PHAL::AlbanyTraits::MPJacobian, Traits, cfunc_traits>::
DirichletCoordFunction(Teuchos::ParameterList& p) :
  DirichletCoordFunction_Base<PHAL::AlbanyTraits::MPJacobian, Traits, cfunc_traits>(p) {
}
// **********************************************************************
template<typename Traits, typename cfunc_traits>
void DirichletCoordFunction<PHAL::AlbanyTraits::MPJacobian, Traits, cfunc_traits>::
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
template<typename Traits, typename cfunc_traits>
DirichletCoordFunction<PHAL::AlbanyTraits::MPTangent, Traits,  cfunc_traits>::
DirichletCoordFunction(Teuchos::ParameterList& p) :
  DirichletCoordFunction_Base<PHAL::AlbanyTraits::MPTangent, Traits,  cfunc_traits>(p) {
}
// **********************************************************************
template<typename Traits, typename cfunc_traits>
void DirichletCoordFunction<PHAL::AlbanyTraits::MPTangent, Traits, cfunc_traits>::
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
#endif //ALBANY_SG_MP

} // namespace LCM

