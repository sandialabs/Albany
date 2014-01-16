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

namespace LCM {

template <typename EvalT, typename Traits>
TorsionBC_Base<EvalT, Traits>::
TorsionBC_Base(Teuchos::ParameterList& p) :
  PHAL::DirichletBase<EvalT, Traits>(p),
  thetaDot(p.get<RealType>("Theta Dot")),
  X0(p.get<RealType>("X0")),
  Y0(p.get<RealType>("Y0"))
{
}

// **********************************************************************
template<typename EvalT, typename Traits>
void
TorsionBC_Base<EvalT, Traits>::
computeBCs(double* coord, ScalarT& Xval, ScalarT& Yval,
           const RealType time)
{
  RealType X(coord[0]);
  RealType Y(coord[1]);
  RealType theta(thetaDot*time);

  // compute displace Xval and Yval. (X0,Y0) is the center of rotation/torsion
  Xval = X0 + (X-X0) * std::cos(theta) - (Y-Y0) * std::sin(theta) - X;
  Yval = Y0 + (X-X0) * std::sin(theta) + (Y-Y0) * std::cos(theta) - Y;

  // a different set of bc, for comparison with analytical solution
  //RealType L = 2.0;
  //Xval = -theta * L * Y;
  //Yval = theta * L * X;
}

// **********************************************************************
// Specialization: Residual
// **********************************************************************
template<typename Traits>
TorsionBC<PHAL::AlbanyTraits::Residual, Traits>::
TorsionBC(Teuchos::ParameterList& p) :
  TorsionBC_Base<PHAL::AlbanyTraits::Residual, Traits>(p)
{
}

// **********************************************************************
template<typename Traits>
void
TorsionBC<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  Teuchos::RCP<Epetra_Vector> f = dirichletWorkset.f;
  Teuchos::RCP<const Epetra_Vector> x = dirichletWorkset.x;
  // Grab the vector off node GIDs for this Node Set ID from the std::map
  const std::vector<std::vector<int> >& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
  const std::vector<double*>& nsNodeCoords =
    dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;

  RealType time = dirichletWorkset.current_time;

  int xlunk, ylunk; // global and local indicies into unknown vector
  double* coord;
  ScalarT Xval, Yval;

  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    xlunk = nsNodes[inode][0];
    ylunk = nsNodes[inode][1];
    coord = nsNodeCoords[inode];

    this->computeBCs(coord, Xval, Yval, time);

    (*f)[xlunk] = ((*x)[xlunk] - Xval);
    (*f)[ylunk] = ((*x)[ylunk] - Yval);
  }
}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************
template<typename Traits>
TorsionBC<PHAL::AlbanyTraits::Jacobian, Traits>::
TorsionBC(Teuchos::ParameterList& p) :
  TorsionBC_Base<PHAL::AlbanyTraits::Jacobian, Traits>(p)
{
}
// **********************************************************************
template<typename Traits>
void TorsionBC<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{

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
  RealType diag=j_coeff;
  bool fillResid = (f != Teuchos::null);

  RealType time = dirichletWorkset.current_time;

  int xlunk, ylunk; // local indicies into unknown vector
  double* coord;
  ScalarT Xval, Yval;
  for (unsigned int inode = 0; inode < nsNodes.size(); inode++)
  {
    xlunk = nsNodes[inode][0];
    ylunk = nsNodes[inode][1];
    coord = nsNodeCoords[inode];

    this->computeBCs(coord, Xval, Yval, time);

    // replace jac values for the X dof
    jac->ExtractMyRowView(xlunk, numEntries, matrixEntries, matrixIndices);
    for (int i=0; i<numEntries; i++) matrixEntries[i]=0;
    jac->ReplaceMyValues(xlunk, 1, &diag, &xlunk);

    // replace jac values for the y dof
    jac->ExtractMyRowView(ylunk, numEntries, matrixEntries, matrixIndices);
    for (int i=0; i<numEntries; i++) matrixEntries[i]=0;
    jac->ReplaceMyValues(ylunk, 1, &diag, &ylunk);

    if (fillResid)
    {
      (*f)[xlunk] = ((*x)[xlunk] - Xval.val());
      (*f)[ylunk] = ((*x)[ylunk] - Yval.val());
    }
  }
}

// **********************************************************************
// Specialization: Tangent
// **********************************************************************
template<typename Traits>
TorsionBC<PHAL::AlbanyTraits::Tangent, Traits>::
TorsionBC(Teuchos::ParameterList& p) :
  TorsionBC_Base<PHAL::AlbanyTraits::Tangent, Traits>(p)
{
}
// **********************************************************************
template<typename Traits>
void TorsionBC<PHAL::AlbanyTraits::Tangent, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
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

  int xlunk, ylunk; // global and local indicies into unknown vector
  double* coord;
  ScalarT Xval, Yval;
  for (unsigned int inode = 0; inode < nsNodes.size(); inode++)
  {
    xlunk = nsNodes[inode][0];
    ylunk = nsNodes[inode][1];
    coord = nsNodeCoords[inode];

    this->computeBCs(coord, Xval, Yval, time);

    if (f != Teuchos::null)
    {
      (*f)[xlunk] = ((*x)[xlunk] - Xval.val());
      (*f)[ylunk] = ((*x)[ylunk] - Yval.val());
    }

    if (JV != Teuchos::null) {
      for (int i=0; i<dirichletWorkset.num_cols_x; i++)
      {
        (*JV)[i][xlunk] = j_coeff*(*Vx)[i][xlunk];
        (*JV)[i][ylunk] = j_coeff*(*Vx)[i][ylunk];
      }
    }

    if (fp != Teuchos::null) {
      for (int i=0; i<dirichletWorkset.num_cols_p; i++)
      {
        (*fp)[i][xlunk] = -Xval.dx(dirichletWorkset.param_offset+i);
        (*fp)[i][ylunk] = -Yval.dx(dirichletWorkset.param_offset+i);
      }
    }

  }
}

// **********************************************************************
// Specialization: DistParamDeriv
// **********************************************************************
template<typename Traits>
TorsionBC<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
TorsionBC(Teuchos::ParameterList& p) :
  TorsionBC_Base<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p)
{
}
// **********************************************************************
template<typename Traits>
void TorsionBC<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
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

  int xlunk, ylunk; // global and local indicies into unknown vector
  // double* coord;
  // ScalarT Xval, Yval;

  // For (df/dp)^T*V we zero out corresponding entries in V
  if (trans) {
    Teuchos::RCP<Epetra_MultiVector> Vp = dirichletWorkset.Vp_bc;
    for (unsigned int inode = 0; inode < nsNodes.size(); inode++)
    {
      xlunk = nsNodes[inode][0];
      ylunk = nsNodes[inode][1];
      // coord = nsNodeCoords[inode];

      // this->computeBCs(coord, Xval, Yval, time);

      for (int col=0; col<num_cols; ++col) {
        (*Vp)[col][xlunk] = 0.0;
        (*Vp)[col][ylunk] = 0.0;
      }
    }
  }

  // for (df/dp)*V we zero out corresponding entries in df/dp
  else {
    for (unsigned int inode = 0; inode < nsNodes.size(); inode++)
    {
      xlunk = nsNodes[inode][0];
      ylunk = nsNodes[inode][1];
      // coord = nsNodeCoords[inode];

      // this->computeBCs(coord, Xval, Yval, time);

      for (int col=0; col<num_cols; ++col) {
        (*fpV)[col][xlunk] = 0.0;
        (*fpV)[col][ylunk] = 0.0;
      }
    }
  }
}

// **********************************************************************
// Specialization: Stochastic Galerkin Residual
// **********************************************************************
#ifdef ALBANY_SG_MP
template<typename Traits>
TorsionBC<PHAL::AlbanyTraits::SGResidual, Traits>::
TorsionBC(Teuchos::ParameterList& p) :
  TorsionBC_Base<PHAL::AlbanyTraits::SGResidual, Traits>(p)
{
}
// **********************************************************************
template<typename Traits>
void TorsionBC<PHAL::AlbanyTraits::SGResidual, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  Teuchos::RCP<Stokhos::EpetraVectorOrthogPoly> f =
    dirichletWorkset.sg_f;
  Teuchos::RCP<const Stokhos::EpetraVectorOrthogPoly> x =
    dirichletWorkset.sg_x;
  const std::vector<std::vector<int> >& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
 const std::vector<double*>& nsNodeCoords =
   dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;

  RealType time = dirichletWorkset.current_time;

  int xlunk, ylunk; // global and local indicies into unknown vector
  double* coord;
  ScalarT Xval, Yval;

  int nblock = x->size();
  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    xlunk = nsNodes[inode][0];
    ylunk = nsNodes[inode][1];
    coord = nsNodeCoords[inode];

    this->computeBCs(coord, Xval, Yval, time);

    for (int block=0; block<nblock; block++) {
      (*f)[block][xlunk] = ((*x)[block][xlunk] - Xval.coeff(block));
      (*f)[block][ylunk] = ((*x)[block][ylunk] - Yval.coeff(block));
    }
  }
}

// **********************************************************************
// Specialization: Stochastic Galerkin Jacobian
// **********************************************************************
template<typename Traits>
TorsionBC<PHAL::AlbanyTraits::SGJacobian, Traits>::
TorsionBC(Teuchos::ParameterList& p) :
  TorsionBC_Base<PHAL::AlbanyTraits::SGJacobian, Traits>(p)
{
}
// **********************************************************************
template<typename Traits>
void TorsionBC<PHAL::AlbanyTraits::SGJacobian, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
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
  RealType diag=j_coeff;
  bool fillResid = (f != Teuchos::null);

  int nblock = 0;
  if (f != Teuchos::null)
    nblock = f->size();
  int nblock_jac = jac->size();

  RealType time = dirichletWorkset.current_time;

  int xlunk, ylunk; // local indicies into unknown vector
  double* coord;
  ScalarT Xval, Yval;
  for (unsigned int inode = 0; inode < nsNodes.size(); inode++)
  {
    xlunk = nsNodes[inode][0];
    ylunk = nsNodes[inode][1];
    coord = nsNodeCoords[inode];

    this->computeBCs(coord, Xval, Yval, time);

    // replace jac values for the X dof
    for (int block=0; block<nblock_jac; block++) {
      (*jac)[block].ExtractMyRowView(xlunk, numEntries, matrixEntries,
                                     matrixIndices);
      for (int i=0; i<numEntries; i++) matrixEntries[i]=0;

      // replace jac values for the y dof
      (*jac)[block].ExtractMyRowView(ylunk, numEntries, matrixEntries,
                                     matrixIndices);
      for (int i=0; i<numEntries; i++) matrixEntries[i]=0;
    }
    (*jac)[0].ReplaceMyValues(xlunk, 1, &diag, &xlunk);
    (*jac)[0].ReplaceMyValues(ylunk, 1, &diag, &ylunk);

    if (fillResid)
    {
      for (int block=0; block<nblock; block++) {
        (*f)[block][xlunk] = ((*x)[block][xlunk] - Xval.val().coeff(block));
        (*f)[block][ylunk] = ((*x)[block][ylunk] - Yval.val().coeff(block));
      }
    }
  }
}

// **********************************************************************
// Specialization: Stochastic Galerkin Tangent
// **********************************************************************
template<typename Traits>
TorsionBC<PHAL::AlbanyTraits::SGTangent, Traits>::
TorsionBC(Teuchos::ParameterList& p) :
  TorsionBC_Base<PHAL::AlbanyTraits::SGTangent, Traits>(p)
{
}
// **********************************************************************
template<typename Traits>
void TorsionBC<PHAL::AlbanyTraits::SGTangent, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
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

  int xlunk, ylunk; // global and local indicies into unknown vector
  double* coord;
  ScalarT Xval, Yval;
  for (unsigned int inode = 0; inode < nsNodes.size(); inode++)
  {
    xlunk = nsNodes[inode][0];
    ylunk = nsNodes[inode][1];
    coord = nsNodeCoords[inode];

    this->computeBCs(coord, Xval, Yval, time);

    if (f != Teuchos::null)
    {
      for (int block=0; block<nblock; block++) {
        (*f)[block][xlunk] = (*x)[block][xlunk] - Xval.val().coeff(block);
        (*f)[block][ylunk] = (*x)[block][ylunk] - Yval.val().coeff(block);
      }
    }

    if (JV != Teuchos::null) {
      for (int i=0; i<dirichletWorkset.num_cols_x; i++)
      {
        (*JV)[0][i][xlunk] = j_coeff*(*Vx)[i][xlunk];
        (*JV)[0][i][ylunk] = j_coeff*(*Vx)[i][ylunk];
      }
    }

    if (fp != Teuchos::null) {
      for (int i=0; i<dirichletWorkset.num_cols_p; i++)
      {
        for (int block=0; block<nblock; block++) {
          (*fp)[block][i][xlunk] =
            -Xval.dx(dirichletWorkset.param_offset+i).coeff(block);
          (*fp)[block][i][ylunk] =
            -Yval.dx(dirichletWorkset.param_offset+i).coeff(block);
        }
      }
    }

  }
}


// **********************************************************************
// Specialization: Multi-point Residual
// **********************************************************************
template<typename Traits>
TorsionBC<PHAL::AlbanyTraits::MPResidual, Traits>::
TorsionBC(Teuchos::ParameterList& p) :
  TorsionBC_Base<PHAL::AlbanyTraits::MPResidual, Traits>(p)
{
}
// **********************************************************************
template<typename Traits>
void TorsionBC<PHAL::AlbanyTraits::MPResidual, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  Teuchos::RCP<Stokhos::ProductEpetraVector> f =
    dirichletWorkset.mp_f;
  Teuchos::RCP<const Stokhos::ProductEpetraVector> x =
    dirichletWorkset.mp_x;
  const std::vector<std::vector<int> >& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
  const std::vector<double*>& nsNodeCoords =
    dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;

  RealType time = dirichletWorkset.current_time;

  int xlunk, ylunk; // global and local indicies into unknown vector
  double* coord;
  ScalarT Xval, Yval;

  int nblock = x->size();
  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    xlunk = nsNodes[inode][0];
    ylunk = nsNodes[inode][1];
    coord = nsNodeCoords[inode];

    this->computeBCs(coord, Xval, Yval, time);

    for (int block=0; block<nblock; block++) {
      (*f)[block][xlunk] = ((*x)[block][xlunk] - Xval.coeff(block));
      (*f)[block][ylunk] = ((*x)[block][ylunk] - Yval.coeff(block));
    }
  }
}

// **********************************************************************
// Specialization: Multi-point Jacobian
// **********************************************************************
template<typename Traits>
TorsionBC<PHAL::AlbanyTraits::MPJacobian, Traits>::
TorsionBC(Teuchos::ParameterList& p) :
  TorsionBC_Base<PHAL::AlbanyTraits::MPJacobian, Traits>(p)
{
}
// **********************************************************************
template<typename Traits>
void TorsionBC<PHAL::AlbanyTraits::MPJacobian, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
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
  RealType diag=j_coeff;
  bool fillResid = (f != Teuchos::null);

  int nblock = 0;
  if (f != Teuchos::null)
    nblock = f->size();
  int nblock_jac = jac->size();

  RealType time = dirichletWorkset.current_time;

  int xlunk, ylunk; // local indicies into unknown vector
  double* coord;
  ScalarT Xval, Yval;
  for (unsigned int inode = 0; inode < nsNodes.size(); inode++)
  {
    xlunk = nsNodes[inode][0];
    ylunk = nsNodes[inode][1];
    coord = nsNodeCoords[inode];

    this->computeBCs(coord, Xval, Yval, time);

    // replace jac values for the X dof
    for (int block=0; block<nblock_jac; block++) {
      (*jac)[block].ExtractMyRowView(xlunk, numEntries, matrixEntries,
                                     matrixIndices);
      for (int i=0; i<numEntries; i++) matrixEntries[i]=0;
      (*jac)[block].ReplaceMyValues(xlunk, 1, &diag, &xlunk);

      // replace jac values for the y dof
      (*jac)[block].ExtractMyRowView(ylunk, numEntries, matrixEntries,
                                     matrixIndices);
      for (int i=0; i<numEntries; i++) matrixEntries[i]=0;
      (*jac)[block].ReplaceMyValues(ylunk, 1, &diag, &ylunk);
    }

    if (fillResid)
    {
      for (int block=0; block<nblock; block++) {
        (*f)[block][xlunk] = ((*x)[block][xlunk] - Xval.val().coeff(block));
        (*f)[block][ylunk] = ((*x)[block][ylunk] - Yval.val().coeff(block));
      }
    }
  }
}

// **********************************************************************
// Specialization: Multi-point Tangent
// **********************************************************************
template<typename Traits>
TorsionBC<PHAL::AlbanyTraits::MPTangent, Traits>::
TorsionBC(Teuchos::ParameterList& p) :
  TorsionBC_Base<PHAL::AlbanyTraits::MPTangent, Traits>(p)
{
}
// **********************************************************************
template<typename Traits>
void TorsionBC<PHAL::AlbanyTraits::MPTangent, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
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

  int xlunk, ylunk; // global and local indicies into unknown vector
  double* coord;
  ScalarT Xval, Yval;
  for (unsigned int inode = 0; inode < nsNodes.size(); inode++)
  {
    xlunk = nsNodes[inode][0];
    ylunk = nsNodes[inode][1];
    coord = nsNodeCoords[inode];

    this->computeBCs(coord, Xval, Yval, time);

    if (f != Teuchos::null)
    {
      for (int block=0; block<nblock; block++) {
        (*f)[block][xlunk] = (*x)[block][xlunk] - Xval.val().coeff(block);
        (*f)[block][ylunk] = (*x)[block][ylunk] - Yval.val().coeff(block);
      }
    }

    if (JV != Teuchos::null) {
      for (int i=0; i<dirichletWorkset.num_cols_x; i++)
      {
        for (int block=0; block<nblock; block++) {
          (*JV)[block][i][xlunk] = j_coeff*(*Vx)[i][xlunk];
          (*JV)[block][i][ylunk] = j_coeff*(*Vx)[i][ylunk];
        }
      }
    }

    if (fp != Teuchos::null) {
      for (int i=0; i<dirichletWorkset.num_cols_p; i++)
      {
        for (int block=0; block<nblock; block++) {
          (*fp)[block][i][xlunk] =
            -Xval.dx(dirichletWorkset.param_offset+i).coeff(block);
          (*fp)[block][i][ylunk] =
            -Yval.dx(dirichletWorkset.param_offset+i).coeff(block);
        }
      }
    }

  }
}
#endif //ALBANY_SG_MP

} // namespace LCM

