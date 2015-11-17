//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

namespace LCM {

template <typename EvalT, typename Traits>
EquilibriumConcentrationBC_Base<EvalT, Traits>::
EquilibriumConcentrationBC_Base(Teuchos::ParameterList& p) :
  offset_(p.get<int>("Equation Offset")),
  PHAL::DirichletBase<EvalT, Traits>(p),
  term1_(p.get<RealType>("Term1")),
  term2_(p.get<RealType>("Term2"))
{
}
//------------------------------------------------------------------------------
//
template<typename EvalT, typename Traits>
void
EquilibriumConcentrationBC_Base<EvalT, Traits>::
computeBCs(ScalarT& pressure, ScalarT& Cval)
{

  Cval = pressure*(term1_/term2_);
/*
//  JTO: I am going to leave this here for now...
     std::cout << "================" << std::endl;
     std::cout.precision(15);
     std::cout << "X : " << X << ", Y: " << Y << ", R: " << R << std::endl;
//     std::cout << "Node : " << nsNodes[inode] << std::endl;
     std::cout << "KI : " << KI << ", KII: " << KII << std::endl;
     std::cout << "theta: " << theta << std::endl;
     std::cout << "coeff_1: " << coeff_1 << ", coeff_2: " << coeff_2 << std::endl;
     std::cout << "KI_X: " << KI_X << ", KI_Y: " << KI_Y << std::endl;
     std::cout << "Xval: " << Xval << ", Yval: " << Yval << std::endl;
     std::cout << "nu: " << nu << std::endl;
//     std::cout << "dx: " << (*x)[xlunk] << std::endl;
//     std::cout << "dy: " << (*x)[ylunk] << std::endl;
//     std::cout << "fx: " << ((*x)[xlunk] - Xval) << std::endl;
//     std::cout << "fy: " << ((*x)[ylunk] - Yval) << std::endl;
     std::cout << "sin(theta/2): " << std::sin( theta / 2.0 ) << std::endl;
     std::cout << "cos(theta/2): " << std::cos( theta / 2.0 ) << std::endl;

*/
}

//------------------------------------------------------------------------------
// Specialization: Residual
//
template<typename Traits>
EquilibriumConcentrationBC<PHAL::AlbanyTraits::Residual, Traits>::
EquilibriumConcentrationBC(Teuchos::ParameterList& p) :
  EquilibriumConcentrationBC_Base<PHAL::AlbanyTraits::Residual, Traits>(p)
{
}
//------------------------------------------------------------------------------
//
template<typename Traits>
void
EquilibriumConcentrationBC<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  Teuchos::RCP<Tpetra_Vector> fT = dirichletWorkset.fT;
  Teuchos::RCP<const Tpetra_Vector> xT = dirichletWorkset.xT;
  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();
  Teuchos::ArrayRCP<ST> fT_nonconstView = fT->get1dViewNonConst();


  // Grab the vector of node GIDs for this Node Set ID from the std::map
  const std::vector<std::vector<int>>& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

  int cunk, punk;
  ScalarT Cval;
  ScalarT pressure;
  
  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    cunk = nsNodes[inode][this->offset_];
    punk = nsNodes[inode][this->offset_+1];
    pressure = xT_constView[punk];
    this->computeBCs(pressure, Cval);

    fT_nonconstView[cunk] = xT_constView[cunk] - Cval;
  }
}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************
template<typename Traits>
EquilibriumConcentrationBC<PHAL::AlbanyTraits::Jacobian, Traits>::
EquilibriumConcentrationBC(Teuchos::ParameterList& p) :
  EquilibriumConcentrationBC_Base<PHAL::AlbanyTraits::Jacobian, Traits>(p)
{
}
// **********************************************************************
template<typename Traits>
void EquilibriumConcentrationBC<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{

  Teuchos::RCP<Tpetra_Vector> fT = dirichletWorkset.fT;
  Teuchos::RCP<const Tpetra_Vector> xT = dirichletWorkset.xT;
  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();
  Teuchos::RCP<Tpetra_CrsMatrix> jacT = dirichletWorkset.JacT;

  const RealType j_coeff = dirichletWorkset.j_coeff;
  const std::vector<std::vector<int>>& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

  bool fillResid = (fT != Teuchos::null);
  Teuchos::ArrayRCP<ST> fT_nonconstView;
  if (fillResid) fT_nonconstView = fT->get1dViewNonConst();

  int cunk, punk;
  ScalarT Cval;
  ScalarT pressure;

  Teuchos::Array<LO> index(1);
  Teuchos::Array<ST> value(1);
  size_t numEntriesT;
  value[0] = j_coeff;
  Teuchos::Array<ST> matrixEntriesT;
  Teuchos::Array<LO> matrixIndicesT;


  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) 
  {
    cunk = nsNodes[inode][this->offset_];
    punk = nsNodes[inode][this->offset_+1];
    pressure = xT_constView[punk];
    this->computeBCs(pressure, Cval);
    
    // replace jac values for the C dof 
    numEntriesT = jacT->getNumEntriesInLocalRow(cunk);
    matrixEntriesT.resize(numEntriesT);
    matrixIndicesT.resize(numEntriesT);
    jacT->getLocalRowCopy(cunk, matrixIndicesT(), matrixEntriesT(), numEntriesT);
    for (int i=0; i<numEntriesT; i++) matrixEntriesT[i]=0;
    jacT->replaceLocalValues(cunk, matrixIndicesT(), matrixEntriesT());

    index[0] = cunk;
    jacT->replaceLocalValues(cunk, index(), value());

    if (fillResid)
    {
      fT_nonconstView[cunk] = xT_constView[cunk] - Cval.val();
    } 
  }
}

// **********************************************************************
// Specialization: Tangent
// **********************************************************************
template<typename Traits>
EquilibriumConcentrationBC<PHAL::AlbanyTraits::Tangent, Traits>::
EquilibriumConcentrationBC(Teuchos::ParameterList& p) :
  EquilibriumConcentrationBC_Base<PHAL::AlbanyTraits::Tangent, Traits>(p)
{
}
// **********************************************************************
template<typename Traits>
void EquilibriumConcentrationBC<PHAL::AlbanyTraits::Tangent, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
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
  const std::vector<std::vector<int>>& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

  int cunk, punk;
  ScalarT Cval;
  ScalarT pressure;

  for (unsigned int inode = 0; inode < nsNodes.size(); inode++)
  {
    cunk = nsNodes[inode][this->offset_];
    punk = nsNodes[inode][this->offset_+1];
    pressure = xT_constView[punk];
    this->computeBCs(pressure, Cval);

    if (fT != Teuchos::null)
    {
      fT_nonconstView[cunk] = xT_constView[cunk] - Cval.val();
    } 

    if (JVT != Teuchos::null) {
      Teuchos::ArrayRCP<ST> JVT_nonconstView; 
      for (int i=0; i<dirichletWorkset.num_cols_x; i++)
      {
        JVT_nonconstView = JVT->getDataNonConst(i); 
        VxT_constView = VxT->getData(i); 
	JVT_nonconstView[cunk] = j_coeff*VxT_constView[cunk];
      }
    }
    
    if (fpT != Teuchos::null) {
      Teuchos::ArrayRCP<ST> fpT_nonconstView; 
      for (int i=0; i<dirichletWorkset.num_cols_p; i++)
      {
        fpT_nonconstView = fpT->getDataNonConst(i); 
	fpT_nonconstView[cunk] = -Cval.dx(dirichletWorkset.param_offset+i);
      }
    }

  }
}

// **********************************************************************
// Specialization: DistParamDeriv
// **********************************************************************
template<typename Traits>
EquilibriumConcentrationBC<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
EquilibriumConcentrationBC(Teuchos::ParameterList& p) :
  EquilibriumConcentrationBC_Base<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p)
{
}
// **********************************************************************
template<typename Traits>
void EquilibriumConcentrationBC<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{

  Teuchos::RCP<Tpetra_MultiVector> fpVT = dirichletWorkset.fpVT;
  Teuchos::ArrayRCP<ST> fpVT_nonconstView; 
  bool trans = dirichletWorkset.transpose_dist_param_deriv;
  int num_cols = fpVT->getNumVectors();

  //
  // We're currently assuming Dirichlet BC's can't be distributed parameters.
  // Thus we don't need to actually evaluate the BC's here.  The code to do
  // so is still here, just commented out for future reference.
  //

  const std::vector<std::vector<int>>& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

  int cunk; // global and local indicies into unknown vector

  // For (df/dp)^T*V we zero out corresponding entries in V
  if (trans) {
    for (unsigned int inode = 0; inode < nsNodes.size(); inode++)
    {
      Teuchos::RCP<Tpetra_MultiVector> VpT = dirichletWorkset.Vp_bcT;
      Teuchos::ArrayRCP<ST> VpT_nonconstView; 
      cunk = nsNodes[inode][this->offset_];

      for (int col=0; col<num_cols; ++col) {
        VpT_nonconstView = VpT->getDataNonConst(col); 
        VpT_nonconstView[cunk] = 0.0; 
      }
    }
  }

  // for (df/dp)*V we zero out corresponding entries in df/dp
  else {
    for (unsigned int inode = 0; inode < nsNodes.size(); inode++)
    {
      cunk = nsNodes[inode][this->offset_];

      for (int col=0; col<num_cols; ++col) {
        fpVT_nonconstView = fpVT->getDataNonConst(col);
        fpVT_nonconstView[cunk] = 0.0; 
      }
    }
  }
}

// **********************************************************************
// Specialization: Stochastic Galerkin Residual
// **********************************************************************
#ifdef ALBANY_SG
template<typename Traits>
EquilibriumConcentrationBC<PHAL::AlbanyTraits::SGResidual, Traits>::
EquilibriumConcentrationBC(Teuchos::ParameterList& p) :
  EquilibriumConcentrationBC_Base<PHAL::AlbanyTraits::SGResidual, Traits>(p)
{
}
// **********************************************************************
template<typename Traits>
void EquilibriumConcentrationBC<PHAL::AlbanyTraits::SGResidual, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  Teuchos::RCP<Stokhos::EpetraVectorOrthogPoly> f =
    dirichletWorkset.sg_f;
  Teuchos::RCP<const Stokhos::EpetraVectorOrthogPoly> x =
    dirichletWorkset.sg_x;
  const std::vector<std::vector<int>>& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

  int cunk, punk;
  ScalarT Cval;
  ScalarT pressure;
  
  int nblock = x->size();
  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    cunk = nsNodes[inode][0];
    pressure = coord = nsNodeCoords[inode];

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
EquilibriumConcentrationBC<PHAL::AlbanyTraits::SGJacobian, Traits>::
EquilibriumConcentrationBC(Teuchos::ParameterList& p) :
  EquilibriumConcentrationBC_Base<PHAL::AlbanyTraits::SGJacobian, Traits>(p)
{
}
// **********************************************************************
template<typename Traits>
void EquilibriumConcentrationBC<PHAL::AlbanyTraits::SGJacobian, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  Teuchos::RCP< Stokhos::EpetraVectorOrthogPoly> f =
    dirichletWorkset.sg_f;
  Teuchos::RCP< Stokhos::VectorOrthogPoly<Epetra_CrsMatrix>> jac =
    dirichletWorkset.sg_Jac;
  Teuchos::RCP<const Stokhos::EpetraVectorOrthogPoly> x =
    dirichletWorkset.sg_x;
  const RealType j_coeff = dirichletWorkset.j_coeff;
  const std::vector<std::vector<int>>& nsNodes =
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

  int cunk, punk;
  ScalarT Cval;
  ScalarT pressure;
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
EquilibriumConcentrationBC<PHAL::AlbanyTraits::SGTangent, Traits>::
EquilibriumConcentrationBC(Teuchos::ParameterList& p) :
  EquilibriumConcentrationBC_Base<PHAL::AlbanyTraits::SGTangent, Traits>(p)
{
}
// **********************************************************************
template<typename Traits>
void EquilibriumConcentrationBC<PHAL::AlbanyTraits::SGTangent, Traits>::
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
  const std::vector<std::vector<int>>& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
  const std::vector<double*>& nsNodeCoords =
    dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;

  int nblock = x->size();

  int cunk, punk;
  ScalarT Cval;
  ScalarT pressure;
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

#endif 
#ifdef ALBANY_ENSEMBLE 

// **********************************************************************
// Specialization: Multi-point Residual
// **********************************************************************
template<typename Traits>
EquilibriumConcentrationBC<PHAL::AlbanyTraits::MPResidual, Traits>::
EquilibriumConcentrationBC(Teuchos::ParameterList& p) :
  EquilibriumConcentrationBC_Base<PHAL::AlbanyTraits::MPResidual, Traits>(p)
{
}
// **********************************************************************
template<typename Traits>
void EquilibriumConcentrationBC<PHAL::AlbanyTraits::MPResidual, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  Teuchos::RCP<Stokhos::ProductEpetraVector> f =
    dirichletWorkset.mp_f;
  Teuchos::RCP<const Stokhos::ProductEpetraVector> x =
    dirichletWorkset.mp_x;
  const std::vector<std::vector<int>>& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
  const std::vector<double*>& nsNodeCoords =
    dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;


  int cunk, punk;
  ScalarT Cval;
  ScalarT pressure;

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
EquilibriumConcentrationBC<PHAL::AlbanyTraits::MPJacobian, Traits>::
EquilibriumConcentrationBC(Teuchos::ParameterList& p) :
  EquilibriumConcentrationBC_Base<PHAL::AlbanyTraits::MPJacobian, Traits>(p)
{
}
// **********************************************************************
template<typename Traits>
void EquilibriumConcentrationBC<PHAL::AlbanyTraits::MPJacobian, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  Teuchos::RCP<Stokhos::ProductEpetraVector> f =
    dirichletWorkset.mp_f;
  Teuchos::RCP< Stokhos::ProductContainer<Epetra_CrsMatrix>> jac =
    dirichletWorkset.mp_Jac;
  Teuchos::RCP<const Stokhos::ProductEpetraVector> x =
    dirichletWorkset.mp_x;
  const RealType j_coeff = dirichletWorkset.j_coeff;
  const std::vector<std::vector<int>>& nsNodes =
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

  int cunk, punk;
  ScalarT Cval;
  ScalarT pressure;
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
EquilibriumConcentrationBC<PHAL::AlbanyTraits::MPTangent, Traits>::
EquilibriumConcentrationBC(Teuchos::ParameterList& p) :
  EquilibriumConcentrationBC_Base<PHAL::AlbanyTraits::MPTangent, Traits>(p)
{
}
// **********************************************************************
template<typename Traits>
void EquilibriumConcentrationBC<PHAL::AlbanyTraits::MPTangent, Traits>::
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
  const std::vector<std::vector<int>>& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
  const std::vector<double*>& nsNodeCoords =
    dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;

  int nblock = x->size();

  int cunk, punk;
  ScalarT Cval;
  ScalarT pressure;
  for (unsigned int inode = 0; inode < nsNodes.size(); inode++)
  {
    cunk = nsNodes[inode][offset_];
    punk = nsNodes[inode][offset_+1];
    pressure = 

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
#endif

} // namespace LCM

