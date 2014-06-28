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
TimeDepBC_Base<EvalT, Traits>::
TimeDepBC_Base(Teuchos::ParameterList& p) :
  offset(p.get<int>("Equation Offset")),

  PHAL::DirichletBase<EvalT, Traits>(p)
{
  timeValues = p.get<Teuchos::Array<RealType> >("Time Values").toVector();
  BCValues = p.get<Teuchos::Array<RealType> >("BC Values").toVector();

  TEUCHOS_TEST_FOR_EXCEPTION( !(timeValues.size() == BCValues.size()),
                              Teuchos::Exceptions::InvalidParameter,
                              "Dimension of \"Time Values\" and \"BC Values\" do not match" );
}

// **********************************************************************
template<typename EvalT, typename Traits>
typename TimeDepBC_Base<EvalT, Traits>::ScalarT
TimeDepBC_Base<EvalT, Traits>::
computeVal(RealType time)
{
  TEUCHOS_TEST_FOR_EXCEPTION( time > timeValues.back(),
                              Teuchos::Exceptions::InvalidParameter,
                              "Time is growing unbounded!" );
  ScalarT Val;
  RealType slope;
  unsigned int Index(0);

  while( timeValues[Index] < time )
    Index++;

  if (Index == 0)
    Val = BCValues[Index];
  else
  {
    slope = ( BCValues[Index] - BCValues[Index - 1] ) / ( timeValues[Index] - timeValues[Index - 1] );
    Val = BCValues[Index-1] + slope * ( time - timeValues[Index - 1] );
  }

  return Val;
}
// **********************************************************************
// Specialization: Residual
// **********************************************************************
template<typename Traits>
TimeDepBC<PHAL::AlbanyTraits::Residual, Traits>::
TimeDepBC(Teuchos::ParameterList& p) :
  TimeDepBC_Base<PHAL::AlbanyTraits::Residual, Traits>(p)
{
}

// **********************************************************************
template<typename Traits>
void
TimeDepBC<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  Teuchos::RCP<Epetra_Vector> f = dirichletWorkset.f;
  Teuchos::RCP<const Epetra_Vector> x = dirichletWorkset.x;

  Teuchos::RCP<Tpetra_Vector> fT = dirichletWorkset.fT;
  Teuchos::RCP<const Tpetra_Vector> xT = dirichletWorkset.xT;
  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();
  Teuchos::ArrayRCP<ST> fT_nonconstView = fT->get1dViewNonConst();

  RealType time = dirichletWorkset.current_time;
  // Grab the vector off node GIDs for this Node Set ID from the std::map
  const std::vector<std::vector<int> >& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

  int lunk; // global and local indicies into unknown vector
  ScalarT Val;

  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    lunk = nsNodes[inode][this->offset];

    Val = this->computeVal(time);

    //(*f)[lunk] = ((*x)[lunk] - Val);
    fT_nonconstView[lunk] = xT_constView[lunk] - Val;

  }
}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************
template<typename Traits>
TimeDepBC<PHAL::AlbanyTraits::Jacobian, Traits>::
TimeDepBC(Teuchos::ParameterList& p) :
  TimeDepBC_Base<PHAL::AlbanyTraits::Jacobian, Traits>(p)
{
}
// **********************************************************************
template<typename Traits>
void TimeDepBC<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{

  Teuchos::RCP<Epetra_Vector> f = dirichletWorkset.f;
  Teuchos::RCP<Epetra_CrsMatrix> jac = dirichletWorkset.Jac;
  Teuchos::RCP<const Epetra_Vector> x = dirichletWorkset.x;

  Teuchos::RCP<Tpetra_Vector> fT = dirichletWorkset.fT;
  Teuchos::RCP<const Tpetra_Vector> xT = dirichletWorkset.xT;
  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();
  Teuchos::RCP<Tpetra_CrsMatrix> jacT = dirichletWorkset.JacT;

  RealType time = dirichletWorkset.current_time;
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
  Teuchos::ArrayRCP<ST> fT_nonconstView;
  if (fillResid) fT_nonconstView = fT->get1dViewNonConst();

  int lunk; // local indicies into unknown vector
  Teuchos::Array<LO> index(1);
  Teuchos::Array<ST> value(1);
  size_t numEntriesT;
  value[0] = j_coeff;
  Teuchos::Array<ST> matrixEntriesT;
  Teuchos::Array<LO> matrixIndicesT;

  ScalarT Val; 
  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) 
  {
    lunk = nsNodes[inode][this->offset];

    Val = this->computeVal(time);
    
    // replace jac values for the X dof
    numEntriesT = jacT->getNumEntriesInLocalRow(lunk);
    matrixEntriesT.resize(numEntriesT);
    matrixIndicesT.resize(numEntriesT);
    jacT->getLocalRowCopy(lunk, matrixIndicesT(), matrixEntriesT(), numEntriesT);
    for (int i=0; i<numEntriesT; i++) matrixEntriesT[i]=0;
    jacT->replaceLocalValues(lunk, matrixIndicesT(), matrixEntriesT());

    index[0] = lunk;
    jacT->replaceLocalValues(lunk, index(), value());


    if (fillResid)  fT_nonconstView[lunk] = xT_constView[lunk] - Val.val();
 
  }
}

// **********************************************************************
// Specialization: Tangent
// **********************************************************************
template<typename Traits>
TimeDepBC<PHAL::AlbanyTraits::Tangent, Traits>::
TimeDepBC(Teuchos::ParameterList& p) :
  TimeDepBC_Base<PHAL::AlbanyTraits::Tangent, Traits>(p)
{
}
// **********************************************************************
template<typename Traits>
void TimeDepBC<PHAL::AlbanyTraits::Tangent, Traits>::
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

  RealType time = dirichletWorkset.current_time;
  const RealType j_coeff = dirichletWorkset.j_coeff;
  const std::vector<std::vector<int> >& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
  const std::vector<double*>& nsNodeCoords =
    dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;

  int lunk; // global and local indicies into unknown vector
  ScalarT Val;

  for (unsigned int inode = 0; inode < nsNodes.size(); inode++)
  {
    lunk = nsNodes[inode][this->offset];

    Val = this->computeVal(time);

    if (fT != Teuchos::null) {
      fT_nonconstView[lunk] = xT_constView[lunk] - Val.val();
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
        fpT_nonconstView[lunk] = -Val.dx(dirichletWorkset.param_offset+i);
      }
    }
  }
}

// **********************************************************************
// Specialization: DistParamDeriv
// **********************************************************************
template<typename Traits>
TimeDepBC<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
TimeDepBC(Teuchos::ParameterList& p) :
  TimeDepBC_Base<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p)
{
}
// **********************************************************************
template<typename Traits>
void TimeDepBC<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
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

  // RealType time = dirichletWorkset.current_time;
  const std::vector<std::vector<int> >& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;
  const std::vector<double*>& nsNodeCoords =
    dirichletWorkset.nodeSetCoords->find(this->nodeSetID)->second;

  int lunk; // global and local indicies into unknown vector
  // ScalarT Val;

  // For (df/dp)^T*V we zero out corresponding entries in V
  if (trans) {
    Teuchos::RCP<Tpetra_MultiVector> VpT = dirichletWorkset.Vp_bcT;
    Teuchos::ArrayRCP<ST> VpT_nonconstView; 
    for (unsigned int inode = 0; inode < nsNodes.size(); inode++)
    {
      lunk = nsNodes[inode][this->offset];

      // Val = this->computeVal(time);

      for (int col=0; col<num_cols; ++col) {
        //(*Vp)[col][lunk] = 0.0;
        VpT_nonconstView = VpT->getDataNonConst(col); 
        VpT_nonconstView[lunk] = 0.0; 
      }
    }
  }

  // for (df/dp)*V we zero out corresponding entries in df/dp
  else {
    for (unsigned int inode = 0; inode < nsNodes.size(); inode++)
    {
      lunk = nsNodes[inode][this->offset];

      // Val = this->computeVal(time);

      for (int col=0; col<num_cols; ++col) {
        //(*fpV)[col][lunk] = 0.0;
        fpVT_nonconstView = fpVT->getDataNonConst(col); 
        fpVT_nonconstView[lunk] = 0.0; 
      }
    }
  }
}


// **********************************************************************
// Specialization: Stochastic Galerkin Residual
// **********************************************************************
#ifdef ALBANY_SG_MP
template<typename Traits>
TimeDepBC<PHAL::AlbanyTraits::SGResidual, Traits>::
TimeDepBC(Teuchos::ParameterList& p) :
  TimeDepBC_Base<PHAL::AlbanyTraits::SGResidual, Traits>(p)
{
}
// **********************************************************************
template<typename Traits>
void TimeDepBC<PHAL::AlbanyTraits::SGResidual, Traits>::
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
  int lunk; // global and local indicies into unknown vector
  ScalarT Val;
  int nblock = x->size();
  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    lunk = nsNodes[inode][this->offset];

    Val = this->computeVal(time);

    for (int block=0; block<nblock; block++)
      (*f)[block][lunk] = ((*x)[block][lunk] - Val.coeff(block));
  }
}

// **********************************************************************
// Specialization: Stochastic Galerkin Jacobian
// **********************************************************************
template<typename Traits>
TimeDepBC<PHAL::AlbanyTraits::SGJacobian, Traits>::
TimeDepBC(Teuchos::ParameterList& p) :
  TimeDepBC_Base<PHAL::AlbanyTraits::SGJacobian, Traits>(p)
{
}
// **********************************************************************
template<typename Traits>
void TimeDepBC<PHAL::AlbanyTraits::SGJacobian, Traits>::
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
  int lunk; // local indicies into unknown vector
  ScalarT Val;

  for (unsigned int inode = 0; inode < nsNodes.size(); inode++)
  {
    lunk = nsNodes[inode][this->offset];

    Val = this->computeVal(time);

    // replace jac values for the X dof
    for (int block=0; block<nblock_jac; block++)
    {
      (*jac)[block].ExtractMyRowView(lunk, numEntries, matrixEntries,
                                     matrixIndices);
      for (int i=0; i<numEntries; i++) matrixEntries[i]=0;
    }
    (*jac)[0].ReplaceMyValues(lunk, 1, &diag, &lunk);

    if (fillResid)
      for (int block=0; block<nblock; block++)
        (*f)[block][lunk] = ((*x)[block][lunk] - Val.val().coeff(block));
  }
}

// **********************************************************************
// Specialization: Stochastic Galerkin Tangent
// **********************************************************************
template<typename Traits>
TimeDepBC<PHAL::AlbanyTraits::SGTangent, Traits>::
TimeDepBC(Teuchos::ParameterList& p) :
  TimeDepBC_Base<PHAL::AlbanyTraits::SGTangent, Traits>(p)
{
}
// **********************************************************************
template<typename Traits>
void TimeDepBC<PHAL::AlbanyTraits::SGTangent, Traits>::
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

  if (JV != Teuchos::null)
    JV->init(0.0);
  if (fp != Teuchos::null)
    fp->init(0.0);

  RealType time = dirichletWorkset.current_time;
  int lunk; // global and local indicies into unknown vector
  ScalarT Val;
  for (unsigned int inode = 0; inode < nsNodes.size(); inode++)
  {
    lunk = nsNodes[inode][this->offset];

    Val = this->computeVal(time);

    if (f != Teuchos::null)
      for (int block=0; block<nblock; block++)
        (*f)[block][lunk] = (*x)[block][lunk] - Val.val().coeff(block);

    if (JV != Teuchos::null)
      for (int i=0; i<dirichletWorkset.num_cols_x; i++)
        (*JV)[0][i][lunk] = j_coeff*(*Vx)[i][lunk];

    if (fp != Teuchos::null)
      for (int i=0; i<dirichletWorkset.num_cols_p; i++)
        for (int block=0; block<nblock; block++)
          (*fp)[block][i][lunk] =
            -Val.dx(dirichletWorkset.param_offset+i).coeff(block);
  }
}


// **********************************************************************
// Specialization: Multi-point Residual
// **********************************************************************
template<typename Traits>
TimeDepBC<PHAL::AlbanyTraits::MPResidual, Traits>::
TimeDepBC(Teuchos::ParameterList& p) :
  TimeDepBC_Base<PHAL::AlbanyTraits::MPResidual, Traits>(p)
{
}
// **********************************************************************
template<typename Traits>
void TimeDepBC<PHAL::AlbanyTraits::MPResidual, Traits>::
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
  int lunk; // global and local indicies into unknown vector
  ScalarT Val;

  int nblock = x->size();
  for (unsigned int inode = 0; inode < nsNodes.size(); inode++)
  {
    lunk = nsNodes[inode][0];

    Val = this->computeVal(time);

    for (int block=0; block<nblock; block++)
      (*f)[block][lunk] = ((*x)[block][lunk] - Val.coeff(block));
  }
}

// **********************************************************************
// Specialization: Multi-point Jacobian
// **********************************************************************
template<typename Traits>
TimeDepBC<PHAL::AlbanyTraits::MPJacobian, Traits>::
TimeDepBC(Teuchos::ParameterList& p) :
  TimeDepBC_Base<PHAL::AlbanyTraits::MPJacobian, Traits>(p)
{
}
// **********************************************************************
template<typename Traits>
void TimeDepBC<PHAL::AlbanyTraits::MPJacobian, Traits>::
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
  int lunk; // local indicies into unknown vector
  ScalarT Val;
  for (unsigned int inode = 0; inode < nsNodes.size(); inode++)
  {
    lunk = nsNodes[inode][this->offset];

    Val = this->computeVal(time);

    // replace jac values
    for (int block=0; block<nblock_jac; block++)
    {
      (*jac)[block].ExtractMyRowView(lunk, numEntries, matrixEntries,
                                     matrixIndices);
      for (int i=0; i<numEntries; i++)
        matrixEntries[i]=0;
      (*jac)[block].ReplaceMyValues(lunk, 1, &diag, &lunk);
    }

    if (fillResid)
      for (int block=0; block<nblock; block++)
        (*f)[block][lunk] = ((*x)[block][lunk] - Val.val().coeff(block));
  }
}

// **********************************************************************
// Specialization: Multi-point Tangent
// **********************************************************************
template<typename Traits>
TimeDepBC<PHAL::AlbanyTraits::MPTangent, Traits>::
TimeDepBC(Teuchos::ParameterList& p) :
  TimeDepBC_Base<PHAL::AlbanyTraits::MPTangent, Traits>(p)
{
}
// **********************************************************************
template<typename Traits>
void TimeDepBC<PHAL::AlbanyTraits::MPTangent, Traits>::
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
  int lunk; // global and local indicies into unknown vector
  ScalarT Val;
  for (unsigned int inode = 0; inode < nsNodes.size(); inode++)
  {
    lunk = nsNodes[inode][this->offset];

    Val = this->computeVal(time);

    if (f != Teuchos::null)
      for (int block=0; block<nblock; block++)
        (*f)[block][lunk] = (*x)[block][lunk] - Val.val().coeff(block);


    if (JV != Teuchos::null)
      for (int i=0; i<dirichletWorkset.num_cols_x; i++)
        for (int block=0; block<nblock; block++)
          (*JV)[block][i][lunk] = j_coeff*(*Vx)[i][lunk];


    if (fp != Teuchos::null)
      for (int i=0; i<dirichletWorkset.num_cols_p; i++)
        for (int block=0; block<nblock; block++)
          (*fp)[block][i][lunk] =
            -Val.dx(dirichletWorkset.param_offset+i).coeff(block);
  }
}
#endif //ALBANY_SG_MP

} // namespace LCM

