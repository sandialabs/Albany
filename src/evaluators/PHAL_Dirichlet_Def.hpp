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

template<typename EvalT,typename Traits>
DirichletBase<EvalT, Traits>::
DirichletBase(Teuchos::ParameterList& p) :
  offset(p.get<int>("Equation Offset")),
  nodeSetID(p.get<std::string>("Node Set ID"))
{
  value = p.get<RealType>("Dirichlet Value");

  std::string name = p.get< std::string >("Dirichlet Name");
  PHX::Tag<ScalarT> fieldTag(name, p.get< Teuchos::RCP<PHX::DataLayout> >("Data Layout"));

  this->addEvaluatedField(fieldTag);

  this->setName(name+PHX::TypeString<EvalT>::value);

  // Set up values as parameters for parameter library
  Teuchos::RCP<ParamLib> paramLib = p.get< Teuchos::RCP<ParamLib> >
               ("Parameter Library", Teuchos::null);

  new Sacado::ParameterRegistration<EvalT, SPL_Traits> (name, this, paramLib);
}

template<typename EvalT, typename Traits>
void DirichletBase<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
}

// **********************************************************************
// Specialization: Residual
// **********************************************************************
template<typename Traits>
Dirichlet<PHAL::AlbanyTraits::Residual, Traits>::
Dirichlet(Teuchos::ParameterList& p) :
  DirichletBase<PHAL::AlbanyTraits::Residual, Traits>(p)
{
}

// **********************************************************************
template<typename Traits>
void Dirichlet<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
  Teuchos::RCP<Epetra_Vector> f = dirichletWorkset.f;
  Teuchos::RCP<const Epetra_Vector> x = dirichletWorkset.x;
  // Grab the vector off node GIDs for this Node Set ID from the std::map
  const std::vector<std::vector<int> >& nsNodes = dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
      int lunk = nsNodes[inode][this->offset];
      (*f)[lunk] = ((*x)[lunk] - this->value);
  }
}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************
template<typename Traits>
Dirichlet<PHAL::AlbanyTraits::Jacobian, Traits>::
Dirichlet(Teuchos::ParameterList& p) :
  DirichletBase<PHAL::AlbanyTraits::Jacobian, Traits>(p)
{
}

// **********************************************************************
template<typename Traits>
void Dirichlet<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{

  Teuchos::RCP<Epetra_Vector> f = dirichletWorkset.f;
  Teuchos::RCP<Epetra_CrsMatrix> jac = dirichletWorkset.Jac;
  Teuchos::RCP<const Epetra_Vector> x = dirichletWorkset.x;
  const RealType j_coeff = dirichletWorkset.j_coeff;
  const std::vector<std::vector<int> >& nsNodes = dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

  RealType* matrixEntries;
  int*    matrixIndices;
  int     numEntries;
  RealType diag=j_coeff;
  bool fillResid = (f != Teuchos::null);

  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
      int lunk = nsNodes[inode][this->offset];
      jac->ExtractMyRowView(lunk, numEntries, matrixEntries, matrixIndices);
      for (int i=0; i<numEntries; i++) matrixEntries[i]=0;
      jac->ReplaceMyValues(lunk, 1, &diag, &lunk);

      if (fillResid) (*f)[lunk] = ((*x)[lunk] - this->value.val());
  }
}

// **********************************************************************
// Specialization: Tangent
// **********************************************************************
template<typename Traits>
Dirichlet<PHAL::AlbanyTraits::Tangent, Traits>::
Dirichlet(Teuchos::ParameterList& p) :
  DirichletBase<PHAL::AlbanyTraits::Tangent, Traits>(p)
{
}

// **********************************************************************
template<typename Traits>
void Dirichlet<PHAL::AlbanyTraits::Tangent, Traits>::
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

  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    int lunk = nsNodes[inode][this->offset];

    if (f != Teuchos::null)
      (*f)[lunk] = ((*x)[lunk] - this->value.val());

    if (JV != Teuchos::null)
      for (int i=0; i<dirichletWorkset.num_cols_x; i++)
        (*JV)[i][lunk] = j_coeff*(*Vx)[i][lunk];

    if (fp != Teuchos::null)
      for (int i=0; i<dirichletWorkset.num_cols_p; i++)
        (*fp)[i][lunk] = -this->value.dx(dirichletWorkset.param_offset+i);
  }
}

// **********************************************************************
// Specialization: DistParamDeriv
// **********************************************************************
template<typename Traits>
Dirichlet<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
Dirichlet(Teuchos::ParameterList& p) :
  DirichletBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p)
{
}

// **********************************************************************
template<typename Traits>
void Dirichlet<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{

  Teuchos::RCP<Epetra_MultiVector> fpV = dirichletWorkset.fpV;
  bool trans = dirichletWorkset.transpose_dist_param_deriv;
  int num_cols = fpV->NumVectors();

  const std::vector<std::vector<int> >& nsNodes =
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

  // For (df/dp)^T*V we zero out corresponding entries in V
  if (trans) {
    Teuchos::RCP<Epetra_MultiVector> Vp = dirichletWorkset.Vp_bc;

    for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
      int lunk = nsNodes[inode][this->offset];

      for (int col=0; col<num_cols; ++col)
        (*Vp)[col][lunk] = 0.0;
    }
  }

  // for (df/dp)*V we zero out corresponding entries in df/dp
  else {
    for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
      int lunk = nsNodes[inode][this->offset];

      for (int col=0; col<num_cols; ++col)
        (*fpV)[col][lunk] = 0.0;
    }
  }
}

// **********************************************************************
// Specialization: Stochastic Galerkin Residual
// **********************************************************************

#ifdef ALBANY_SG_MP
template<typename Traits>
Dirichlet<PHAL::AlbanyTraits::SGResidual, Traits>::
Dirichlet(Teuchos::ParameterList& p) :
  DirichletBase<PHAL::AlbanyTraits::SGResidual, Traits>(p)
{
}
// **********************************************************************
template<typename Traits>
void Dirichlet<PHAL::AlbanyTraits::SGResidual, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
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
        (*f)[block][lunk] = ((*x)[block][lunk] - this->value.coeff(block));
  }
}

// **********************************************************************
// Specialization: Stochastic Galerkin Jacobian
// **********************************************************************
template<typename Traits>
Dirichlet<PHAL::AlbanyTraits::SGJacobian, Traits>::
Dirichlet(Teuchos::ParameterList& p) :
  DirichletBase<PHAL::AlbanyTraits::SGJacobian, Traits>(p)
{
}

// **********************************************************************
template<typename Traits>
void Dirichlet<PHAL::AlbanyTraits::SGJacobian, Traits>::
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
          (*f)[block][lunk] =
            (*x)[block][lunk] - this->value.val().coeff(block);
      }
  }
}

// **********************************************************************
// Specialization: Stochastic Galerkin Tangent
// **********************************************************************
template<typename Traits>
Dirichlet<PHAL::AlbanyTraits::SGTangent, Traits>::
Dirichlet(Teuchos::ParameterList& p) :
  DirichletBase<PHAL::AlbanyTraits::SGTangent, Traits>(p)
{
}

// **********************************************************************
template<typename Traits>
void Dirichlet<PHAL::AlbanyTraits::SGTangent, Traits>::
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

  int nblock = x->size();

  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    int lunk = nsNodes[inode][this->offset];

    if (f != Teuchos::null)
      for (int block=0; block<nblock; block++)
        (*f)[block][lunk] =
          (*x)[block][lunk] - this->value.val().coeff(block);

    if (JV != Teuchos::null)
      for (int i=0; i<dirichletWorkset.num_cols_x; i++)
        (*JV)[0][i][lunk] = j_coeff*(*Vx)[i][lunk];

    if (fp != Teuchos::null)
      for (int i=0; i<dirichletWorkset.num_cols_p; i++)
        for (int block=0; block<nblock; block++)
          (*fp)[block][i][lunk] =
            -this->value.dx(dirichletWorkset.param_offset+i).coeff(block);
  }
}

// **********************************************************************
// Specialization: Multi-point Residual
// **********************************************************************

template<typename Traits>
Dirichlet<PHAL::AlbanyTraits::MPResidual, Traits>::
Dirichlet(Teuchos::ParameterList& p) :
  DirichletBase<PHAL::AlbanyTraits::MPResidual, Traits>(p)
{
}
// **********************************************************************
template<typename Traits>
void Dirichlet<PHAL::AlbanyTraits::MPResidual, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{
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
        (*f)[block][lunk] = ((*x)[block][lunk] - this->value.coeff(block));
  }
}

// **********************************************************************
// Specialization: Multi-point Jacobian
// **********************************************************************
template<typename Traits>
Dirichlet<PHAL::AlbanyTraits::MPJacobian, Traits>::
Dirichlet(Teuchos::ParameterList& p) :
  DirichletBase<PHAL::AlbanyTraits::MPJacobian, Traits>(p)
{
}

// **********************************************************************
template<typename Traits>
void Dirichlet<PHAL::AlbanyTraits::MPJacobian, Traits>::
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
          (*f)[block][lunk] =
            (*x)[block][lunk] - this->value.val().coeff(block);
      }
  }
}

// **********************************************************************
// Specialization: Multi-point Tangent
// **********************************************************************
template<typename Traits>
Dirichlet<PHAL::AlbanyTraits::MPTangent, Traits>::
Dirichlet(Teuchos::ParameterList& p) :
  DirichletBase<PHAL::AlbanyTraits::MPTangent, Traits>(p)
{
}

// **********************************************************************
template<typename Traits>
void Dirichlet<PHAL::AlbanyTraits::MPTangent, Traits>::
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

  int nblock = x->size();

  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    int lunk = nsNodes[inode][this->offset];

    if (f != Teuchos::null)
      for (int block=0; block<nblock; block++)
        (*f)[block][lunk] =
          (*x)[block][lunk] - this->value.val().coeff(block);

    if (JV != Teuchos::null)
      for (int i=0; i<dirichletWorkset.num_cols_x; i++)
        for (int block=0; block<nblock; block++)
          (*JV)[block][i][lunk] = j_coeff*(*Vx)[i][lunk];

    if (fp != Teuchos::null)
      for (int i=0; i<dirichletWorkset.num_cols_p; i++)
        for (int block=0; block<nblock; block++)
          (*fp)[block][i][lunk] =
            -this->value.dx(dirichletWorkset.param_offset+i).coeff(block);
  }
}
#endif //ALBANY_SG_MP

// **********************************************************************
// Simple evaluator to aggregate all Dirichlet BCs into one "field"
// **********************************************************************

template<typename EvalT, typename Traits>
DirichletAggregator<EvalT, Traits>::
DirichletAggregator(Teuchos::ParameterList& p)
{
  Teuchos::RCP<PHX::DataLayout> dl =  p.get< Teuchos::RCP<PHX::DataLayout> >("Data Layout");

  std::vector<std::string>& dbcs = *(p.get<std::vector<std::string>* >("DBC Names"));

  for (unsigned int i=0; i<dbcs.size(); i++) {
    PHX::Tag<ScalarT> fieldTag(dbcs[i], dl);
    this->addDependentField(fieldTag);
  }

  PHX::Tag<ScalarT> fieldTag(p.get<std::string>("DBC Aggregator Name"), dl);
  this->addEvaluatedField(fieldTag);

  this->setName("Dirichlet Aggregator"+PHX::TypeString<EvalT>::value);
}

// **********************************************************************
}

