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

namespace QCAD {

template<typename EvalT,typename Traits>
SchrodingerDirichletBase<EvalT, Traits>::
SchrodingerDirichletBase(Teuchos::ParameterList& p) :
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
void SchrodingerDirichletBase<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
}

// **********************************************************************
// Specialization: Residual
// **********************************************************************
template<typename Traits>
SchrodingerDirichlet<PHAL::AlbanyTraits::Residual, Traits>::
SchrodingerDirichlet(Teuchos::ParameterList& p) :
  SchrodingerDirichletBase<PHAL::AlbanyTraits::Residual, Traits>(p)
{
}

// **********************************************************************
template<typename Traits>
void SchrodingerDirichlet<PHAL::AlbanyTraits::Residual, Traits>::
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
SchrodingerDirichlet<PHAL::AlbanyTraits::Jacobian, Traits>::
SchrodingerDirichlet(Teuchos::ParameterList& p) :
  SchrodingerDirichletBase<PHAL::AlbanyTraits::Jacobian, Traits>(p)
{
}

// **********************************************************************
template<typename Traits>
void SchrodingerDirichlet<PHAL::AlbanyTraits::Jacobian, Traits>::
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

  int nMyRows = jac->NumMyRows();
  RealType zero = 0.0;
  //std::vector<int> globalRows(nMyRows);
  //jac->RowMap().MyGlobalElements(&globalRows[0]);

  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
      int lunk = nsNodes[inode][this->offset];
      jac->ExtractMyRowView(lunk, numEntries, matrixEntries, matrixIndices);
      for (int i=0; i<numEntries; i++) matrixEntries[i]=0; // zero out row
      for (int i=0; i<nMyRows; i++) jac->ReplaceMyValues(i, 1, &zero, &lunk); //zero out col
      //int gunk = globalRows[lunk]; // convert local row index -> global index

      jac->ReplaceMyValues(lunk, 1, &diag, &lunk); //set diagonal element = j_coeff

      if (fillResid) (*f)[lunk] = ((*x)[lunk] - this->value.val());
  }
}

// **********************************************************************
// Specialization: Tangent
// **********************************************************************
template<typename Traits>
SchrodingerDirichlet<PHAL::AlbanyTraits::Tangent, Traits>::
SchrodingerDirichlet(Teuchos::ParameterList& p) :
  SchrodingerDirichletBase<PHAL::AlbanyTraits::Tangent, Traits>(p)
{
}

// **********************************************************************
template<typename Traits>
void SchrodingerDirichlet<PHAL::AlbanyTraits::Tangent, Traits>::
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
SchrodingerDirichlet<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
SchrodingerDirichlet(Teuchos::ParameterList& p) :
  SchrodingerDirichletBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p)
{
}

// **********************************************************************
template<typename Traits>
void SchrodingerDirichlet<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData dirichletWorkset)
{

  const std::vector<std::vector<int> >& nsNodes = 
    dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

  Teuchos::RCP<Epetra_MultiVector> fpV = dirichletWorkset.fpV;
  bool trans = dirichletWorkset.transpose_dist_param_deriv;
  int num_cols = fpV->NumVectors();

  if (trans) {
    Teuchos::RCP<Epetra_MultiVector> Vp = dirichletWorkset.Vp_bc;
    for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
      int lunk = nsNodes[inode][this->offset];
  
      for (int i=0; i<num_cols; i++)
	(*Vp)[i][lunk] = 0.0;
    }
  }

  else {
    for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
      int lunk = nsNodes[inode][this->offset];

      for (int i=0; i<num_cols; i++)
  	(*fpV)[i][lunk] = 0.0;
    }
  }
}

// **********************************************************************
// Specialization: Stochastic Galerkin Residual
// **********************************************************************

#ifdef ALBANY_SG_MP
template<typename Traits>
SchrodingerDirichlet<PHAL::AlbanyTraits::SGResidual, Traits>::
SchrodingerDirichlet(Teuchos::ParameterList& p) :
  SchrodingerDirichletBase<PHAL::AlbanyTraits::SGResidual, Traits>(p)
{
}
// **********************************************************************
template<typename Traits>
void SchrodingerDirichlet<PHAL::AlbanyTraits::SGResidual, Traits>::
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
SchrodingerDirichlet<PHAL::AlbanyTraits::SGJacobian, Traits>::
SchrodingerDirichlet(Teuchos::ParameterList& p) :
  SchrodingerDirichletBase<PHAL::AlbanyTraits::SGJacobian, Traits>(p)
{
}

// **********************************************************************
template<typename Traits>
void SchrodingerDirichlet<PHAL::AlbanyTraits::SGJacobian, Traits>::
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
SchrodingerDirichlet<PHAL::AlbanyTraits::SGTangent, Traits>::
SchrodingerDirichlet(Teuchos::ParameterList& p) :
  SchrodingerDirichletBase<PHAL::AlbanyTraits::SGTangent, Traits>(p)
{
}

// **********************************************************************
template<typename Traits>
void SchrodingerDirichlet<PHAL::AlbanyTraits::SGTangent, Traits>::
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
SchrodingerDirichlet<PHAL::AlbanyTraits::MPResidual, Traits>::
SchrodingerDirichlet(Teuchos::ParameterList& p) :
  SchrodingerDirichletBase<PHAL::AlbanyTraits::MPResidual, Traits>(p)
{
}
// **********************************************************************
template<typename Traits>
void SchrodingerDirichlet<PHAL::AlbanyTraits::MPResidual, Traits>::
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
SchrodingerDirichlet<PHAL::AlbanyTraits::MPJacobian, Traits>::
SchrodingerDirichlet(Teuchos::ParameterList& p) :
  SchrodingerDirichletBase<PHAL::AlbanyTraits::MPJacobian, Traits>(p)
{
}

// **********************************************************************
template<typename Traits>
void SchrodingerDirichlet<PHAL::AlbanyTraits::MPJacobian, Traits>::
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
SchrodingerDirichlet<PHAL::AlbanyTraits::MPTangent, Traits>::
SchrodingerDirichlet(Teuchos::ParameterList& p) :
  SchrodingerDirichletBase<PHAL::AlbanyTraits::MPTangent, Traits>(p)
{
}

// **********************************************************************
template<typename Traits>
void SchrodingerDirichlet<PHAL::AlbanyTraits::MPTangent, Traits>::
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
// Simple evaluator to aggregate all SchrodingerDirichlet BCs into one "field"
// **********************************************************************

template<typename EvalT, typename Traits>
SchrodingerDirichletAggregator<EvalT, Traits>::
SchrodingerDirichletAggregator(Teuchos::ParameterList& p) 
{
  Teuchos::RCP<PHX::DataLayout> dl =  p.get< Teuchos::RCP<PHX::DataLayout> >("Data Layout");

  std::vector<std::string>& dbcs = *(p.get<std::vector<std::string>* >("DBC Names"));

  for (unsigned int i=0; i<dbcs.size(); i++) {
    PHX::Tag<ScalarT> fieldTag(dbcs[i], dl);
    this->addDependentField(fieldTag);
  }

  PHX::Tag<ScalarT> fieldTag(p.get<std::string>("DBC Aggregator Name"), dl);
  this->addEvaluatedField(fieldTag);

  this->setName("SchrodingerDirichlet Aggregator"+PHX::TypeString<EvalT>::value);
}

// **********************************************************************
}

