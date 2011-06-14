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
  neq(p.get<int>("Number of Equations")),
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
  const std::vector<int>& nsNodes = dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

  int gunk, lunk; // global and local indicies into unknown vector
  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
      gunk = nsNodes[inode] * this->neq + this->offset;
      lunk = f->Map().LID(gunk);
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
  const std::vector<int>& nsNodes = dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

  const Epetra_Map& map = jac->RowMap();

  RealType* matrixEntries;
  int*    matrixIndices;
  int     numEntries;
  RealType diag=j_coeff;
  bool fillResid = (f != Teuchos::null);

  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    const unsigned nodeid = nsNodes[inode];
      const int gunk = nodeid * this->neq + this->offset;
      int lunk = map.LID(gunk);
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
  const std::vector<int>& nsNodes = dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

  const Epetra_BlockMap& map = x->Map();
  bool fillResid = (f != Teuchos::null);

  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
    const unsigned nodeid = nsNodes[inode];
      const int gunk = nodeid * this->neq + this->offset;
      int lunk = map.LID(gunk);

      if (fillResid) (*f)[lunk] = ((*x)[lunk] - this->value.val());

      for (int i=0; i<dirichletWorkset.num_cols_x; i++)
	(*JV)[i][lunk] = j_coeff*(*Vx)[i][lunk];

      for (int i=0; i<dirichletWorkset.num_cols_p; i++)
	(*fp)[i][lunk] = -this->value.dx(dirichletWorkset.param_offset+i);
  }
}

// **********************************************************************
// Specialization: Stochastic Galerkin Residual
// **********************************************************************

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
  Teuchos::RCP< Stokhos::VectorOrthogPoly<Epetra_Vector> > f = 
    dirichletWorkset.sg_f;
  Teuchos::RCP< const Stokhos::VectorOrthogPoly<Epetra_Vector> > x = 
    dirichletWorkset.sg_x;
  const std::vector<int>& nsNodes = dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

  int nblock = x->size();
  int gunk, lunk; // global and local indicies into unknown vector
  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
      gunk = nsNodes[inode] * this->neq + this->offset;
      lunk = (*f)[0].Map().LID(gunk);
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

  Teuchos::RCP< Stokhos::VectorOrthogPoly<Epetra_Vector> > f = 
    dirichletWorkset.sg_f;
  Teuchos::RCP< Stokhos::VectorOrthogPoly<Epetra_CrsMatrix> > jac = 
    dirichletWorkset.sg_Jac;
  Teuchos::RCP<const Stokhos::VectorOrthogPoly<Epetra_Vector> > x = 
    dirichletWorkset.sg_x;
  const RealType j_coeff = dirichletWorkset.j_coeff;
  const std::vector<int>& nsNodes = dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

  const Epetra_Map& map = (*jac)[0].RowMap();

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
    const unsigned nodeid = nsNodes[inode];
      const int gunk = nodeid * this->neq + this->offset;
      int lunk = map.LID(gunk);
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
  Teuchos::RCP< Stokhos::ProductContainer<Epetra_Vector> > f = 
    dirichletWorkset.mp_f;
  Teuchos::RCP< const Stokhos::ProductContainer<Epetra_Vector> > x = 
    dirichletWorkset.mp_x;
  const std::vector<int>& nsNodes = dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

  int nblock = x->size();
  int gunk, lunk; // global and local indicies into unknown vector
  for (unsigned int inode = 0; inode < nsNodes.size(); inode++) {
      gunk = nsNodes[inode] * this->neq + this->offset;
      lunk = (*f)[0].Map().LID(gunk);
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

  Teuchos::RCP< Stokhos::ProductContainer<Epetra_Vector> > f = 
    dirichletWorkset.mp_f;
  Teuchos::RCP< Stokhos::ProductContainer<Epetra_CrsMatrix> > jac = 
    dirichletWorkset.mp_Jac;
  Teuchos::RCP<const Stokhos::ProductContainer<Epetra_Vector> > x = 
    dirichletWorkset.mp_x;
  const RealType j_coeff = dirichletWorkset.j_coeff;
  const std::vector<int>& nsNodes = dirichletWorkset.nodeSets->find(this->nodeSetID)->second;

  const Epetra_Map& map = (*jac)[0].RowMap();

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
    const unsigned nodeid = nsNodes[inode];
      const int gunk = nodeid * this->neq + this->offset;
      int lunk = map.LID(gunk);
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

