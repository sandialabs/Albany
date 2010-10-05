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


#include <vector>
#include <string>

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

namespace PHAL {

template<typename EvalT, typename Traits>
GatherSolutionBase<EvalT,Traits>::
GatherSolutionBase(const Teuchos::ParameterList& p) 
{ 
  transient = p.get<bool>("Is Transient");
  if (p.isType<bool>("Vector Field"))
    vectorField = p.get<bool>("Vector Field");
  else vectorField = false;

  std::vector<std::string> solution_names; 
  if (p.getEntryPtr("Solution Names")) {
    solution_names = *(p.get< Teuchos::RCP< std::vector<std::string> > >("Solution Names"));
  }
  Teuchos::RCP<PHX::DataLayout> dl = 
    p.get< Teuchos::RCP<PHX::DataLayout> >("Data Layout");

  if (!vectorField) {
    val.resize(solution_names.size());
    for (std::size_t eq = 0; eq < solution_names.size(); ++eq) {
      PHX::MDField<ScalarT,Cell,Node> f(solution_names[eq],dl);
      val[eq] = f;
      this->addEvaluatedField(val[eq]);
    }
    // repeat for xdot if transient
    if (transient) {
      const std::vector<std::string>& names_dot = 
        *(p.get< Teuchos::RCP< std::vector<std::string> > >("Time Dependent Solution Names"));
  
      val_dot.resize(names_dot.size());
      for (std::size_t eq = 0; eq < names_dot.size(); ++eq) {
        PHX::MDField<ScalarT,Cell,Node> f(names_dot[eq],dl);
        val_dot[eq] = f;
        this->addEvaluatedField(val_dot[eq]);
      }
    }
    numFieldsBase = val.size();
  } 
  else {
    valVec.resize(1);
    PHX::MDField<ScalarT,Cell,Node,Dim> f(solution_names[0],dl);
    valVec[0] = f;
    this->addEvaluatedField(valVec[0]);
    // repeat for xdot if transient
    if (transient) {
      const std::vector<std::string>& names_dot = 
        *(p.get< Teuchos::RCP< std::vector<std::string> > >("Time Dependent Solution Names"));
  
      valVec_dot.resize(1);
      PHX::MDField<ScalarT,Cell,Node,Dim> f(names_dot[0],dl);
      valVec_dot[0] = f;
      this->addEvaluatedField(valVec_dot[0]);
    }
    numFieldsBase = dl->dimension(2);
  }

  if (p.isType<int>("Offset of First DOF"))
    offset = p.get<int>("Offset of First DOF");
  else offset = 0;
  if (p.isType<int>("Number of DOF per Node"))
    neqBase = p.get<int>("Number of DOF per Node");
  else neqBase = numFieldsBase; // Defaults to all


  this->setName("Gather Solution");
}

// **********************************************************************
template<typename EvalT, typename Traits>
void GatherSolutionBase<EvalT,Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  if (!vectorField) {
    for (std::size_t eq = 0; eq < numFieldsBase; ++eq)
      this->utils.setFieldData(val[eq],fm);
    if (transient) {
      for (std::size_t eq = 0; eq < val_dot.size(); ++eq)
        this->utils.setFieldData(val_dot[eq],fm);
    }
    numNodes = val[0].dimension(1);
  }
  else {
    this->utils.setFieldData(valVec[0],fm);
    if (transient) this->utils.setFieldData(valVec_dot[0],fm);
    numNodes = valVec[0].dimension(1);
  }
}

// **********************************************************************

// **********************************************************************
// Specialization: Residual
// **********************************************************************

template<typename Traits>
GatherSolution<PHAL::AlbanyTraits::Residual, Traits>::
GatherSolution(const Teuchos::ParameterList& p) :
  GatherSolutionBase<PHAL::AlbanyTraits::Residual, Traits>(p),
  neq(GatherSolutionBase<PHAL::AlbanyTraits::Residual,Traits>::neqBase),
  numFields(GatherSolutionBase<PHAL::AlbanyTraits::Residual,Traits>::numFieldsBase)
{  
}

// **********************************************************************
template<typename Traits>
void GatherSolution<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{ 
  ScalarT* valptr;

  Teuchos::RCP<const Epetra_Vector> x = workset.x;
  Teuchos::RCP<const Epetra_Vector> xdot = workset.xdot;

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<int>& nodeID  = workset.elNodeID[workset.firstCell+cell];

    for (std::size_t node = 0; node < this->numNodes; ++node) {
      int firstDOF =    nodeID[node] * neq + this->offset;
      for (std::size_t eq = 0; eq < numFields; eq++) {
        if (this->vectorField) valptr = &(this->valVec[0])(cell,node,eq);
        else                   valptr = &(this->val[eq])(cell,node);
	*valptr = (*x)[firstDOF + eq];
      }
      if (this->transient) {
        for (std::size_t eq = 0; eq < numFields; eq++) {
          if (this->vectorField) valptr = &(this->valVec_dot[0])(cell,node,eq);
          else                   valptr = &(this->val_dot[eq])(cell,node);
  	  *valptr = (*xdot)[firstDOF + eq];
        }
      }
    }
  }
}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************

template<typename Traits>
GatherSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
GatherSolution(const Teuchos::ParameterList& p) :
  GatherSolutionBase<PHAL::AlbanyTraits::Jacobian, Traits>(p),
  neq(GatherSolutionBase<PHAL::AlbanyTraits::Jacobian,Traits>::neqBase),
  numFields(GatherSolutionBase<PHAL::AlbanyTraits::Jacobian,Traits>::numFieldsBase)
{ 
}

// **********************************************************************
template<typename Traits>
void GatherSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{ 
  const std::size_t num_dof = neq * this->numNodes;

  Teuchos::RCP<const Epetra_Vector> x = workset.x;
  Teuchos::RCP<const Epetra_Vector> xdot = workset.xdot;
  ScalarT* valptr;

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<int>& nodeID  = workset.elNodeID[workset.firstCell+cell];

    for (std::size_t node = 0; node < this->numNodes; ++node) {
      int firstDOF = nodeID[node] * neq + this->offset;
      for (std::size_t eq = 0; eq < numFields; eq++) {
        if (this->vectorField) valptr = &((this->valVec[0])(cell,node,eq));
        else                   valptr = &(this->val[eq])(cell,node);
	*valptr = FadType(num_dof, (*x)[firstDOF + eq]);
	valptr->fastAccessDx(neq * node + eq + this->offset) = workset.j_coeff;
      }
      if (this->transient) {
        for (std::size_t eq = 0; eq < numFields; eq++) {
          if (this->vectorField) valptr = &(this->valVec_dot[0])(cell,node,eq);
          else                   valptr = &(this->val_dot[eq])(cell,node);
  	  *valptr = FadType(num_dof, (*xdot)[firstDOF + eq]);
	  valptr->fastAccessDx(neq * node + eq + this->offset) = workset.m_coeff;
        }
      }
    }
  }
}

// **********************************************************************

// **********************************************************************
// Specialization: Tangent
// **********************************************************************

template<typename Traits>
GatherSolution<PHAL::AlbanyTraits::Tangent, Traits>::
GatherSolution(const Teuchos::ParameterList& p) :
  GatherSolutionBase<PHAL::AlbanyTraits::Tangent, Traits>(p),
  neq(GatherSolutionBase<PHAL::AlbanyTraits::Tangent,Traits>::neqBase),
  numFields(GatherSolutionBase<PHAL::AlbanyTraits::Tangent,Traits>::numFieldsBase)
{ 
}

// **********************************************************************
template<typename Traits>
void GatherSolution<PHAL::AlbanyTraits::Tangent, Traits>::
evaluateFields(typename Traits::EvalData workset)
{ 
  const std::size_t num_dof = neq * this->numNodes;

  Teuchos::RCP<const Epetra_Vector> x = workset.x;
  Teuchos::RCP<const Epetra_Vector> xdot = workset.xdot;
  Teuchos::RCP<const Epetra_MultiVector> Vx = workset.Vx;
  Teuchos::RCP<const Epetra_MultiVector> Vxdot = workset.Vxdot;
  Teuchos::RCP<const Epetra_MultiVector> Vp = workset.Vp;
  Teuchos::RCP<ParamVec> params = workset.params;
  int num_cols_tot = workset.param_offset + workset.num_cols_p;
  ScalarT* valptr;

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<int>& nodeID  = workset.elNodeID[workset.firstCell+cell];

    for (std::size_t node = 0; node < this->numNodes; ++node) {
      int firstDOF = nodeID[node] * neq + this->offset;
      for (std::size_t eq = 0; eq < numFields; eq++) {
        if (this->vectorField) valptr = &(this->valVec[0])(cell,node,eq);
        else                   valptr = &(this->val[eq])(cell,node);
	if (Vx != Teuchos::null && workset.j_coeff != 0.0) {
	  *valptr = FadType(num_cols_tot, (*x)[firstDOF + eq]);
	  for (int k=0; k<workset.num_cols_x; k++)
	    valptr->fastAccessDx(k) = 
	      workset.j_coeff*(*Vx)[k][firstDOF+eq];
	}
	else
	  *valptr = FadType((*x)[firstDOF + eq]);
      }
      if (this->transient) {
        for (std::size_t eq = 0; eq < numFields; eq++) {
          if (this->vectorField) valptr = &(this->valVec_dot[0])(cell,node,eq);
          else                   valptr = &(this->val_dot[eq])(cell,node);
	  if (Vxdot != Teuchos::null && workset.m_coeff != 0.0) {
	    *valptr = FadType(num_cols_tot, (*xdot)[firstDOF + eq]);
	    for (int k=0; k<workset.num_cols_x; k++)
	      valptr->fastAccessDx(k) = 
		workset.m_coeff*(*Vxdot)[k][firstDOF+eq];
	  }
	  else
	    *valptr = FadType((*xdot)[firstDOF + eq]);
        }
      }
    }
  }
}

// **********************************************************************

// **********************************************************************
// Specialization: Stochastic Galerkin Residual
// **********************************************************************

template<typename Traits>
GatherSolution<PHAL::AlbanyTraits::SGResidual, Traits>::
GatherSolution(const Teuchos::ParameterList& p) :
  GatherSolutionBase<PHAL::AlbanyTraits::SGResidual, Traits>(p),
  neq(GatherSolutionBase<PHAL::AlbanyTraits::SGResidual,Traits>::neqBase),
  numFields(GatherSolutionBase<PHAL::AlbanyTraits::SGResidual,Traits>::numFieldsBase)
{  
}

// **********************************************************************
template<typename Traits>
void GatherSolution<PHAL::AlbanyTraits::SGResidual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{ 
  Teuchos::RCP<Stokhos::OrthogPolyExpansion<int,RealType> > sg_expansion =
    workset.sg_expansion;
  Teuchos::RCP<const Stokhos::VectorOrthogPoly<Epetra_Vector> > x = 
    workset.sg_x;
  Teuchos::RCP<const Stokhos::VectorOrthogPoly<Epetra_Vector> > xdot = 
    workset.sg_xdot;
  ScalarT* valptr;

  int nblock = x->size();
  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<int>& nodeID  = workset.elNodeID[workset.firstCell+cell];

    for (std::size_t node = 0; node < this->numNodes; ++node) {
      int firstDOF = nodeID[node] * neq + this->offset;
      for (std::size_t eq = 0; eq < numFields; eq++) {
        if (this->vectorField) valptr = &(this->valVec[0])(cell,node,eq);
        else                   valptr = &(this->val[eq])(cell,node);
	valptr->reset(sg_expansion);
	valptr->copyForWrite();
	for (int block=0; block<nblock; block++)
	  valptr->fastAccessCoeff(block) = 
	    (*x)[block][firstDOF + eq];
      }
      if (this->transient) {
        for (std::size_t eq = 0; eq < numFields; eq++) {
          if (this->vectorField) valptr = &(this->valVec_dot[0])(cell,node,eq);
          else                   valptr = &(this->val_dot[eq])(cell,node);
	  valptr->reset(sg_expansion);
	  valptr->copyForWrite();
	  for (int block=0; block<nblock; block++)
	    valptr->fastAccessCoeff(block) = 
	      (*xdot)[block][firstDOF + eq];
        }
      }
    }
  }
}

// **********************************************************************

// **********************************************************************
// Specialization: Stochastic Galerkin Jacobian
// **********************************************************************

template<typename Traits>
GatherSolution<PHAL::AlbanyTraits::SGJacobian, Traits>::
GatherSolution(const Teuchos::ParameterList& p) :
  GatherSolutionBase<PHAL::AlbanyTraits::SGJacobian, Traits>(p),
  neq(GatherSolutionBase<PHAL::AlbanyTraits::SGJacobian,Traits>::neqBase),
  numFields(GatherSolutionBase<PHAL::AlbanyTraits::SGJacobian,Traits>::numFieldsBase)
{ 
}

// **********************************************************************
template<typename Traits>
void GatherSolution<PHAL::AlbanyTraits::SGJacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{ 
  const std::size_t num_dof = neq * this->numNodes;

  Teuchos::RCP<Stokhos::OrthogPolyExpansion<int,RealType> > sg_expansion =
    workset.sg_expansion;
  Teuchos::RCP<const Stokhos::VectorOrthogPoly<Epetra_Vector> > x = 
    workset.sg_x;
  Teuchos::RCP<const Stokhos::VectorOrthogPoly<Epetra_Vector> > xdot = 
    workset.sg_xdot;
  ScalarT* valptr;
  
  int nblock = x->size();
  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<int>& nodeID  = workset.elNodeID[workset.firstCell+cell];

    for (std::size_t node = 0; node < this->numNodes; ++node) {
      int firstDOF = nodeID[node] * neq + this->offset;

      for (std::size_t eq = 0; eq < numFields; eq++) {
        if (this->vectorField) valptr = &(this->valVec[0])(cell,node,eq);
        else                   valptr = &(this->val[eq])(cell,node);
	*valptr = SGFadType(num_dof, 0.0);
	valptr->fastAccessDx(neq * node + eq + this->offset) = workset.j_coeff;
	valptr->val().reset(sg_expansion);
	valptr->val().copyForWrite();
	for (int block=0; block<nblock; block++)
	  valptr->val().fastAccessCoeff(block) = (*x)[block][firstDOF + eq];
      }
      if (this->transient) {
        for (std::size_t eq = 0; eq < numFields; eq++) {
          if (this->vectorField) valptr = &(this->valVec_dot[0])(cell,node,eq);
          else                   valptr = &(this->val_dot[eq])(cell,node);
  	  *valptr = SGFadType(num_dof, 0.0);
	  valptr->fastAccessDx(neq * node + eq + this->offset) = workset.m_coeff;
	  valptr->val().reset(sg_expansion);
	  valptr->val().copyForWrite();
	  for (int block=0; block<nblock; block++)
	    valptr->val().fastAccessCoeff(block) = (*xdot)[block][firstDOF + eq];
        }
      }
    }
  }
}

}

