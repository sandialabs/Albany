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
  if (p.isType<bool>("Vector Field"))
    vectorField = p.get<bool>("Vector Field");
  else
    vectorField = false;

  if (p.isType<bool>("Disable Transient"))
    enableTransient = !p.get<bool>("Disable Transient");
  else enableTransient = true;

  Teuchos::ArrayRCP<std::string> solution_names;
  if (p.getEntryPtr("Solution Names")) {
    solution_names = p.get< Teuchos::ArrayRCP<std::string> >("Solution Names");
  }
  Teuchos::RCP<PHX::DataLayout> dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("Data Layout");

  // scalar
  if (!vectorField ) {
    val.resize(solution_names.size());
    for (std::size_t eq = 0; eq < solution_names.size(); ++eq) {
      PHX::MDField<ScalarT,Cell,Node> f(solution_names[eq],dl);
      val[eq] = f;
      this->addEvaluatedField(val[eq]);
    }
    // repeat for xdot if transient is enabled
    if (enableTransient) {
      const Teuchos::ArrayRCP<std::string>& names_dot =
        p.get< Teuchos::ArrayRCP<std::string> >("Time Dependent Solution Names");

      val_dot.resize(names_dot.size());
      for (std::size_t eq = 0; eq < names_dot.size(); ++eq) {
        PHX::MDField<ScalarT,Cell,Node> f(names_dot[eq],dl);
        val_dot[eq] = f;
        this->addEvaluatedField(val_dot[eq]);
      }
    }
    numFieldsBase = val.size();
  }
  // vector
  else {
    valVec.resize(1);
    PHX::MDField<ScalarT,Cell,Node,VecDim> f(solution_names[0],dl);
    valVec[0] = f;
    this->addEvaluatedField(valVec[0]);
    // repeat for xdot if transient is enabled
    if (enableTransient) {
      const Teuchos::ArrayRCP<std::string>& names_dot =
        p.get< Teuchos::ArrayRCP<std::string> >("Time Dependent Solution Names");

      valVec_dot.resize(1);
      PHX::MDField<ScalarT,Cell,Node,VecDim> f(names_dot[0],dl);
      valVec_dot[0] = f;
      this->addEvaluatedField(valVec_dot[0]);
    }
    numFieldsBase = dl->dimension(2);
  }

  if (p.isType<int>("Offset of First DOF"))
    offset = p.get<int>("Offset of First DOF");
  else offset = 0;

  this->setName("Gather Solution"+PHX::TypeString<EvalT>::value);
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
    if (enableTransient) {
      for (std::size_t eq = 0; eq < val_dot.size(); ++eq)
        this->utils.setFieldData(val_dot[eq],fm);
    }
    numNodes = val[0].dimension(1);
  }
  else {
    this->utils.setFieldData(valVec[0],fm);
    if (enableTransient) this->utils.setFieldData(valVec_dot[0],fm);
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
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

    for (std::size_t node = 0; node < this->numNodes; ++node) {
      for (std::size_t eq = 0; eq < numFields; eq++) {
        if (this->vectorField) valptr = &(this->valVec[0])(cell,node,eq);
        else                   valptr = &(this->val[eq])(cell,node);
        *valptr = (*x)[nodeID[node][this->offset + eq]];
      }
      if (workset.transientTerms && this->enableTransient) {
        for (std::size_t eq = 0; eq < numFields; eq++) {
          if (this->vectorField) valptr = &(this->valVec_dot[0])(cell,node,eq);
          else                   valptr = &(this->val_dot[eq])(cell,node);
          *valptr = (*xdot)[nodeID[node][this->offset + eq]];
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
  numFields(GatherSolutionBase<PHAL::AlbanyTraits::Jacobian,Traits>::numFieldsBase)
{
}

// **********************************************************************
template<typename Traits>
void GatherSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::RCP<const Epetra_Vector> x = workset.x;
  Teuchos::RCP<const Epetra_Vector> xdot = workset.xdot;
  ScalarT* valptr;

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

    for (std::size_t node = 0; node < this->numNodes; ++node) {
      int neq = nodeID[node].size();
      std::size_t num_dof = neq * this->numNodes;
      for (std::size_t eq = 0; eq < numFields; eq++) {
        if (this->vectorField) valptr = &((this->valVec[0])(cell,node,eq));
        else                   valptr = &(this->val[eq])(cell,node);
        *valptr = FadType(num_dof, (*x)[nodeID[node][this->offset + eq]]);
        valptr->setUpdateValue(!workset.ignore_residual);
        valptr->fastAccessDx(neq * node + eq + this->offset) = workset.j_coeff;
      }
      if (workset.transientTerms && this->enableTransient) {
        for (std::size_t eq = 0; eq < numFields; eq++) {
          if (this->vectorField) valptr = &(this->valVec_dot[0])(cell,node,eq);
          else                   valptr = &(this->val_dot[eq])(cell,node);
          *valptr = FadType(num_dof, (*xdot)[nodeID[node][this->offset + eq]]);
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
  numFields(GatherSolutionBase<PHAL::AlbanyTraits::Tangent,Traits>::numFieldsBase)
{
}

// **********************************************************************
template<typename Traits>
void GatherSolution<PHAL::AlbanyTraits::Tangent, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

  Teuchos::RCP<const Epetra_Vector> x = workset.x;
  Teuchos::RCP<const Epetra_Vector> xdot = workset.xdot;
  Teuchos::RCP<const Epetra_MultiVector> Vx = workset.Vx;
  Teuchos::RCP<const Epetra_MultiVector> Vxdot = workset.Vxdot;
  Teuchos::RCP<const Epetra_MultiVector> Vp = workset.Vp;
  Teuchos::RCP<ParamVec> params = workset.params;
  int num_cols_tot = workset.param_offset + workset.num_cols_p;
  ScalarT* valptr;

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

    for (std::size_t node = 0; node < this->numNodes; ++node) {
      for (std::size_t eq = 0; eq < numFields; eq++) {
        if (this->vectorField) valptr = &(this->valVec[0])(cell,node,eq);
        else                   valptr = &(this->val[eq])(cell,node);
        if (Vx != Teuchos::null && workset.j_coeff != 0.0) {
          *valptr = FadType(num_cols_tot, (*x)[nodeID[node][this->offset + eq]]);
          for (int k=0; k<workset.num_cols_x; k++)
            valptr->fastAccessDx(k) =
              workset.j_coeff*(*Vx)[k][nodeID[node][this->offset + eq]];
        }
        else
          *valptr = FadType((*x)[nodeID[node][this->offset + eq]]);
      }
      if (workset.transientTerms && this->enableTransient) {
        for (std::size_t eq = 0; eq < numFields; eq++) {
          if (this->vectorField) valptr = &(this->valVec_dot[0])(cell,node,eq);
          else                   valptr = &(this->val_dot[eq])(cell,node);
          if (Vxdot != Teuchos::null && workset.m_coeff != 0.0) {
            *valptr = FadType(num_cols_tot, (*xdot)[nodeID[node][this->offset + eq]]);
            for (int k=0; k<workset.num_cols_x; k++)
              valptr->fastAccessDx(k) =
                workset.m_coeff*(*Vxdot)[k][nodeID[node][this->offset + eq]];
          }
          else
            *valptr = FadType((*xdot)[nodeID[node][this->offset + eq]]);
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
  Teuchos::RCP<const Stokhos::EpetraVectorOrthogPoly > x =
    workset.sg_x;
  Teuchos::RCP<const Stokhos::EpetraVectorOrthogPoly > xdot =
    workset.sg_xdot;
  ScalarT* valptr;

  int nblock = x->size();
  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

    for (std::size_t node = 0; node < this->numNodes; ++node) {
      for (std::size_t eq = 0; eq < numFields; eq++) {
        if (this->vectorField) valptr = &(this->valVec[0])(cell,node,eq);
        else                   valptr = &(this->val[eq])(cell,node);
        valptr->reset(sg_expansion);
        valptr->copyForWrite();
        for (int block=0; block<nblock; block++)
          valptr->fastAccessCoeff(block) =
            (*x)[block][nodeID[node][this->offset + eq]];
      }
      if (workset.transientTerms && this->enableTransient) {
        for (std::size_t eq = 0; eq < numFields; eq++) {
          if (this->vectorField) valptr = &(this->valVec_dot[0])(cell,node,eq);
          else                   valptr = &(this->val_dot[eq])(cell,node);
          valptr->reset(sg_expansion);
          valptr->copyForWrite();
          for (int block=0; block<nblock; block++)
            valptr->fastAccessCoeff(block) =
              (*xdot)[block][nodeID[node][this->offset + eq]];
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
  numFields(GatherSolutionBase<PHAL::AlbanyTraits::SGJacobian,Traits>::numFieldsBase)
{
}

// **********************************************************************
template<typename Traits>
void GatherSolution<PHAL::AlbanyTraits::SGJacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::RCP<Stokhos::OrthogPolyExpansion<int,RealType> > sg_expansion =
    workset.sg_expansion;
  Teuchos::RCP<const Stokhos::EpetraVectorOrthogPoly > x =
    workset.sg_x;
  Teuchos::RCP<const Stokhos::EpetraVectorOrthogPoly > xdot =
    workset.sg_xdot;
  ScalarT* valptr;

  int nblock = x->size();
  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

    for (std::size_t node = 0; node < this->numNodes; ++node) {
      int neq = nodeID[node].size();
      std::size_t num_dof = neq * this->numNodes;

      for (std::size_t eq = 0; eq < numFields; eq++) {
        if (this->vectorField) valptr = &(this->valVec[0])(cell,node,eq);
        else                   valptr = &(this->val[eq])(cell,node);
        *valptr = SGFadType(num_dof, 0.0);
        valptr->setUpdateValue(!workset.ignore_residual);
        valptr->fastAccessDx(neq * node + eq + this->offset) = workset.j_coeff;
        valptr->val().reset(sg_expansion);
        valptr->val().copyForWrite();
        for (int block=0; block<nblock; block++)
          valptr->val().fastAccessCoeff(block) = (*x)[block][nodeID[node][this->offset + eq]];
      }
      if (workset.transientTerms && this->enableTransient) {
        for (std::size_t eq = 0; eq < numFields; eq++) {
          if (this->vectorField) valptr = &(this->valVec_dot[0])(cell,node,eq);
          else                   valptr = &(this->val_dot[eq])(cell,node);
          *valptr = SGFadType(num_dof, 0.0);
          valptr->fastAccessDx(neq * node + eq + this->offset) = workset.m_coeff;
          valptr->val().reset(sg_expansion);
          valptr->val().copyForWrite();
          for (int block=0; block<nblock; block++)
            valptr->val().fastAccessCoeff(block) = (*xdot)[block][nodeID[node][this->offset + eq]];
        }
      }
    }
  }

}

// **********************************************************************

// **********************************************************************
// Specialization: Stochastic Galerkin Tangent
// **********************************************************************

template<typename Traits>
GatherSolution<PHAL::AlbanyTraits::SGTangent, Traits>::
GatherSolution(const Teuchos::ParameterList& p) :
  GatherSolutionBase<PHAL::AlbanyTraits::SGTangent, Traits>(p),
  numFields(GatherSolutionBase<PHAL::AlbanyTraits::SGTangent,Traits>::numFieldsBase)
{
}

// **********************************************************************
template<typename Traits>
void GatherSolution<PHAL::AlbanyTraits::SGTangent, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

  Teuchos::RCP<Stokhos::OrthogPolyExpansion<int,RealType> > sg_expansion =
    workset.sg_expansion;
  Teuchos::RCP<const Stokhos::EpetraVectorOrthogPoly > x =
    workset.sg_x;
  Teuchos::RCP<const Stokhos::EpetraVectorOrthogPoly > xdot =
    workset.sg_xdot;
  Teuchos::RCP<const Epetra_MultiVector> Vx = workset.Vx;
  Teuchos::RCP<const Epetra_MultiVector> Vxdot = workset.Vxdot;
  Teuchos::RCP<const Epetra_MultiVector> Vp = workset.Vp;
  Teuchos::RCP<ParamVec> params = workset.params;
  int num_cols_tot = workset.param_offset + workset.num_cols_p;
  ScalarT* valptr;

  int nblock = x->size();
  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

    for (std::size_t node = 0; node < this->numNodes; ++node) {
      for (std::size_t eq = 0; eq < numFields; eq++) {
        if (this->vectorField) valptr = &(this->valVec[0])(cell,node,eq);
        else                   valptr = &(this->val[eq])(cell,node);
        if (Vx != Teuchos::null && workset.j_coeff != 0.0) {
          *valptr = SGFadType(num_cols_tot, 0.0);
          for (int k=0; k<workset.num_cols_x; k++)
            valptr->fastAccessDx(k) =
              workset.j_coeff*(*Vx)[k][nodeID[node][this->offset + eq]];
        }
        else
          *valptr = SGFadType(0.0);
        valptr->val().reset(sg_expansion);
        valptr->val().copyForWrite();
        for (int block=0; block<nblock; block++)
          valptr->val().fastAccessCoeff(block) = (*x)[block][nodeID[node][this->offset + eq]];
      }
      if (workset.transientTerms && this->enableTransient) {
        for (std::size_t eq = 0; eq < numFields; eq++) {
          if (this->vectorField) valptr = &(this->valVec_dot[0])(cell,node,eq);
          else                   valptr = &(this->val_dot[eq])(cell,node);
          if (Vxdot != Teuchos::null && workset.m_coeff != 0.0) {
            *valptr = SGFadType(num_cols_tot, 0.0);
            for (int k=0; k<workset.num_cols_x; k++)
              valptr->fastAccessDx(k) =
                workset.m_coeff*(*Vxdot)[k][nodeID[node][this->offset + eq]];
          }
          else
            *valptr = SGFadType(0.0);
          valptr->val().reset(sg_expansion);
          valptr->val().copyForWrite();
          for (int block=0; block<nblock; block++)
            valptr->val().fastAccessCoeff(block) = (*xdot)[block][nodeID[node][this->offset + eq]];
        }
      }
    }
  }

}

// **********************************************************************

// **********************************************************************
// Specialization: Multi-point Residual
// **********************************************************************

template<typename Traits>
GatherSolution<PHAL::AlbanyTraits::MPResidual, Traits>::
GatherSolution(const Teuchos::ParameterList& p) :
  GatherSolutionBase<PHAL::AlbanyTraits::MPResidual, Traits>(p),
  numFields(GatherSolutionBase<PHAL::AlbanyTraits::MPResidual,Traits>::numFieldsBase)
{
}

// **********************************************************************
template<typename Traits>
void GatherSolution<PHAL::AlbanyTraits::MPResidual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::RCP<const Stokhos::ProductEpetraVector > x =
    workset.mp_x;
  Teuchos::RCP<const Stokhos::ProductEpetraVector > xdot =
    workset.mp_xdot;
  ScalarT* valptr;

  int nblock = x->size();
  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

    for (std::size_t node = 0; node < this->numNodes; ++node) {
      for (std::size_t eq = 0; eq < numFields; eq++) {
        if (this->vectorField) valptr = &(this->valVec[0])(cell,node,eq);
        else                   valptr = &(this->val[eq])(cell,node);
        valptr->reset(nblock);
        valptr->copyForWrite();
        for (int block=0; block<nblock; block++)
          valptr->fastAccessCoeff(block) =
            (*x)[block][nodeID[node][this->offset + eq]];
      }
      if (workset.transientTerms && this->enableTransient) {
        for (std::size_t eq = 0; eq < numFields; eq++) {
          if (this->vectorField) valptr = &(this->valVec_dot[0])(cell,node,eq);
          else                   valptr = &(this->val_dot[eq])(cell,node);
          valptr->reset(nblock);
          valptr->copyForWrite();
          for (int block=0; block<nblock; block++)
            valptr->fastAccessCoeff(block) =
              (*xdot)[block][nodeID[node][this->offset + eq]];
        }
      }
    }
  }

}

// **********************************************************************

// **********************************************************************
// Specialization: Mulit-point Jacobian
// **********************************************************************

template<typename Traits>
GatherSolution<PHAL::AlbanyTraits::MPJacobian, Traits>::
GatherSolution(const Teuchos::ParameterList& p) :
  GatherSolutionBase<PHAL::AlbanyTraits::MPJacobian, Traits>(p),
  numFields(GatherSolutionBase<PHAL::AlbanyTraits::MPJacobian,Traits>::numFieldsBase)
{
}

// **********************************************************************
template<typename Traits>
void GatherSolution<PHAL::AlbanyTraits::MPJacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::RCP<const Stokhos::ProductEpetraVector > x =
    workset.mp_x;
  Teuchos::RCP<const Stokhos::ProductEpetraVector > xdot =
    workset.mp_xdot;
  ScalarT* valptr;

  int nblock = x->size();
  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

    for (std::size_t node = 0; node < this->numNodes; ++node) {
      int neq = nodeID[node].size();
      std::size_t num_dof = neq * this->numNodes;

      for (std::size_t eq = 0; eq < numFields; eq++) {
        if (this->vectorField) valptr = &(this->valVec[0])(cell,node,eq);
        else                   valptr = &(this->val[eq])(cell,node);
        *valptr = MPFadType(num_dof, 0.0);
        valptr->setUpdateValue(!workset.ignore_residual);
        valptr->fastAccessDx(neq * node + eq + this->offset) = workset.j_coeff;
        valptr->val().reset(nblock);
        valptr->val().copyForWrite();
        for (int block=0; block<nblock; block++)
          valptr->val().fastAccessCoeff(block) = (*x)[block][nodeID[node][this->offset + eq]];
      }
      if (workset.transientTerms && this->enableTransient) {
        for (std::size_t eq = 0; eq < numFields; eq++) {
          if (this->vectorField) valptr = &(this->valVec_dot[0])(cell,node,eq);
          else                   valptr = &(this->val_dot[eq])(cell,node);
          *valptr = MPFadType(num_dof, 0.0);
          valptr->fastAccessDx(neq * node + eq + this->offset) = workset.m_coeff;
          valptr->val().reset(nblock);
          valptr->val().copyForWrite();
          for (int block=0; block<nblock; block++)
            valptr->val().fastAccessCoeff(block) = (*xdot)[block][nodeID[node][this->offset + eq]];
        }
      }
    }
  }

}

// **********************************************************************

// **********************************************************************
// Specialization: Multi-point Galerkin Tangent
// **********************************************************************

template<typename Traits>
GatherSolution<PHAL::AlbanyTraits::MPTangent, Traits>::
GatherSolution(const Teuchos::ParameterList& p) :
  GatherSolutionBase<PHAL::AlbanyTraits::MPTangent, Traits>(p),
  numFields(GatherSolutionBase<PHAL::AlbanyTraits::MPTangent,Traits>::numFieldsBase)
{
}

// **********************************************************************
template<typename Traits>
void GatherSolution<PHAL::AlbanyTraits::MPTangent, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

  Teuchos::RCP<const Stokhos::ProductEpetraVector > x =
    workset.mp_x;
  Teuchos::RCP<const Stokhos::ProductEpetraVector > xdot =
    workset.mp_xdot;
  Teuchos::RCP<const Epetra_MultiVector> Vx = workset.Vx;
  Teuchos::RCP<const Epetra_MultiVector> Vxdot = workset.Vxdot;
  Teuchos::RCP<const Epetra_MultiVector> Vp = workset.Vp;
  Teuchos::RCP<ParamVec> params = workset.params;
  int num_cols_tot = workset.param_offset + workset.num_cols_p;
  ScalarT* valptr;

  int nblock = x->size();
  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

    for (std::size_t node = 0; node < this->numNodes; ++node) {
      for (std::size_t eq = 0; eq < numFields; eq++) {
        if (this->vectorField) valptr = &(this->valVec[0])(cell,node,eq);
        else                   valptr = &(this->val[eq])(cell,node);
        if (Vx != Teuchos::null && workset.j_coeff != 0.0) {
          *valptr = MPFadType(num_cols_tot, 0.0);
          for (int k=0; k<workset.num_cols_x; k++)
            valptr->fastAccessDx(k) =
              workset.j_coeff*(*Vx)[k][nodeID[node][this->offset + eq]];
        }
        else
          *valptr = MPFadType(0.0);
        valptr->val().reset(nblock);
        valptr->val().copyForWrite();
        for (int block=0; block<nblock; block++)
          valptr->val().fastAccessCoeff(block) = (*x)[block][nodeID[node][this->offset + eq]];
      }
      if (workset.transientTerms && this->enableTransient) {
        for (std::size_t eq = 0; eq < numFields; eq++) {
          if (this->vectorField) valptr = &(this->valVec_dot[0])(cell,node,eq);
          else                   valptr = &(this->val_dot[eq])(cell,node);
          if (Vxdot != Teuchos::null && workset.m_coeff != 0.0) {
            *valptr = MPFadType(num_cols_tot, 0.0);
            for (int k=0; k<workset.num_cols_x; k++)
              valptr->fastAccessDx(k) =
                workset.m_coeff*(*Vxdot)[k][nodeID[node][this->offset + eq]];
          }
          else
            *valptr = MPFadType(0.0);
          valptr->val().reset(nblock);
          valptr->val().copyForWrite();
          for (int block=0; block<nblock; block++)
            valptr->val().fastAccessCoeff(block) = (*xdot)[block][nodeID[node][this->offset + eq]];
        }
      }
    }
  }

}

}
