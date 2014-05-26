//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <vector>
#include <string>

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Aeras_Layouts.hpp"

namespace Aeras {


template<typename EvalT, typename Traits>
GatherSolutionBase<EvalT,Traits>::
GatherSolutionBase(const Teuchos::ParameterList& p,
                   const Teuchos::RCP<Aeras::Layouts>& dl): 
numNodes(0), worksetSize(0)
{
  numLevels = p.get< int >("Number of Vertical Levels");

  Teuchos::ArrayRCP<std::string> solution_names;
  if (p.getEntryPtr("Solution Names")) {
    solution_names = p.get< Teuchos::ArrayRCP<std::string> >("Solution Names");
  }

  val.resize(solution_names.size());
  for (int eq = 0; eq < solution_names.size(); ++eq) {
    PHX::MDField<ScalarT,Cell,Node> f(solution_names[eq],dl->node_scalar_level);
    val[eq] = f;
    this->addEvaluatedField(val[eq]);
  }   
  const Teuchos::ArrayRCP<std::string>& names_dot =
    p.get< Teuchos::ArrayRCP<std::string> >("Time Dependent Solution Names");

  val_dot.resize(names_dot.size());
  for (int eq = 0; eq < names_dot.size(); ++eq) {
    PHX::MDField<ScalarT,Cell,Node> f(names_dot[eq],dl->node_scalar_level);
    val_dot[eq] = f;
    this->addEvaluatedField(val_dot[eq]);
  }   

  numFields = val.size();
  this->setName("Aeras_GatherSolution"+PHX::TypeString<EvalT>::value);
}


template<typename EvalT, typename Traits>
void GatherSolutionBase<EvalT,Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm) 
{
  for (int eq = 0; eq < val.size();     ++eq) this->utils.setFieldData(val[eq],fm);
  for (int eq = 0; eq < val_dot.size(); ++eq) this->utils.setFieldData(val_dot[eq],fm);

  typename std::vector< typename PHX::template MDField<MeshScalarT,Cell,Vertex,Dim>::size_type > dims;
  val[0].dimensions(dims); //get dimensions
  worksetSize = dims[0];
  numNodes    = dims[1];
}




// **********************************************************************
// **********************************************************************
// Specialization: Residual
// **********************************************************************

template<typename Traits>
GatherSolution<PHAL::AlbanyTraits::Residual, Traits>::
GatherSolution(const Teuchos::ParameterList& p,
               const Teuchos::RCP<Aeras::Layouts>& dl) :
  GatherSolutionBase<PHAL::AlbanyTraits::Residual, Traits>(p,dl)
{}

template<typename Traits>
void GatherSolution<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{ 
  Teuchos::RCP<const Epetra_Vector> x = workset.x;
  Teuchos::RCP<const Epetra_Vector> xdot = workset.xdot;

  for (int cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];
 
    for (int node = 0; node < this->numNodes; ++node) {
    const Teuchos::ArrayRCP<int>& eqID  = nodeID[node];
      for (int eq = 0; eq < this->numFields; eq++) 
        for (int level = 0; level < this->numLevels; level++) { 
          int n=eq+this->numFields*level;
          (this->val[eq])(cell,node,level) = (*x)[eqID[n]];
          (this->val_dot[eq])(cell,node,level) = (*xdot)[eqID[n]];
        }
     }
  }
}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************

template<typename Traits>
GatherSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
GatherSolution(const Teuchos::ParameterList& p,
               const Teuchos::RCP<Aeras::Layouts>& dl) :
  GatherSolutionBase<PHAL::AlbanyTraits::Jacobian, Traits>(p,dl)
{}

// **********************************************************************
template<typename Traits>
void GatherSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  const Teuchos::RCP<const Epetra_Vector>    x = workset.x;
  const Teuchos::RCP<const Epetra_Vector> xdot = workset.xdot;

  for (int cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];
    const int neq = nodeID[0].size();
    const int num_dof = neq * this->numNodes;

    for (int node = 0; node < this->numNodes; ++node) {
      const Teuchos::ArrayRCP<int>& eqID  = nodeID[node];
      const int firstunk = neq * node;
      for (int eq = 0; eq < this->numFields; eq++) {
        for (int level = 0; level < this->numLevels; level++) { 
          const int n = eq + this->numFields*level;
          ScalarT* valptr = &(this->val[eq])(cell,node,level);
          *valptr = FadType(num_dof, (*x)[eqID[n]]);
          valptr->setUpdateValue(!workset.ignore_residual);
          valptr->fastAccessDx(firstunk + n) = workset.j_coeff;

        }
      }
      if (workset.transientTerms) {
        for (int eq = 0; eq < this->numFields; eq++) {
          for (int level = 0; level < this->numLevels; level++) { 
            const int n= eq + this->numFields*level;
            ScalarT* valptr = &(this->val_dot[eq])(cell,node,level);
            *valptr = FadType(num_dof, (*xdot)[eqID[n]]);
            valptr->fastAccessDx(firstunk + n) = workset.m_coeff;
          }
        }

      }
    }
  }
}


// **********************************************************************
// Specialization: Tangent
// **********************************************************************

template<typename Traits>
GatherSolution<PHAL::AlbanyTraits::Tangent, Traits>::
GatherSolution(const Teuchos::ParameterList& p,
               const Teuchos::RCP<Aeras::Layouts>& dl) :
  GatherSolutionBase<PHAL::AlbanyTraits::Tangent, Traits>(p,dl)
{ }

// **********************************************************************
template<typename Traits>
void GatherSolution<PHAL::AlbanyTraits::Tangent, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

  Teuchos::RCP<const Epetra_Vector> x = workset.x;
  Teuchos::RCP<const Epetra_Vector> xdot = workset.xdot;
  Teuchos::RCP<const Epetra_MultiVector> Vx = workset.Vx;
  Teuchos::RCP<const Epetra_MultiVector> Vxdot = workset.Vxdot;

  Teuchos::RCP<ParamVec> params = workset.params;
  int num_cols_tot = workset.param_offset + workset.num_cols_p;
  ScalarT* valptr;

  for (int cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

    for (int node = 0; node < this->numNodes; ++node) {
      const Teuchos::ArrayRCP<int>& eqID  = nodeID[node];
      for (int eq = 0; eq < this->numFields; eq++) {
        for (int level = 0; level < this->numLevels; level++) { 
          const int n=eq+this->numFields*level;
          valptr = &(this->val[eq])(cell,node,level);
          if (Vx != Teuchos::null && workset.j_coeff != 0.0) {
            *valptr = TanFadType(num_cols_tot, (*x)[eqID[n]]);
            for (int k=0; k<workset.num_cols_x; k++)
              valptr->fastAccessDx(k) = workset.j_coeff*(*Vx)[k][eqID[n]];
          }
          else
            *valptr = TanFadType((*x)[eqID[n]]);
        }
      }
      if (workset.transientTerms) {
        for (int eq = 0; eq < this->numFields; eq++) {
          for (int level = 0; level < this->numLevels; level++) { 
            const int n=eq+this->numFields*level;
            valptr = &(this->val_dot[eq])(cell,node,level);
            if (Vxdot != Teuchos::null && workset.m_coeff != 0.0) {
              *valptr = TanFadType(num_cols_tot, (*xdot)[eqID[n]]);
              for (int k=0; k<workset.num_cols_x; k++)
                valptr->fastAccessDx(k) =
                  workset.m_coeff*(*Vxdot)[k][eqID[n]];
            }
            else
              *valptr = TanFadType((*xdot)[eqID[n]]);
          }
        }
      }
    }
  }
}

}
