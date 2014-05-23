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
ScatterResidualBase<EvalT, Traits>::
ScatterResidualBase(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Aeras::Layouts>& dl) :
  numFields(p.get< Teuchos::ArrayRCP<std::string> >("Residual Names").size()),
  numLevels(p.get<int>("Number of Vertical Levels"))
{
  const std::string fieldName = p.get<std::string>("Scatter Field Name");
  
  scatter_operation = Teuchos::rcp(new PHX::Tag<ScalarT>(fieldName, dl->dummy));

  const Teuchos::ArrayRCP<std::string>& names =
    p.get< Teuchos::ArrayRCP<std::string> >("Residual Names");

  val.resize(numFields);

  for (std::size_t eq = 0; eq < numFields; ++eq) {
    PHX::MDField<ScalarT,Cell,Node> mdf(names[eq],dl->node_scalar_level);
    val[eq] = mdf;
    this->addDependentField(val[eq]);
  }   

  this->addEvaluatedField(*scatter_operation);

  this->setName(fieldName+PHX::TypeString<EvalT>::value);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void ScatterResidualBase<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm) 
{
    for (std::size_t eq = 0; eq < numFields; ++eq) this->utils.setFieldData(val[eq],fm);
    numNodes = val[0].dimension(1);
}

// **********************************************************************
// Specialization: Residual
// **********************************************************************
template<typename Traits>
ScatterResidual<PHAL::AlbanyTraits::Residual,Traits>::
ScatterResidual(const Teuchos::ParameterList& p,
                       const Teuchos::RCP<Aeras::Layouts>& dl)
  : ScatterResidualBase<PHAL::AlbanyTraits::Residual,Traits>(p,dl)
{}

template<typename Traits>
void ScatterResidual<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::RCP<Epetra_Vector> f = workset.f;

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];
    for (std::size_t node = 0; node < this->numNodes; ++node) {
      for (std::size_t eq = 0; eq < this->numFields; eq++) {
        for (std::size_t level = 0; level < this->numLevels; level++) { 
          const int n=eq+this->numFields*level;
          (*f)[nodeID[node][n]] += (this->val[eq])(cell,node,level);
        }
      }
    }
  }
}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************

template<typename Traits>
ScatterResidual<PHAL::AlbanyTraits::Jacobian, Traits>::
ScatterResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Aeras::Layouts>& dl)
  : ScatterResidualBase<PHAL::AlbanyTraits::Jacobian,Traits>(p,dl)
{ }

// **********************************************************************
template<typename Traits>
void ScatterResidual<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::RCP<Epetra_Vector> f = workset.f;
  Teuchos::RCP<Epetra_CrsMatrix> Jac = workset.Jac;
  ScalarT *valptr;

  bool loadResid = (f != Teuchos::null);
  int row;
  std::vector<int> col;

  int neq = workset.wsElNodeEqID[0][0].size();
  int nunk = neq*this->numNodes;
  col.resize(nunk);


  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
  const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];
    // Local Unks: Loop over nodes in element, Loop over equations per node

    for (unsigned int node_col=0, i=0; node_col<this->numNodes; node_col++){
      for (unsigned int eq_col=0; eq_col<neq; eq_col++) {
        col[neq * node_col + eq_col] =  nodeID[node_col][eq_col];
      }
    }

    for (std::size_t node = 0; node < this->numNodes; ++node) {

      for (std::size_t eq = 0; eq < this->numFields; eq++) {
        for (std::size_t level = 0; level < this->numLevels; level++) { 
          const int n=eq+this->numFields*level;
          valptr = &(this->val[eq])(cell,node,level);

          row = nodeID[node][n];
          if (loadResid) f->SumIntoMyValue(row, 0, valptr->val());

          if (valptr->hasFastAccess()) {
            if (workset.is_adjoint) {
              // Sum Jacobian transposed
              for (unsigned int lunk=0; lunk<nunk; lunk++)
                Jac->SumIntoMyValues(col[lunk], 1, &(valptr->fastAccessDx(lunk)), &row);
            }
            else {
              // Sum Jacobian entries all at once
              Jac->SumIntoMyValues(row, nunk, &(valptr->fastAccessDx(0)), &col[0]);
            }
          } // has fast access
        }
      }
    }
  }
}

// **********************************************************************
// Specialization: Tangent
// **********************************************************************

template<typename Traits>
ScatterResidual<PHAL::AlbanyTraits::Tangent, Traits>::
ScatterResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Aeras::Layouts>& dl)
  : ScatterResidualBase<PHAL::AlbanyTraits::Tangent,Traits>(p,dl)
{ }

// **********************************************************************
template<typename Traits>
void ScatterResidual<PHAL::AlbanyTraits::Tangent, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::RCP<Epetra_Vector> f = workset.f;
  Teuchos::RCP<Epetra_MultiVector> JV = workset.JV;
  Teuchos::RCP<Epetra_MultiVector> fp = workset.fp;
  ScalarT *valptr;

  const Epetra_BlockMap *row_map = NULL;
  if (f != Teuchos::null)
    row_map = &(f->Map());
  else if (JV != Teuchos::null)
    row_map = &(JV->Map());
  else if (fp != Teuchos::null)
    row_map = &(fp->Map());
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                     "One of f, JV, or fp must be non-null! " << std::endl);

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

    for (std::size_t node = 0; node < this->numNodes; ++node) {
      for (std::size_t eq = 0; eq < this->numFields; eq++) {
        for (std::size_t level = 0; level < this->numLevels; level++) { 
          const int n=eq+this->numFields*level;
          valptr = &(this->val[eq])(cell,node,level);

          const int row = nodeID[node][n];

          if (f != Teuchos::null) f->SumIntoMyValue(row, 0, valptr->val());

          if (JV != Teuchos::null)
            for (int col=0; col<workset.num_cols_x; col++)
              JV->SumIntoMyValue(row, col, valptr->dx(col));

          if (fp != Teuchos::null)
            for (int col=0; col<workset.num_cols_p; col++)
              fp->SumIntoMyValue(row, col, valptr->dx(col+workset.param_offset));
        }
      }
    }
  }
}


}
