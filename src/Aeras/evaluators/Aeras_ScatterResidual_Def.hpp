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
  worksetSize(dl->node_scalar             ->dimension(0)),
  numNodes   (dl->node_scalar             ->dimension(1)),
  numLevels  (dl->node_scalar_level       ->dimension(2)), 
  numFields  (0), numNodeVar(0), numLevelVar(0), numTracerVar(0)
{
  const Teuchos::ArrayRCP<std::string> node_names       = p.get< Teuchos::ArrayRCP<std::string> >("Node Residual Names");
  const Teuchos::ArrayRCP<std::string> level_names      = p.get< Teuchos::ArrayRCP<std::string> >("Level Residual Names");
  const Teuchos::ArrayRCP<std::string> tracer_names     = p.get< Teuchos::ArrayRCP<std::string> >("Tracer Residual Names");

  numNodeVar   = node_names  .size();
  numLevelVar  = level_names .size();
  numTracerVar = tracer_names.size();
  numFields = numNodeVar +  numLevelVar + numTracerVar;

  val.resize(numFields);

  int eq = 0;
  for (int i = 0; i < numNodeVar; ++i, ++eq) {
    PHX::MDField<ScalarT,Cell,Node> mdf(node_names[i],dl->node_scalar);
    val[eq] = mdf;
    this->addDependentField(val[eq]);
  }   
  for (int i = 0; i < numLevelVar; ++i, ++eq) {
    PHX::MDField<ScalarT,Cell,Node> mdf(level_names[i],dl->node_scalar_level);
    val[eq] = mdf;
    this->addDependentField(val[eq]);
  }
  for (int i = 0; i < numTracerVar; ++i, ++eq) {
    PHX::MDField<ScalarT,Cell,Node> mdf(tracer_names[i],dl->node_scalar_level);
    val[eq] = mdf;
    this->addDependentField(val[eq]);
  }

  const std::string fieldName = p.get<std::string>("Scatter Field Name");
  scatter_operation = Teuchos::rcp(new PHX::Tag<ScalarT>(fieldName, dl->dummy));

  this->addEvaluatedField(*scatter_operation);

  this->setName(fieldName+PHX::TypeString<EvalT>::value);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void ScatterResidualBase<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm) 
{
  for (int eq = 0; eq < numFields; ++eq) this->utils.setFieldData(val[eq],fm);
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

  for (int cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];
    for (int node = 0; node < this->numNodes; ++node) {
      const Teuchos::ArrayRCP<int>& eqID  = nodeID[node];
      int n = 0, eq = 0;
      for (int j = eq; j < eq+this->numNodeVar; ++j, ++n) {
        (*f)[eqID[n]] += (this->val[j])(cell,node);
      }
      eq += this->numNodeVar;
      for (int level = 0; level < this->numLevels; level++) { 
        for (int j = eq; j < eq+this->numLevelVar; ++j, ++n) {
          (*f)[eqID[n]] += (this->val[j])(cell,node,level);
        }
      }
      eq += this->numLevelVar;
      for (int level = 0; level < this->numLevels; ++level) { 
        for (int j = eq; j < eq+this->numTracerVar; ++j, ++n) {
          (*f)[eqID[n]] += (this->val[j])(cell,node,level);
        }
      }
      eq += this->numTracerVar;
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
  Teuchos::RCP<Epetra_Vector>      f = workset.f;
  Teuchos::RCP<Epetra_CrsMatrix> Jac = workset.Jac;

  const bool loadResid = (f != Teuchos::null);

  for (int cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

    const int neq = nodeID[0].size();
    std::vector<int> col(neq * this->numNodes);
    for (int node=0; node<this->numNodes; node++){
      for (int eq_col=0; eq_col<neq; eq_col++) {
        col[neq * node + eq_col] =  nodeID[node][eq_col];
      }
    }


    for (int node = 0; node < this->numNodes; ++node) {
      const Teuchos::ArrayRCP<int>& eqID  = nodeID[node];
      int n = 0, eq = 0;
      for (int j = eq; j < eq+this->numNodeVar; ++j, ++n) {
        const ScalarT *valptr = &(this->val[j])(cell,node);
        if (loadResid) f->SumIntoMyValue(eqID[n], 0, valptr->val());
        if (valptr->hasFastAccess()) {
          if (workset.is_adjoint) {
            // Sum Jacobian transposed
            for (int i=0; i<col.size(); ++i)
              Jac->SumIntoMyValues(col[i], 1, &(valptr->fastAccessDx(i)), &eqID[n]);
          } else {
            // Sum Jacobian entries all at once
            Jac->SumIntoMyValues(eqID[n], col.size(), &(valptr->fastAccessDx(0)), &col[0]);
          }
        } // has fast access
      }
      eq += this->numNodeVar;
      for (int level = 0; level < this->numLevels; level++) { 
        for (int j = eq; j < eq+this->numLevelVar; ++j, ++n) {
          const ScalarT *valptr = &(this->val[j])(cell,node,level);
          if (loadResid) f->SumIntoMyValue(eqID[n], 0, valptr->val());
          if (valptr->hasFastAccess()) {
            if (workset.is_adjoint) {
              // Sum Jacobian transposed
              for (int i=0; i<col.size(); ++i)
                Jac->SumIntoMyValues(col[i], 1, &(valptr->fastAccessDx(i)), &eqID[n]);
            } else {
              // Sum Jacobian entries all at once
              Jac->SumIntoMyValues(eqID[n], col.size(), &(valptr->fastAccessDx(0)), &col[0]);
            }
          } // has fast access
        }
      }
      eq += this->numLevelVar;
      for (int level = 0; level < this->numLevels; ++level) { 
        for (int j = eq; j < eq+this->numTracerVar; ++j, ++n) {
          const ScalarT *valptr = &(this->val[j])(cell,node,level);
          if (loadResid) f->SumIntoMyValue(eqID[n], 0, valptr->val());
          if (valptr->hasFastAccess()) {
            if (workset.is_adjoint) {
              // Sum Jacobian transposed
              for (int i=0; i<col.size(); ++i)
                Jac->SumIntoMyValues(col[i], 1, &(valptr->fastAccessDx(i)), &eqID[n]);
            } else {
              // Sum Jacobian entries all at once
              Jac->SumIntoMyValues(eqID[n], col.size(), &(valptr->fastAccessDx(0)), &col[0]);
            }
          } // has fast access
        }
      }
      eq += this->numTracerVar;
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
  Teuchos::RCP<Epetra_Vector>       f = workset.f;
  Teuchos::RCP<Epetra_MultiVector> JV = workset.JV;
  Teuchos::RCP<Epetra_MultiVector> fp = workset.fp;
  ScalarT *valptr;

  const Epetra_BlockMap *row_map = NULL;
  if (f != Teuchos::null)       row_map = &( f->Map());
  else if (JV != Teuchos::null) row_map = &(JV->Map());
  else if (fp != Teuchos::null) row_map = &(fp->Map());
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                     "One of f, JV, or fp must be non-null! " << std::endl);

  for (int cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

    for (int node = 0; node < this->numNodes; ++node) {
      const Teuchos::ArrayRCP<int>& eqID  = nodeID[node];
      int n = 0, eq = 0;
      for (int j = eq; j < eq+this->numNodeVar; ++j, ++n) {
        valptr = &(this->val[j])(cell,node);
        if (f != Teuchos::null) f->SumIntoMyValue(eqID[n], 0, valptr->val());
        if (JV != Teuchos::null)
          for (int col=0; col<workset.num_cols_x; col++)
            JV->SumIntoMyValue(eqID[n], col, valptr->dx(col));
        if (fp != Teuchos::null)
          for (int col=0; col<workset.num_cols_p; col++)
            fp->SumIntoMyValue(eqID[n], col, valptr->dx(col+workset.param_offset));
      }
      eq += this->numNodeVar;
      for (int level = 0; level < this->numLevels; level++) { 
        for (int j = eq; j < eq+this->numLevelVar; ++j, ++n) {
          valptr = &(this->val[j])(cell,node,level);
          if (f != Teuchos::null) f->SumIntoMyValue(eqID[n], 0, valptr->val());
          if (JV != Teuchos::null)
            for (int col=0; col<workset.num_cols_x; col++)
              JV->SumIntoMyValue(eqID[n], col, valptr->dx(col));
          if (fp != Teuchos::null)
            for (int col=0; col<workset.num_cols_p; col++)
              fp->SumIntoMyValue(eqID[n], col, valptr->dx(col+workset.param_offset));
        }
      }
      eq += this->numLevelVar;
      for (int level = 0; level < this->numLevels; ++level) { 
        for (int j = eq; j < eq+this->numTracerVar; ++j, ++n) {
          valptr = &(this->val[j])(cell,node,level);
          if (f != Teuchos::null) f->SumIntoMyValue(eqID[n], 0, valptr->val());
          if (JV != Teuchos::null)
            for (int col=0; col<workset.num_cols_x; col++)
              JV->SumIntoMyValue(eqID[n], col, valptr->dx(col));
          if (fp != Teuchos::null)
            for (int col=0; col<workset.num_cols_p; col++)
              fp->SumIntoMyValue(eqID[n], col, valptr->dx(col+workset.param_offset));
        }
      }
      eq += this->numTracerVar;
    }
  }
}


}
