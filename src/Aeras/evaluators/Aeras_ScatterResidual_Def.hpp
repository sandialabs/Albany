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
  numDims    (dl->node_qp_gradient        ->dimension(3)),
  numLevels  (dl->node_scalar_level       ->dimension(2)), 
  numFields  (0), numNodeVar(0), numVectorLevelVar(0),  numScalarLevelVar(0), numTracerVar(0)
{
  const Teuchos::ArrayRCP<std::string> node_names         = p.get< Teuchos::ArrayRCP<std::string> >("Node Residual Names");
  const Teuchos::ArrayRCP<std::string> vector_level_names = p.get< Teuchos::ArrayRCP<std::string> >("Vector Level Residual Names");
  const Teuchos::ArrayRCP<std::string> scalar_level_names = p.get< Teuchos::ArrayRCP<std::string> >("Scalar Level Residual Names");
  const Teuchos::ArrayRCP<std::string> tracer_names       = p.get< Teuchos::ArrayRCP<std::string> >("Tracer Residual Names");

  numNodeVar   = node_names  .size();
  numVectorLevelVar  = vector_level_names .size();
  numScalarLevelVar  = scalar_level_names .size();
  numTracerVar = tracer_names.size();
  numFields = numNodeVar +  numVectorLevelVar + numScalarLevelVar +  numTracerVar;

  val.resize(numFields);

  int eq = 0;
  for (int i = 0; i < numNodeVar; ++i, ++eq) {
    PHX::MDField<ScalarT,Cell,Node> mdf(node_names[i],dl->node_scalar);
    val[eq] = mdf;
    this->addDependentField(val[eq]);
  }   
  for (int i = 0; i < numVectorLevelVar; ++i, ++eq) {
    PHX::MDField<ScalarT,Cell,Node> mdf(vector_level_names[i],dl->node_vector_level); val[eq] = mdf;
    this->addDependentField(val[eq]);
  }
  for (int i = 0; i < numScalarLevelVar; ++i, ++eq) {
    PHX::MDField<ScalarT,Cell,Node> mdf(scalar_level_names[i],dl->node_scalar_level); val[eq] = mdf;
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

  this->setName(fieldName);
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
  Teuchos::RCP<Tpetra_Vector> fT = workset.fT;
  //get non-const view of fT 
  Teuchos::ArrayRCP<ST> fT_nonconstView = fT->get1dViewNonConst();

  for (int cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];
    for (int node = 0; node < this->numNodes; ++node) {
      const Teuchos::ArrayRCP<int>& eqID  = nodeID[node];
      int n = 0, eq = 0;
      for (int j = eq; j < eq+this->numNodeVar; ++j, ++n) {
        fT_nonconstView[eqID[n]] += (this->val[j])(cell,node);
      }
      eq += this->numNodeVar;
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "no impl");
//Irina TOFIX
/*
      for (int level = 0; level < this->numLevels; level++) { 
        for (int j = eq; j < eq+this->numVectorLevelVar; ++j) {
          for (int dim = 0; dim < this->numDims; ++dim, ++n) {
            fT_nonconstView[eqID[n]] += (this->val[j])(cell,node,level,dim);
          }
        }
        for (int j = eq+this->numVectorLevelVar; 
                 j < eq+this->numVectorLevelVar + this->numScalarLevelVar; ++j, ++n) {
          fT_nonconstView[eqID[n]] += (this->val[j])(cell,node,level);
        }
      }
      eq += this->numVectorLevelVar + this->numScalarLevelVar;
      for (int level = 0; level < this->numLevels; ++level) { 
        for (int j = eq; j < eq+this->numTracerVar; ++j, ++n) {
          fT_nonconstView[eqID[n]] += (this->val[j])(cell,node,level);
        }
      }
      eq += this->numTracerVar;
*/    }
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
  Teuchos::RCP<Tpetra_Vector>      fT = workset.fT;
  Teuchos::RCP<Tpetra_CrsMatrix> JacT = workset.JacT;

  const bool loadResid = (fT != Teuchos::null);
  LO rowT; 
  Teuchos::Array<LO> colT; 

  for (int cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

    const int neq = nodeID[0].size();
    colT.resize(neq * this->numNodes); 
    
    for (int node=0; node<this->numNodes; node++){
      for (int eq_col=0; eq_col<neq; eq_col++) {
        colT[neq * node + eq_col] =  nodeID[node][eq_col];
      }
    }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "no impl");
//Irina TOFIX
/*
 for (int node = 0; node < this->numNodes; ++node) {
      const Teuchos::ArrayRCP<int>& eqID  = nodeID[node];
      int n = 0, eq = 0;
      for (int j = eq; j < eq+this->numNodeVar; ++j, ++n) {
        const ScalarT *valptr = &(this->val[j])(cell,node);
        rowT = eqID[n];
        if (loadResid) fT->sumIntoLocalValue(rowT, valptr->val());
        if (valptr->hasFastAccess()) {
          if (workset.is_adjoint) {
          // Sum Jacobian transposed
          for (unsigned int i=0; i<colT.size(); ++i) {
              //Jac->SumIntoMyValues(colT[i], 1, &(valptr->fastAccessDx(i)), &eqID[n]);
              JacT->sumIntoLocalValues(colT[i], Teuchos::arrayView(&rowT,1), Teuchos::arrayView(&(valptr->fastAccessDx(i)),1));
            }
          } else {
            // Sum Jacobian entries all at once
            //             //Jac->SumIntoMyValues(eqID[n], colT.size(), &(valptr->fastAccessDx(0)), &colT[0]);
            JacT->sumIntoLocalValues(rowT, colT, Teuchos::arrayView(&(valptr->fastAccessDx(0)), colT.size()));
          }
        } // has fast access
      }
      eq += this->numNodeVar;
      for (int level = 0; level < this->numLevels; level++) {
        for (int j = eq; j < eq+this->numVectorLevelVar; ++j) {
          for (int dim = 0; dim < this->numDims; ++dim, ++n) {
            const ScalarT *valptr = &(this->val[j])(cell,node,level,dim);
            rowT = eqID[n];
            if (loadResid) fT->sumIntoLocalValue(rowT, valptr->val());
            if (valptr->hasFastAccess()) {
              if (workset.is_adjoint) {
                // Sum Jacobian transposed
                for (int i=0; i<colT.size(); ++i)
                  JacT->sumIntoLocalValues(colT[i], Teuchos::arrayView(&rowT,1), Teuchos::arrayView(&(valptr->fastAccessDx(i)),1));
              } else {
                // Sum Jacobian entries all at once
                JacT->sumIntoLocalValues(rowT, colT, Teuchos::arrayView(&(valptr->fastAccessDx(0)), colT.size()));
              }
            }
          }
        }
        for (int j = eq+this->numVectorLevelVar;
                 j < eq+this->numVectorLevelVar+this->numScalarLevelVar; ++j, ++n) {
          const ScalarT *valptr = &(this->val[j])(cell,node,level);
          if (loadResid) fT->sumIntoLocalValue(eqID[n], valptr->val());
          if (valptr->hasFastAccess()) {
            if (workset.is_adjoint) {
              // Sum Jacobian transposed
              for (unsigned int i=0; i<colT.size(); ++i) {
                //Jac->SumIntoMyValues(colT[i], 1, &(valptr->fastAccessDx(i)), &eqID[n]);
                JacT->sumIntoLocalValues(colT[i], Teuchos::arrayView(&rowT,1), Teuchos::arrayView(&(valptr->fastAccessDx(i)),1));
              }
            } else {
              // Sum Jacobian entries all at once
              //Jac->SumIntoMyValues(eqID[n], colT.size(), &(valptr->fastAccessDx(0)), &colT[0]);
              JacT->sumIntoLocalValues(rowT, colT, Teuchos::arrayView(&(valptr->fastAccessDx(0)), colT.size()));
            }
          } // has fast access
        }
      }
      eq += this->numVectorLevelVar+this->numScalarLevelVar;
      for (int level = 0; level < this->numLevels; ++level) {
        for (int j = eq; j < eq+this->numTracerVar; ++j, ++n) {
          const ScalarT *valptr = &(this->val[j])(cell,node,level);
          if (loadResid) fT->sumIntoLocalValue(eqID[n], valptr->val());
          if (valptr->hasFastAccess()) {
            if (workset.is_adjoint) {
              // Sum Jacobian transposed
              for (unsigned int i=0; i<colT.size(); ++i) {
                //Jac->SumIntoMyValues(colT[i], 1, &(valptr->fastAccessDx(i)), &eqID[n]);
                JacT->sumIntoLocalValues(colT[i], Teuchos::arrayView(&rowT,1), Teuchos::arrayView(&(valptr->fastAccessDx(i)),1));
              }
            } else {
              // Sum Jacobian entries all at once
              //Jac->SumIntoMyValues(eqID[n], colT.size(), &(valptr->fastAccessDx(0)), &colT[0]);
              JacT->sumIntoLocalValues(rowT, colT, Teuchos::arrayView(&(valptr->fastAccessDx(0)), colT.size()));
            }
          } // has fast access
        }
      }
      eq += this->numTracerVar;
    }

 */

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
  Teuchos::RCP<Tpetra_Vector>       fT = workset.fT;
  Teuchos::RCP<Tpetra_MultiVector> JVT = workset.JVT;
  Teuchos::RCP<Tpetra_MultiVector> fpT = workset.fpT;
  ScalarT *valptr;

  int rowT; 

  //IK, 6/27/14: I think you don't actually need row_map here the way this function is written right now...
  //const Epetra_BlockMap *row_map = NULL;
  //if (f != Teuchos::null)       row_map = &( f->Map());
  //else if (JV != Teuchos::null) row_map = &(JV->Map());
  //else if (fp != Teuchos::null) row_map = &(fp->Map());
  //else
  if ((fT == Teuchos::null) & (JVT == Teuchos::null) & (fpT == Teuchos::null)) {  
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                     "One of f, JV, or fp must be non-null! " << std::endl);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "no impl");
  //Irina TOFIX
  /*
  for (int cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

    for (int node = 0; node < this->numNodes; ++node) {
      const Teuchos::ArrayRCP<int>& eqID  = nodeID[node];
      int n = 0, eq = 0;
      for (int j = eq; j < eq+this->numNodeVar; ++j, ++n) {
        valptr = &(this->val[j])(cell,node);
        rowT = eqID[n];
        if (fT != Teuchos::null) fT->sumIntoLocalValue(rowT, valptr->val());
        if (JVT != Teuchos::null)
          for (int col=0; col<workset.num_cols_x; col++)
            JVT->sumIntoLocalValue(rowT, col, valptr->dx(col));
        if (fpT != Teuchos::null)
          for (int col=0; col<workset.num_cols_p; col++)
            fpT->sumIntoLocalValue(rowT, col, valptr->dx(col+workset.param_offset));
      }
      eq += this->numNodeVar;
      for (int level = 0; level < this->numLevels; level++) {
        for (int j = eq; j < eq+this->numVectorLevelVar; ++j) {
          for (int dim = 0; dim < this->numDims; ++dim, ++n) {
            valptr = &(this->val[j])(cell,node,level,dim);
            rowT = eqID[n];
            if (fT != Teuchos::null) fT->sumIntoLocalValue(rowT, valptr->val());
            if (JVT != Teuchos::null)
              for (int col=0; col<workset.num_cols_x; col++)
                JVT->sumIntoLocalValue(rowT, col, valptr->dx(col));
            if (fpT != Teuchos::null)
              for (int col=0; col<workset.num_cols_p; col++)
                fpT->sumIntoLocalValue(rowT, col, valptr->dx(col+workset.param_offset));
          }
        }
        for (int j = eq+this->numVectorLevelVar;
                 j < eq+this->numVectorLevelVar+this->numScalarLevelVar; ++j,++n) {
          valptr = &(this->val[j])(cell,node,level);
          if (fT != Teuchos::null) fT->sumIntoLocalValue(rowT, valptr->val());
          if (JVT != Teuchos::null)
            for (int col=0; col<workset.num_cols_x; col++)
              JVT->sumIntoLocalValue(rowT, col, valptr->dx(col));
          if (fpT != Teuchos::null)
            for (int col=0; col<workset.num_cols_p; col++)
              fpT->sumIntoLocalValue(rowT, col, valptr->dx(col+workset.param_offset));
        }
      }
     eq += this->numVectorLevelVar+this->numScalarLevelVar;
      for (int level = 0; level < this->numLevels; ++level) {
        for (int j = eq; j < eq+this->numTracerVar; ++j, ++n) {
          valptr = &(this->val[j])(cell,node,level);
          if (fT != Teuchos::null) fT->sumIntoLocalValue(rowT, valptr->val());
          if (JVT != Teuchos::null)
            for (int col=0; col<workset.num_cols_x; col++)
              JVT->sumIntoLocalValue(rowT, col, valptr->dx(col));
          if (fpT != Teuchos::null)
            for (int col=0; col<workset.num_cols_p; col++)
              fpT->sumIntoLocalValue(rowT, col, valptr->dx(col+workset.param_offset));
        }
      }
      eq += this->numTracerVar;
    }
  }

 
 */
}

#ifdef ALBANY_SG_MP
// **********************************************************************
// Specialization: Multi-point Residual
// **********************************************************************
template<typename Traits>
ScatterResidual<PHAL::AlbanyTraits::MPResidual,Traits>::
ScatterResidual(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Aeras::Layouts>& dl)
  : ScatterResidualBase<PHAL::AlbanyTraits::MPResidual,Traits>(p,dl)
{}

template<typename Traits>
void ScatterResidual<PHAL::AlbanyTraits::MPResidual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::RCP<Stokhos::ProductEpetraVector> f = workset.mp_f;
  //get non-const view of fT 
//  Teuchos::ArrayRCP<ST> fT_nonconstView = fT->get1dViewNonConst();

  int nblock = f->size();
  for (int cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];
    for (int node = 0; node < this->numNodes; ++node) {
      const Teuchos::ArrayRCP<int>& eqID  = nodeID[node];
      int n = 0, eq = 0;
      for (int j = eq; j < eq+this->numNodeVar; ++j, ++n) {
//        fT_nonconstView[eqID[n]] += (this->val[j])(cell,node);
        ScalarT *valptr = &(this->val[j])(cell,node);
        for (int block=0; block<nblock; block++)
          (*f)[block][nodeID[node][n]] += valptr->coeff(block);
      }
      eq += this->numNodeVar;
      for (int level = 0; level < this->numLevels; level++) { 
        for (int j = eq; j < eq+this->numVectorLevelVar; ++j) {
          for (int dim = 0; dim < this->numDims; ++dim, ++n) {
//            fT_nonconstView[eqID[n]] += (this->val[j])(cell,node,level,dim);
            ScalarT *valptr = &(this->val[j])(cell,node,level,dim);
            for (int block=0; block<nblock; block++)
              (*f)[block][nodeID[node][n]] += valptr->coeff(block);
          }
        }
        for (int j = eq+this->numVectorLevelVar; 
                 j < eq+this->numVectorLevelVar + this->numScalarLevelVar; ++j, ++n) {
//          fT_nonconstView[eqID[n]] += (this->val[j])(cell,node,level);
          ScalarT *valptr = &(this->val[j])(cell,node,level);
          for (int block=0; block<nblock; block++)
            (*f)[block][nodeID[node][n]] += valptr->coeff(block);
        }
      }
      eq += this->numVectorLevelVar + this->numScalarLevelVar;
      for (int level = 0; level < this->numLevels; ++level) { 
        for (int j = eq; j < eq+this->numTracerVar; ++j, ++n) {
//          fT_nonconstView[eqID[n]] += (this->val[j])(cell,node,level);
          ScalarT *valptr = &(this->val[j])(cell,node,level);
          for (int block=0; block<nblock; block++)
            (*f)[block][nodeID[node][n]] += valptr->coeff(block);
        }
      }
      eq += this->numTracerVar;
    }
  }
}

// **********************************************************************
// Specialization: Multi-point Jacobian
// **********************************************************************

template<typename Traits>
ScatterResidual<PHAL::AlbanyTraits::MPJacobian, Traits>::
ScatterResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Aeras::Layouts>& dl)
  : ScatterResidualBase<PHAL::AlbanyTraits::MPJacobian,Traits>(p,dl)
{ }

// **********************************************************************
template<typename Traits>
void ScatterResidual<PHAL::AlbanyTraits::MPJacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::RCP<Stokhos::ProductEpetraVector>      f = workset.mp_f;
  Teuchos::RCP<Stokhos::ProductContainer<Epetra_CrsMatrix> > Jac = workset.mp_Jac;

  const bool loadResid = (f != Teuchos::null);
  int nblock = 0;
  if (loadResid)
    nblock = f->size();
  int nblock_jac = Jac->size();

  int row, lcol, col;
  double c;
//  LO rowT; 
//  Teuchos::Array<LO> colT; 

  for (int cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

    const int neq = nodeID[0].size();
//    colT.resize(neq * this->numNodes); 
    
//    for (int node=0; node<this->numNodes; node++){
//      for (int eq_col=0; eq_col<neq; eq_col++) {
//        colT[neq * node + eq_col] =  nodeID[node][eq_col];
//      }
//    }


    for (int node = 0; node < this->numNodes; ++node) {
      const Teuchos::ArrayRCP<int>& eqID  = nodeID[node];
      int n = 0, eq = 0;
      for (int j = eq; j < eq+this->numNodeVar; ++j, ++n) {
        const ScalarT *valptr = &(this->val[j])(cell,node);
        row = eqID[n]; 
        if (loadResid) {
//          fT->sumIntoLocalValue(rowT, valptr->val());
          for (int block=0; block<nblock; block++)
            (*f)[block].SumIntoMyValue(row, 0, valptr->val().coeff(block));
        }

//        if (valptr->hasFastAccess()) {
//          if (workset.is_adjoint) {
            // Sum Jacobian transposed
//            for (unsigned int i=0; i<colT.size(); ++i) {
              //Jac->SumIntoMyValues(colT[i], 1, &(valptr->fastAccessDx(i)), &eqID[n]);
//              JacT->sumIntoLocalValues(colT[i], Teuchos::arrayView(&rowT,1), Teuchos::arrayView(&(valptr->fastAccessDx(i)),1));
//            }
//          } else {
            // Sum Jacobian entries all at once
            //Jac->SumIntoMyValues(eqID[n], colT.size(), &(valptr->fastAccessDx(0)), &colT[0]);
//            JacT->sumIntoLocalValues(rowT, colT, Teuchos::arrayView(&(valptr->fastAccessDx(0)), colT.size()));
//          }
//        } // has fast access

        if (valptr->hasFastAccess()) {
          for (int node_col=0; node_col<this->numNodes; node_col++){
            for (int eq_col=0; eq_col<neq; eq_col++) {
              lcol = neq * node_col + eq_col;
              col = nodeID[node_col][eq_col];
              for (int block=0; block<nblock_jac; block++) {
                c = valptr->fastAccessDx(lcol).coeff(block);
                (*Jac)[block].SumIntoMyValues(row, 1, &c, &col);
              }
            }
          }
        }
      }
      eq += this->numNodeVar;
      for (int level = 0; level < this->numLevels; level++) { 
        for (int j = eq; j < eq+this->numVectorLevelVar; ++j) {
          for (int dim = 0; dim < this->numDims; ++dim, ++n) {
            const ScalarT *valptr = &(this->val[j])(cell,node,level,dim);
            row = eqID[n]; 
            if (loadResid) {
              for (int block=0; block<nblock; block++)
                (*f)[block].SumIntoMyValue(row, 0, valptr->val().coeff(block));
            }
            if (valptr->hasFastAccess()) {
              for (int node_col=0; node_col<this->numNodes; node_col++){
                for (int eq_col=0; eq_col<neq; eq_col++) {
                  lcol = neq * node_col + eq_col;
                  col = nodeID[node_col][eq_col];
                  for (int block=0; block<nblock_jac; block++) {
                    c = valptr->fastAccessDx(lcol).coeff(block);
                    (*Jac)[block].SumIntoMyValues(row, 1, &c, &col);
                  }
                }
              }
            }
          } 
        }
        for (int j = eq+this->numVectorLevelVar; 
                 j < eq+this->numVectorLevelVar+this->numScalarLevelVar; ++j, ++n) {
          const ScalarT *valptr = &(this->val[j])(cell,node,level);
          if (loadResid) {
            for (int block=0; block<nblock; block++)
              (*f)[block].SumIntoMyValue(eqID[n], 0, valptr->val().coeff(block));
          }
          if (valptr->hasFastAccess()) {
            for (int node_col=0; node_col<this->numNodes; node_col++){
              for (int eq_col=0; eq_col<neq; eq_col++) {
                lcol = neq * node_col + eq_col;
                col = nodeID[node_col][eq_col];
                for (int block=0; block<nblock_jac; block++) {
                  c = valptr->fastAccessDx(lcol).coeff(block);
                  (*Jac)[block].SumIntoMyValues(row, 1, &c, &col);
                }
              }
            }
          }
        }
      }
      eq += this->numVectorLevelVar+this->numScalarLevelVar;
      for (int level = 0; level < this->numLevels; ++level) { 
        for (int j = eq; j < eq+this->numTracerVar; ++j, ++n) {
          const ScalarT *valptr = &(this->val[j])(cell,node,level);
          if (loadResid) {
            for (int block=0; block<nblock; block++)
              (*f)[block].SumIntoMyValue(eqID[n], 0, valptr->val().coeff(block));
          }
          if (valptr->hasFastAccess()) {
            for (int node_col=0; node_col<this->numNodes; node_col++){
              for (int eq_col=0; eq_col<neq; eq_col++) {
                lcol = neq * node_col + eq_col;
                col = nodeID[node_col][eq_col];
                for (int block=0; block<nblock_jac; block++) {
                  c = valptr->fastAccessDx(lcol).coeff(block);
                  (*Jac)[block].SumIntoMyValues(row, 1, &c, &col);
                }
              }
            }
          }
        }
      }
      eq += this->numTracerVar;
    }
  }
}
#endif //ALBANY_SG_MP

}
