//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <vector>
#include <string>

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Aeras_Layouts.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ThyraUtils.hpp"

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
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  val_kokkos.resize(numFields);
#endif

  int eq = 0;
  for (int i = 0; i < numNodeVar; ++i, ++eq) {
    PHX::MDField<const ScalarT> mdf(node_names[i],dl->node_scalar);
    val[eq] = mdf;
    this->addDependentField(val[eq]);
  }   
  for (int i = 0; i < numVectorLevelVar; ++i, ++eq) {
    PHX::MDField<const ScalarT> mdf(vector_level_names[i],dl->node_vector_level); val[eq] = mdf;
    this->addDependentField(val[eq]);
  }
  for (int i = 0; i < numScalarLevelVar; ++i, ++eq) {
    PHX::MDField<const ScalarT> mdf(scalar_level_names[i],dl->node_scalar_level); val[eq] = mdf;
    this->addDependentField(val[eq]);
  }
  for (int i = 0; i < numTracerVar; ++i, ++eq) {
    PHX::MDField<const ScalarT> mdf(tracer_names[i],dl->node_scalar_level);
    val[eq] = mdf;
    this->addDependentField(val[eq]);
  }

  const std::string fieldName = p.get<std::string>("Scatter Field Name");
  scatter_operation = Teuchos::rcp(new PHX::Tag<ScalarT>(fieldName, dl->dummy));

  this->addEvaluatedField(*scatter_operation);

  this->setName("Aeras_ScatterResidual"+PHX::typeAsString<EvalT>());
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
  : ScatterResidualBase<PHAL::AlbanyTraits::Residual,Traits>(p,dl) {}

// **********************************************************************
// Kokkos kernels
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
template<typename Traits>
KOKKOS_INLINE_FUNCTION
void ScatterResidual<PHAL::AlbanyTraits::Residual, Traits>::
operator() (const Aeras_ScatterRes_Tag&, const int& cell) const{
  for (int node = 0; node < this->numNodes; ++node) {
    int n = 0, eq = 0;
    for (int j = eq; j < eq+this->numNodeVar; ++j, ++n) {
       Kokkos::atomic_fetch_add(&f_kokkos[nodeID(cell,node,n)], d_val_kokkos[j](cell,node));
    }
    eq += this->numNodeVar;
    for (int level = 0; level < this->numLevels; level++) {
      for (int j = eq; j < eq+this->numVectorLevelVar; ++j) {
        for (int dim = 0; dim < this->numDims; ++dim, ++n) {
            Kokkos::atomic_fetch_add(&f_kokkos[nodeID(cell,node,n)], d_val_kokkos[j](cell,node,level,dim));
        }
      }
      for (int j = eq+this->numVectorLevelVar;
               j < eq+this->numVectorLevelVar + this->numScalarLevelVar; ++j, ++n) {
              Kokkos::atomic_fetch_add(&f_kokkos[nodeID(cell,node,n)], d_val_kokkos[j](cell,node,level));
      }
    }
    eq += this->numVectorLevelVar + this->numScalarLevelVar;
    for (int level = 0; level < this->numLevels; ++level) {
      for (int j = eq; j < eq+this->numTracerVar; ++j, ++n) {
          Kokkos::atomic_fetch_add(&f_kokkos[nodeID(cell,node,n)], d_val_kokkos[j](cell,node,level));
      }
    }
    eq += this->numTracerVar;
  }
}

#endif

// **********************************************************************
template<typename Traits>
void ScatterResidual<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  auto nodeID = workset.wsElNodeEqID;

  //get non-const view of residual 
  Teuchos::ArrayRCP<ST> f_nonconstView = Albany::getNonconstLocalData(workset.f);

  for (int cell=0; cell < workset.numCells; ++cell ) {
    for (int node = 0; node < this->numNodes; ++node) {
      int n = 0, eq = 0;
      for (int j = eq; j < eq+this->numNodeVar; ++j, ++n) {
        f_nonconstView[nodeID(cell,node,n)] += (this->val[j])(cell,node);
      }
      eq += this->numNodeVar;
      for (int level = 0; level < this->numLevels; level++) { 
        for (int j = eq; j < eq+this->numVectorLevelVar; ++j) {
          for (int dim = 0; dim < this->numDims; ++dim, ++n) {
            f_nonconstView[nodeID(cell,node,n)] += (this->val[j])(cell,node,level,dim);
          }
        }
        for (int j = eq+this->numVectorLevelVar; 
                 j < eq+this->numVectorLevelVar + this->numScalarLevelVar; ++j, ++n) {
          f_nonconstView[nodeID(cell,node,n)] += (this->val[j])(cell,node,level);
        }
      }
      eq += this->numVectorLevelVar + this->numScalarLevelVar;
      for (int level = 0; level < this->numLevels; ++level) { 
        for (int j = eq; j < eq+this->numTracerVar; ++j, ++n) {
          f_nonconstView[nodeID(cell,node,n)] += (this->val[j])(cell,node,level);
        }
      }
      eq += this->numTracerVar;
    }
  }

#else
  // Get map for local data structures
  nodeID = workset.wsElNodeEqID;

  // Get Tpetra vector view from a specific device
  f_kokkos = Albany::getNonconstDeviceData(workset.f);

  // Get MDField views from std::vector
  for (int i = 0; i < this->numFields; i++) {
    val_kokkos[i] = this->val[i].get_view();
  }
  d_val_kokkos = val_kokkos.template view<ExecutionSpace>();

  Kokkos::parallel_for(Aeras_ScatterRes_Policy(0,workset.numCells),*this);
  cudaCheckError();
#endif
}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************
template<typename Traits>
ScatterResidual<PHAL::AlbanyTraits::Jacobian, Traits>::
ScatterResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Aeras::Layouts>& dl)
  : ScatterResidualBase<PHAL::AlbanyTraits::Jacobian,Traits>(p,dl) {}

// *********************************************************************
// Kokkos kernels
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
template<typename Traits>
KOKKOS_INLINE_FUNCTION
void ScatterResidual<PHAL::AlbanyTraits::Jacobian, Traits>::
operator() (const Aeras_ScatterRes_Tag&, const int& cell) const{
  for (int node = 0; node < this->numNodes; ++node) {
    int n = 0, eq = 0;
    for (int j = eq; j < eq+this->numNodeVar; ++j, ++n) {
       Kokkos::atomic_fetch_add(&f_kokkos[nodeID(cell,node,n)], (d_val_kokkos[j](cell,node)).val());
    }
    eq += this->numNodeVar;
    for (int level = 0; level < this->numLevels; level++) {
      for (int j = eq; j < eq+this->numVectorLevelVar; ++j) {
        for (int dim = 0; dim < this->numDims; ++dim, ++n) {
            Kokkos::atomic_fetch_add(&f_kokkos[nodeID(cell,node,n)], (d_val_kokkos[j](cell,node,level,dim)).val());
        }
      }
      for (int j = eq+this->numVectorLevelVar;
               j < eq+this->numVectorLevelVar + this->numScalarLevelVar; ++j, ++n) {
              Kokkos::atomic_fetch_add(&f_kokkos[nodeID(cell,node,n)], (d_val_kokkos[j](cell,node,level)).val());
      }
    }
    eq += this->numVectorLevelVar + this->numScalarLevelVar;
    for (int level = 0; level < this->numLevels; ++level) {
      for (int j = eq; j < eq+this->numTracerVar; ++j, ++n) {
          Kokkos::atomic_fetch_add(&f_kokkos[nodeID(cell,node,n)], (d_val_kokkos[j](cell,node,level)).val());
      }
    }
    eq += this->numTracerVar;
  }
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void ScatterResidual<PHAL::AlbanyTraits::Jacobian, Traits>::
operator() (const Aeras_ScatterJac_Adjoint_Tag&, const int& cell) const{
  LO col, row;
  ST val;
  for (int node = 0; node < this->numNodes; ++node) {
    int n = 0, eq = 0;
    for (int j = eq; j < eq+this->numNodeVar; ++j, ++n) {
      auto valptr = d_val_kokkos[j](cell,node);
      row = nodeID(cell,node,n);
      for (int node_col=0, i=0; node_col<this->numNodes; node_col++){
        for (int eq_col=0; eq_col<neq; eq_col++, i++) {
          col = nodeID(cell,node_col,eq_col);
          val = valptr.fastAccessDx(i);
          if (val != 0) 
            Jac_kokkos.sumIntoValues(col, &row,  1, &val, false, true);
        }
      }
    }

    eq += this->numNodeVar;
    for (int level = 0; level < this->numLevels; level++) {
      for (int j = eq; j < eq+this->numVectorLevelVar; ++j) {
        for (int dim = 0; dim < this->numDims; ++dim, ++n) {
          auto valptr = d_val_kokkos[j](cell,node,level,dim);
          row = nodeID(cell,node,n);
          for (int node_col=0, i=0; node_col<this->numNodes; node_col++){
            for (int eq_col=0; eq_col<neq; eq_col++, i++) {
              col = nodeID(cell,node_col,eq_col);
              val = valptr.fastAccessDx(i);
              if (val != 0) 
                Jac_kokkos.sumIntoValues(col, &row,  1, &val, false, true);
            }
          }
        }

        for (int j = eq+this->numVectorLevelVar;
             j < eq+this->numVectorLevelVar+this->numScalarLevelVar; ++j, ++n) {
          auto valptr = d_val_kokkos[j](cell,node,level);
          row = nodeID(cell,node,n);
          for (int node_col=0, i=0; node_col<this->numNodes; node_col++){
            for (int eq_col=0; eq_col<neq; eq_col++, i++) {
              col = nodeID(cell,node_col,eq_col);
              val = valptr.fastAccessDx(i);
              if (val != 0) 
                Jac_kokkos.sumIntoValues(col, &row,  1, &val, false, true);
            }
          }
        }
      }
    }

    eq += this->numVectorLevelVar+this->numScalarLevelVar;
    for (int level = 0; level < this->numLevels; ++level) {
      for (int j = eq; j < eq+this->numTracerVar; ++j, ++n) {
        auto valptr = d_val_kokkos[j](cell,node,level);
        row = nodeID(cell,node,n);
        for (int node_col=0, i=0; node_col<this->numNodes; node_col++){
          for (int eq_col=0; eq_col<neq; eq_col++, i++) {
            col = nodeID(cell,node_col,eq_col);
            val = valptr.fastAccessDx(i);
            if (val != 0) 
              Jac_kokkos.sumIntoValues(col, &row,  1, &val, false, true);
          }
        }
      }
    }
    eq += this->numTracerVar;
  }
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void ScatterResidual<PHAL::AlbanyTraits::Jacobian, Traits>::
operator() (const Aeras_ScatterJac_Tag&, const int& cell) const{
  LO row, col;
  ST val;
  for (int node = 0; node < this->numNodes; ++node) {
    int n = 0, eq = 0;
    for (int j = eq; j < eq+this->numNodeVar; ++j, ++n) {
      auto valptr = d_val_kokkos[j](cell,node);
      row = nodeID(cell,node,n);
      for (int node_col=0, i=0; node_col<this->numNodes; node_col++){
        for (int eq_col=0; eq_col<neq; eq_col++, i++) {
          col = nodeID(cell,node_col,eq_col);
          val = valptr.fastAccessDx(i);
          if (val != 0) 
            Jac_kokkos.sumIntoValues(row, &col,  1, &val, false, true);
        }
      }
    }

    eq += this->numNodeVar;
    for (int level = 0; level < this->numLevels; level++) {
      for (int j = eq; j < eq+this->numVectorLevelVar; ++j) {
        for (int dim = 0; dim < this->numDims; ++dim, ++n) {
          auto valptr = d_val_kokkos[j](cell,node,level,dim);
          row = nodeID(cell,node,n);
          for (int node_col=0, i=0; node_col<this->numNodes; node_col++){
            for (int eq_col=0; eq_col<neq; eq_col++, i++) {
              col = nodeID(cell,node_col,eq_col);
              val = valptr.fastAccessDx(i);
              if (val != 0) 
                Jac_kokkos.sumIntoValues(row, &col,  1, &val, false, true);
            }
          }
        }

        for (int j = eq+this->numVectorLevelVar;
             j < eq+this->numVectorLevelVar+this->numScalarLevelVar; ++j, ++n) {
          auto valptr = d_val_kokkos[j](cell,node,level);
          row = nodeID(cell,node,n);
          for (int node_col=0, i=0; node_col<this->numNodes; node_col++){
            for (int eq_col=0; eq_col<neq; eq_col++, i++) {
              col = nodeID(cell,node_col,eq_col);
              val = valptr.fastAccessDx(i);
              if (val != 0) 
                Jac_kokkos.sumIntoValues(row, &col,  1, &val, false, true);
            }
          }
        }
      }
    }

    eq += this->numVectorLevelVar+this->numScalarLevelVar;
    for (int level = 0; level < this->numLevels; ++level) {
      for (int j = eq; j < eq+this->numTracerVar; ++j, ++n) {
        auto valptr = d_val_kokkos[j](cell,node,level);
        row = nodeID(cell,node,n);
        for (int node_col=0, i=0; node_col<this->numNodes; node_col++){
          for (int eq_col=0; eq_col<neq; eq_col++, i++) {
            col = nodeID(cell,node_col,eq_col);
            val = valptr.fastAccessDx(i);
            if (val != 0) 
              Jac_kokkos.sumIntoValues(row, &col,  1, &val, false, true);
          }
        }
      }
    }
    eq += this->numTracerVar;
  }
}

#endif

// **********************************************************************
template<typename Traits>
void ScatterResidual<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  auto nodeID = workset.wsElNodeEqID;
  Teuchos::RCP<Thyra_Vector>     f = workset.f;
  Teuchos::RCP<Thyra_LinearOp> Jac = workset.Jac;

  auto f_view = Albany::getNonconstLocalData(workset.f);

  const bool loadResid = (f != Teuchos::null);
  LO row; 
  Teuchos::Array<LO> cols; 
  Teuchos::Array<ST> vals; 

  for (int cell=0; cell < workset.numCells; ++cell ) {
    const int neq = nodeID.dimension(2);
    cols.resize(neq * this->numNodes);
    vals.resize(neq * this->numNodes);
    
    for (int node=0; node<this->numNodes; node++){
      for (int eq_col=0; eq_col<neq; eq_col++) {
        cols[neq * node + eq_col] =  nodeID(cell,node,eq_col);
      }
    }
    for (int node = 0; node < this->numNodes; ++node) {
      int n = 0, eq = 0;
      for (int j = eq; j < eq+this->numNodeVar; ++j, ++n) {
        auto valptr = (this->val[j])(cell,node);
        row = nodeID(cell,node,n);
        if (loadResid) {
          f_view[row] += valptr.val();
        }
        if (valptr.hasFastAccess()) {
          if (workset.is_adjoint) {
            // Sum Jacobian transposed
            for (unsigned int i=0; i<col.size(); ++i) {
              vals[0] = valptr.fastAccessDx(i); 
              Albany::addToLocalRowValues(Jac, col[i], Teuchos::arrayView(&row,1), vals.view(0,1));
            }
          } else {
            // Sum Jacobian entries all at once
            for (unsigned int i=0; i<neq*this->numNodes; ++i) {
              vals[i] = valptr.fastAccessDx(i);
            }
            Albany::addToLocalRowValues(Jac, row, cols(), vals());
          }
        } // has fast access
      }
      eq += this->numNodeVar;
      for (int level = 0; level < this->numLevels; level++) {
        for (int j = eq; j < eq+this->numVectorLevelVar; ++j) {
          for (int dim = 0; dim < this->numDims; ++dim, ++n) {
            auto valptr = (this->val[j])(cell,node,level,dim);
            row = nodeID(cell,node,n);
            if (loadResid) {
              f_view[row] += valptr.val();
            }
            if (valptr.hasFastAccess()) {
              if (workset.is_adjoint) {
                // Sum Jacobian transposed
                for (int i=0; i<col.size(); ++i) {
                  vals[0] = valptr.fastAccessDx(i); 
                  Albany::addToLocalRowValues(Jac, col[i], Teuchos::arrayView(&row,1), vals.view(0,1));
                }
              } else {
                // Sum Jacobian entries all at once
                for (unsigned int i=0; i<neq*this->numNodes; ++i) {
                  vals[i] = valptr.fastAccessDx(i);
                }
                Albany::addToLocalRowValues(Jac, row, cols(), vals());
              }
            }
          }
        }
        for (int j = eq+this->numVectorLevelVar;
                 j < eq+this->numVectorLevelVar+this->numScalarLevelVar; ++j, ++n) {
          auto valptr = (this->val[j])(cell,node,level);
          row = nodeID(cell,node,n);
          if (loadResid) {
            f_view[nodeID(cell,node,n)] += valptr.val();
          }
          if (valptr.hasFastAccess()) {
            if (workset.is_adjoint) {
              // Sum Jacobian transposed
              for (unsigned int i=0; i<colT.size(); ++i) {
                vals[0] = valptr.fastAccessDx(i); 
                Albany::addToLocalRowValues(Jac, col[i], Teuchos::arrayView(&row,1), vals.view(0,1));
              }
            } else {
              // Sum Jacobian entries all at once
              for (unsigned int i=0; i<neq*this->numNodes; ++i) {
                vals[i] = valptr.fastAccessDx(i);
              }
              Albany::addToLocalRowValues(Jac, row, cols(), vals());
            }
          } // has fast access
        }
      }
      eq += this->numVectorLevelVar+this->numScalarLevelVar;
      for (int level = 0; level < this->numLevels; ++level) {
        for (int j = eq; j < eq+this->numTracerVar; ++j, ++n) {
          auto valptr = (this->val[j])(cell,node,level);
          row = nodeID(cell,node,n);
          if (loadResid) {
            f_view[nodeID(cell,node,n)] += valptr.val();
          }
          if (valptr.hasFastAccess()) {
            if (workset.is_adjoint) {
              // Sum Jacobian transposed
              for (unsigned int i=0; i<colT.size(); ++i) {
                vals[0] = valptr.fastAccessDx(i); 
                Albany::addToLocalRowValues(Jac, col[i], Teuchos::arrayView(&row,1), vals.view(0,1));
              }
            } else {
              // Sum Jacobian entries all at once
              for (unsigned int i=0; i<neq*this->numNodes; ++i) {
                val[i] = valptr.fastAccessDx(i);
              }
              Albany::addToLocalRowValues(Jac, row, cols(), vals());
            }
          } // has fast access
        }
      }
      eq += this->numTracerVar;
    }
  }

#else
  // Get map for local data structures
  nodeID = workset.wsElNodeEqID;

  // Get dimensions
  neq = nodeID.dimension(2);
  nunk = neq*this->numNodes;

  // Get Tpetra vector view and local matrix
  const bool loadResid = Teuchos::nonnull(workset.f);
  if (loadResid) {
    f_kokkos = Albany::getNonconstDeviceData(workset.f);
  }

  if (Albany::isFillActive(workset.Jac)) {
    Albany::fillComplete(workset.Jac);
  }
  Jac_kokkos = Albany::getNonconstDeviceData(workset.Jac);

  // Get MDField views from std::vector
  for (int i = 0; i < this->numFields; i++) {
    val_kokkos[i] = this->val[i].get_view();
  }
  d_val_kokkos = val_kokkos.template view<ExecutionSpace>();

  if (loadResid) {
    Kokkos::parallel_for(Aeras_ScatterRes_Policy(0,workset.numCells),*this);
    cudaCheckError();
  }
  if (workset.is_adjoint) {
    Kokkos::parallel_for(Aeras_ScatterJac_Adjoint_Policy(0,workset.numCells),*this);
    cudaCheckError();
  }
  else {
    Kokkos::parallel_for(Aeras_ScatterJac_Policy(0,workset.numCells),*this);
    cudaCheckError();
  }

  if (!Albany::isFillActive(workset.Jac)) {
    Albany::resumeFill(workset.Jac);
  }

#endif
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
  auto nodeID = workset.wsElNodeEqID;
  Teuchos::RCP<Thyra_Vector>       f = workset.f;
  Teuchos::RCP<Thyra_MultiVector> JV = workset.JV;
  Teuchos::RCP<Thyra_MultiVector> fp = workset.fp;

  Teuchos::ArrayRCP<ST> f_view;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>> JV_view;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>> fp_view;

  if (!f.is_null()) {
    f_view = Albany::getNonconstLocalData(f);
  }
  if (!JV.is_null()) {
    JV_view = Albany::getNonconstLocalData(JV);
  }
  if (!fp.is_null()) {
    fp_view = Albany::getNonconstLocalData(fp);
  }
  
  int row; 

  //IK, 6/27/14: I think you don't actually need row_map here the way this function is written right now...
  //const Epetra_BlockMap *row_map = NULL;
  //if (f != Teuchos::null)       row_map = &( f->Map());
  //else if (JV != Teuchos::null) row_map = &(JV->Map());
  //else if (fp != Teuchos::null) row_map = &(fp->Map());
  //else
  if (f.is_null() && JV.is_null() && fp.is_null()) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                     "One of f, JV, or fp must be non-null! " << std::endl);
  }

  for (int cell=0; cell < workset.numCells; ++cell ) {
    for (int node = 0; node < this->numNodes; ++node) {
      int n = 0, eq = 0;
      for (int j = eq; j < eq+this->numNodeVar; ++j, ++n) {
        auto valptr = (this->val[j])(cell,node);
        row = nodeID(cell,node,n);
        if (f != Teuchos::null) {
          f_view[row] += valptr.val();
        }
        if (JV != Teuchos::null) {
          for (int col=0; col<workset.num_cols_x; col++) {
            JV_view[col][row] += valptr.dx(col);
          }
        }
        if (fp != Teuchos::null) {
          for (int col=0; col<workset.num_cols_p; col++) {
            fp_view[col][row] += valptr.dx(col+workset.param_offset);
          }
        }
      }
      eq += this->numNodeVar;
      for (int level = 0; level < this->numLevels; level++) {
        for (int j = eq; j < eq+this->numVectorLevelVar; ++j) {
          for (int dim = 0; dim < this->numDims; ++dim, ++n) {
            auto valptr = (this->val[j])(cell,node,level,dim);
            row = nodeID(cell,node,n);
            if (f != Teuchos::null) {
              f_view[row] += valptr.val();
            }
            if (JV != Teuchos::null) {
              for (int col=0; col<workset.num_cols_x; col++) {
                JV_view[col][row] += valptr.dx(col);
              }
            }
            if (fp != Teuchos::null) {
              for (int col=0; col<workset.num_cols_p; col++) {
                fp_view[col][row] += valptr.dx(col+workset.param_offset);
              }
            }
          }
        }
        for (int j = eq+this->numVectorLevelVar;
                 j < eq+this->numVectorLevelVar+this->numScalarLevelVar; ++j,++n) {
          auto valptr = (this->val[j])(cell,node,level);
          row = nodeID(cell,node,n);
          if (f != Teuchos::null) {
            f_view[row] += valptr.val();
          }
          if (JV != Teuchos::null) {
            for (int col=0; col<workset.num_cols_x; col++) {
              JV_view[col][row] += valptr.dx(col);
            }
          }
          if (fp != Teuchos::null) {
            for (int col=0; col<workset.num_cols_p; col++) {
              fp_view[col][row] += valptr.dx(col+workset.param_offset);
            }
          }
        }
      }
     eq += this->numVectorLevelVar+this->numScalarLevelVar;
      for (int level = 0; level < this->numLevels; ++level) {
        for (int j = eq; j < eq+this->numTracerVar; ++j, ++n) {
          auto valptr = (this->val[j])(cell,node,level);
          row = nodeID(cell,node,n);
          if (f != Teuchos::null) {
            f_view[row] += valptr.val();
          }
          if (JV != Teuchos::null) {
            for (int col=0; col<workset.num_cols_x; col++) {
              JV_view[col][row] += valptr.dx(col);
            }
          }
          if (fp != Teuchos::null) {
            for (int col=0; col<workset.num_cols_p; col++) {
              fp_view[col][row] += valptr.dx(col+workset.param_offset);
            }
          }
        }
      }
      eq += this->numTracerVar;
    }
  } 
}

} // namespace Aeras
