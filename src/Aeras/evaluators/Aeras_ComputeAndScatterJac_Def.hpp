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
ComputeAndScatterJacBase<EvalT, Traits>::
ComputeAndScatterJacBase(const Teuchos::ParameterList& p,
                    const Teuchos::RCP<Aeras::Layouts>& dl) :
  BF            (p.get<std::string>  ("BF Name"),           dl->node_qp_scalar),
  wBF           (p.get<std::string>  ("Weighted BF Name"),  dl->node_qp_scalar),
  GradBF        (p.get<std::string>  ("Gradient BF Name"),  dl->node_qp_gradient),
  wGradBF       (p.get<std::string>  ("Weighted Gradient BF Name"), dl->node_qp_gradient),
  worksetSize(dl->node_scalar             ->dimension(0)),
  numNodes   (dl->node_scalar             ->dimension(1)),
  numDims    (dl->node_qp_gradient        ->dimension(3)),
  numLevels  (dl->node_scalar_level       ->dimension(2)), 
  numFields  (0), numNodeVar(0), numVectorLevelVar(0),  numScalarLevelVar(0), numTracerVar(0)
{
  std::cout << "DEBUG: " << __PRETTY_FUNCTION__ << "\n";

  const Teuchos::ArrayRCP<std::string> node_names         = p.get< Teuchos::ArrayRCP<std::string> >("Node Residual Names");
  const Teuchos::ArrayRCP<std::string> vector_level_names = p.get< Teuchos::ArrayRCP<std::string> >("Vector Level Residual Names");
  const Teuchos::ArrayRCP<std::string> scalar_level_names = p.get< Teuchos::ArrayRCP<std::string> >("Scalar Level Residual Names");
  const Teuchos::ArrayRCP<std::string> tracer_names       = p.get< Teuchos::ArrayRCP<std::string> >("Tracer Residual Names");

  numNodeVar   = node_names  .size();
  numVectorLevelVar  = vector_level_names .size();
  numScalarLevelVar  = scalar_level_names .size();
  numTracerVar = tracer_names.size();
  numFields = numNodeVar +  numVectorLevelVar + numScalarLevelVar +  numTracerVar;

  const std::string fieldName = p.get<std::string>("Scatter Field Name");
  scatter_operation = Teuchos::rcp(new PHX::Tag<ScalarT>(fieldName, dl->dummy));
  
  this->addDependentField(BF);
  this->addDependentField(wBF);
  this->addDependentField(GradBF);
  this->addDependentField(wGradBF);

  this->addEvaluatedField(*scatter_operation);

  this->setName(fieldName);
}

// **********************************************************************
template<typename EvalT, typename Traits>
void ComputeAndScatterJacBase<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm) 
{
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(wBF,fm);
  this->utils.setFieldData(GradBF,fm);
  this->utils.setFieldData(wGradBF,fm);
}


// **********************************************************************
// Specialization: Jacobian
// **********************************************************************

template<typename Traits>
ComputeAndScatterJac<PHAL::AlbanyTraits::Jacobian, Traits>::
ComputeAndScatterJac(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Aeras::Layouts>& dl)
  : ComputeAndScatterJacBase<PHAL::AlbanyTraits::Jacobian,Traits>(p,dl)
{ }
// *********************************************************************
// Kokkos kernels
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
template<typename Traits>
KOKKOS_INLINE_FUNCTION
void ComputeAndScatterJac<PHAL::AlbanyTraits::Jacobian, Traits>::
operator() (const ScatterResid_noFastAccess_Tag& tag, const int& i) const{

}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void ComputeAndScatterJac<PHAL::AlbanyTraits::Jacobian, Traits>::
operator() (const ScatterResid_hasFastAccess_is_adjoint_Tag& tag, const int& cell) const{

 LO* colT = new LO[nunk];
 LO rowT;

  for (int node_col=0, i=0; node_col<this->numNodes; node_col++){
      for (int eq_col=0; eq_col<neq; eq_col++) {
        colT[neq * node_col + eq_col] =  Index(cell,node_col,eq_col);
      }
    }
  for (int node = 0; node < this->numNodes; ++node) {
      int n = 0, eq = 0;
      for (int j = eq; j < eq+this->numNodeVar; ++j, ++n) {
        const typename PHAL::Ref<ScalarT>::type valptr = (this->val[j])(cell,node);
        rowT = Index(cell,node,n);
        if (loadResid) fT->sumIntoLocalValue(rowT, valptr.val());
        for (unsigned int i=0; i<nunk; ++i) {
          ST val = valptr.fastAccessDx(i);
          if (val != 0) 
            jacobian.sumIntoValues(colT[i], &rowT,  1, &val,true);
        }
      }
      eq += this->numNodeVar;
      for (int level = 0; level < this->numLevels; level++) {
        for (int j = eq; j < eq+this->numVectorLevelVar; ++j) {
          for (int dim = 0; dim < this->numDims; ++dim, ++n) {
            const typename PHAL::Ref<ScalarT>::type valptr = (this->val[j])(cell,node,level,dim);
            rowT = Index(cell,node,n);
            if (loadResid) fT->sumIntoLocalValue(rowT, valptr.val());
            for (int i=0; i<nunk; ++i){
              ST val = valptr.fastAccessDx(i);
              if (val != 0) 
                jacobian.sumIntoValues(colT[i], &rowT,  1, &val,true);
         }
        }
         for (int j = eq+this->numVectorLevelVar;
                 j < eq+this->numVectorLevelVar+this->numScalarLevelVar; ++j, ++n) {
          const typename PHAL::Ref<ScalarT>::type valptr = (this->val[j])(cell,node,level);
          rowT = Index(cell,node,n);
          if (loadResid) fT->sumIntoLocalValue(Index(cell,node,n), valptr.val());
          for (int i=0; i<nunk; ++i){ 
            ST val = valptr.fastAccessDx(i);
            if (val != 0) 
              jacobian.sumIntoValues(colT[i], &rowT,  1, &val,true);
          }
        }
      }
     }
      eq += this->numVectorLevelVar+this->numScalarLevelVar;
      for (int level = 0; level < this->numLevels; ++level) {
        for (int j = eq; j < eq+this->numTracerVar; ++j, ++n) {
          const typename PHAL::Ref<ScalarT>::type valptr = (this->val[j])(cell,node,level);
          rowT = Index(cell,node,n);
          if (loadResid) fT->sumIntoLocalValue(Index(cell,node,n), valptr.val());
          for (int i=0; i<nunk; ++i) {
            ST val = valptr.fastAccessDx(i);
            if (val != 0) 
              jacobian.sumIntoValues(colT[i], &rowT,  1, &val,true);
        }
      }
     }
      eq += this->numTracerVar;
    }    
    delete colT; 
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void ComputeAndScatterJac<PHAL::AlbanyTraits::Jacobian, Traits>::
operator() (const ScatterResid_hasFastAccess_no_adjoint_Tag& tag, const int& cell) const{

  LO* colT = new LO[nunk];
  LO rowT;
 
  for (int node_col=0, i=0; node_col<this->numNodes; node_col++){
      for (int eq_col=0; eq_col<neq; eq_col++) {
        colT[neq * node_col + eq_col] =  Index(cell,node_col,eq_col);
      }
    }
  for (int node = 0; node < this->numNodes; ++node) {
      int n = 0, eq = 0;
      for (int j = eq; j < eq+this->numNodeVar; ++j, ++n) {
        const typename PHAL::Ref<ScalarT>::type valptr = (this->val[j])(cell,node);
        rowT = Index(cell,node,n);
        if (loadResid) fT->sumIntoLocalValue(rowT, valptr.val());
        for (unsigned int i=0; i<nunk; ++i) { 
          ST val = valptr.fastAccessDx(i);
          if (val != 0) 
            jacobian.sumIntoValues(rowT, &colT[i],  1, &val, true);
        }
        //jacobian.sumIntoValues(rowT, colT, nunk,  vals, true); 
      }
      eq += this->numNodeVar;
      for (int level = 0; level < this->numLevels; level++) {
        for (int j = eq; j < eq+this->numVectorLevelVar; ++j) {
          for (int dim = 0; dim < this->numDims; ++dim, ++n) {
            const typename PHAL::Ref<ScalarT>::type valptr = (this->val[j])(cell,node,level,dim);
            rowT = Index(cell,node,n);
            if (loadResid) fT->sumIntoLocalValue(rowT, valptr.val());
            for (int i=0; i<nunk; ++i) { 
              ST val = valptr.fastAccessDx(i);
              if (val != 0) 
                jacobian.sumIntoValues(rowT, &colT[i],  1, &val, true);
            }
            //jacobian.sumIntoValues(rowT, colT, nunk,  vals, true);
        }
         for (int j = eq+this->numVectorLevelVar;
                 j < eq+this->numVectorLevelVar+this->numScalarLevelVar; ++j, ++n) {
          const typename PHAL::Ref<ScalarT>::type valptr = (this->val[j])(cell,node,level);
          rowT = Index(cell,node,n);
          if (loadResid) fT->sumIntoLocalValue(Index(cell,node,n), valptr.val());
          for (int i=0; i<nunk; ++i) {
            ST val = valptr.fastAccessDx(i);
            if (val != 0) 
              jacobian.sumIntoValues(rowT, &colT[i],  1, &val, true);
          }
          //jacobian.sumIntoValues(rowT, colT, nunk,  vals, true);
        }
      }
     }
      eq += this->numVectorLevelVar+this->numScalarLevelVar;
      for (int level = 0; level < this->numLevels; ++level) {
        for (int j = eq; j < eq+this->numTracerVar; ++j, ++n) {
          const typename PHAL::Ref<ScalarT>::type valptr = (this->val[j])(cell,node,level);
          rowT = Index(cell,node,n);
          if (loadResid) fT->sumIntoLocalValue(Index(cell,node,n), valptr.val());
          for (int i=0; i<nunk; ++i) {
            ST val = valptr.fastAccessDx(i);
            if (val != 0) 
              jacobian.sumIntoValues(rowT, &colT[i],  1, &val, true);
          }
          //jacobian.sumIntoValues(rowT, colT, nunk,  vals, true);
      }
     }
      eq += this->numTracerVar;
    }
    delete colT;
}

#endif
// **********************************************************************
template<typename Traits>
void ComputeAndScatterJac<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{

//FIXME: this function needs to be rewritten to not use AD.  
//First, we need to compute the local mass and laplacian matrices 
//(checking the n_coeff flag for whether the laplacian is needed) as follows: 
//Mass:
//loop over cells, c
//  loop over levels, l
//    loop over nodes,n
//      q=n; m=n
//      diag(c,l,n) = BF(n,q)*wBF(m,q)
//
//Laplacian:
//loop over cells, c
//  loop over levels, l
//    loop over nodes,n
//      loop over nodes,m
//        loop over qp, q
//          loop over dim, d
//            laplace(c,l,n,m) += gradBF(n,q,d)*wGradBF(m,q,d)
//
//(There’s also a loop over unknowns per node.)
//
//Then the values of these matrices need to be scattered into the global Jacobian.

#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT

  Teuchos::RCP<Tpetra_Vector>      fT = workset.fT;
  Teuchos::RCP<Tpetra_CrsMatrix> JacT = workset.JacT;

  const bool loadResid = (fT != Teuchos::null);
  LO rowT; 
  Teuchos::Array<LO> colT; 

  std::cout << "DEBUG in ComputeAndScatterJac::EvaluateFields: " << __PRETTY_FUNCTION__ << "\n";
  std::cout << "LOAD RESIDUAL? " << loadResid << "\n";

#define OLDSCATTER 0
#if OLDSCATTER
  for (int cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

    //OG (the way I understand this code): nodeID is an array of LIDs (row map, or map for Tpetra Vector),
    //that is, wsElNodeEqID[cell][node][equation] gives a map to LIDs.
    //If a proc owes cell, then it owes all nodes and equations (layers of eqns) at this cell.
    //That is the jacobian is overlapped here, later jac->export(overlap_jac) is called for unique mapping.
    //TPetra vector (say, residual vector) is ordered by
    //ps(node0),u_lev0(node0),v_lev0(node0),T_lev0(node0),trA_lev0(node0),trB_lev0(node0),...,u_lev1(node0),v_lev1(node0),..., ps(node1),u_lev0(node1)...
    //In our case of hydrostatic problem numNodeVar = 1 (ps), numVectorLevelVar = 1 (velocity),
    //numScalarLevelVar = 1 (temperature), numTracerVar = # of tracers, tracers are also leveled vars.


    const int neq = nodeID[0].size();
    colT.resize(neq * this->numNodes);
    
    for (int node=0; node<this->numNodes; node++){
      for (int eq_col=0; eq_col<neq; eq_col++) {
        colT[neq * node + eq_col] =  nodeID[node][eq_col];
      }
    }
    for (int node = 0; node < this->numNodes; ++node) {
      const Teuchos::ArrayRCP<int>& eqID  = nodeID[node];
      int n = 0, eq = 0;
      for (int j = eq; j < eq+this->numNodeVar; ++j, ++n) {
        const typename PHAL::Ref<ScalarT>::type valptr = (this->val[j])(cell,node);
        rowT = eqID[n];
        if (loadResid) fT->sumIntoLocalValue(rowT, valptr.val());
        if (valptr.hasFastAccess()) {
          if (workset.is_adjoint) {
            // Sum Jacobian transposed
            for (unsigned int i=0; i<colT.size(); ++i) {
              ST val = valptr.fastAccessDx(i); 
              if (val != 0.0) { 
                JacT->sumIntoLocalValues(colT[i], Teuchos::arrayView(&rowT,1), Teuchos::arrayView(&val,1));
              }
            }
          } 
          else {
            //Sum Jacobian entries 
            for (unsigned int i=0; i<neq*this->numNodes; ++i) {
              ST val = valptr.fastAccessDx(i);
              if (val != 0.0) {
                JacT->sumIntoLocalValues(rowT, Teuchos::arrayView(&colT[i],1), Teuchos::arrayView(&val,1));
              }
            }
            // Sum Jacobian entries all at once
            // JacT->sumIntoLocalValues(rowT, colT, Teuchos::arrayView(&(valptr.fastAccessDx(0)), colT.size()));
          }
        } // has fast access
      }
      eq += this->numNodeVar;
      for (int level = 0; level < this->numLevels; level++) {
        for (int j = eq; j < eq+this->numVectorLevelVar; ++j) {
          for (int dim = 0; dim < this->numDims; ++dim, ++n) {
            const typename PHAL::Ref<ScalarT>::type valptr = (this->val[j])(cell,node,level,dim);
            rowT = eqID[n];
            if (loadResid) fT->sumIntoLocalValue(rowT, valptr.val());
            if (valptr.hasFastAccess()) {
              if (workset.is_adjoint) {
                // Sum Jacobian transposed
                for (int i=0; i<colT.size(); ++i) {
                  ST val = valptr.fastAccessDx(i); 
                  if (val != 0.0) { 
                    JacT->sumIntoLocalValues(colT[i], Teuchos::arrayView(&rowT,1), Teuchos::arrayView(&val,1));
                  }
                }
              } 
              else {
                //Sum Jacobian entries 
                for (unsigned int i=0; i<neq*this->numNodes; ++i) {
                  ST val = valptr.fastAccessDx(i);
                  if (val != 0.0) {
                    JacT->sumIntoLocalValues(rowT, Teuchos::arrayView(&colT[i],1), Teuchos::arrayView(&val,1));
                  }
                }
                // Sum Jacobian entries all at once
                //JacT->sumIntoLocalValues(rowT, colT, Teuchos::arrayView(&(valptr.fastAccessDx(0)), colT.size()));
              }
            }
          }
        }
        for (int j = eq+this->numVectorLevelVar;
                 j < eq+this->numVectorLevelVar+this->numScalarLevelVar; ++j, ++n) {
          const typename PHAL::Ref<ScalarT>::type valptr = (this->val[j])(cell,node,level);
          rowT = eqID[n];
          if (loadResid) fT->sumIntoLocalValue(eqID[n], valptr.val());
          if (valptr.hasFastAccess()) {
            if (workset.is_adjoint) {
              // Sum Jacobian transposed
              for (unsigned int i=0; i<colT.size(); ++i) {
                ST val = valptr.fastAccessDx(i); 
                if (val != 0.0) { 
                  JacT->sumIntoLocalValues(colT[i], Teuchos::arrayView(&rowT,1), Teuchos::arrayView(&val,1));
                }
              }
            } 
            else {
                //Sum Jacobian entries 
                for (unsigned int i=0; i<neq*this->numNodes; ++i) {
                  ST val = valptr.fastAccessDx(i);
                  if (val != 0.0) {
                    JacT->sumIntoLocalValues(rowT, Teuchos::arrayView(&colT[i],1), Teuchos::arrayView(&val,1));
                  }
                }
              // Sum Jacobian entries all at once
              //JacT->sumIntoLocalValues(rowT, colT, Teuchos::arrayView(&(valptr.fastAccessDx(0)), colT.size()));
            }
          } // has fast access
        }
      }
      eq += this->numVectorLevelVar+this->numScalarLevelVar;
      for (int level = 0; level < this->numLevels; ++level) {
        for (int j = eq; j < eq+this->numTracerVar; ++j, ++n) {
          const typename PHAL::Ref<ScalarT>::type valptr = (this->val[j])(cell,node,level);
          rowT = eqID[n];
          if (loadResid) fT->sumIntoLocalValue(eqID[n], valptr.val());
          if (valptr.hasFastAccess()) {
            if (workset.is_adjoint) {
              // Sum Jacobian transposed
              for (unsigned int i=0; i<colT.size(); ++i) {
                ST val = valptr.fastAccessDx(i); 
                if (val != 0.0) { 
                  JacT->sumIntoLocalValues(colT[i], Teuchos::arrayView(&rowT,1), Teuchos::arrayView(&val,1));
                }
              }
            } 
            else {
              //Sum Jacobian entries 
                for (unsigned int i=0; i<neq*this->numNodes; ++i) {
                  ST val = valptr.fastAccessDx(i);
                  if (val != 0.0) {
                    JacT->sumIntoLocalValues(rowT, Teuchos::arrayView(&colT[i],1), Teuchos::arrayView(&val,1));
                  }
                }
              // Sum Jacobian entries all at once
              //JacT->sumIntoLocalValues(rowT, colT, Teuchos::arrayView(&(valptr.fastAccessDx(0)), colT.size()));
            }
          } // has fast access
        }
      }
      eq += this->numTracerVar;
    }
  }

#endif

#if !OLDSCATTER
  int numcells_ = workset.numCells,
	  numnodes_ = this->numNodes;
  //for mass we do not really need even this
  /*
  Intrepid2::FieldContainer_Kokkos<ScalarT, PHX::Layout, PHX::Device> localMassMatr(numcells_,numnodes_,numnodes_);
  for(int cell = 0; cell < numcells_; cell++){
	  for(int node = 0; node < numnodes_; node++)
	     localMassMatr(cell, node, node) = this -> wBF(cell, node, node);
  }*/

  for (int cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

    const int neq = nodeID[0].size();

    for (int node = 0; node < this->numNodes; ++node) {
      const Teuchos::ArrayRCP<int>& eqID  = nodeID[node];
      int n = 0, eq = 0;
      for (int j = eq; j < eq+this->numNodeVar; ++j, ++n) {
        rowT = eqID[n];
        ST val2 = - this -> wBF(cell, node, node);
        JacT->sumIntoLocalValues(rowT, Teuchos::arrayView(&rowT,1), Teuchos::arrayView(&val2,1));
      }
      eq += this->numNodeVar;
      for (int level = 0; level < this->numLevels; level++) {
        for (int j = eq; j < eq+this->numVectorLevelVar; ++j) {
          for (int dim = 0; dim < this->numDims; ++dim, ++n) {
            rowT = eqID[n];
            ST val2 = - this -> wBF(cell, node, node);
            JacT->sumIntoLocalValues(rowT, Teuchos::arrayView(&rowT,1), Teuchos::arrayView(&val2,1));
          }
        }
        for (int j = eq+this->numVectorLevelVar;
                 j < eq+this->numVectorLevelVar+this->numScalarLevelVar; ++j, ++n) {
          rowT = eqID[n];
          ST val2 = - this -> wBF(cell, node, node);
          JacT->sumIntoLocalValues(rowT, Teuchos::arrayView(&rowT,1), Teuchos::arrayView(&val2,1));
        }
      }
      eq += this->numVectorLevelVar+this->numScalarLevelVar;
      for (int level = 0; level < this->numLevels; ++level) {
        for (int j = eq; j < eq+this->numTracerVar; ++j, ++n) {
          rowT = eqID[n];
          //Minus!
          ST val2 = - this -> wBF(cell, node, node);
          JacT->sumIntoLocalValues(rowT, Teuchos::arrayView(&rowT,1), Teuchos::arrayView(&val2,1));
        }
      }
      eq += this->numTracerVar;
    }
  }

#endif






#else
   fT = workset.fT;
   JacT = workset.JacT;
   Index=workset.wsElNodeEqID_kokkos;
   neq = workset.wsElNodeEqID[0][0].size();
   nunk = neq*this->numNodes;

   if (!JacT->isFillComplete())
      JacT->fillComplete();

   jacobian=JacT->getLocalMatrix();

   loadResid = (fT != Teuchos::null);

       if (workset.is_adjoint)
           Kokkos::parallel_for(ScatterResid_hasFastAccess_is_adjoint_Policy(0,workset.numCells),*this);
       else
           Kokkos::parallel_for(ScatterResid_hasFastAccess_no_adjoint_Policy(0,workset.numCells),*this);


  if (JacT->isFillComplete())
    JacT->resumeFill();

#endif
}

#ifdef ALBANY_ENSEMBLE
// **********************************************************************
// Specialization: Multi-point Residual
// **********************************************************************
template<typename Traits>
ComputeAndScatterJac<PHAL::AlbanyTraits::MPResidual,Traits>::
ComputeAndScatterJac(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Aeras::Layouts>& dl)
  : ComputeAndScatterJacBase<PHAL::AlbanyTraits::MPResidual,Traits>(p,dl)
{}

template<typename Traits>
void ComputeAndScatterJac<PHAL::AlbanyTraits::MPResidual, Traits>::
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
        typename PHAL::Ref<ScalarT>::type valptr = (this->val[j])(cell,node);
        for (int block=0; block<nblock; block++)
          (*f)[block][nodeID[node][n]] += valptr.coeff(block);
      }
      eq += this->numNodeVar;
      for (int level = 0; level < this->numLevels; level++) { 
        for (int j = eq; j < eq+this->numVectorLevelVar; ++j) {
          for (int dim = 0; dim < this->numDims; ++dim, ++n) {
//            fT_nonconstView[eqID[n]] += (this->val[j])(cell,node,level,dim);
            typename PHAL::Ref<ScalarT>::type valptr = (this->val[j])(cell,node,level,dim);
            for (int block=0; block<nblock; block++)
              (*f)[block][nodeID[node][n]] += valptr.coeff(block);
          }
        }
        for (int j = eq+this->numVectorLevelVar; 
                 j < eq+this->numVectorLevelVar + this->numScalarLevelVar; ++j, ++n) {
//          fT_nonconstView[eqID[n]] += (this->val[j])(cell,node,level);
          typename PHAL::Ref<ScalarT>::type valptr = (this->val[j])(cell,node,level);
          for (int block=0; block<nblock; block++)
            (*f)[block][nodeID[node][n]] += valptr.coeff(block);
        }
      }
      eq += this->numVectorLevelVar + this->numScalarLevelVar;
      for (int level = 0; level < this->numLevels; ++level) { 
        for (int j = eq; j < eq+this->numTracerVar; ++j, ++n) {
//          fT_nonconstView[eqID[n]] += (this->val[j])(cell,node,level);
          typename PHAL::Ref<ScalarT>::type valptr = (this->val[j])(cell,node,level);
          for (int block=0; block<nblock; block++)
            (*f)[block][nodeID[node][n]] += valptr.coeff(block);
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
ComputeAndScatterJac<PHAL::AlbanyTraits::MPJacobian, Traits>::
ComputeAndScatterJac(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Aeras::Layouts>& dl)
  : ComputeAndScatterJacBase<PHAL::AlbanyTraits::MPJacobian,Traits>(p,dl)
{ }

// **********************************************************************
template<typename Traits>
void ComputeAndScatterJac<PHAL::AlbanyTraits::MPJacobian, Traits>::
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
        const typename PHAL::Ref<ScalarT>::type valptr = (this->val[j])(cell,node);
        row = eqID[n]; 
        if (loadResid) {
//          fT->sumIntoLocalValue(rowT, valptr.val());
          for (int block=0; block<nblock; block++)
            (*f)[block].SumIntoMyValue(row, 0, valptr.val().coeff(block));
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

        if (valptr.hasFastAccess()) {
          for (int node_col=0; node_col<this->numNodes; node_col++){
            for (int eq_col=0; eq_col<neq; eq_col++) {
              lcol = neq * node_col + eq_col;
              col = nodeID[node_col][eq_col];
              for (int block=0; block<nblock_jac; block++) {
                c = valptr.fastAccessDx(lcol).coeff(block);
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
            const typename PHAL::Ref<ScalarT>::type valptr = (this->val[j])(cell,node,level,dim);
            row = eqID[n]; 
            if (loadResid) {
              for (int block=0; block<nblock; block++)
                (*f)[block].SumIntoMyValue(row, 0, valptr.val().coeff(block));
            }
            if (valptr.hasFastAccess()) {
              for (int node_col=0; node_col<this->numNodes; node_col++){
                for (int eq_col=0; eq_col<neq; eq_col++) {
                  lcol = neq * node_col + eq_col;
                  col = nodeID[node_col][eq_col];
                  for (int block=0; block<nblock_jac; block++) {
                    c = valptr.fastAccessDx(lcol).coeff(block);
                    (*Jac)[block].SumIntoMyValues(row, 1, &c, &col);
                  }
                }
              }
            }
          } 
        }
        for (int j = eq+this->numVectorLevelVar; 
                 j < eq+this->numVectorLevelVar+this->numScalarLevelVar; ++j, ++n) {
          const typename PHAL::Ref<ScalarT>::type valptr = (this->val[j])(cell,node,level);
          row = eqID[n]; 
          if (loadResid) {
            for (int block=0; block<nblock; block++)
              (*f)[block].SumIntoMyValue(eqID[n], 0, valptr.val().coeff(block));
          }
          if (valptr.hasFastAccess()) {
            for (int node_col=0; node_col<this->numNodes; node_col++){
              for (int eq_col=0; eq_col<neq; eq_col++) {
                lcol = neq * node_col + eq_col;
                col = nodeID[node_col][eq_col];
                for (int block=0; block<nblock_jac; block++) {
                  c = valptr.fastAccessDx(lcol).coeff(block);
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
          const typename PHAL::Ref<ScalarT>::type valptr = (this->val[j])(cell,node,level);
          row = eqID[n]; 
          if (loadResid) {
            for (int block=0; block<nblock; block++)
              (*f)[block].SumIntoMyValue(eqID[n], 0, valptr.val().coeff(block));
          }
          if (valptr.hasFastAccess()) {
            for (int node_col=0; node_col<this->numNodes; node_col++){
              for (int eq_col=0; eq_col<neq; eq_col++) {
                lcol = neq * node_col + eq_col;
                col = nodeID[node_col][eq_col];
                for (int block=0; block<nblock_jac; block++) {
                  c = valptr.fastAccessDx(lcol).coeff(block);
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
#endif

}
