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
#include "Aeras_Dimension.hpp"

namespace Aeras {


template<typename EvalT, typename Traits>
GatherSolutionBase<EvalT,Traits>::
GatherSolutionBase(const Teuchos::ParameterList& p,
                   const Teuchos::RCP<Aeras::Layouts>& dl): 
worksetSize(dl->node_scalar             ->dimension(0)),
numNodes   (dl->node_scalar             ->dimension(1)),
numDims    (dl->node_qp_gradient        ->dimension(3)),
numLevels  (dl->node_scalar_level       ->dimension(2)), 
numFields  (0), numNodeVar(0), numVectorLevelVar(0), numScalarLevelVar(0), numTracerVar(0)
{
  const Teuchos::ArrayRCP<std::string> node_names              = p.get< Teuchos::ArrayRCP<std::string> >("Node Names");
  const Teuchos::ArrayRCP<std::string> vector_level_names      = p.get< Teuchos::ArrayRCP<std::string> >("Vector Level Names");
  const Teuchos::ArrayRCP<std::string> scalar_level_names      = p.get< Teuchos::ArrayRCP<std::string> >("Scalar Level Names");
  const Teuchos::ArrayRCP<std::string> tracer_names            = p.get< Teuchos::ArrayRCP<std::string> >("Tracer Names");
  const Teuchos::ArrayRCP<std::string> node_names_dot          = p.get< Teuchos::ArrayRCP<std::string> >("Time Dependent Node Names");
  const Teuchos::ArrayRCP<std::string> vector_level_names_dot  = p.get< Teuchos::ArrayRCP<std::string> >("Time Dependent Vector Level Names");
  const Teuchos::ArrayRCP<std::string> scalar_level_names_dot  = p.get< Teuchos::ArrayRCP<std::string> >("Time Dependent Scalar Level Names");
  const Teuchos::ArrayRCP<std::string> tracer_names_dot        = p.get< Teuchos::ArrayRCP<std::string> >("Time Dependent Tracer Names");

  numNodeVar         = node_names         .size();
  numVectorLevelVar  = vector_level_names .size();
  numScalarLevelVar  = scalar_level_names .size();
  numTracerVar       = tracer_names       .size();
  numFields          = numNodeVar +  numVectorLevelVar + numScalarLevelVar + numTracerVar;

  val.resize(numFields);
  val_dot.resize(numFields);
  int eq = 0;

  for (int i = 0; i < numNodeVar; ++i,++eq) {
    PHX::MDField<ScalarT> f(node_names[i],dl->node_scalar);
    val[eq] = f;
    this->addEvaluatedField(val[eq]);
  }   
  for (int i = 0; i < numVectorLevelVar; ++i,++eq) {
    PHX::MDField<ScalarT> f(vector_level_names[i],dl->node_vector_level); val[eq] = f;
    this->addEvaluatedField(val[eq]);
  }   
  for (int i = 0; i < numScalarLevelVar; ++i,++eq) {
    PHX::MDField<ScalarT> f(scalar_level_names[i],dl->node_scalar_level); val[eq] = f;
    this->addEvaluatedField(val[eq]);
  }   
  for (int i = 0; i < numTracerVar; ++i,++eq) {
    PHX::MDField<ScalarT> f(tracer_names[i],dl->node_scalar_level);
    val[eq] = f;
    this->addEvaluatedField(val[eq]);
  }   

  eq = 0;
  for (int i = 0; i < numNodeVar; ++i, ++eq) {
    PHX::MDField<ScalarT> f(node_names_dot[i],dl->node_scalar);
    val_dot[eq] = f;
    this->addEvaluatedField(val_dot[eq]);
  }   
  for (int i = 0; i < numVectorLevelVar; ++i, ++eq) {
    PHX::MDField<ScalarT> f(vector_level_names_dot[i],dl->node_vector_level); val_dot[eq] = f;
    this->addEvaluatedField(val_dot[eq]);
  }   
  for (int i = 0; i < numScalarLevelVar; ++i, ++eq) {
    PHX::MDField<ScalarT> f(scalar_level_names_dot[i],dl->node_scalar_level); val_dot[eq] = f;
    this->addEvaluatedField(val_dot[eq]);
  }   
  for (int i = 0; i < numTracerVar; ++i, ++eq) {
    PHX::MDField<ScalarT> f(tracer_names_dot[i],dl->node_scalar_level);
    val_dot[eq] = f;
    this->addEvaluatedField(val_dot[eq]);
  }

/*#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
   for (int i =0; i<numFields;i++){
     val_kokkosvec[i]=val[i];//.get_kokkos_view();
     val_dot_kokkosvec[i]=val_dot[i];//.get_kokkos_view();
   }

   d_val=val_kokkosvec.template view<executionSpace>();
   d_val_dot=val_dot_kokkosvec.template view<executionSpace>();
#endif
*/
  this->setName("Aeras_GatherSolution" +PHX::typeAsString<EvalT>());
}


// **********************************************************************
template<typename EvalT, typename Traits>
void GatherSolutionBase<EvalT,Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm) 
{
  for (int eq = 0; eq < val.size();     ++eq) this->utils.setFieldData(val[eq],fm);
  for (int eq = 0; eq < val_dot.size(); ++eq) this->utils.setFieldData(val_dot[eq],fm);
}


// **********************************************************************
// **********************************************************************
// GENERIC: MP and SG specializations not yet implemented
// **********************************************************************
template<typename Traits>
GatherSolution<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
GatherSolution(const Teuchos::ParameterList& p,
               const Teuchos::RCP<Aeras::Layouts>& dl) :
  GatherSolutionBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p,dl)
{}

template<typename Traits>
void GatherSolution<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData workset)
{ 
    throw "Aeras::GatherSolution not implemented for all tempate specializations";
}

template<typename EvalT, typename Traits>
GatherSolution<EvalT, Traits>::
GatherSolution(const Teuchos::ParameterList& p,
               const Teuchos::RCP<Aeras::Layouts>& dl) :
  GatherSolutionBase<EvalT, Traits>(p,dl)
{
}

template<typename EvalT, typename Traits>
void GatherSolution<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{ 
    throw "Aeras::GatherSolution not implemented for all tempate specializations";
}


// **********************************************************************
// **********************************************************************
// Specialization: Residual
// **********************************************************************
//Kokkos kernel Residual
/*#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherSolution<PHAL::AlbanyTraits::Residual, Traits>::
operator() (const int &cell) const{

 for (int node = 0; node < this->numNodes; ++node) {
      int n = 0, eq = 0;
      for (int j = eq; j < eq+this->numNodeVar; ++j, ++n) {
        (this->d_val    [j])(cell,node) = xT_constView[wsID_kokkos(cell, node,n)];
        (this->d_val_dot[j])(cell,node) = xdotT_constView[wsID_kokkos(cell, node,n)];
      }
      eq += this->numNodeVar;
      for (int level = 0; level < this->numLevels; level++) {
        for (int j = eq; j < eq+this->numVectorLevelVar; ++j) {
          for (int dim = 0; dim < this->numDims; ++dim, ++n) {
            (this->d_val    [j])(cell,node,level,dim) = xT_constView   [wsID_kokkos(cell, node,n)];
            (this->d_val_dot[j])(cell,node,level,dim) = xdotT_constView[wsID_kokkos(cell, node,n)];
          }
        }
        for (int j = eq+this->numVectorLevelVar;
                 j < eq+this->numVectorLevelVar+this->numScalarLevelVar; ++j, ++n) {
          (this->d_val    [j])(cell,node,level) = xT_constView   [wsID_kokkos(cell, node,n)];
          (this->d_val_dot[j])(cell,node,level) = xdotT_constView[wsID_kokkos(cell, node,n)];
        }
      }
      eq += this->numScalarLevelVar + this->numVectorLevelVar;
      for (int level = 0; level < this->numLevels; ++level) {
        for (int j = eq; j < eq+this->numTracerVar; ++j, ++n) {
          (this->d_val    [j])(cell,node,level) = xT_constView[wsID_kokkos(cell, node,n)];
          (this->d_val_dot[j])(cell,node,level) = xdotT_constView[wsID_kokkos(cell, node,n)];
        }
      }
      eq += this->numTracerVar;
    }


}
#endif
*/
// ***********************************************************************
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
  Teuchos::RCP<const Tpetra_Vector> xT = workset.xT;
  Teuchos::RCP<const Tpetra_Vector> xdotT = workset.xdotT;

  //Get const view of xT and xdotT 
//  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();
//  Teuchos::ArrayRCP<const ST> xdotT_constView = xdotT->get1dView();
  xT_constView = xT->get1dView();
  xdotT_constView = xdotT->get1dView();

//#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT

  for (int cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];
    for (int node = 0; node < this->numNodes; ++node) {
      const Teuchos::ArrayRCP<int>& eqID  = nodeID[node];
      int n = 0, eq = 0;
      for (int j = eq; j < eq+this->numNodeVar; ++j, ++n) {
        (this->val    [j])(cell,node) = xT_constView[eqID[n]];
        (this->val_dot[j])(cell,node) = xdotT_constView[eqID[n]];
      }
      eq += this->numNodeVar;
      for (int level = 0; level < this->numLevels; level++) { 
        for (int j = eq; j < eq+this->numVectorLevelVar; ++j) {
          for (int dim = 0; dim < this->numDims; ++dim, ++n) {
            (this->val    [j])(cell,node,level,dim) = xT_constView   [eqID[n]];
            (this->val_dot[j])(cell,node,level,dim) = xdotT_constView[eqID[n]];
          }
        }
        for (int j = eq+this->numVectorLevelVar; 
                 j < eq+this->numVectorLevelVar+this->numScalarLevelVar; ++j, ++n) {
          (this->val    [j])(cell,node,level) = xT_constView   [eqID[n]];
          (this->val_dot[j])(cell,node,level) = xdotT_constView[eqID[n]];
        }
      }
      eq += this->numScalarLevelVar + this->numVectorLevelVar;
      for (int level = 0; level < this->numLevels; ++level) { 
        for (int j = eq; j < eq+this->numTracerVar; ++j, ++n) {
          (this->val    [j])(cell,node,level) = xT_constView[eqID[n]];
          (this->val_dot[j])(cell,node,level) = xdotT_constView[eqID[n]];
        }
      }
      eq += this->numTracerVar;
    }
  }
/*#else
   wsID_kokkos=workset.wsElNodeEqID_kokkos;
  Kokkos::parallel_for(workset.numCells,*this);

#endif
*/
}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************
//Kokkos kernels Jacobian
/*#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
gather_solution(const int &cell, const int &node, const int &neq, const int &num_dof, const int &firstunk) const{
 
   int eq=0, n=0;

    for (int j = eq; j < eq+this->numNodeVar; ++j, ++n) {
        typename PHAL::Ref<ScalarT>::type valptr = (this->d_val[j])(cell,node);
        valptr = FadType(num_dof, xT_constView[wsID_kokkos(cell, node,n)]);
        valptr.setUpdateValue(!ignore_residual);
        valptr.fastAccessDx(firstunk + n) = j_coeff;
      }
      eq += this->numNodeVar;
      for (int level = 0; level < this->numLevels; level++) {
        for (int j = eq; j < eq+this->numVectorLevelVar; j++) {
          for (int dim = 0; dim < this->numDims; ++dim, ++n) {
            typename PHAL::Ref<ScalarT>::type valptr = (this->d_val[j])(cell,node,level,dim);
            valptr = FadType(num_dof, xT_constView[wsID_kokkos(cell, node,n)]);
            valptr.setUpdateValue(!ignore_residual);
            valptr.fastAccessDx(firstunk + n) = j_coeff;
          }
        }
        for (int j = eq+this->numVectorLevelVar;
                 j < eq+this->numVectorLevelVar+this->numScalarLevelVar; ++j,++n) {
          typename PHAL::Ref<ScalarT>::type valptr = (this->d_val[j])(cell,node,level);
          valptr = FadType(num_dof, xT_constView[wsID_kokkos(cell, node,n)]);
          valptr.setUpdateValue(!ignore_residual);
          valptr.fastAccessDx(firstunk + n) = j_coeff;
        }
      }
      eq += this->numVectorLevelVar+this->numScalarLevelVar;
      for (int level = 0; level < this->numLevels; ++level) {
        for (int j = eq; j < eq+this->numTracerVar; ++j, ++n) {
          typename PHAL::Ref<ScalarT>::type valptr = (this->d_val[j])(cell,node,level);
          valptr = FadType(num_dof, xT_constView[wsID_kokkos(cell, node,n)]);
          valptr.setUpdateValue(!ignore_residual);
          valptr.fastAccessDx(firstunk + n) = j_coeff;
        }
      }
      eq += this->numTracerVar;

}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
gather_solution_transientTerms(const int &cell, const int &node, const int &neq, const int &num_dof, const int &firstunk) const{

      int  n = 0, eq = 0;
        for (int j = eq; j < eq+this->numNodeVar; ++j, ++n) {
          typename PHAL::Ref<ScalarT>::type valptr = (this->d_val_dot[j])(cell,node);
          valptr = FadType(num_dof, xdotT_constView[wsID_kokkos(cell, node,n)]);
          valptr.fastAccessDx(firstunk + n) = m_coeff;
        }
        eq += this->numNodeVar;
        for (int level = 0; level < this->numLevels; level++) {
          for (int j = eq; j < eq+this->numVectorLevelVar; j++) {
            for (int dim = 0; dim < this->numDims; ++dim, ++n) {
              typename PHAL::Ref<ScalarT>::type valptr = (this->d_val_dot[j])(cell,node,level,dim);
              valptr = FadType(num_dof, xdotT_constView[wsID_kokkos(cell, node,n)]);
              valptr.fastAccessDx(firstunk + n) = m_coeff;
            }
          }
          for (int j = eq+this->numVectorLevelVar;
                   j < eq+this->numVectorLevelVar+this->numScalarLevelVar; j++,++n) {
            typename PHAL::Ref<ScalarT>::type valptr = (this->d_val_dot[j])(cell,node,level);
            valptr = FadType(num_dof, xdotT_constView[wsID_kokkos(cell, node,n)]);
            valptr.fastAccessDx(firstunk + n) = m_coeff;
          }
        }
        eq += this->numVectorLevelVar+this->numScalarLevelVar;
        for (int level = 0; level < this->numLevels; ++level) {
          for (int j = eq; j < eq+this->numTracerVar; ++j, ++n) {
            typename PHAL::Ref<ScalarT>::type valptr = (this->d_val_dot[j])(cell,node,level);
            valptr = FadType(num_dof, xdotT_constView[wsID_kokkos(cell, node,n)]);
            valptr.fastAccessDx(firstunk + n) =m_coeff;
          }
        }
        eq += this->numTracerVar;

}


template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
operator() (const GatherSolution_Tag &tag, const int &cell) const{

  const int neq=wsID_kokkos.dimension(2);
  const int num_dof = neq * this->numNodes;

  for (int node = 0; node < this->numNodes; ++node) {
   const int firstunk = neq * node;
   int n = 0, eq = 0;
   gather_solution(cell, node, neq, num_dof, firstunk);
  }
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
operator() (const GatherSolution_transientTerms_Tag &tag, const int &cell) const{

  const int neq=wsID_kokkos.dimension(2);
  const int num_dof = neq * this->numNodes;

  for (int node = 0; node < this->numNodes; ++node) {
   const int firstunk = neq * node;
   int n = 0, eq = 0;
   gather_solution(cell, node, neq, num_dof, firstunk);
   gather_solution_transientTerms(cell, node, neq, num_dof, firstunk);
  }
}
#endif
*/
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
  const Teuchos::RCP<const Tpetra_Vector>    xT = workset.xT;
  const Teuchos::RCP<const Tpetra_Vector> xdotT = workset.xdotT;

//#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT

  //get const view of xT and xdotT   
  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();
  Teuchos::ArrayRCP<const ST> xdotT_constView = xdotT->get1dView();

  for (int cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];
    const int neq = nodeID[0].size();
    const int num_dof = neq * this->numNodes;


    for (int node = 0; node < this->numNodes; ++node) {
      const Teuchos::ArrayRCP<int>& eqID  = nodeID[node];
      const int firstunk = neq * node;
      int n = 0, eq = 0;
      for (int j = eq; j < eq+this->numNodeVar; ++j, ++n) {
        typename PHAL::Ref<ScalarT>::type valptr = (this->val[j])(cell,node);
        valptr = FadType(num_dof, xT_constView[eqID[n]]);
        valptr.setUpdateValue(!workset.ignore_residual);
        valptr.fastAccessDx(firstunk + n) = workset.j_coeff;
      }
      eq += this->numNodeVar;
      for (int level = 0; level < this->numLevels; level++) {
        for (int j = eq; j < eq+this->numVectorLevelVar; j++) {
          for (int dim = 0; dim < this->numDims; ++dim, ++n) {
            typename PHAL::Ref<ScalarT>::type valptr = (this->val[j])(cell,node,level,dim);
            valptr = FadType(num_dof, xT_constView[eqID[n]]);
            valptr.setUpdateValue(!workset.ignore_residual);
            valptr.fastAccessDx(firstunk + n) = workset.j_coeff;
          }
        }
        for (int j = eq+this->numVectorLevelVar;
                 j < eq+this->numVectorLevelVar+this->numScalarLevelVar; ++j,++n) {
          typename PHAL::Ref<ScalarT>::type valptr = (this->val[j])(cell,node,level);
          valptr = FadType(num_dof, xT_constView[eqID[n]]);
          valptr.setUpdateValue(!workset.ignore_residual);
          valptr.fastAccessDx(firstunk + n) = workset.j_coeff;
        }
      }
      eq += this->numVectorLevelVar+this->numScalarLevelVar;
      for (int level = 0; level < this->numLevels; ++level) {
        for (int j = eq; j < eq+this->numTracerVar; ++j, ++n) {
          typename PHAL::Ref<ScalarT>::type valptr = (this->val[j])(cell,node,level);
          valptr = FadType(num_dof, xT_constView[eqID[n]]);
          valptr.setUpdateValue(!workset.ignore_residual);
          valptr.fastAccessDx(firstunk + n) = workset.j_coeff;
        }
      }
      eq += this->numTracerVar;

      if (workset.transientTerms) {
        int n = 0, eq = 0;
        for (int j = eq; j < eq+this->numNodeVar; ++j, ++n) {
          typename PHAL::Ref<ScalarT>::type valptr = (this->val_dot[j])(cell,node);
          valptr = FadType(num_dof, xdotT_constView[eqID[n]]);
          valptr.fastAccessDx(firstunk + n) = workset.m_coeff;
        }
        eq += this->numNodeVar;
        for (int level = 0; level < this->numLevels; level++) {
          for (int j = eq; j < eq+this->numVectorLevelVar; j++) {
            for (int dim = 0; dim < this->numDims; ++dim, ++n) {
              typename PHAL::Ref<ScalarT>::type valptr = (this->val_dot[j])(cell,node,level,dim);
              valptr = FadType(num_dof, xdotT_constView[eqID[n]]);
              valptr.fastAccessDx(firstunk + n) = workset.m_coeff;
            }
          }
          for (int j = eq+this->numVectorLevelVar;
                   j < eq+this->numVectorLevelVar+this->numScalarLevelVar; j++,++n) {
            typename PHAL::Ref<ScalarT>::type valptr = (this->val_dot[j])(cell,node,level);
            valptr = FadType(num_dof, xdotT_constView[eqID[n]]);
            valptr.fastAccessDx(firstunk + n) = workset.m_coeff;
          }
        }
        eq += this->numVectorLevelVar+this->numScalarLevelVar;
        for (int level = 0; level < this->numLevels; ++level) {
          for (int j = eq; j < eq+this->numTracerVar; ++j, ++n) {
            typename PHAL::Ref<ScalarT>::type valptr = (this->val_dot[j])(cell,node,level);
            valptr = FadType(num_dof, xdotT_constView[eqID[n]]);
            valptr.fastAccessDx(firstunk + n) = workset.m_coeff;
          }
        }
        eq += this->numTracerVar;
      }
    }
  }

/*#else
 xT_constView = xT->get1dView();
 xdotT_constView = xdotT->get1dView();
 ignore_residual=workset.ignore_residual;

 j_coeff=workset.j_coeff;
 m_coeff=workset.m_coeff; 

 wsID_kokkos=workset.wsElNodeEqID_kokkos;

 if (workset.transientTerms) 
     Kokkos::parallel_for(GatherSolution_transientTerms_Policy(0,workset.numCells),*this);
 else
     Kokkos::parallel_for(GatherSolution_Policy(0,workset.numCells),*this);

#endif
*/
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
  Teuchos::RCP<const Tpetra_Vector> xT = workset.xT;
  Teuchos::RCP<const Tpetra_Vector> xdotT = workset.xdotT;
  Teuchos::RCP<const Tpetra_MultiVector> VxT = workset.VxT;
  Teuchos::RCP<const Tpetra_MultiVector> VxdotT = workset.VxdotT;

  //get const views of xT and xdotT  
  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();
  Teuchos::ArrayRCP<const ST> xdotT_constView = xdotT->get1dView();

  TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "no impl");
 
  Teuchos::RCP<ParamVec> params = workset.params;
  int num_cols_tot = workset.param_offset + workset.num_cols_p;

  for (int cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

    for (int node = 0; node < this->numNodes; ++node) {
      const Teuchos::ArrayRCP<int>& eqID  = nodeID[node];
      int n = 0, eq = 0;
      for (int j = eq; j < eq+this->numNodeVar; j++, ++n) {
        typename PHAL::Ref<ScalarT>::type valptr = (this->val[j])(cell,node);
        if (VxT != Teuchos::null && workset.j_coeff != 0.0) {
          valptr = TanFadType(num_cols_tot, xT_constView[eqID[n]]);
          for (int k=0; k<workset.num_cols_x; k++)
            valptr.fastAccessDx(k) = workset.j_coeff*VxT->getData(k)[eqID[n]];
        }
        else
          valptr = TanFadType(xT_constView[eqID[n]]);
      }
      eq += this->numNodeVar;
      for (int level = 0; level < this->numLevels; level++) {
        for (int j = eq; j < eq+this->numVectorLevelVar; j++) {
          for (int dim = 0; dim < this->numDims; ++dim, ++n) {
            typename PHAL::Ref<ScalarT>::type valptr = (this->val[j])(cell,node,level,dim);
            if (VxT != Teuchos::null && workset.j_coeff != 0.0) {
              valptr = TanFadType(num_cols_tot, xT_constView[eqID[n]]);
              for (int k=0; k<workset.num_cols_x; k++)
                valptr.fastAccessDx(k) = workset.j_coeff*VxT->getData(k)[eqID[n]];
            }
            else
              valptr = TanFadType(xT_constView[eqID[n]]);
          }
        }
        for (int j = eq+this->numVectorLevelVar;
                 j < eq+this->numVectorLevelVar+this->numScalarLevelVar; j++, ++n) {
          typename PHAL::Ref<ScalarT>::type valptr = (this->val[j])(cell,node,level);
          if (VxT != Teuchos::null && workset.j_coeff != 0.0) {
            valptr = TanFadType(num_cols_tot, xT_constView[eqID[n]]);
            for (int k=0; k<workset.num_cols_x; k++)
              valptr.fastAccessDx(k) = workset.j_coeff*VxT->getData(k)[eqID[n]];
          }
          else
            valptr = TanFadType(xT_constView[eqID[n]]);
        }
      }
     eq += this->numVectorLevelVar+this->numScalarLevelVar;
      for (int level = 0; level < this->numLevels; ++level) {
        for (int j = eq; j < eq+this->numTracerVar; ++j, ++n) {
          typename PHAL::Ref<ScalarT>::type valptr = (this->val[j])(cell,node,level);
          if (VxT != Teuchos::null && workset.j_coeff != 0.0) {
            valptr = TanFadType(num_cols_tot, xT_constView[eqID[n]]);
            for (int k=0; k<workset.num_cols_x; k++)
              valptr.fastAccessDx(k) = workset.j_coeff*VxT->getData(k)[eqID[n]];
          }
          else
            valptr = TanFadType(xT_constView[eqID[n]]);
        }
      }
      eq += this->numTracerVar;
      if (workset.transientTerms) {
        int n = 0, eq = 0;
        for (int j = eq; j < eq+this->numNodeVar; j++, ++n) {
          typename PHAL::Ref<ScalarT>::type valptr = (this->val_dot[j])(cell,node);
          if (VxdotT != Teuchos::null && workset.m_coeff != 0.0) {
            valptr = TanFadType(num_cols_tot, xdotT_constView[eqID[n]]);
            for (int k=0; k<workset.num_cols_x; k++)
              valptr.fastAccessDx(k) =
                workset.m_coeff*VxdotT->getData(k)[eqID[n]];
          }
          else
            valptr = TanFadType(xdotT_constView[eqID[n]]);
        }
        eq += this->numNodeVar;
        for (int level = 0; level < this->numLevels; level++) {
          for (int j = eq; j < eq+this->numVectorLevelVar; j++) {
            for (int dim = 0; dim < this->numDims; ++dim, ++n) {
              typename PHAL::Ref<ScalarT>::type valptr = (this->val_dot[j])(cell,node,level,dim);
              if (VxdotT != Teuchos::null && workset.m_coeff != 0.0) {
                valptr = TanFadType(num_cols_tot, xdotT_constView[eqID[n]]);
                for (int k=0; k<workset.num_cols_x; k++)
                  valptr.fastAccessDx(k) =
                    workset.m_coeff*VxdotT->getData(k)[eqID[n]];
              }
              else
                valptr = TanFadType(xdotT_constView[eqID[n]]);
            }
          }
          for (int j = eq+this->numVectorLevelVar;
                   j < eq+this->numScalarLevelVar+this->numScalarLevelVar; j++,++n) {
            typename PHAL::Ref<ScalarT>::type valptr = (this->val_dot[j])(cell,node,level);
            if (VxdotT != Teuchos::null && workset.m_coeff != 0.0) {
              valptr = TanFadType(num_cols_tot, xdotT_constView[eqID[n]]);
              for (int k=0; k<workset.num_cols_x; k++)
                valptr.fastAccessDx(k) =
                  workset.m_coeff*VxdotT->getData(k)[eqID[n]];
            }
            else
              valptr = TanFadType(xdotT_constView[eqID[n]]);
          }
        }
       eq += this->numVectorLevelVar+this->numScalarLevelVar;
        for (int level = 0; level < this->numLevels; ++level) {
          for (int j = eq; j < eq+this->numTracerVar; ++j, ++n) {
            typename PHAL::Ref<ScalarT>::type valptr = (this->val_dot[j])(cell,node,level);
            if (VxdotT != Teuchos::null && workset.m_coeff != 0.0) {
              valptr = TanFadType(num_cols_tot, xdotT_constView[eqID[n]]);
              for (int k=0; k<workset.num_cols_x; k++)
                valptr.fastAccessDx(k) =
                  workset.m_coeff*VxdotT->getData(k)[eqID[n]];
            }
            else
              valptr = TanFadType(xdotT_constView[eqID[n]]);
          }
        }
        eq += this->numTracerVar;
      }
    }
  }

}

#ifdef ALBANY_ENSEMBLE 
// **********************************************************************
// Specialization: Multi-point Residual
// **********************************************************************

template<typename Traits>
GatherSolution<PHAL::AlbanyTraits::MPResidual, Traits>::
GatherSolution(const Teuchos::ParameterList& p,
               const Teuchos::RCP<Aeras::Layouts>& dl) :
  GatherSolutionBase<PHAL::AlbanyTraits::MPResidual, Traits>(p,dl)
{}

template<typename Traits>
void GatherSolution<PHAL::AlbanyTraits::MPResidual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{ 
  Teuchos::RCP<const Stokhos::ProductEpetraVector> x = workset.mp_x;
  Teuchos::RCP<const Stokhos::ProductEpetraVector> xdot = workset.mp_xdot;

  //Get const view of xT and xdotT 
//  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();
//  Teuchos::ArrayRCP<const ST> xdotT_constView = xdotT->get1dView();

  int nblock = x->size();
  for (int cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

    for (int node = 0; node < this->numNodes; ++node) {
      const Teuchos::ArrayRCP<int>& eqID  = nodeID[node];
      int n = 0, eq = 0;
      for (int j = eq; j < eq+this->numNodeVar; ++j, ++n) {
//        (this->val    [j])(cell,node) = xT_constView[eqID[n]];
        typename PHAL::Ref<ScalarT>::type valptr = (this->val[j])(cell,node);
        valptr.reset(nblock);
        valptr.copyForWrite();
        for (int block=0; block<nblock; block++)
          valptr.fastAccessCoeff(block) = (*x)[block][nodeID[node][n]];
      }
      eq += this->numNodeVar;
      for (int level = 0; level < this->numLevels; level++) { 
        for (int j = eq; j < eq+this->numVectorLevelVar; ++j) {
          for (int dim = 0; dim < this->numDims; ++dim, ++n) {
//            (this->val    [j])(cell,node,level,dim) = xT_constView   [eqID[n]];
            typename PHAL::Ref<ScalarT>::type valptr = (this->val[j])(cell,node,level,dim);
            valptr.reset(nblock);
            valptr.copyForWrite();
            for (int block=0; block<nblock; block++)
              valptr.fastAccessCoeff(block) = (*x)[block][nodeID[node][n]];
          }
        }
        for (int j = eq+this->numVectorLevelVar; 
                 j < eq+this->numVectorLevelVar+this->numScalarLevelVar; ++j, ++n) {
//          (this->val    [j])(cell,node,level) = xT_constView   [eqID[n]];
          typename PHAL::Ref<ScalarT>::type valptr = (this->val[j])(cell,node,level);
          valptr.reset(nblock);
          valptr.copyForWrite();
          for (int block=0; block<nblock; block++)
            valptr.fastAccessCoeff(block) = (*x)[block][nodeID[node][n]];
        }
      }
      eq += this->numScalarLevelVar + this->numVectorLevelVar;
      for (int level = 0; level < this->numLevels; ++level) { 
        for (int j = eq; j < eq+this->numTracerVar; ++j, ++n) {
//          (this->val    [j])(cell,node,level) = xT_constView[eqID[n]];
          typename PHAL::Ref<ScalarT>::type valptr = (this->val[j])(cell,node,level);
          valptr.reset(nblock);
          valptr.copyForWrite();
          for (int block=0; block<nblock; block++)
            valptr.fastAccessCoeff(block) = (*x)[block][nodeID[node][n]];
        }
      }
      eq += this->numTracerVar;

      if (workset.transientTerms) {
        int n = 0, eq = 0;
        for (int j = eq; j < eq+this->numNodeVar; ++j, ++n) {
//        (this->val_dot[j])(cell,node) = xdotT_constView[eqID[n]];
          typename PHAL::Ref<ScalarT>::type valptr = (this->val_dot[j])(cell,node);
          valptr.reset(nblock);
          valptr.copyForWrite();
          for (int block=0; block<nblock; block++)
            valptr.fastAccessCoeff(block) = (*xdot)[block][nodeID[node][n]];
        }
        eq += this->numNodeVar;
        for (int level = 0; level < this->numLevels; level++) { 
          for (int j = eq; j < eq+this->numVectorLevelVar; j++) {
            for (int dim = 0; dim < this->numDims; ++dim, ++n) {
//            (this->val_dot[j])(cell,node,level,dim) = xdotT_constView[eqID[n]];
              typename PHAL::Ref<ScalarT>::type valptr = (this->val_dot[j])(cell,node,level,dim);
              valptr.reset(nblock);
              valptr.copyForWrite();
              for (int block=0; block<nblock; block++)
                valptr.fastAccessCoeff(block) = (*xdot)[block][nodeID[node][n]];
            }
          }
          for (int j = eq+this->numVectorLevelVar; 
                   j < eq+this->numVectorLevelVar+this->numScalarLevelVar; j++,++n) {
//          (this->val_dot[j])(cell,node,level) = xdotT_constView[eqID[n]];
            typename PHAL::Ref<ScalarT>::type valptr = (this->val_dot[j])(cell,node,level);
            valptr.reset(nblock);
            valptr.copyForWrite();
            for (int block=0; block<nblock; block++)
              valptr.fastAccessCoeff(block) = (*xdot)[block][nodeID[node][n]];
          }
        }
        eq += this->numVectorLevelVar+this->numScalarLevelVar;
        for (int level = 0; level < this->numLevels; ++level) { 
          for (int j = eq; j < eq+this->numTracerVar; ++j, ++n) {
//          (this->val_dot[j])(cell,node,level) = xdotT_constView[eqID[n]];
            typename PHAL::Ref<ScalarT>::type valptr = (this->val_dot[j])(cell,node,level);
            valptr.reset(nblock);
            valptr.copyForWrite();
            for (int block=0; block<nblock; block++)
              valptr.fastAccessCoeff(block) = (*xdot)[block][nodeID[node][n]];
          }
        }
        eq += this->numTracerVar;
      }
    }
  }
}

// **********************************************************************
// Specialization: Multi-point Jacobian
// **********************************************************************

template<typename Traits>
GatherSolution<PHAL::AlbanyTraits::MPJacobian, Traits>::
GatherSolution(const Teuchos::ParameterList& p,
               const Teuchos::RCP<Aeras::Layouts>& dl) :
  GatherSolutionBase<PHAL::AlbanyTraits::MPJacobian, Traits>(p,dl)
{}

// **********************************************************************
template<typename Traits>
void GatherSolution<PHAL::AlbanyTraits::MPJacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  const Teuchos::RCP<const Stokhos::ProductEpetraVector>    x = workset.mp_x;
  const Teuchos::RCP<const Stokhos::ProductEpetraVector> xdot = workset.mp_xdot;

  //get const view of xT and xdotT   
//  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();
//  Teuchos::ArrayRCP<const ST> xdotT_constView = xdotT->get1dView();

  int nblock = x->size();
  for (int cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];
    const int neq = nodeID[0].size();
    const int num_dof = neq * this->numNodes;
    //JF  move above 2 lines into node loop as in PHAL case?

    for (int node = 0; node < this->numNodes; ++node) {
      const Teuchos::ArrayRCP<int>& eqID  = nodeID[node];
      const int firstunk = neq * node;
      int n = 0, eq = 0;
      for (int j = eq; j < eq+this->numNodeVar; ++j, ++n) {
        typename PHAL::Ref<ScalarT>::type valptr = (this->val[j])(cell,node);
//        valptr = FadType(num_dof, xT_constView[eqID[n]]);
        valptr = MPFadType(num_dof, 0.0);
        valptr.setUpdateValue(!workset.ignore_residual);
        valptr.fastAccessDx(firstunk + n) = workset.j_coeff;
        valptr.val().reset(nblock);
        valptr.val().copyForWrite();
        for (int block=0; block<nblock; block++)
          valptr.val().fastAccessCoeff(block) = (*x)[block][nodeID[node][n]];
      }
      eq += this->numNodeVar;
      for (int level = 0; level < this->numLevels; level++) { 
        for (int j = eq; j < eq+this->numVectorLevelVar; j++) {
          for (int dim = 0; dim < this->numDims; ++dim, ++n) {
            typename PHAL::Ref<ScalarT>::type valptr = (this->val[j])(cell,node,level,dim);
//            valptr = FadType(num_dof, xT_constView[eqID[n]]);
            valptr = MPFadType(num_dof, 0.0);
            valptr.setUpdateValue(!workset.ignore_residual);
            valptr.fastAccessDx(firstunk + n) = workset.j_coeff;
            valptr.val().reset(nblock);
            valptr.val().copyForWrite();
            for (int block=0; block<nblock; block++)
              valptr.val().fastAccessCoeff(block) = (*x)[block][nodeID[node][n]];
          } 
        }
        for (int j = eq+this->numVectorLevelVar; 
                 j < eq+this->numVectorLevelVar+this->numScalarLevelVar; ++j,++n) {
          typename PHAL::Ref<ScalarT>::type valptr = (this->val[j])(cell,node,level);
//          valptr = FadType(num_dof, xT_constView[eqID[n]]);
          valptr = MPFadType(num_dof, 0.0);
          valptr.setUpdateValue(!workset.ignore_residual);
          valptr.fastAccessDx(firstunk + n) = workset.j_coeff;
          valptr.val().reset(nblock);
          valptr.val().copyForWrite();
          for (int block=0; block<nblock; block++)
            valptr.val().fastAccessCoeff(block) = (*x)[block][nodeID[node][n]];
        }
      }
      eq += this->numVectorLevelVar+this->numScalarLevelVar;
      for (int level = 0; level < this->numLevels; ++level) { 
        for (int j = eq; j < eq+this->numTracerVar; ++j, ++n) {
          typename PHAL::Ref<ScalarT>::type valptr = (this->val[j])(cell,node,level);
//          valptr = FadType(num_dof, xT_constView[eqID[n]]);
          valptr = MPFadType(num_dof, 0.0);
          valptr.setUpdateValue(!workset.ignore_residual);
          valptr.fastAccessDx(firstunk + n) = workset.j_coeff;
          valptr.val().reset(nblock);
          valptr.val().copyForWrite();
          for (int block=0; block<nblock; block++)
            valptr.val().fastAccessCoeff(block) = (*x)[block][nodeID[node][n]];
        }
      }
      eq += this->numTracerVar;

      if (workset.transientTerms) {
        int n = 0, eq = 0;
        for (int j = eq; j < eq+this->numNodeVar; ++j, ++n) {
          typename PHAL::Ref<ScalarT>::type valptr = (this->val_dot[j])(cell,node);
          valptr = MPFadType(num_dof, 0.0);
          valptr.fastAccessDx(firstunk + n) = workset.m_coeff;
          valptr.val().reset(nblock);
          valptr.val().copyForWrite();
          for (int block=0; block<nblock; block++)
            valptr.val().fastAccessCoeff(block) = (*xdot)[block][nodeID[node][n]];
        }
        eq += this->numNodeVar;
        for (int level = 0; level < this->numLevels; level++) { 
          for (int j = eq; j < eq+this->numVectorLevelVar; j++) {
            for (int dim = 0; dim < this->numDims; ++dim, ++n) {
              typename PHAL::Ref<ScalarT>::type valptr = (this->val_dot[j])(cell,node,level,dim);
              valptr = MPFadType(num_dof, 0.0);
              valptr.fastAccessDx(firstunk + n) = workset.m_coeff;
              valptr.val().reset(nblock);
              valptr.val().copyForWrite();
              for (int block=0; block<nblock; block++)
                valptr.val().fastAccessCoeff(block) = (*xdot)[block][nodeID[node][n]];
            }
          }
          for (int j = eq+this->numVectorLevelVar; 
                   j < eq+this->numVectorLevelVar+this->numScalarLevelVar; j++,++n) {
            typename PHAL::Ref<ScalarT>::type valptr = (this->val_dot[j])(cell,node,level);
            valptr = MPFadType(num_dof, 0.0);
            valptr.fastAccessDx(firstunk + n) = workset.m_coeff;
            valptr.val().reset(nblock);
            valptr.val().copyForWrite();
            for (int block=0; block<nblock; block++)
              valptr.val().fastAccessCoeff(block) = (*xdot)[block][nodeID[node][n]];
          }
        }
        eq += this->numVectorLevelVar+this->numScalarLevelVar;
        for (int level = 0; level < this->numLevels; ++level) { 
          for (int j = eq; j < eq+this->numTracerVar; ++j, ++n) {
            typename PHAL::Ref<ScalarT>::type valptr = (this->val_dot[j])(cell,node,level);
            valptr = MPFadType(num_dof, 0.0);
            valptr.fastAccessDx(firstunk + n) = workset.m_coeff;
            valptr.val().reset(nblock);
            valptr.val().copyForWrite();
            for (int block=0; block<nblock; block++)
              valptr.val().fastAccessCoeff(block) = (*xdot)[block][nodeID[node][n]];
          }
        }
        eq += this->numTracerVar;
      }
    }
  }
}
#endif

}

