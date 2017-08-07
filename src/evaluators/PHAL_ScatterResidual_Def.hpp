//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#ifdef ALBANY_TIMER
#include <chrono>
#endif
#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Albany_Utils.hpp"

// **********************************************************************
// Base Class Generic Implemtation
// **********************************************************************
namespace PHAL {

template<typename EvalT, typename Traits>
ScatterResidualBase<EvalT, Traits>::
ScatterResidualBase(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
{
  std::string fieldName;
  if (p.isType<std::string>("Scatter Field Name"))
    fieldName = p.get<std::string>("Scatter Field Name");
  else fieldName = "Scatter";

  scatter_operation = Teuchos::rcp(new PHX::Tag<ScalarT>
    (fieldName, dl->dummy));

  const Teuchos::ArrayRCP<std::string>& names =
    p.get< Teuchos::ArrayRCP<std::string> >("Residual Names");


  tensorRank = p.get<int>("Tensor Rank");

  // scalar
  if (tensorRank == 0 ) {
    numFieldsBase = names.size();
    const std::size_t num_val = numFieldsBase;
    val.resize(num_val);
    for (std::size_t eq = 0; eq < numFieldsBase; ++eq) {
      PHX::MDField<ScalarT const,Cell,Node> mdf(names[eq],dl->node_scalar);
      val[eq] = mdf;
      this->addDependentField(val[eq]);
    }
  }
  // vector
  else
  if (tensorRank == 1 ) {
    PHX::MDField<ScalarT const,Cell,Node,Dim> mdf(names[0],dl->node_vector);
    valVec= mdf;
    this->addDependentField(valVec);
    numFieldsBase = dl->node_vector->dimension(2);
  }
  // tensor
  else
  if (tensorRank == 2 ) {
    PHX::MDField<ScalarT const,Cell,Node,Dim,Dim> mdf(names[0],dl->node_tensor);
    valTensor = mdf;
    this->addDependentField(valTensor);
    numFieldsBase = (dl->node_tensor->dimension(2))*(dl->node_tensor->dimension(3));
  }

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  if (tensorRank == 0)
    val_kokkos.resize(numFieldsBase);
#endif

  if (p.isType<int>("Offset of First DOF"))
    offset = p.get<int>("Offset of First DOF");
  else offset = 0;

  this->addEvaluatedField(*scatter_operation);

  this->setName(fieldName+PHX::typeAsString<EvalT>());
}

// **********************************************************************
template<typename EvalT, typename Traits>
void ScatterResidualBase<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  if (tensorRank == 0) {
    for (std::size_t eq = 0; eq < numFieldsBase; ++eq)
      this->utils.setFieldData(val[eq],fm);
    numNodes = val[0].dimension(1);
  }
  else 
  if (tensorRank == 1) {
    this->utils.setFieldData(valVec,fm);
    numNodes = valVec.dimension(1);
  }
  else 
  if (tensorRank == 2) {
    this->utils.setFieldData(valTensor,fm);
    numNodes = valTensor.dimension(1);
  }
}

// **********************************************************************
// Specialization: Residual
// **********************************************************************
template<typename Traits>
ScatterResidual<PHAL::AlbanyTraits::Residual,Traits>::
ScatterResidual(const Teuchos::ParameterList& p,
                       const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidualBase<PHAL::AlbanyTraits::Residual,Traits>(p,dl),
  numFields(ScatterResidualBase<PHAL::AlbanyTraits::Residual,Traits>::numFieldsBase) {}

// **********************************************************************
// Kokkos kernels
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
template<typename Traits>
KOKKOS_INLINE_FUNCTION
void ScatterResidual<PHAL::AlbanyTraits::Residual,Traits>::
operator() (const PHAL_ScatterResRank0_Tag&, const int& cell) const
{
  for (std::size_t node = 0; node < this->numNodes; node++)
    for (std::size_t eq = 0; eq < numFields; eq++) {
      const LO id = nodeID(cell,node,this->offset + eq);
      Kokkos::atomic_fetch_add(&fT_kokkos(id), val_kokkos[eq](cell,node));
    }
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void ScatterResidual<PHAL::AlbanyTraits::Residual,Traits>::
operator() (const PHAL_ScatterResRank1_Tag&, const int& cell) const
{
  for (std::size_t node = 0; node < this->numNodes; node++)
    for (std::size_t eq = 0; eq < numFields; eq++) {
      const LO id = nodeID(cell,node,this->offset + eq);
      Kokkos::atomic_fetch_add(&fT_kokkos(id), this->valVec(cell,node,eq));
    }
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void ScatterResidual<PHAL::AlbanyTraits::Residual,Traits>::
operator() (const PHAL_ScatterResRank2_Tag&, const int& cell) const
{
  for (std::size_t node = 0; node < this->numNodes; node++)
    for (std::size_t i = 0; i < numDims; i++)
      for (std::size_t j = 0; j < numDims; j++) {
        const LO id = nodeID(cell,node,this->offset + i*numDims + j);
        Kokkos::atomic_fetch_add(&fT_kokkos(id), this->valTensor(cell,node,i,j)); 
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
  Teuchos::RCP<Tpetra_Vector> fT = workset.fT;

  //get nonconst (read and write) view of fT
  Teuchos::ArrayRCP<ST> f_nonconstView = fT->get1dViewNonConst();

  if (this->tensorRank == 0) {
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      for (std::size_t node = 0; node < this->numNodes; ++node)
        for (std::size_t eq = 0; eq < numFields; eq++)
          f_nonconstView[nodeID(cell,node,this->offset + eq)] += (this->val[eq])(cell,node);
    }
  } else 
  if (this->tensorRank == 1) {
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      for (std::size_t node = 0; node < this->numNodes; ++node)
        for (std::size_t eq = 0; eq < numFields; eq++)
          f_nonconstView[nodeID(cell,node,this->offset + eq)] += (this->valVec)(cell,node,eq);
    }
  } else
  if (this->tensorRank == 2) {
    int numDims = this->valTensor.dimension(2);
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      for (std::size_t node = 0; node < this->numNodes; ++node)
        for (std::size_t i = 0; i < numDims; i++)
          for (std::size_t j = 0; j < numDims; j++)
            f_nonconstView[nodeID(cell,node,this->offset + i*numDims + j)] += (this->valTensor)(cell,node,i,j);
  
    }
  }

#else
#ifdef ALBANY_TIMER
  auto start = std::chrono::high_resolution_clock::now();
#endif
  // Get map for local data structures
  nodeID = workset.wsElNodeEqID;

  // Get Tpetra vector view from a specific device
  auto fT_2d = workset.fT->template getLocalView<PHX::Device>();
  fT_kokkos = Kokkos::subview(fT_2d, Kokkos::ALL(), 0);

  if (this->tensorRank == 0) {
    // Get MDField views from std::vector
    for (int i = 0; i < numFields; i++)
      val_kokkos[i] = this->val[i].get_view();

    Kokkos::parallel_for(PHAL_ScatterResRank0_Policy(0,workset.numCells),*this);
    cudaCheckError();
  }
  else if (this->tensorRank == 1) {
    Kokkos::parallel_for(PHAL_ScatterResRank1_Policy(0,workset.numCells),*this);
    cudaCheckError();
  }
  else if (this->tensorRank == 2) {
    numDims = this->valTensor.dimension(2);
    Kokkos::parallel_for(PHAL_ScatterResRank2_Policy(0,workset.numCells),*this);
    cudaCheckError();
  }

#ifdef ALBANY_TIMER
  PHX::Device::fence();
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  long long millisec= std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
  std::cout<< "Scatter Residual time = "  << millisec << "  "  << microseconds << std::endl;
#endif
#endif
}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************

template<typename Traits>
ScatterResidual<PHAL::AlbanyTraits::Jacobian, Traits>::
ScatterResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidualBase<PHAL::AlbanyTraits::Jacobian,Traits>(p,dl),
  numFields(ScatterResidualBase<PHAL::AlbanyTraits::Jacobian,Traits>::numFieldsBase) {}

// **********************************************************************
// Kokkos kernels
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
template<typename Traits>
KOKKOS_INLINE_FUNCTION
void ScatterResidual<PHAL::AlbanyTraits::Jacobian,Traits>::
operator() (const PHAL_ScatterResRank0_Tag&, const int& cell) const
{
  for (std::size_t node = 0; node < this->numNodes; node++)
    for (std::size_t eq = 0; eq < numFields; eq++) {
      const LO id = nodeID(cell,node,this->offset + eq);
      Kokkos::atomic_fetch_add(&fT_kokkos(id), (val_kokkos[eq](cell,node)).val());
    }
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void ScatterResidual<PHAL::AlbanyTraits::Jacobian,Traits>::
operator() (const PHAL_ScatterJacRank0_Adjoint_Tag&, const int& cell) const
{
  //const int neq = nodeID.dimension(2);
  //const int nunk = neq*this->numNodes;
  // Irina TOFIX replace 500 with nunk with Kokkos::malloc is available
  LO colT[500];
  LO rowT;
  //std::vector<LO> colT(nunk);
  //colT=(LO*) Kokkos::cuda_malloc<PHX::Device>(nunk*sizeof(LO));

  if (nunk>500) Kokkos::abort("ERROR (ScatterResidual): nunk > 500");

  for (int node_col=0; node_col<this->numNodes; node_col++) {
    for (int eq_col=0; eq_col<neq; eq_col++) {
      colT[neq * node_col + eq_col] =  nodeID(cell,node_col,eq_col);
    }
  }

  for (int node = 0; node < this->numNodes; ++node) {
    for (int eq = 0; eq < numFields; eq++) {
      rowT = nodeID(cell,node,this->offset + eq);
      auto valptr = val_kokkos[eq](cell,node);
      for (int lunk=0; lunk<nunk; lunk++) {
        ST val = valptr.fastAccessDx(lunk);
        JacT_kokkos.sumIntoValues(colT[lunk], &rowT, 1, &val, false, true); 
      }
    }
  }
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void ScatterResidual<PHAL::AlbanyTraits::Jacobian,Traits>::
operator() (const PHAL_ScatterJacRank0_Tag&, const int& cell) const
{
  //const int neq = nodeID.dimension(2);
  //const int nunk = neq*this->numNodes;
  // Irina TOFIX replace 500 with nunk with Kokkos::malloc is available
  //colT=(LO*) Kokkos::cuda_malloc<PHX::Device>(nunk*sizeof(LO));
  LO rowT;
  LO colT[500];
  ST vals[500];
  //std::vector<LO> colT(nunk);
  //std::vector<ST> vals(nunk);

  if (nunk>500) Kokkos::abort("ERROR (ScatterResidual): nunk > 500");

  for (int node_col=0, i=0; node_col<this->numNodes; node_col++) {
    for (int eq_col=0; eq_col<neq; eq_col++) {
      colT[neq * node_col + eq_col] = nodeID(cell,node_col,eq_col);
    }
  }

  for (int node = 0; node < this->numNodes; ++node) {
    for (int eq = 0; eq < numFields; eq++) {
      rowT = nodeID(cell,node,this->offset + eq);
      auto valptr = val_kokkos[eq](cell,node);
      for (int i = 0; i < nunk; ++i) vals[i] = valptr.fastAccessDx(i);
      JacT_kokkos.sumIntoValues(rowT, colT, nunk, vals, false, true);
    }
  }
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void ScatterResidual<PHAL::AlbanyTraits::Jacobian,Traits>::
operator() (const PHAL_ScatterResRank1_Tag&, const int& cell) const
{
  for (std::size_t node = 0; node < this->numNodes; node++)
    for (std::size_t eq = 0; eq < numFields; eq++) {
      const LO id = nodeID(cell,node,this->offset + eq);
      Kokkos::atomic_fetch_add(&fT_kokkos(id), (this->valVec(cell,node,eq)).val());
    }
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void ScatterResidual<PHAL::AlbanyTraits::Jacobian,Traits>::
operator() (const PHAL_ScatterJacRank1_Adjoint_Tag&, const int& cell) const
{
  //const int neq = nodeID.dimension(2);
  //const int nunk = neq*this->numNodes;
  // Irina TOFIX replace 500 with nunk with Kokkos::malloc is available
  LO colT[500];
  LO rowT;
  ST vals[500];
  //std::vector<ST> vals(nunk);
  //std::vector<LO> colT(nunk);
  //colT=(LO*) Kokkos::malloc<PHX::Device>(nunk*sizeof(LO));

  if (nunk>500) Kokkos::abort("ERROR (ScatterResidual): nunk > 500");

  for (int node_col=0, i=0; node_col<this->numNodes; node_col++) {
    for (int eq_col=0; eq_col<neq; eq_col++) {
      colT[neq * node_col + eq_col] = nodeID(cell,node_col,eq_col);
    }
  }

  for (int node = 0; node < this->numNodes; ++node) {
    for (int eq = 0; eq < numFields; eq++) {
      rowT = nodeID(cell,node,this->offset + eq);
      if (((this->valVec)(cell,node,eq)).hasFastAccess()) {
        for (int lunk=0; lunk<nunk; lunk++){
          ST val = ((this->valVec)(cell,node,eq)).fastAccessDx(lunk);
          JacT_kokkos.sumIntoValues(colT[lunk], &rowT, 1, &val, false, true);
        }
      }//has fast access
    }
  }
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void ScatterResidual<PHAL::AlbanyTraits::Jacobian,Traits>::
operator() (const PHAL_ScatterJacRank1_Tag&, const int& cell) const
{
  //const int neq = nodeID.dimension(2);
  //const int nunk = neq*this->numNodes;
  // Irina TOFIX replace 500 with nunk with Kokkos::malloc is available
  LO colT[500];
  LO rowT;
  ST vals[500];
  //std::vector<LO> colT(nunk);
  //colT=(LO*) Kokkos::malloc<PHX::Device>(nunk*sizeof(LO));
  //std::vector<ST> vals(nunk);

  if (nunk>500) Kokkos::abort ("ERROR (ScatterResidual): nunk > 500");

  for (int node_col=0, i=0; node_col<this->numNodes; node_col++) {
    for (int eq_col=0; eq_col<neq; eq_col++) {
      colT[neq * node_col + eq_col] = nodeID(cell,node_col,eq_col);
    }
  }

  for (int node = 0; node < this->numNodes; ++node) {
    for (int eq = 0; eq < numFields; eq++) {
      rowT = nodeID(cell,node,this->offset + eq);
      if (((this->valVec)(cell,node,eq)).hasFastAccess()) {
        for (int i = 0; i < nunk; ++i) vals[i] = (this->valVec)(cell,node,eq).fastAccessDx(i);
        JacT_kokkos.sumIntoValues(rowT, colT, nunk, vals, false, true);
      }
    }
  }
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void ScatterResidual<PHAL::AlbanyTraits::Jacobian,Traits>::
operator() (const PHAL_ScatterResRank2_Tag&, const int& cell) const
{
  for (std::size_t node = 0; node < this->numNodes; node++)
    for (std::size_t i = 0; i < numDims; i++)
      for (std::size_t j = 0; j < numDims; j++) {
        const LO id = nodeID(cell,node,this->offset + i*numDims + j);
        Kokkos::atomic_fetch_add(&fT_kokkos(id), (this->valTensor(cell,node,i,j)).val()); 
      }
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void ScatterResidual<PHAL::AlbanyTraits::Jacobian,Traits>::
operator() (const PHAL_ScatterJacRank2_Adjoint_Tag&, const int& cell) const
{
  //const int neq = nodeID.dimension(2);
  //const int nunk = neq*this->numNodes;
  // Irina TOFIX replace 500 with nunk with Kokkos::malloc is available
  LO colT[500];
  LO rowT;
  //std::vector<LO> colT(nunk);
  //colT=(LO*) Kokkos::malloc<PHX::Device>(nunk*sizeof(LO));

  if (nunk>500) Kokkos::abort("ERROR (ScatterResidual): nunk > 500");

  for (int node_col=0, i=0; node_col<this->numNodes; node_col++) {
    for (int eq_col=0; eq_col<neq; eq_col++) {
      colT[neq * node_col + eq_col] = nodeID(cell,node_col,eq_col);
    }
  }

  for (int node = 0; node < this->numNodes; ++node) {
    for (int eq = 0; eq < numFields; eq++) {
      rowT = nodeID(cell,node,this->offset + eq);
      if (((this->valTensor)(cell,node, eq/numDims, eq%numDims)).hasFastAccess()) {
        for (int lunk=0; lunk<nunk; lunk++) {
          ST val = ((this->valTensor)(cell,node, eq/numDims, eq%numDims)).fastAccessDx(lunk);
          JacT_kokkos.sumIntoValues (colT[lunk], &rowT, 1, &val, false, true);
        }
      }//has fast access
    }
  }
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void ScatterResidual<PHAL::AlbanyTraits::Jacobian,Traits>::
operator() (const PHAL_ScatterJacRank2_Tag&, const int& cell) const
{
  //const int neq = nodeID.dimension(2);
  //const int nunk = neq*this->numNodes;
  // Irina TOFIX replace 500 with nunk with Kokkos::malloc is available
  LO colT[500];
  LO rowT;
  ST vals[500];
  //std::vector<LO> colT(nunk);
  //colT=(LO*) Kokkos::malloc<PHX::Device>(nunk*sizeof(LO));
  //std::vector<ST> vals(nunk);

  if (nunk>500) Kokkos::abort("ERROR (ScatterResidual): nunk > 500");

  for (int node_col=0, i=0; node_col<this->numNodes; node_col++) {
    for (int eq_col=0; eq_col<neq; eq_col++) {
      colT[neq * node_col + eq_col] = nodeID(cell,node_col,eq_col);
    }
  }

  for (int node = 0; node < this->numNodes; ++node) {
    for (int eq = 0; eq < numFields; eq++) {
      rowT = nodeID(cell,node,this->offset + eq);
      if (((this->valTensor)(cell,node, eq/numDims, eq%numDims)).hasFastAccess()) {
        for (int i = 0; i < nunk; ++i) vals[i] = (this->valTensor)(cell,node, eq/numDims, eq%numDims).fastAccessDx(i);
        JacT_kokkos.sumIntoValues(rowT, colT, nunk,  vals, false, true);
      }
    }
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
  Teuchos::RCP<Tpetra_Vector> fT = workset.fT;
  Teuchos::RCP<Tpetra_CrsMatrix> JacT = workset.JacT;
  const bool loadResid = Teuchos::nonnull(fT);
  Teuchos::Array<LO> colT;
  const int neq = nodeID.dimension(2);
  const int nunk = neq*this->numNodes;
  colT.resize(nunk);
  int numDims = 0;
  if (this->tensorRank==2) numDims = this->valTensor.dimension(2);

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    // Local Unks: Loop over nodes in element, Loop over equations per node
    for (unsigned int node_col=0, i=0; node_col<this->numNodes; node_col++){
      for (unsigned int eq_col=0; eq_col<neq; eq_col++) {
        colT[neq * node_col + eq_col] = nodeID(cell,node_col,eq_col);
      }
    }
    for (std::size_t node = 0; node < this->numNodes; ++node) {
      for (std::size_t eq = 0; eq < numFields; eq++) {
        typename PHAL::Ref<ScalarT const>::type
          valptr = (this->tensorRank == 0 ? this->val[eq](cell,node) :
                    this->tensorRank == 1 ? this->valVec(cell,node,eq) :
                    this->valTensor(cell,node, eq/numDims, eq%numDims));
        const LO rowT = nodeID(cell,node,this->offset + eq);
        if (loadResid)
          fT->sumIntoLocalValue(rowT, valptr.val());
        // Check derivative array is nonzero
        if (valptr.hasFastAccess()) {
          if (workset.is_adjoint) {
            // Sum Jacobian transposed
            for (unsigned int lunk = 0; lunk < nunk; lunk++)
              JacT->sumIntoLocalValues(
                colT[lunk], Teuchos::arrayView(&rowT, 1),
                Teuchos::arrayView(&(valptr.fastAccessDx(lunk)), 1));
          }
          else {
            // Sum Jacobian entries all at once
            JacT->sumIntoLocalValues(
              rowT, colT, Teuchos::arrayView(&(valptr.fastAccessDx(0)), nunk));
          }
        } // has fast access
      }
    }
  }

#else
#ifdef ALBANY_TIMER
  auto start = std::chrono::high_resolution_clock::now();
#endif
  // Get map for local data structures
  nodeID = workset.wsElNodeEqID;

  // Get dimensions
  neq = nodeID.dimension(2);
  nunk = neq*this->numNodes;

  // Get Tpetra vector view and local matrix
  const bool loadResid = Teuchos::nonnull(workset.fT);
  if (loadResid) {
    auto fT_2d = workset.fT->template getLocalView<PHX::Device>();
    fT_kokkos = Kokkos::subview(fT_2d, Kokkos::ALL(), 0);
  }
  JacT_kokkos = workset.JacT->getLocalMatrix();

  if (this->tensorRank == 0) {
    // Get MDField views from std::vector
    for (int i = 0; i < numFields; i++)
      val_kokkos[i] = this->val[i].get_view();

    if (loadResid) {
      Kokkos::parallel_for(PHAL_ScatterResRank0_Policy(0,workset.numCells),*this);
      cudaCheckError();
    }

    if (workset.is_adjoint) {
      Kokkos::parallel_for(PHAL_ScatterJacRank0_Adjoint_Policy(0,workset.numCells),*this);  
      cudaCheckError();
    }
    else {
      Kokkos::parallel_for(PHAL_ScatterJacRank0_Policy(0,workset.numCells),*this);
      cudaCheckError();
    }
  }
  else  if (this->tensorRank == 1) {
    if (loadResid) {
      Kokkos::parallel_for(PHAL_ScatterResRank1_Policy(0,workset.numCells),*this);
      cudaCheckError();
    }

    if (workset.is_adjoint) {
      Kokkos::parallel_for(PHAL_ScatterJacRank1_Adjoint_Policy(0,workset.numCells),*this);
      cudaCheckError();
    }
    else {
      Kokkos::parallel_for(PHAL_ScatterJacRank1_Policy(0,workset.numCells),*this);
      cudaCheckError();
    }
  }
  else if (this->tensorRank == 2) {
    numDims = this->valTensor.dimension(2);

    if (loadResid) {
      Kokkos::parallel_for(PHAL_ScatterResRank2_Policy(0,workset.numCells),*this);
      cudaCheckError();
    }

    if (workset.is_adjoint) {
      Kokkos::parallel_for(PHAL_ScatterJacRank2_Adjoint_Policy(0,workset.numCells),*this);
    }
    else {
      Kokkos::parallel_for(PHAL_ScatterJacRank2_Policy(0,workset.numCells),*this);
      cudaCheckError();
    }
  }

#ifdef ALBANY_TIMER
  PHX::Device::fence();
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  long long millisec= std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
  std::cout<< "Scatter Jacobian time = "  << millisec << "  "  << microseconds << std::endl;
#endif 
#endif
}

// **********************************************************************
// Specialization: Tangent
// **********************************************************************

template<typename Traits>
ScatterResidual<PHAL::AlbanyTraits::Tangent, Traits>::
ScatterResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidualBase<PHAL::AlbanyTraits::Tangent,Traits>(p,dl),
  numFields(ScatterResidualBase<PHAL::AlbanyTraits::Tangent,Traits>::numFieldsBase)
{
}

// **********************************************************************
template<typename Traits>
void ScatterResidual<PHAL::AlbanyTraits::Tangent, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  auto nodeID = workset.wsElNodeEqID;
  Teuchos::RCP<Tpetra_Vector> fT = workset.fT;
  Teuchos::RCP<Tpetra_MultiVector> JVT = workset.JVT;
  Teuchos::RCP<Tpetra_MultiVector> fpT = workset.fpT;

  int numDims = 0;
  if (this->tensorRank == 2) numDims = this->valTensor.dimension(2);

  for (std::size_t cell = 0; cell < workset.numCells; ++cell ) {
    for (std::size_t node = 0; node < this->numNodes; ++node) {
      for (std::size_t eq = 0; eq < numFields; eq++) {
        typename PHAL::Ref<ScalarT const>::type valref = (
            this->tensorRank == 0 ? this->val[eq] (cell, node) :
            this->tensorRank == 1 ? this->valVec (cell, node, eq) :
            this->valTensor (cell, node, eq / numDims, eq % numDims));

        const LO row = nodeID(cell,node,this->offset + eq);

        if (Teuchos::nonnull (fT))
          fT->sumIntoLocalValue (row, valref.val ());

        if (Teuchos::nonnull (JVT))
          for (int col = 0; col < workset.num_cols_x; col++)
            JVT->sumIntoLocalValue (row, col, valref.dx (col));

        if (Teuchos::nonnull (fpT))
          for (int col = 0; col < workset.num_cols_p; col++)
            fpT->sumIntoLocalValue (row, col, valref.dx (col + workset.param_offset));
      }
    }
  }
}

// **********************************************************************
// Specialization: DistParamDeriv
// **********************************************************************

template<typename Traits>
ScatterResidual<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
ScatterResidual(const Teuchos::ParameterList& p,
                const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidualBase<PHAL::AlbanyTraits::DistParamDeriv,Traits>(p,dl),
  numFields(ScatterResidualBase<PHAL::AlbanyTraits::DistParamDeriv,Traits>::numFieldsBase)
{
}

// **********************************************************************
template<typename Traits>
void ScatterResidual<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  auto nodeID = workset.wsElNodeEqID;
  Teuchos::RCP<Tpetra_MultiVector> fpVT = workset.fpVT;
  bool trans = workset.transpose_dist_param_deriv;
  int num_cols = workset.VpT->getNumVectors();

  if(workset.local_Vp[0].size() == 0) return; //In case the parameter has not been gathered, e.g. parameter is used only in Dirichlet conditions.

  int numDims= (this->tensorRank==2) ? this->valTensor.dimension(2) : 0;

  if (trans) {
    const int neq = nodeID.dimension(2);
    const Albany::IDArray&  wsElDofs = workset.distParamLib->get(workset.dist_param_deriv_name)->workset_elem_dofs()[workset.wsIndex];
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      const Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> >& local_Vp =
        workset.local_Vp[cell];
      const int num_deriv = this->numNodes;//local_Vp.size()/numFields;
      for (int i=0; i<num_deriv; i++) {
        for (int col=0; col<num_cols; col++) {
          double val = 0.0;
          for (std::size_t node = 0; node < this->numNodes; ++node) {
            for (std::size_t eq = 0; eq < numFields; eq++) {
              typename PHAL::Ref<ScalarT const>::type
                        valref = (this->tensorRank == 0 ? this->val[eq](cell,node) :
                                  this->tensorRank == 1 ? this->valVec(cell,node,eq) :
                                  this->valTensor(cell,node, eq/numDims, eq%numDims));
              val += valref.dx(i)*local_Vp[node*neq+eq+this->offset][col];  //numField can be less then neq
            }
          }
          const LO row = wsElDofs((int)cell,i,0);
          if(row >=0)
            fpVT->sumIntoLocalValue(row, col, val);
        }
      }
    }
  }
  else {
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      const Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> >& local_Vp =
        workset.local_Vp[cell];
      const int num_deriv = local_Vp.size();

      for (std::size_t node = 0; node < this->numNodes; ++node) {
        for (std::size_t eq = 0; eq < numFields; eq++) {
          typename PHAL::Ref<ScalarT const>::type
                    valref = (this->tensorRank == 0 ? this->val[eq](cell,node) :
                              this->tensorRank == 1 ? this->valVec(cell,node,eq) :
                              this->valTensor(cell,node, eq/numDims, eq%numDims));
          const int row = nodeID(cell,node,this->offset + eq);
          for (int col=0; col<num_cols; col++) {
            double val = 0.0;
            for (int i=0; i<num_deriv; ++i)
              val += valref.dx(i)*local_Vp[i][col];
            fpVT->sumIntoLocalValue(row, col, val);
          }
        }
      }
    }
  }
}

// **********************************************************************
template<typename Traits>
void ScatterResidualWithExtrudedParams<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
 
  if(workset.local_Vp[0].size() == 0) return; //In case the parameter has not been gathered, e.g. parameter is used only in Dirichlet conditions.

  auto level_it = extruded_params_levels->find(workset.dist_param_deriv_name);
  if(level_it == extruded_params_levels->end()) //if parameter is not extruded use usual scatter.
    return ScatterResidual<PHAL::AlbanyTraits::DistParamDeriv, Traits>::evaluateFields(workset);

  auto nodeID = workset.wsElNodeEqID;
  int fieldLevel = level_it->second;
  Teuchos::RCP<Tpetra_MultiVector> fpVT = workset.fpVT;
  bool trans = workset.transpose_dist_param_deriv;
  int num_cols = workset.VpT->getNumVectors();

  int numDims= (this->tensorRank==2) ? this->valTensor.dimension(2) : 0;

  if (trans) {
    const int neq = nodeID.dimension(2);
    const Albany::LayeredMeshNumbering<LO>& layeredMeshNumbering = *workset.disc->getLayeredMeshNumbering();

    int numLayers = layeredMeshNumbering.numLayers;
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> >& wsElNodeID  = workset.disc->getWsElNodeID()[workset.wsIndex];

    const Albany::IDArray&  wsElDofs = workset.distParamLib->get(workset.dist_param_deriv_name)->workset_elem_dofs()[workset.wsIndex];
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      const Teuchos::ArrayRCP<GO>& elNodeID = wsElNodeID[cell];
      const Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> >& local_Vp =
        workset.local_Vp[cell];
      const int num_deriv = this->numNodes;//local_Vp.size()/this->numFields;
      for (int i=0; i<num_deriv; i++) {
        LO lnodeId = workset.disc->getOverlapNodeMapT()->getLocalElement(elNodeID[i]);
        LO base_id, ilayer;
        layeredMeshNumbering.getIndices(lnodeId, base_id, ilayer);
        LO inode = layeredMeshNumbering.getId(base_id, fieldLevel);
        GO ginode = workset.disc->getOverlapNodeMapT()->getGlobalElement(inode);

        for (int col=0; col<num_cols; col++) {
          double val = 0.0;
          for (std::size_t node = 0; node < this->numNodes; ++node) {
            for (std::size_t eq = 0; eq < this->numFields; eq++) {
              typename PHAL::Ref<ScalarT const>::type
                        valref = (this->tensorRank == 0 ? this->val[eq](cell,node) :
                                  this->tensorRank == 1 ? this->valVec(cell,node,eq) :
                                  this->valTensor(cell,node, eq/numDims, eq%numDims));
              val += valref.dx(i)*local_Vp[node*neq+eq+this->offset][col];  //numField can be less then neq
            }
          }
          const LO row = workset.distParamLib->get(workset.dist_param_deriv_name)->overlap_map()->getLocalElement(ginode);
          if(row >=0)
            fpVT->sumIntoLocalValue(row, col, val);
        }
      }
    }
  }
  else {
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      const Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> >& local_Vp =
        workset.local_Vp[cell];
      const int num_deriv = local_Vp.size();

      for (std::size_t node = 0; node < this->numNodes; ++node) {
        for (std::size_t eq = 0; eq < this->numFields; eq++) {
          typename PHAL::Ref<ScalarT const>::type
                    valref = (this->tensorRank == 0 ? this->val[eq](cell,node) :
                              this->tensorRank == 1 ? this->valVec(cell,node,eq) :
                              this->valTensor(cell,node, eq/numDims, eq%numDims));
          const int row = nodeID(cell,node,this->offset + eq);
          for (int col=0; col<num_cols; col++) {
            double val = 0.0;
            for (int i=0; i<num_deriv; ++i)
              val += valref.dx(i)*local_Vp[i][col];
            fpVT->sumIntoLocalValue(row, col, val);
          }
        }
      }
    }
  }
}

// **********************************************************************
// Specialization: Stochastic Galerkin Residual
// **********************************************************************

#ifdef ALBANY_SG
template<typename Traits>
ScatterResidual<PHAL::AlbanyTraits::SGResidual, Traits>::
ScatterResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidualBase<PHAL::AlbanyTraits::SGResidual,Traits>(p,dl),
  numFields(ScatterResidualBase<PHAL::AlbanyTraits::SGResidual,Traits>::numFieldsBase)
{
}

// **********************************************************************
template<typename Traits>
void ScatterResidual<PHAL::AlbanyTraits::SGResidual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  auto nodeID = workset.wsElNodeEqID;
  Teuchos::RCP< Stokhos::EpetraVectorOrthogPoly > f = workset.sg_f;

  int numDims=0;
  if(this->tensorRank==2)
    numDims = this->valTensor.dimension(2);

  int nblock = f->size();
  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    for (std::size_t node = 0; node < this->numNodes; ++node) {

      for (std::size_t eq = 0; eq < numFields; eq++) {
        typename PHAL::Ref<ScalarT const>::type
          valptr = (this->tensorRank == 0 ? this->val[eq](cell,node) :
                    this->tensorRank == 1 ? this->valVec(cell,node,eq) :
                    this->valTensor(cell,node, eq/numDims, eq%numDims));

        for (int block=0; block<nblock; block++)
          (*f)[block][nodeID(cell,node,this->offset + eq)] += valptr.coeff(block);
      }
    }
  }
}

// **********************************************************************
// Specialization: Stochastic Galerkin Jacobian
// **********************************************************************

template<typename Traits>
ScatterResidual<PHAL::AlbanyTraits::SGJacobian, Traits>::
ScatterResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidualBase<PHAL::AlbanyTraits::SGJacobian,Traits>(p,dl),
  numFields(ScatterResidualBase<PHAL::AlbanyTraits::SGJacobian,Traits>::numFieldsBase)
{
}

// **********************************************************************
template<typename Traits>
void ScatterResidual<PHAL::AlbanyTraits::SGJacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  auto nodeID = workset.wsElNodeEqID;
  Teuchos::RCP< Stokhos::EpetraVectorOrthogPoly > f = workset.sg_f;
  Teuchos::RCP< Stokhos::VectorOrthogPoly<Epetra_CrsMatrix> > Jac =
    workset.sg_Jac;

  int row, lcol, col;
  int nblock = 0;
  if (f != Teuchos::null)
    nblock = f->size();
  int nblock_jac = Jac->size();
  double c; // use double since it goes into CrsMatrix

  int numDims=0;
  if(this->tensorRank==2)
    numDims = this->valTensor.dimension(2);

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    for (std::size_t node = 0; node < this->numNodes; ++node) {

      for (std::size_t eq = 0; eq < numFields; eq++) {
        typename PHAL::Ref<ScalarT const>::type
          valptr = (this->tensorRank == 0 ? this->val[eq](cell,node) :
                    this->tensorRank == 1 ? this->valVec(cell,node,eq) :
                    this->valTensor(cell,node, eq/numDims, eq%numDims));

        row = nodeID(cell,node,this->offset + eq);
        int neq = nodeID.dimension(2);

        if (f != Teuchos::null) {
          for (int block=0; block<nblock; block++)
            (*f)[block].SumIntoMyValue(row, 0, valptr.val().coeff(block));
        }

        // Check derivative array is nonzero
        if (valptr.hasFastAccess()) {

          // Loop over nodes in element
          for (unsigned int node_col=0; node_col<this->numNodes; node_col++){

            // Loop over equations per node
            for (unsigned int eq_col=0; eq_col<neq; eq_col++) {
              lcol = neq * node_col + eq_col;

              // Global column
              col =  nodeID(cell,node_col,eq_col);

              // Sum Jacobian
              for (int block=0; block<nblock_jac; block++) {
                c = valptr.fastAccessDx(lcol).coeff(block);
                if (workset.is_adjoint) {
                  (*Jac)[block].SumIntoMyValues(col, 1, &c, &row);
                }
                else {
                  (*Jac)[block].SumIntoMyValues(row, 1, &c, &col);
                }
              }
            } // column equations
          } // column nodes
        } // has fast access
      }
    }
  }
}

// **********************************************************************
// Specialization: Stochastic Galerkin Tangent
// **********************************************************************

template<typename Traits>
ScatterResidual<PHAL::AlbanyTraits::SGTangent, Traits>::
ScatterResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidualBase<PHAL::AlbanyTraits::SGTangent,Traits>(p,dl),
  numFields(ScatterResidualBase<PHAL::AlbanyTraits::SGTangent,Traits>::numFieldsBase)
{
}

// **********************************************************************
template<typename Traits>
void ScatterResidual<PHAL::AlbanyTraits::SGTangent, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  auto nodeID = workset.wsElNodeEqID;
  Teuchos::RCP< Stokhos::EpetraVectorOrthogPoly > f = workset.sg_f;
  Teuchos::RCP< Stokhos::EpetraMultiVectorOrthogPoly > JV = workset.sg_JV;
  Teuchos::RCP< Stokhos::EpetraMultiVectorOrthogPoly > fp = workset.sg_fp;

  int nblock = 0;
  if (f != Teuchos::null)
    nblock = f->size();
  else if (JV != Teuchos::null)
    nblock = JV->size();
  else if (fp != Teuchos::null)
    nblock = fp->size();
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                       "One of sg_f, sg_JV, or sg_fp must be non-null! " <<
                       std::endl);

  int numDims=0;
  if(this->tensorRank==2)
    numDims = this->valTensor.dimension(2);

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    for (std::size_t node = 0; node < this->numNodes; ++node) {
      for (std::size_t eq = 0; eq < numFields; eq++) {
        typename PHAL::Ref<ScalarT const>::type
          valptr = (this->tensorRank == 0 ? this->val[eq](cell,node) :
                    this->tensorRank == 1 ? this->valVec(cell,node,eq) :
                    this->valTensor(cell,node, eq/numDims, eq%numDims));

        int row = nodeID(cell,node,this->offset + eq);

        if (f != Teuchos::null)
          for (int block=0; block<nblock; block++)
            (*f)[block].SumIntoMyValue(row, 0, valptr.val().coeff(block));

        if (JV != Teuchos::null)
          for (int col=0; col<workset.num_cols_x; col++)
            for (int block=0; block<nblock; block++)
              (*JV)[block].SumIntoMyValue(row, col, valptr.dx(col).coeff(block));

        if (fp != Teuchos::null)
          for (int col=0; col<workset.num_cols_p; col++)
            for (int block=0; block<nblock; block++)
              (*fp)[block].SumIntoMyValue(row, col, valptr.dx(col+workset.param_offset).coeff(block));
      }
    }
  }
}
#endif 
#ifdef ALBANY_ENSEMBLE 

// **********************************************************************
// Specialization: Multi-point Residual
// **********************************************************************

template<typename Traits>
ScatterResidual<PHAL::AlbanyTraits::MPResidual, Traits>::
ScatterResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidualBase<PHAL::AlbanyTraits::MPResidual,Traits>(p,dl),
  numFields(ScatterResidualBase<PHAL::AlbanyTraits::MPResidual,Traits>::numFieldsBase)
{
}

// **********************************************************************
template<typename Traits>
void ScatterResidual<PHAL::AlbanyTraits::MPResidual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  auto nodeID = workset.wsElNodeEqID;
  Teuchos::RCP< Stokhos::ProductEpetraVector > f = workset.mp_f;

  int numDims=0;
  if(this->tensorRank==2)
    numDims = this->valTensor.dimension(2);

  int nblock = f->size();
  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    for (std::size_t node = 0; node < this->numNodes; ++node) {

      for (std::size_t eq = 0; eq < numFields; eq++) {
        typename PHAL::Ref<ScalarT const>::type
          valptr = (this->tensorRank == 0 ? this->val[eq](cell,node) :
                    this->tensorRank == 1 ? this->valVec(cell,node,eq) :
                    this->valTensor(cell,node, eq/numDims, eq%numDims));
        for (int block=0; block<nblock; block++)
          (*f)[block][nodeID(cell,node,this->offset + eq)] += valptr.coeff(block);
      }
    }
  }
}

// **********************************************************************
// Specialization: Multi-point Jacobian
// **********************************************************************

template<typename Traits>
ScatterResidual<PHAL::AlbanyTraits::MPJacobian, Traits>::
ScatterResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidualBase<PHAL::AlbanyTraits::MPJacobian,Traits>(p,dl),
  numFields(ScatterResidualBase<PHAL::AlbanyTraits::MPJacobian,Traits>::numFieldsBase)
{
}

// **********************************************************************
template<typename Traits>
void ScatterResidual<PHAL::AlbanyTraits::MPJacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  auto nodeID = workset.wsElNodeEqID;
  Teuchos::RCP< Stokhos::ProductEpetraVector > f = workset.mp_f;
  Teuchos::RCP< Stokhos::ProductContainer<Epetra_CrsMatrix> > Jac =
    workset.mp_Jac;

  int row, lcol;
  int nblock = 0;
  if (f != Teuchos::null)
    nblock = f->size();
  int nblock_jac = Jac->size();
  const int neq = nodeID.dimension(2);
  const int nunk = neq*this->numNodes;
  Teuchos::Array<double> val(nunk); // use double since it goes into CrsMatrix
  Teuchos::Array<int> col(nunk);

  int numDims=0;
  if(this->tensorRank==2)
    numDims = this->valTensor.dimension(2);

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    for (std::size_t node = 0; node < this->numNodes; ++node) {

      for (std::size_t eq = 0; eq < numFields; eq++) {
        typename PHAL::Ref<ScalarT const>::type
          valptr = (this->tensorRank == 0 ? this->val[eq](cell,node) :
                    this->tensorRank == 1 ? this->valVec(cell,node,eq) :
                    this->valTensor(cell,node, eq/numDims, eq%numDims));

        row = nodeID(cell,node,this->offset + eq);

        if (f != Teuchos::null) {
          for (int block=0; block<nblock; block++)
            (*f)[block].SumIntoMyValue(row, 0, valptr.val().coeff(block));
        }

        // Check derivative array is nonzero
        if (valptr.hasFastAccess()) {

          // Loop over blocks -- here we pull out one ensemble value at at
          // a time some we can sum in a whole row at once.  This is much
          // faster due to the search of column indices in SumIntoMyValues.
          for (int block=0; block<nblock_jac; block++) {

            // Loop over nodes in element
            for (unsigned int node_col=0; node_col<this->numNodes; node_col++){

              // Loop over equations per node
              for (unsigned int eq_col=0; eq_col<neq; eq_col++) {
                lcol = neq * node_col + eq_col;

                // Global column
                col[lcol] =  nodeID(cell,node_col,eq_col);

                // Matrix value
                val[lcol] = valptr.fastAccessDx(lcol).coeff(block);

              } // column equations

            } // column nodes

            // Sum Jacobian
            (*Jac)[block].SumIntoMyValues(row, nunk,
                                          val.getRawPtr(), col.getRawPtr());

          } // has fast access

        } // block

      } // row equations

    } // row nodes

  } // cells
}

// **********************************************************************
// Specialization: Multi-point Tangent
// **********************************************************************

template<typename Traits>
ScatterResidual<PHAL::AlbanyTraits::MPTangent, Traits>::
ScatterResidual(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
  : ScatterResidualBase<PHAL::AlbanyTraits::MPTangent,Traits>(p,dl),
  numFields(ScatterResidualBase<PHAL::AlbanyTraits::MPTangent,Traits>::numFieldsBase)
{
}

// **********************************************************************
template<typename Traits>
void ScatterResidual<PHAL::AlbanyTraits::MPTangent, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::RCP< Stokhos::ProductEpetraVector > f = workset.mp_f;
  Teuchos::RCP< Stokhos::ProductEpetraMultiVector > JV = workset.mp_JV;
  Teuchos::RCP< Stokhos::ProductEpetraMultiVector > fp = workset.mp_fp;

  int nblock = 0;
  if (f != Teuchos::null)
    nblock = f->size();
  else if (JV != Teuchos::null)
    nblock = JV->size();
  else if (fp != Teuchos::null)
    nblock = fp->size();
  else
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                       "One of mp_f, mp_JV, or mp_fp must be non-null! " <<
                       std::endl);

  int numDims=0;
  if(this->tensorRank==2)
    numDims = this->valTensor.dimension(2);

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    for (std::size_t node = 0; node < this->numNodes; ++node) {
      for (std::size_t eq = 0; eq < numFields; eq++) {
        typename PHAL::Ref<ScalarT const>::type
          valptr = (this->tensorRank == 0 ? this->val[eq](cell,node) :
                    this->tensorRank == 1 ? this->valVec(cell,node,eq) :
                    this->valTensor(cell,node, eq/numDims, eq%numDims));

        int row = nodeID(cell,node,this->offset + eq);

        if (f != Teuchos::null)
          for (int block=0; block<nblock; block++)
            (*f)[block].SumIntoMyValue(row, 0, valptr.val().coeff(block));

        if (JV != Teuchos::null)
          for (int col=0; col<workset.num_cols_x; col++)
            for (int block=0; block<nblock; block++)
              (*JV)[block].SumIntoMyValue(row, col, valptr.dx(col).coeff(block));

        if (fp != Teuchos::null)
          for (int col=0; col<workset.num_cols_p; col++)
            for (int block=0; block<nblock; block++)
              (*fp)[block].SumIntoMyValue(row, col, valptr.dx(col+workset.param_offset).coeff(block));
      }
    }
  }
}
#endif

}

