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

#include "PHAL_ScatterResidual.hpp"
#include "Albany_Macros.hpp"
#include "Albany_Utils.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_AbstractDiscretization.hpp"
#include "Albany_DistributedParameterLibrary.hpp"
#include "Albany_GlobalLocalIndexer.hpp"

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

  scatter_operation = Teuchos::rcp(new PHX::Tag<ScalarT>(fieldName, dl->dummy));

  Teuchos::ArrayRCP<std::string> names;
  if (p.isType<Teuchos::ArrayRCP<std::string>>("Residual Names")) {
    names = p.get< Teuchos::ArrayRCP<std::string> >("Residual Names");
  } else if (p.isType<std::string>("Residual Name")) {
    names = Teuchos::ArrayRCP<std::string>(1,p.get<std::string>("Residual Name"));
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error, "Error! You must specify either the std::string 'Residual Name', "
                                                          "or the Teuchos::ArrayRCP<std::string> 'Residual Names'.\n");
  }

  tensorRank = p.get<int>("Tensor Rank");

  if (tensorRank == 0 ) {
    // scalar
    numFieldsBase = names.size();
    const std::size_t num_val = numFieldsBase;
    val.resize(num_val);
    for (std::size_t eq = 0; eq < numFieldsBase; ++eq) {
      PHX::MDField<ScalarT const,Cell,Node> mdf(names[eq],dl->node_scalar);
      val[eq] = mdf;
      this->addDependentField(val[eq]);
    }
  } else if (tensorRank == 1 ) {
    // vector
    PHX::MDField<ScalarT const,Cell,Node,Dim> mdf(names[0],dl->node_vector);
    valVec= mdf;
    this->addDependentField(valVec);
    numFieldsBase = dl->node_vector->extent(2);
  } else if (tensorRank == 2 ) {
    // tensor
    PHX::MDField<ScalarT const,Cell,Node,Dim,Dim> mdf(names[0],dl->node_tensor);
    valTensor = mdf;
    this->addDependentField(valTensor);
    numFieldsBase = (dl->node_tensor->extent(2))*(dl->node_tensor->extent(3));
  }

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  if (tensorRank == 0) {
    val_kokkos.resize(numFieldsBase);
  }
#endif

  if (p.isType<int>("Offset of First DOF")) {
    offset = p.get<int>("Offset of First DOF");
  } else {
    offset = 0;
  }

  this->addEvaluatedField(*scatter_operation);

  this->setName(fieldName+PHX::print<EvalT>());
}

// **********************************************************************
template<typename EvalT, typename Traits>
void ScatterResidualBase<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  if (tensorRank == 0) {
    for (std::size_t eq = 0; eq < numFieldsBase; ++eq) {
      this->utils.setFieldData(val[eq],fm);
    }
    numNodes = val[0].extent(1);
  } else  if (tensorRank == 1) {
    this->utils.setFieldData(valVec,fm);
    numNodes = valVec.extent(1);
  } else  if (tensorRank == 2) {
    this->utils.setFieldData(valTensor,fm);
    numNodes = valTensor.extent(1);
  }
  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
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
      Kokkos::atomic_fetch_add(&f_kokkos(id), val_kokkos[eq](cell,node));
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
      Kokkos::atomic_fetch_add(&f_kokkos(id), this->valVec(cell,node,eq));
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
        Kokkos::atomic_fetch_add(&f_kokkos(id), this->valTensor(cell,node,i,j)); 
      }
}
#endif

// **********************************************************************
template<typename Traits>
void ScatterResidual<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::RCP<Thyra_Vector> f = workset.f;

#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  auto nodeID = workset.wsElNodeEqID;

  //get nonconst (read and write) view of f
  Teuchos::ArrayRCP<ST> f_nonconstView = Albany::getNonconstLocalData(f);

  if (this->tensorRank == 0) {
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      for (std::size_t node = 0; node < this->numNodes; ++node)
        for (std::size_t eq = 0; eq < numFields; eq++)
          f_nonconstView[nodeID(cell,node,this->offset + eq)] += (this->val[eq])(cell,node);
    }
  } else if (this->tensorRank == 1) {
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      for (std::size_t node = 0; node < this->numNodes; ++node)
        for (std::size_t eq = 0; eq < numFields; eq++)
          f_nonconstView[nodeID(cell,node,this->offset + eq)] += (this->valVec)(cell,node,eq);
    }
  } else if (this->tensorRank == 2) {
    int numDims = this->valTensor.extent(2);
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

  // Get device data
  f_kokkos = Albany::getNonconstDeviceData(f);

  if (this->tensorRank == 0) {
    // Get MDField views from std::vector
    for (int i = 0; i < numFields; i++)
      val_kokkos[i] = this->val[i].get_view();

    Kokkos::parallel_for(PHAL_ScatterResRank0_Policy(0,workset.numCells),*this);
    cudaCheckError();
  } else if (this->tensorRank == 1) {
    Kokkos::parallel_for(PHAL_ScatterResRank1_Policy(0,workset.numCells),*this);
    cudaCheckError();
  } else if (this->tensorRank == 2) {
    numDims = this->valTensor.extent(2);
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
      Kokkos::atomic_fetch_add(&f_kokkos(id), (val_kokkos[eq](cell,node)).val());
    }
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void ScatterResidual<PHAL::AlbanyTraits::Jacobian,Traits>::
operator() (const PHAL_ScatterJacRank0_Adjoint_Tag&, const int& cell) const
{
  //const int neq = nodeID.extent(2);
  //const int nunk = neq*this->numNodes;
  // Irina TOFIX replace 500 with nunk with Kokkos::malloc is available
  LO col[500];
  LO row;

  if (nunk>500) {
    Kokkos::abort("ERROR (ScatterResidual): nunk > 500");
  }

  for (int node_col=0; node_col<this->numNodes; node_col++) {
    for (int eq_col=0; eq_col<neq; eq_col++) {
      col[neq * node_col + eq_col] =  nodeID(cell,node_col,eq_col);
    }
  }

  for (int node = 0; node < this->numNodes; ++node) {
    for (int eq = 0; eq < numFields; eq++) {
      row = nodeID(cell,node,this->offset + eq);
      auto valptr = val_kokkos[eq](cell,node);
      for (int lunk=0; lunk<nunk; lunk++) {
        ST val = valptr.fastAccessDx(lunk);
        Jac_kokkos.sumIntoValues(col[lunk], &row, 1, &val, false, true); 
      }
    }
  }
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void ScatterResidual<PHAL::AlbanyTraits::Jacobian,Traits>::
operator() (const PHAL_ScatterJacRank0_Tag&, const int& cell) const
{
  //const int neq = nodeID.extent(2);
  //const int nunk = neq*this->numNodes;
  // Irina TOFIX replace 500 with nunk with Kokkos::malloc is available
  LO row;
  LO col[500];
  ST vals[500];

  if (nunk>500) {
    Kokkos::abort("ERROR (ScatterResidual): nunk > 500");
  }

  for (int node_col=0; node_col<this->numNodes; node_col++) {
    for (int eq_col=0; eq_col<neq; eq_col++) {
      col[neq * node_col + eq_col] = nodeID(cell,node_col,eq_col);
    }
  }

  for (int node = 0; node < this->numNodes; ++node) {
    for (int eq = 0; eq < numFields; eq++) {
      row = nodeID(cell,node,this->offset + eq);
      auto valptr = val_kokkos[eq](cell,node);
      for (int i = 0; i < nunk; ++i) vals[i] = valptr.fastAccessDx(i);
      Jac_kokkos.sumIntoValues(row, col, nunk, vals, false, true);
    }
  }
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void ScatterResidual<PHAL::AlbanyTraits::Jacobian,Traits>::
operator() (const PHAL_ScatterResRank1_Tag&, const int& cell) const
{
  for (std::size_t node = 0; node < this->numNodes; node++) {
    for (std::size_t eq = 0; eq < numFields; eq++) {
      const LO id = nodeID(cell,node,this->offset + eq);
      Kokkos::atomic_fetch_add(&f_kokkos(id), (this->valVec(cell,node,eq)).val());
    }
  }
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void ScatterResidual<PHAL::AlbanyTraits::Jacobian,Traits>::
operator() (const PHAL_ScatterJacRank1_Adjoint_Tag&, const int& cell) const
{
  //const int neq = nodeID.extent(2);
  //const int nunk = neq*this->numNodes;
  // Irina TOFIX replace 500 with nunk with Kokkos::malloc is available
  LO col[500];
  LO row;

  if (nunk>500) {
    Kokkos::abort("ERROR (ScatterResidual): nunk > 500");
  }

  for (int node_col=0; node_col<this->numNodes; node_col++) {
    for (int eq_col=0; eq_col<neq; eq_col++) {
      col[neq * node_col + eq_col] = nodeID(cell,node_col,eq_col);
    }
  }

  for (int node = 0; node < this->numNodes; ++node) {
    for (int eq = 0; eq < numFields; eq++) {
      row = nodeID(cell,node,this->offset + eq);
      if (((this->valVec)(cell,node,eq)).hasFastAccess()) {
        for (int lunk=0; lunk<nunk; lunk++){
          ST val = ((this->valVec)(cell,node,eq)).fastAccessDx(lunk);
          Jac_kokkos.sumIntoValues(col[lunk], &row, 1, &val, false, true);
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
  //const int neq = nodeID.extent(2);
  //const int nunk = neq*this->numNodes;
  // Irina TOFIX replace 500 with nunk with Kokkos::malloc is available
  LO col[500];
  LO row;
  ST vals[500];

  if (nunk>500) {
    Kokkos::abort("ERROR (ScatterResidual): nunk > 500");
  }

  for (int node_col=0; node_col<this->numNodes; node_col++) {
    for (int eq_col=0; eq_col<neq; eq_col++) {
      col[neq * node_col + eq_col] = nodeID(cell,node_col,eq_col);
    }
  }

  for (int node = 0; node < this->numNodes; ++node) {
    for (int eq = 0; eq < numFields; eq++) {
      row = nodeID(cell,node,this->offset + eq);
      if (((this->valVec)(cell,node,eq)).hasFastAccess()) {
        for (int i = 0; i < nunk; ++i) vals[i] = (this->valVec)(cell,node,eq).fastAccessDx(i);
        Jac_kokkos.sumIntoValues(row, col, nunk, vals, false, true);
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
        Kokkos::atomic_fetch_add(&f_kokkos(id), (this->valTensor(cell,node,i,j)).val()); 
      }
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void ScatterResidual<PHAL::AlbanyTraits::Jacobian,Traits>::
operator() (const PHAL_ScatterJacRank2_Adjoint_Tag&, const int& cell) const
{
  //const int neq = nodeID.extent(2);
  //const int nunk = neq*this->numNodes;
  // Irina TOFIX replace 500 with nunk with Kokkos::malloc is available
  LO col[500];
  LO row;

  if (nunk>500) {
    Kokkos::abort("ERROR (ScatterResidual): nunk > 500");
  }

  for (int node_col=0; node_col<this->numNodes; node_col++) {
    for (int eq_col=0; eq_col<neq; eq_col++) {
      col[neq * node_col + eq_col] = nodeID(cell,node_col,eq_col);
    }
  }

  for (int node = 0; node < this->numNodes; ++node) {
    for (int eq = 0; eq < numFields; eq++) {
      row = nodeID(cell,node,this->offset + eq);
      if (((this->valTensor)(cell,node, eq/numDims, eq%numDims)).hasFastAccess()) {
        for (int lunk=0; lunk<nunk; lunk++) {
          ST val = ((this->valTensor)(cell,node, eq/numDims, eq%numDims)).fastAccessDx(lunk);
          Jac_kokkos.sumIntoValues (col[lunk], &row, 1, &val, false, true);
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
  //const int neq = nodeID.extent(2);
  //const int nunk = neq*this->numNodes;
  // Irina TOFIX replace 500 with nunk with Kokkos::malloc is available
  LO col[500];
  LO row;
  ST vals[500];

  if (nunk>500) {
    Kokkos::abort("ERROR (ScatterResidual): nunk > 500");
  }

  for (int node_col=0; node_col<this->numNodes; node_col++) {
    for (int eq_col=0; eq_col<neq; eq_col++) {
      col[neq * node_col + eq_col] = nodeID(cell,node_col,eq_col);
    }
  }

  for (int node = 0; node < this->numNodes; ++node) {
    for (int eq = 0; eq < numFields; eq++) {
      row = nodeID(cell,node,this->offset + eq);
      if (((this->valTensor)(cell,node, eq/numDims, eq%numDims)).hasFastAccess()) {
        for (int i = 0; i < nunk; ++i) vals[i] = (this->valTensor)(cell,node, eq/numDims, eq%numDims).fastAccessDx(i);
        Jac_kokkos.sumIntoValues(row, col, nunk,  vals, false, true);
      }
    }
  }
}
#endif // ALBANY_KOKKOS_UNDER_DEVELOPMENT

// **********************************************************************
template<typename Traits>
void ScatterResidual<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  const bool use_device = Albany::build_type()==Albany::BuildType::Tpetra;
#else
  const bool use_device = false;
#endif
  if (use_device) {
    evaluateFieldsDevice(workset);
  } else {
    evaluateFieldsHost(workset);
  }
}

template<typename Traits>
void ScatterResidual<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFieldsDevice(typename Traits::EvalData workset)
{
#ifdef ALBANY_TIMER
  auto start = std::chrono::high_resolution_clock::now();
#endif
  // Get map for local data structures
  nodeID = workset.wsElNodeEqID;

  // Get dimensions
  neq = nodeID.extent(2);
  nunk = neq*this->numNodes;

  // Get Kokkos vector view and local matrix
  const bool loadResid = Teuchos::nonnull(workset.f);
  if (loadResid) {
    f_kokkos = workset.f_kokkos;
  }
  Jac_kokkos = workset.Jac_kokkos;

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
    } else {
      Kokkos::parallel_for(PHAL_ScatterJacRank0_Policy(0,workset.numCells),*this);
      cudaCheckError();
    }
  } else  if (this->tensorRank == 1) {
    if (loadResid) {
      Kokkos::parallel_for(PHAL_ScatterResRank1_Policy(0,workset.numCells),*this);
      cudaCheckError();
    }

    if (workset.is_adjoint) {
      Kokkos::parallel_for(PHAL_ScatterJacRank1_Adjoint_Policy(0,workset.numCells),*this);
      cudaCheckError();
    } else {
      Kokkos::parallel_for(PHAL_ScatterJacRank1_Policy(0,workset.numCells),*this);
      cudaCheckError();
    }
  } else if (this->tensorRank == 2) {
    numDims = this->valTensor.extent(2);

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
}

template<typename Traits>
void ScatterResidual<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFieldsHost(typename Traits::EvalData workset)
{
  Teuchos::RCP<Thyra_Vector>   f   = workset.f;
  Teuchos::RCP<Thyra_LinearOp> Jac = workset.Jac;

  auto nodeID = workset.wsElNodeEqID;
  const bool loadResid = Teuchos::nonnull(f);
  Teuchos::Array<LO> col;
  neq = nodeID.extent(2);
  nunk = neq*this->numNodes;
  col.resize(nunk);
  numDims = 0;
  if (this->tensorRank==2) {
    numDims = this->valTensor.extent(2);
  }
  Teuchos::ArrayRCP<ST> f_nonconstView;
  if (loadResid) {
    f_nonconstView = Albany::getNonconstLocalData(f);
  }

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    // Local Unks: Loop over nodes in element, Loop over equations per node
    for (unsigned int node_col=0; node_col<this->numNodes; node_col++){
      for (unsigned int eq_col=0; eq_col<neq; eq_col++) {
        col[neq * node_col + eq_col] = nodeID(cell,node_col,eq_col);
      }
    }
    for (std::size_t node = 0; node < this->numNodes; ++node) {
      for (std::size_t eq = 0; eq < numFields; eq++) {
        typename PHAL::Ref<ScalarT const>::type
          valptr = (this->tensorRank == 0 ? this->val[eq](cell,node) :
                    this->tensorRank == 1 ? this->valVec(cell,node,eq) :
                    this->valTensor(cell,node, eq/numDims, eq%numDims));
        const LO row = nodeID(cell,node,this->offset + eq);
        if (loadResid) {
          f_nonconstView[row] += valptr.val();
        }
        // Check derivative array is nonzero
        if (valptr.hasFastAccess()) {
          if (workset.is_adjoint) {
            // Sum Jacobian transposed
            for (unsigned int lunk = 0; lunk < nunk; lunk++)
              Albany::addToLocalRowValues(Jac,
                col[lunk], Teuchos::arrayView(&row, 1),
                Teuchos::arrayView(&(valptr.fastAccessDx(lunk)), 1));
          } else {
            // Sum Jacobian entries all at once
            Albany::addToLocalRowValues(Jac,
              row, col, Teuchos::arrayView(&(valptr.fastAccessDx(0)), nunk));
          }
        } // has fast access
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
  Teuchos::RCP<Thyra_Vector> f = workset.f;
  Teuchos::RCP<Thyra_MultiVector> JV = workset.JV;
  Teuchos::RCP<Thyra_MultiVector> fp = workset.fp;

  Teuchos::ArrayRCP<ST> f_nonconstView;

  Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>> JV_nonconst2dView;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>> fp_nonconst2dView;

  if (!f.is_null()) {
    f_nonconstView = Albany::getNonconstLocalData(f);
  }
  if (!JV.is_null()) {
    JV_nonconst2dView = Albany::getNonconstLocalData(JV);
  }
  if (!fp.is_null()) {
    fp_nonconst2dView = Albany::getNonconstLocalData(fp);
  }

  int numDims = 0;
  if (this->tensorRank == 2) numDims = this->valTensor.extent(2);

  for (std::size_t cell = 0; cell < workset.numCells; ++cell ) {
    for (std::size_t node = 0; node < this->numNodes; ++node) {
      for (std::size_t eq = 0; eq < numFields; eq++) {
        typename PHAL::Ref<ScalarT const>::type valref = (
            this->tensorRank == 0 ? this->val[eq] (cell, node) :
            this->tensorRank == 1 ? this->valVec (cell, node, eq) :
            this->valTensor (cell, node, eq / numDims, eq % numDims));

        const LO row = nodeID(cell,node,this->offset + eq);

        if (Teuchos::nonnull (f)) {
          f_nonconstView[row] += valref.val();
        }

        if (Teuchos::nonnull (JV)) {
          for (int col = 0; col < workset.num_cols_x; col++) {
            JV_nonconst2dView[col][row] += valref.dx(col);
        }}

        if (Teuchos::nonnull (fp)) {
          for (int col = 0; col < workset.num_cols_p; col++) {
            fp_nonconst2dView[col][row] += valref.dx(col + workset.param_offset);
        }}
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
  Teuchos::RCP<Thyra_MultiVector> fpV = workset.fpV;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>> fpV_nonconst2dView = Albany::getNonconstLocalData(fpV);

  bool trans = workset.transpose_dist_param_deriv;
  int num_cols = workset.Vp->domain()->dim();

  if(workset.local_Vp[0].size() == 0) { return; } //In case the parameter has not been gathered, e.g. parameter is used only in Dirichlet conditions.

  int numDims= (this->tensorRank==2) ? this->valTensor.extent(2) : 0;

  if (trans) {
    const int neq = nodeID.extent(2);
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
          if(row >=0) {
            fpV_nonconst2dView[col][row] += val;
          }
        }
      }
    }
  } else {
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
            for (int i=0; i<num_deriv; ++i) {
              val += valref.dx(i)*local_Vp[i][col];
            }
            fpV_nonconst2dView[col][row] += val;
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
  if(workset.local_Vp[0].size() == 0) { return; } //In case the parameter has not been gathered, e.g. parameter is used only in Dirichlet conditions.

  auto level_it = extruded_params_levels->find(workset.dist_param_deriv_name);
  if(level_it == extruded_params_levels->end()) //if parameter is not extruded use usual scatter.
    return ScatterResidual<PHAL::AlbanyTraits::DistParamDeriv, Traits>::evaluateFields(workset);

  auto nodeID = workset.wsElNodeEqID;
  int fieldLevel = level_it->second;
  Teuchos::RCP<Thyra_MultiVector> fpV = workset.fpV;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<ST>> fpV_nonconst2dView = Albany::getNonconstLocalData(fpV);

  bool trans = workset.transpose_dist_param_deriv;
  int num_cols = workset.Vp->domain()->dim();

  int numDims= (this->tensorRank==2) ? this->valTensor.extent(2) : 0;

  if (trans) {
    const int neq = nodeID.extent(2);
    const Albany::LayeredMeshNumbering<LO>& layeredMeshNumbering = *workset.disc->getLayeredMeshNumbering();

    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> >& wsElNodeID  = workset.disc->getWsElNodeID()[workset.wsIndex];

    auto overlapVS = workset.distParamLib->get(workset.dist_param_deriv_name)->overlap_vector_space();
    auto overlapNodeVS = workset.disc->getOverlapNodeVectorSpace();
    auto node_indexer = Albany::createGlobalLocalIndexer(overlapNodeVS);
    auto indexer = Albany::createGlobalLocalIndexer(overlapVS);
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      const Teuchos::ArrayRCP<GO>& elNodeID = wsElNodeID[cell];
      const Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> >& local_Vp =
        workset.local_Vp[cell];
      const int num_deriv = this->numNodes;//local_Vp.size()/this->numFields;
      for (int i=0; i<num_deriv; i++) {
        const LO lnodeId = node_indexer->getLocalElement(elNodeID[i]);
        LO base_id, ilayer;
        layeredMeshNumbering.getIndices(lnodeId, base_id, ilayer);
        const LO inode = layeredMeshNumbering.getId(base_id, fieldLevel);
        const GO ginode = node_indexer->getGlobalElement(inode);

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
          const LO row = indexer->getLocalElement(ginode);
          if(row >=0) {
            fpV_nonconst2dView[col][row] += val;
          }
        }
      }
    }
  } else {
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
            fpV_nonconst2dView[col][row] += val;
          }
        }
      }
    }
  }
}

} // namespace PHAL
