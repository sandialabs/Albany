//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <vector>
#include <string>
#include <chrono>

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Albany_Macros.hpp"
#include "Albany_ThyraUtils.hpp"

#include "PHAL_GatherSolution.hpp"

namespace PHAL {

template<typename EvalT, typename Traits>
GatherSolutionBase<EvalT,Traits>::
GatherSolutionBase(const Teuchos::ParameterList& p,
                   const Teuchos::RCP<Albany::Layouts>& dl): numNodes(0)
{
  if (p.isType<int>("Tensor Rank")) {
    tensorRank = p.get<int>("Tensor Rank");
  } else if (p.isType<bool>("Vector Field")) {
    if (p.get<bool>("Vector Field") == true) {
      tensorRank = 1;
    } else {
      tensorRank = 0;
    }
  }

  if (p.isType<bool>("Disable Transient")) {
    enableTransient = !p.get<bool>("Disable Transient");
  } else {
    enableTransient = true;
  }

  if (p.isType<bool>("Enable Acceleration")) {
    enableAcceleration = p.get<bool>("Enable Acceleration");
  } else {
    enableAcceleration = false;
  }

  Teuchos::ArrayRCP<std::string> solution_names;
  if (p.getEntryPtr("Solution Names")) {
    solution_names = p.get< Teuchos::ArrayRCP<std::string> >("Solution Names");
  }

  // scalar
  if ( tensorRank == 0 ) {
    val.resize(solution_names.size());
    for (int eq = 0; eq < solution_names.size(); ++eq) {
      PHX::MDField<ScalarT,Cell,Node> f(solution_names[eq],dl->node_scalar);
      val[eq] = f;
      this->addEvaluatedField(val[eq]);
    }
    // repeat for xdot if transient is enabled
    if (enableTransient) {
      const Teuchos::ArrayRCP<std::string>& names_dot =
        p.get< Teuchos::ArrayRCP<std::string> >("Time Dependent Solution Names");

      val_dot.resize(names_dot.size());
      for (int eq = 0; eq < names_dot.size(); ++eq) {
        PHX::MDField<ScalarT,Cell,Node> f(names_dot[eq],dl->node_scalar);
        val_dot[eq] = f;
        this->addEvaluatedField(val_dot[eq]);
      }
    }
    // repeat for xdotdot if acceleration is enabled
    if (enableAcceleration) {
      const Teuchos::ArrayRCP<std::string>& names_dotdot =
        p.get< Teuchos::ArrayRCP<std::string> >("Solution Acceleration Names");

      val_dotdot.resize(names_dotdot.size());
      for (int eq = 0; eq < names_dotdot.size(); ++eq) {
        PHX::MDField<ScalarT,Cell,Node> f(names_dotdot[eq],dl->node_scalar);
        val_dotdot[eq] = f;
        this->addEvaluatedField(val_dotdot[eq]);
      }
    }
    numFields = val.size();
  } else if ( tensorRank == 1 ) {
    // vector
    PHX::MDField<ScalarT,Cell,Node,VecDim> f(solution_names[0],dl->node_vector);
    valVec= f;
    this->addEvaluatedField(valVec);
    // repeat for xdot if transient is enabled
    if (enableTransient) {
      const Teuchos::ArrayRCP<std::string>& names_dot =
        p.get< Teuchos::ArrayRCP<std::string> >("Time Dependent Solution Names");

      PHX::MDField<ScalarT,Cell,Node,VecDim> fdot(names_dot[0],dl->node_vector);
      valVec_dot= fdot;
      this->addEvaluatedField(valVec_dot);
    }
    // repeat for xdotdot if acceleration is enabled
    if (enableAcceleration) {
      const Teuchos::ArrayRCP<std::string>& names_dotdot =
        p.get< Teuchos::ArrayRCP<std::string> >("Solution Acceleration Names");

      PHX::MDField<ScalarT,Cell,Node,VecDim> fdotdot(names_dotdot[0],dl->node_vector);
      valVec_dotdot = fdotdot;
      this->addEvaluatedField(valVec_dotdot);
    }
    numFields = dl->node_vector->extent(2);
  } else if ( tensorRank == 2 ) {
    // tensor
    PHX::MDField<ScalarT,Cell,Node,VecDim,VecDim> f(solution_names[0],dl->node_tensor);
    valTensor = f;
    this->addEvaluatedField(valTensor);
    numDim = this->valTensor.extent(2);
    // repeat for xdot if transient is enabled
    if (enableTransient) {
      const Teuchos::ArrayRCP<std::string>& names_dot =
        p.get< Teuchos::ArrayRCP<std::string> >("Time Dependent Solution Names");

      PHX::MDField<ScalarT,Cell,Node,VecDim,VecDim> fdot(names_dot[0],dl->node_tensor);
      valTensor_dot = fdot;
      this->addEvaluatedField(valTensor_dot);
    }
    // repeat for xdotdot if acceleration is enabled
    if (enableAcceleration) {
      const Teuchos::ArrayRCP<std::string>& names_dotdot =
        p.get< Teuchos::ArrayRCP<std::string> >("Solution Acceleration Names");

      PHX::MDField<ScalarT,Cell,Node,VecDim,VecDim> fdotdot(names_dotdot[0],dl->node_tensor);
      valTensor_dotdot = fdotdot;
      this->addEvaluatedField(valTensor_dotdot);
    }
    numFields = (dl->node_tensor->extent(2))*(dl->node_tensor->extent(3));
  }

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  if ( tensorRank == 0 ) {
    val_kokkos.resize(numFields);
    if (enableTransient)
      val_dot_kokkos.resize(numFields);
    if (enableAcceleration)
      val_dotdot_kokkos.resize(numFields);
  }
#endif

  if (p.isType<int>("Offset of First DOF"))
    offset = p.get<int>("Offset of First DOF");
  else offset = 0;

  this->setName("Gather Solution"+PHX::print<EvalT>() );
}

// **********************************************************************
template<typename EvalT, typename Traits>
void GatherSolutionBase<EvalT,Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  if (tensorRank == 0) {
    for (int eq = 0; eq < numFields; ++eq)
      this->utils.setFieldData(val[eq],fm);
    if (enableTransient) {
      for (std::size_t eq = 0; eq < val_dot.size(); ++eq)
        this->utils.setFieldData(val_dot[eq],fm);
    }
    if (enableAcceleration) {
      for (std::size_t eq = 0; eq < val_dotdot.size(); ++eq)
        this->utils.setFieldData(val_dotdot[eq],fm);
    }
    numNodes = val[0].extent(1);
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
    // Get MDField views from std::vector
    for (int i =0; i<numFields;i++){
      val_kokkos[i]=this->val[i].get_static_view();
      if (enableTransient){
        val_dot_kokkos[i]=this->val_dot[i].get_static_view();
      }
      if (enableAcceleration){
        val_dotdot_kokkos[i]=this->val_dotdot[i].get_static_view();
      }
    }

    d_val=val_kokkos.template view<ExecutionSpace>();
    d_val_dot=val_dot_kokkos.template view<ExecutionSpace>();
    d_val_dotdot=val_dotdot_kokkos.template view<ExecutionSpace>();
#endif
  } else if (tensorRank == 1) {
    this->utils.setFieldData(valVec,fm);
    if (enableTransient) this->utils.setFieldData(valVec_dot,fm);
    if (enableAcceleration) this->utils.setFieldData(valVec_dotdot,fm);
    numNodes = valVec.extent(1);
  } else if (tensorRank == 2) {
    this->utils.setFieldData(valTensor,fm);
    if (enableTransient) this->utils.setFieldData(valTensor_dot,fm);
    if (enableAcceleration) this->utils.setFieldData(valTensor_dotdot,fm);
    numNodes = valTensor.extent(1);
  }
  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields(),false);
}

// **********************************************************************

// **********************************************************************
// Specialization: Residual
// **********************************************************************

template<typename Traits>
GatherSolution<PHAL::AlbanyTraits::Residual, Traits>::
GatherSolution(const Teuchos::ParameterList& p,
               const Teuchos::RCP<Albany::Layouts>& dl)
 : GatherSolutionBase<PHAL::AlbanyTraits::Residual, Traits>(p,dl)
{
  // Nothing else to do
}

template<typename Traits>
GatherSolution<PHAL::AlbanyTraits::Residual, Traits>::
GatherSolution(const Teuchos::ParameterList& p)
 : GatherSolutionBase<PHAL::AlbanyTraits::Residual, Traits>(p,p.get<Teuchos::RCP<Albany::Layouts> >("Layouts Struct"))
{
  // Nothing else to do
}

// ********************************************************************
// Kokkos functors for Residual
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherSolution<PHAL::AlbanyTraits::Residual, Traits>::
operator() (const PHAL_GatherSol_Tag&, const int& cell) const{
  for (size_t node = 0; node < this->numNodes; ++node)
    for (int eq = 0; eq < numFields; eq++)
      get_ref(cell,node,eq) = x_constView(nodeID(cell, node, this->offset+eq));
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherSolution<PHAL::AlbanyTraits::Residual, Traits>::
operator() (const PHAL_GatherSol_Transient_Tag&, const int& cell) const{
  for (size_t node = 0; node < this->numNodes; ++node)
    for (int eq = 0; eq < numFields; eq++)
      get_ref_dot(cell,node,eq)= xdot_constView(nodeID(cell, node, this->offset+eq));
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherSolution<PHAL::AlbanyTraits::Residual, Traits>::
operator() (const PHAL_GatherSol_Acceleration_Tag&, const int& cell) const{
  for (size_t node = 0; node < this->numNodes; ++node)
    for (int eq = 0; eq < numFields; eq++)
      get_ref_dotdot(cell,node,eq)= xdotdot_constView(nodeID(cell, node, this->offset+eq));
}
#endif

// **********************************************************************
template<typename Traits>
void GatherSolution<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  const auto& x       = workset.x;
  const auto& xdot    = workset.xdot;
  const auto& xdotdot = workset.xdotdot;

#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  auto nodeID = workset.wsElNodeEqID;
  Teuchos::ArrayRCP<const ST> x_constView, xdot_constView, xdotdot_constView;
  x_constView = Albany::getLocalData(x);
  if(!xdot.is_null()) {
    xdot_constView = Albany::getLocalData(xdot);
  }
  if(!xdotdot.is_null()) {
    xdotdot_constView = Albany::getLocalData(xdotdot);
  }

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    for (int node = 0; node < this->numNodes; ++node) {
      for (int eq = 0; eq < numFields; eq++)
        get_ref(cell,node,eq) = x_constView[nodeID(cell,node,this->offset + eq)];
        if (workset.transientTerms && this->enableTransient) {
          get_ref_dot(cell,node,eq) = xdot_constView[nodeID(cell,node,this->offset + eq)];
        }
        if (workset.accelerationTerms && this->enableAcceleration) {
          get_ref_dotdot(cell,node,eq) = xdotdot_constView[nodeID(cell,node,this->offset + eq)];
        }
      }
    }
  }

#else
#ifdef ALBANY_TIMER
  auto start = std::chrono::high_resolution_clock::now();
#endif

  // Get map for local data structures
  nodeID = workset.wsElNodeEqID;

  // Get vector view from a specific device
  x_constView = Albany::getDeviceData(x);
  if(!xdot.is_null()) {
    xdot_constView = Albany::getDeviceData(xdot);
  }
  if(!xdotdot.is_null()) {
    xdotdot_constView = Albany::getDeviceData(xdotdot);
  }

  Kokkos::parallel_for(PHAL_GatherSol_Policy(0,workset.numCells),*this);
  cudaCheckError();

  if (workset.transientTerms && this->enableTransient) {
    Kokkos::parallel_for(PHAL_GatherSol_Transient_Policy(0,workset.numCells),*this);
    cudaCheckError();
  }

  if (workset.accelerationTerms && this->enableAcceleration) {
    Kokkos::parallel_for(PHAL_GatherSol_Acceleration_Policy(0,workset.numCells),*this);
    cudaCheckError();
  }

#ifdef ALBANY_TIMER
  PHX::Device::fence();
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  long long millisec= std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
  std::cout<< "GaTher Solution Residual time = "  << millisec << "  "  << microseconds << std::endl;
#endif
#endif
}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************

template<typename Traits>
GatherSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
GatherSolution(const Teuchos::ParameterList& p,
               const Teuchos::RCP<Albany::Layouts>& dl)
 : GatherSolutionBase<PHAL::AlbanyTraits::Jacobian, Traits>(p,dl)
{
  // Nothing else to do
}

template<typename Traits>
GatherSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
GatherSolution(const Teuchos::ParameterList& p)
 : GatherSolutionBase<PHAL::AlbanyTraits::Jacobian, Traits>(p,p.get<Teuchos::RCP<Albany::Layouts> >("Layouts Struct"))
{
  // Nothing else to do
}

//********************************************************************
////Kokkos functors for Jacobian
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
operator() (const PHAL_GatherJac_Tag&, const int& cell) const{
  for (size_t node = 0; node < this->numNodes; ++node){
    int firstunk = neq * node + this->offset;
    for (int eq = 0; eq < numFields; eq++){
      ref_t valref = get_ref(cell,node,eq);
      valref=FadType(valref.size(), x_constView(nodeID(cell,node,this->offset+eq)));
      valref.fastAccessDx(firstunk + eq) =j_coeff;
    }
  }
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
operator() (const PHAL_GatherJac_Transient_Tag&, const int& cell) const{
  for (size_t node = 0; node < this->numNodes; ++node){
    int firstunk = neq * node + this->offset;
    for (int eq = 0; eq < numFields; eq++){
      ref_t valref = get_ref_dot(cell,node,eq);
      valref =FadType(valref.size(), xdot_constView(nodeID(cell,node,this->offset+eq)));
      valref.fastAccessDx(firstunk + eq) =m_coeff;
    }
  }
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
operator() (const PHAL_GatherJac_Acceleration_Tag&, const int& cell) const{
  for (size_t node = 0; node < this->numNodes; ++node){
    int firstunk = neq * node + this->offset;
    for (int eq = 0; eq < numFields; eq++){
      ref_t valref = get_ref_dotdot(cell,node,eq);
      valref=FadType(valref.size(), xdotdot_constView(nodeID(cell,node,this->offset+eq)));
      valref.fastAccessDx(firstunk + eq) =n_coeff;
    }
  }
}

#endif

// **********************************************************************
template<typename Traits>
void GatherSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  const auto& x       = workset.x;
  const auto& xdot    = workset.xdot;
  const auto& xdotdot = workset.xdotdot;

#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  auto nodeID = workset.wsElNodeEqID;
  Teuchos::ArrayRCP<const ST> x_constView, xdot_constView, xdotdot_constView;
  x_constView = Albany::getLocalData(x);
  if(!xdot.is_null()) {
    xdot_constView = Albany::getLocalData(xdot);
  }
  if(!xdotdot.is_null()) {
    xdot_constView = Albany::getLocalData(xdot);
  }

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const int neq = nodeID.extent(2);

    for (int node = 0; node < this->numNodes; ++node) {
      int firstunk = neq * node + this->offset;
      for (int eq = 0; eq < numFields; eq++) {
        ref_t valref = get_ref(cell,node,eq);
        valref = FadType(valref.size(), x_constView[nodeID(cell,node,this->offset + eq)]);
        valref.fastAccessDx(firstunk + eq) = workset.j_coeff;

        if (workset.transientTerms && this->enableTransient) {
          ref_t valref_dot = get_ref_dot (cell,node,eq);
          valref_dot = FadType(valref_dot.size(), xdot_constView[nodeID(cell,node,this->offset + eq)]);
          valref_dot.fastAccessDx(firstunk + eq) = workset.m_coeff;
        }

        if (workset.accelerationTerms && this->enableAcceleration) {
          ref_t valref_dotdot = get_ref_dotdot(cell,node,eq);
          valref_dotdot = FadType(valref_dotdot.size(), xdotdot_constView[nodeID(cell,node,this->offset + eq)]);
          valref_dotdot.fastAccessDx(firstunk + eq) = workset.n_coeff;
        }
      }
    }
  }

#else
#ifdef ALBANY_TIMER
  auto start = std::chrono::high_resolution_clock::now();
#endif

  // Get map for local data structures
  nodeID = workset.wsElNodeEqID;

  // Get dimensions and coefficients
  neq = nodeID.extent(2);
  j_coeff=workset.j_coeff;
  m_coeff=workset.m_coeff;
  n_coeff=workset.n_coeff;

  // Get vector view from a specific device
  x_constView = Albany::getDeviceData(x);
  if(!xdot.is_null()) {
    xdot_constView = Albany::getDeviceData(xdot);
  }
  if(!xdotdot.is_null()) {
    xdotdot_constView = Albany::getDeviceData(xdotdot);
  }

  Kokkos::parallel_for(PHAL_GatherJac_Policy(0,workset.numCells),*this);
  cudaCheckError();

  if (workset.transientTerms && this->enableTransient) {
    Kokkos::parallel_for(PHAL_GatherJac_Transient_Policy(0,workset.numCells),*this);
    cudaCheckError();
  }

  if (workset.accelerationTerms && this->enableAcceleration) {
    Kokkos::parallel_for(PHAL_GatherJac_Acceleration_Policy(0,workset.numCells),*this);
    cudaCheckError();
  }

#ifdef ALBANY_TIMER
  PHX::Device::fence();
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  long long millisec= std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
  std::cout<< "GaTher Solution Jacobian time = "  << millisec << "  "  << microseconds << std::endl;
#endif
#endif
}

// **********************************************************************

// **********************************************************************
// Specialization: Tangent
// **********************************************************************

template<typename Traits>
GatherSolution<PHAL::AlbanyTraits::Tangent, Traits>::
GatherSolution(const Teuchos::ParameterList& p,
               const Teuchos::RCP<Albany::Layouts>& dl)
 : GatherSolutionBase<PHAL::AlbanyTraits::Tangent, Traits>(p,dl)
{
  // Nothing else to do
}

template<typename Traits>
GatherSolution<PHAL::AlbanyTraits::Tangent, Traits>::
GatherSolution(const Teuchos::ParameterList& p)
 : GatherSolutionBase<PHAL::AlbanyTraits::Tangent, Traits>(p,p.get<Teuchos::RCP<Albany::Layouts> >("Layouts Struct"))
{
  // Nothing else to do
}

// **********************************************************************
template<typename Traits>
void GatherSolution<PHAL::AlbanyTraits::Tangent, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  auto nodeID = workset.wsElNodeEqID;

  const auto& x       = workset.x;
  const auto& xdot    = workset.xdot;
  const auto& xdotdot = workset.xdotdot;

  const auto& Vx       = workset.Vx;
  const auto& Vxdot    = workset.Vxdot;
  const auto& Vxdotdot = workset.Vxdotdot;

  //get const (read-only) view of x
  using const_data_t = Teuchos::ArrayRCP<const ST>;
  using const_mv_data_t = Teuchos::ArrayRCP<Teuchos::ArrayRCP<const ST>>;
  const_data_t x_data, xdot_data, xdotdot_data;
  const_mv_data_t Vx_data, Vxdot_data, Vxdotdot_data;

  x_data = Albany::getLocalData(x);

  if (xdot!=Teuchos::null)
    xdot_data = Albany::getLocalData(xdot);
  if (xdotdot!=Teuchos::null)
    xdotdot_data = Albany::getLocalData(xdotdot);

  if (Vx!=Teuchos::null)
    Vx_data = Albany::getLocalData(Vx);
  if (Vxdot!=Teuchos::null)
    Vxdot_data = Albany::getLocalData(Vx);
  if (Vxdotdot!=Teuchos::null)
    Vxdotdot_data = Albany::getLocalData(Vx);

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    for (int node = 0; node < this->numNodes; ++node) {
      for (int eq = 0; eq < numFields; eq++) {
        ref_t valref = get_ref(cell,node,eq);
        if (Vx != Teuchos::null && workset.j_coeff != 0.0) {
          valref = TanFadType(valref.size(), x_data[nodeID(cell,node,this->offset + eq)]);
          for (int k=0; k<workset.num_cols_x; k++)
            valref.fastAccessDx(k) =
              workset.j_coeff*Vx_data[k][nodeID(cell,node,this->offset + eq)];
        } else {
          valref = TanFadType(x_data[nodeID(cell,node,this->offset + eq)]);
        }

        if (workset.transientTerms && this->enableTransient) {
          ref_t valref_dot = get_ref_dot(cell,node,eq);
          valref_dot = TanFadType(valref_dot.size(), xdot_data[nodeID(cell,node,this->offset + eq)]);
          if (Vxdot != Teuchos::null && workset.m_coeff != 0.0) {
            for (int k=0; k<workset.num_cols_x; k++)
              valref_dot.fastAccessDx(k) =
                workset.m_coeff*Vxdot_data[k][nodeID(cell,node,this->offset + eq)];
          }
        }

        if (workset.accelerationTerms && this->enableAcceleration) {
          ref_t valref_dotdot = get_ref_dotdot(cell,node,eq);
          valref_dotdot = TanFadType(valref_dotdot.size(), xdotdot_data[nodeID(cell,node,this->offset + eq)]);
          if (Vxdotdot != Teuchos::null && workset.n_coeff != 0.0) {
            for (int k=0; k<workset.num_cols_x; k++)
              valref_dotdot.fastAccessDx(k) =
                workset.n_coeff*Vxdotdot_data[k][nodeID(cell,node,this->offset + eq)];
          }
        }
      }
    }
  }
}

// **********************************************************************

// **********************************************************************
// Specialization: Distributed Parameter Derivative
// **********************************************************************

template<typename Traits>
GatherSolution<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
GatherSolution(const Teuchos::ParameterList& p,
               const Teuchos::RCP<Albany::Layouts>& dl)
 : GatherSolutionBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p,dl)
{
  // Nothing else to do
}

template<typename Traits>
GatherSolution<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
GatherSolution(const Teuchos::ParameterList& p)
 : GatherSolutionBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p,p.get<Teuchos::RCP<Albany::Layouts> >("Layouts Struct"))
{
  // Nothing else to do
}

// **********************************************************************
template<typename Traits>
void GatherSolution<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  auto nodeID = workset.wsElNodeEqID;
  const auto& x       = workset.x;
  const auto& xdot    = workset.xdot;
  const auto& xdotdot = workset.xdotdot;

  //get const (read-only) view of x and xdot
  using const_data_t = Teuchos::ArrayRCP<const ST>;
  const_data_t x_data, xdot_data, xdotdot_data;

  x_data = Albany::getLocalData(x);
  if (xdot!=Teuchos::null)
    xdot_data = Albany::getLocalData(xdot);
  if (xdotdot!=Teuchos::null)
    xdotdot_data = Albany::getLocalData(xdotdot);

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    for (int node = 0; node < this->numNodes; ++node) {
      for (int eq = 0; eq < numFields; eq++) {
        get_ref(cell,node,eq) = x_data[nodeID(cell,node,this->offset + eq)];

        if (workset.transientTerms && this->enableTransient) {
          get_ref_dot(cell,node,eq) = xdot_data[nodeID(cell,node,this->offset + eq)];
        }

        if (workset.accelerationTerms && this->enableAcceleration) {
          get_ref_dotdot(cell,node,eq) = xdotdot_data[nodeID(cell,node,this->offset + eq)];
        }
      }
    }
  }
}

// **********************************************************************
// Specialization: HessianVec
// **********************************************************************

template<typename Traits>
GatherSolution<PHAL::AlbanyTraits::HessianVec, Traits>::
GatherSolution(const Teuchos::ParameterList& p,
               const Teuchos::RCP<Albany::Layouts>& dl)
 : GatherSolutionBase<PHAL::AlbanyTraits::HessianVec, Traits>(p,dl)
{
  // Nothing else to do
}

template<typename Traits>
GatherSolution<PHAL::AlbanyTraits::HessianVec, Traits>::
GatherSolution(const Teuchos::ParameterList& p)
 : GatherSolutionBase<PHAL::AlbanyTraits::HessianVec, Traits>(p,p.get<Teuchos::RCP<Albany::Layouts> >("Layouts Struct"))
{
  // Nothing else to do
}

// **********************************************************************
template<typename Traits>
void GatherSolution<PHAL::AlbanyTraits::HessianVec, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  const auto& x       = workset.x;
  const auto& xdot    = workset.xdot;
  const auto& xdotdot = workset.xdotdot;

  Teuchos::RCP<const Thyra_MultiVector> direction_x = workset.hessianWorkset.direction_x;

  auto nodeID = workset.wsElNodeEqID;

  bool g_xx_is_active = !workset.hessianWorkset.hess_vec_prod_g_xx.is_null();
  bool g_xp_is_active = !workset.hessianWorkset.hess_vec_prod_g_xp.is_null();
  bool g_px_is_active = !workset.hessianWorkset.hess_vec_prod_g_px.is_null();
  bool f_xx_is_active = !workset.hessianWorkset.hess_vec_prod_f_xx.is_null();
  bool f_xp_is_active = !workset.hessianWorkset.hess_vec_prod_f_xp.is_null();
  bool f_px_is_active = !workset.hessianWorkset.hess_vec_prod_f_px.is_null();

  //get const (read-only) view of x and xdot
  using const_data_t = Teuchos::ArrayRCP<const ST>;
  const_data_t x_data, xdot_data, xdotdot_data, direction_x_data;

  x_data = Albany::getLocalData(x);
  if (xdot!=Teuchos::null)
    xdot_data = Albany::getLocalData(xdot);
  if (xdotdot!=Teuchos::null)
    xdotdot_data = Albany::getLocalData(xdotdot);

  // is_x_active is true if we compute the Hessian-vector product contributions of either:
  // Hv_g_xx, Hv_g_xp, Hv_f_xx, or Hv_f_xp, i.e. if the first derivative is w.r.t. the solution.
  // If one of those is active, we have to initialize the first level of AD derivatives:
  // .fastAccessDx().val().
  const bool is_x_active = g_xx_is_active || g_xp_is_active || f_xx_is_active || f_xp_is_active;

  // is_x_direction_active is true if we compute the Hessian-vector product contributions of either:
  // Hv_g_xx, Hv_g_px, Hv_f_xx, or Hv_f_px, i.e. if the second derivative is w.r.t. the solution direction.
  // If one of those is active, we have to initialize the second level of AD derivatives:
  // .val().fastAccessDx().
  const bool is_x_direction_active = g_xx_is_active || g_px_is_active || f_xx_is_active || f_px_is_active;

  if(is_x_direction_active) {
    TEUCHOS_TEST_FOR_EXCEPTION(
        direction_x.is_null(),
        Teuchos::Exceptions::InvalidParameter,
        "\nError in GatherSolution<HessianVec, Traits>: "
        "direction_x is not set and the direction is active.\n");
    direction_x_data = Albany::getLocalData(direction_x->col(0));
  }

  const int neq = nodeID.extent(2);
  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {

    for (int node = 0; node < this->numNodes; ++node) {
      int firstunk = neq * node + this->offset;
      for (int eq = 0; eq < numFields; eq++) {
        ref_t valref = get_ref(cell,node,eq);
        RealType xvec_val = x_data[nodeID(cell,node,this->offset + eq)];

        valref = HessianVecFad(valref.size(), xvec_val);
        // If we differentiate w.r.t. the solution, we have to set the first
        // derivative to 1
        if (is_x_active)
          valref.fastAccessDx(firstunk + eq).val() = 1;
        // If we differentiate w.r.t. the solution direction, we have to set
        // the second derivative to the related direction value
        if (is_x_direction_active)
          valref.val().fastAccessDx(0) = direction_x_data[nodeID(cell,node,this->offset + eq)];

        if (workset.transientTerms && this->enableTransient) {
          ref_t valref_dot = get_ref_dot(cell,node,eq);
          valref_dot = xdot_data[nodeID(cell,node,this->offset + eq)];
        }

        if (workset.accelerationTerms && this->enableAcceleration) {
          ref_t valref_dotdot = get_ref_dotdot(cell,node,eq);
          valref_dotdot = xdotdot_data[nodeID(cell,node,this->offset + eq)];
        }
      }
    }
  }
}

} // namespace PHAL
