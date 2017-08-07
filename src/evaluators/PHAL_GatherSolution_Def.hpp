//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/13/14: only Epetra is SG and MP 

#include <vector>
#include <string>
#include <chrono>

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Albany_Utils.hpp"

namespace PHAL {

template<typename EvalT, typename Traits>
GatherSolutionBase<EvalT,Traits>::
GatherSolutionBase(const Teuchos::ParameterList& p,
                   const Teuchos::RCP<Albany::Layouts>& dl): numNodes(0)
{
  if (p.isType<int>("Tensor Rank"))
    tensorRank = p.get<int>("Tensor Rank");
  else
  if (p.isType<bool>("Vector Field")){
    if (p.get<bool>("Vector Field") == true)
      tensorRank = 1;
    else tensorRank = 0;
  }

  if (p.isType<bool>("Disable Transient"))
    enableTransient = !p.get<bool>("Disable Transient");
  else enableTransient = true;

  if (p.isType<bool>("Enable Acceleration"))
    enableAcceleration = p.get<bool>("Enable Acceleration");
  else enableAcceleration = false;

  Teuchos::ArrayRCP<std::string> solution_names;
  if (p.getEntryPtr("Solution Names")) {
    solution_names = p.get< Teuchos::ArrayRCP<std::string> >("Solution Names");
  }

  // scalar
  if ( tensorRank == 0 ) {
    val.resize(solution_names.size());
    for (std::size_t eq = 0; eq < solution_names.size(); ++eq) {
      PHX::MDField<ScalarT,Cell,Node> f(solution_names[eq],dl->node_scalar);
      val[eq] = f;
      this->addEvaluatedField(val[eq]);
    }
    // repeat for xdot if transient is enabled
    if (enableTransient) {
      const Teuchos::ArrayRCP<std::string>& names_dot =
        p.get< Teuchos::ArrayRCP<std::string> >("Time Dependent Solution Names");

      val_dot.resize(names_dot.size());
      for (std::size_t eq = 0; eq < names_dot.size(); ++eq) {
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
      for (std::size_t eq = 0; eq < names_dotdot.size(); ++eq) {
        PHX::MDField<ScalarT,Cell,Node> f(names_dotdot[eq],dl->node_scalar);
        val_dotdot[eq] = f;
        this->addEvaluatedField(val_dotdot[eq]);
      }
    }
    numFieldsBase = val.size();
  }
  // vector
  else 
  if ( tensorRank == 1 ) {
    PHX::MDField<ScalarT,Cell,Node,VecDim> f(solution_names[0],dl->node_vector);
    valVec= f;
    this->addEvaluatedField(valVec);
    // repeat for xdot if transient is enabled
    if (enableTransient) {
      const Teuchos::ArrayRCP<std::string>& names_dot =
        p.get< Teuchos::ArrayRCP<std::string> >("Time Dependent Solution Names");

      PHX::MDField<ScalarT,Cell,Node,VecDim> f(names_dot[0],dl->node_vector);
      valVec_dot= f;
      this->addEvaluatedField(valVec_dot);
    }
    // repeat for xdotdot if acceleration is enabled
    if (enableAcceleration) {
      const Teuchos::ArrayRCP<std::string>& names_dotdot =
        p.get< Teuchos::ArrayRCP<std::string> >("Solution Acceleration Names");

      PHX::MDField<ScalarT,Cell,Node,VecDim> f(names_dotdot[0],dl->node_vector);
      valVec_dotdot = f;
      this->addEvaluatedField(valVec_dotdot);
    }
    numFieldsBase = dl->node_vector->dimension(2);
  }
  // tensor
  else 
  if ( tensorRank == 2 ) {
    PHX::MDField<ScalarT,Cell,Node,VecDim,VecDim> f(solution_names[0],dl->node_tensor);
    valTensor = f;
    this->addEvaluatedField(valTensor);
    // repeat for xdot if transient is enabled
    if (enableTransient) {
      const Teuchos::ArrayRCP<std::string>& names_dot =
        p.get< Teuchos::ArrayRCP<std::string> >("Time Dependent Solution Names");

      PHX::MDField<ScalarT,Cell,Node,VecDim,VecDim> f(names_dot[0],dl->node_tensor);
      valTensor_dot = f;
      this->addEvaluatedField(valTensor_dot);
    }
    // repeat for xdotdot if acceleration is enabled
    if (enableAcceleration) {
      const Teuchos::ArrayRCP<std::string>& names_dotdot =
        p.get< Teuchos::ArrayRCP<std::string> >("Solution Acceleration Names");

      PHX::MDField<ScalarT,Cell,Node,VecDim,VecDim> f(names_dotdot[0],dl->node_tensor);
      valTensor_dotdot = f;
      this->addEvaluatedField(valTensor_dotdot);
    }
    numFieldsBase = (dl->node_tensor->dimension(2))*(dl->node_tensor->dimension(3));
  }

#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT 
  if ( tensorRank == 0 ) {
    val_kokkos.resize(numFieldsBase);
    if (enableTransient)
      val_dot_kokkos.resize(numFieldsBase);
    if (enableAcceleration)
      val_dotdot_kokkos.resize(numFieldsBase);
  }
#endif

  if (p.isType<int>("Offset of First DOF"))
    offset = p.get<int>("Offset of First DOF");
  else offset = 0;

  this->setName("Gather Solution"+PHX::typeAsString<EvalT>() );
}

// **********************************************************************
template<typename EvalT, typename Traits>
void GatherSolutionBase<EvalT,Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  if (tensorRank == 0) {
    for (std::size_t eq = 0; eq < numFieldsBase; ++eq)
      this->utils.setFieldData(val[eq],fm);
    if (enableTransient) {
      for (std::size_t eq = 0; eq < val_dot.size(); ++eq)
        this->utils.setFieldData(val_dot[eq],fm);
    }
    if (enableAcceleration) {
      for (std::size_t eq = 0; eq < val_dotdot.size(); ++eq)
        this->utils.setFieldData(val_dotdot[eq],fm);
    }
    numNodes = val[0].dimension(1);
  }
  else 
  if (tensorRank == 1) {
    this->utils.setFieldData(valVec,fm);
    if (enableTransient) this->utils.setFieldData(valVec_dot,fm);
    if (enableAcceleration) this->utils.setFieldData(valVec_dotdot,fm);
    numNodes = valVec.dimension(1);
  }
  else 
  if (tensorRank == 2) {
    this->utils.setFieldData(valTensor,fm);
    if (enableTransient) this->utils.setFieldData(valTensor_dot,fm);
    if (enableAcceleration) this->utils.setFieldData(valTensor_dotdot,fm);
    numNodes = valTensor.dimension(1);
  }
}

// **********************************************************************

// **********************************************************************
// Specialization: Residual
// **********************************************************************

template<typename Traits>
GatherSolution<PHAL::AlbanyTraits::Residual, Traits>::
GatherSolution(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
  GatherSolutionBase<PHAL::AlbanyTraits::Residual, Traits>(p,dl),
  numFields(GatherSolutionBase<PHAL::AlbanyTraits::Residual,Traits>::numFieldsBase)
{
}

template<typename Traits>
GatherSolution<PHAL::AlbanyTraits::Residual, Traits>::
GatherSolution(const Teuchos::ParameterList& p) :
  GatherSolutionBase<PHAL::AlbanyTraits::Residual, Traits>(p,p.get<Teuchos::RCP<Albany::Layouts> >("Layouts Struct")),
  numFields(GatherSolutionBase<PHAL::AlbanyTraits::Residual,Traits>::numFieldsBase)
{
}

// ********************************************************************
// Kokkos functors for Residual
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT 
template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherSolution<PHAL::AlbanyTraits::Residual, Traits>::
operator() (const PHAL_GatherSolRank1_Tag&, const int& cell) const{
  for (int node = 0; node < this->numNodes; ++node) 
    for (int eq = 0; eq < numFields; eq++) 
      (this->valVec)(cell,node,eq)= xT_constView(nodeID(cell, node,this->offset+eq));
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherSolution<PHAL::AlbanyTraits::Residual, Traits>::
operator() (const PHAL_GatherSolRank1_Transient_Tag&, const int& cell) const{
  for (int node = 0; node < this->numNodes; ++node)
    for (int eq = 0; eq < numFields; eq++)
      (this->valVec_dot)(cell,node,eq)= xdotT_constView(nodeID(cell, node, this->offset+eq));
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherSolution<PHAL::AlbanyTraits::Residual, Traits>::
operator() (const PHAL_GatherSolRank1_Acceleration_Tag&, const int& cell) const{
  for (int node = 0; node < this->numNodes; ++node)
    for (int eq = 0; eq < numFields; eq++)
      (this->valVec_dotdot)(cell,node,eq)= xdotdotT_constView(nodeID(cell, node, this->offset+eq));
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherSolution<PHAL::AlbanyTraits::Residual, Traits>::
operator() (const PHAL_GatherSolRank2_Tag&, const int& cell) const{
  for (int node = 0; node < this->numNodes; ++node)
    for (int eq = 0; eq < numFields; eq++)
      (this->valTensor)(cell,node,eq/numDim,eq%numDim)= xT_constView(nodeID(cell, node, this->offset+eq));
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherSolution<PHAL::AlbanyTraits::Residual, Traits>::
operator() (const PHAL_GatherSolRank2_Transient_Tag&, const int& cell) const{
  for (int node = 0; node < this->numNodes; ++node)
    for (int eq = 0; eq < numFields; eq++)
      (this->valTensor_dot)(cell,node,eq/numDim,eq%numDim)= xdotT_constView(nodeID(cell, node, this->offset+eq));
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherSolution<PHAL::AlbanyTraits::Residual, Traits>::
operator() (const PHAL_GatherSolRank2_Acceleration_Tag&, const int& cell) const{
  for (int node = 0; node < this->numNodes; ++node)
    for (int eq = 0; eq < numFields; eq++)
      (this->valTensor_dotdot)(cell,node,eq/numDim,eq%numDim)= xdotdotT_constView(nodeID(cell, node, this->offset+eq));
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherSolution<PHAL::AlbanyTraits::Residual, Traits>::
operator() (const PHAL_GatherSolRank0_Tag&, const int& cell) const{
  for (int node = 0; node < this->numNodes; ++node)
    for (int eq = 0; eq < numFields; eq++)
      d_val[eq](cell,node)= xT_constView(nodeID(cell, node, this->offset+eq));
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherSolution<PHAL::AlbanyTraits::Residual, Traits>::
operator() (const PHAL_GatherSolRank0_Transient_Tag&, const int& cell) const{
  for (int node = 0; node < this->numNodes; ++node)
    for (int eq = 0; eq < numFields; eq++)
      d_val_dot[eq](cell,node)= xdotT_constView(nodeID(cell, node, this->offset+eq));
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherSolution<PHAL::AlbanyTraits::Residual, Traits>::
operator() (const PHAL_GatherSolRank0_Acceleration_Tag&, const int& cell) const{
  for (int node = 0; node < this->numNodes; ++node)
    for (int eq = 0; eq < numFields; eq++)
      d_val_dotdot[eq](cell,node)= xdotdotT_constView(nodeID(cell, node, this->offset+eq));
}

#endif

// **********************************************************************
template<typename Traits>
void GatherSolution<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  auto nodeID = workset.wsElNodeEqID;
  Teuchos::RCP<const Tpetra_Vector> xT = workset.xT;
  Teuchos::RCP<const Tpetra_Vector> xdotT = workset.xdotT;
  Teuchos::RCP<const Tpetra_Vector> xdotdotT = workset.xdotdotT;
  Teuchos::ArrayRCP<const ST> xT_constView, xdotT_constView, xdotdotT_constView;
  xT_constView = xT->get1dView();
  if(!xdotT.is_null())
    xdotT_constView = xdotT->get1dView();
  if(!xdotdotT.is_null())
    xdotdotT_constView = xdotdotT->get1dView();

  if (this->tensorRank == 1) {
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      for (std::size_t node = 0; node < this->numNodes; ++node) {
        for (std::size_t eq = 0; eq < numFields; eq++) 
          (this->valVec)(cell,node,eq) = xT_constView[nodeID(cell,node,this->offset + eq)];
        if (workset.transientTerms && this->enableTransient) {
          for (std::size_t eq = 0; eq < numFields; eq++) 
            (this->valVec_dot)(cell,node,eq) = xdotT_constView[nodeID(cell,node,this->offset + eq)];
        }
        if (workset.accelerationTerms && this->enableAcceleration) {
          for (std::size_t eq = 0; eq < numFields; eq++) 
            (this->valVec_dotdot)(cell,node,eq) = xdotdotT_constView[nodeID(cell,node,this->offset + eq)];
        }
      }
    }
  } else 
  if (this->tensorRank == 2) {
    int numDim = this->valTensor.dimension(2);
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      for (std::size_t node = 0; node < this->numNodes; ++node) {
        for (std::size_t eq = 0; eq < numFields; eq++) 
          (this->valTensor)(cell,node,eq/numDim,eq%numDim) = xT_constView[nodeID(cell,node,this->offset + eq)];
        if (workset.transientTerms && this->enableTransient) {
          for (std::size_t eq = 0; eq < numFields; eq++) 
            (this->valTensor_dot)(cell,node,eq/numDim,eq%numDim) = xdotT_constView[nodeID(cell,node,this->offset + eq)];
        }
        if (workset.accelerationTerms && this->enableAcceleration) {
          for (std::size_t eq = 0; eq < numFields; eq++) 
            (this->valTensor_dotdot)(cell,node,eq/numDim,eq%numDim) = xdotdotT_constView[nodeID(cell,node,this->offset + eq)];
        }
      }
    }
  } else {
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      for (std::size_t node = 0; node < this->numNodes; ++node) {
        for (std::size_t eq = 0; eq < numFields; eq++) 
          (this->val[eq])(cell,node) = xT_constView[nodeID(cell,node,this->offset + eq)];
        if (workset.transientTerms && this->enableTransient) {
          for (std::size_t eq = 0; eq < numFields; eq++) 
            (this->val_dot[eq])(cell,node) = xdotT_constView[nodeID(cell,node,this->offset + eq)];
        }
        if (workset.accelerationTerms && this->enableAcceleration) {
          for (std::size_t eq = 0; eq < numFields; eq++) 
            (this->val_dotdot[eq])(cell,node) = xdotdotT_constView[nodeID(cell,node,this->offset + eq)];
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

  // Get Tpetra vector view from a specific device 
  auto xT_2d = workset.xT->template getLocalView<PHX::Device>();
  xT_constView = Kokkos::subview(xT_2d, Kokkos::ALL(), 0);
  if(!workset.xdotT.is_null()) {
    auto xdotT_2d = workset.xdotT->template getLocalView<PHX::Device>();
    xdotT_constView = Kokkos::subview(xdotT_2d, Kokkos::ALL(), 0);
  }
  if(!workset.xdotdotT.is_null()) {
    auto xdotdotT_2d = workset.xdotdotT->template getLocalView<PHX::Device>();
    xdotdotT_constView = Kokkos::subview(xdotdotT_2d, Kokkos::ALL(), 0);
  }

  if (this->tensorRank == 2){
    numDim = this->valTensor.dimension(2);
    Kokkos::parallel_for(PHAL_GatherSolRank2_Policy(0,workset.numCells),*this);
    cudaCheckError();

    if (workset.transientTerms && this->enableTransient) {
      Kokkos::parallel_for(PHAL_GatherSolRank2_Transient_Policy(0,workset.numCells),*this);
      cudaCheckError();
    }

    if (workset.accelerationTerms && this->enableAcceleration) {
      Kokkos::parallel_for(PHAL_GatherSolRank2_Acceleration_Policy(0,workset.numCells),*this);
      cudaCheckError();
    }
  }
  
  else if (this->tensorRank == 1){
    Kokkos::parallel_for(PHAL_GatherSolRank1_Policy(0,workset.numCells),*this);
    cudaCheckError();

    if (workset.transientTerms && this->enableTransient) {
      Kokkos::parallel_for(PHAL_GatherSolRank1_Transient_Policy(0,workset.numCells),*this);
      cudaCheckError();
    }

    if (workset.accelerationTerms && this->enableAcceleration) {
      Kokkos::parallel_for(PHAL_GatherSolRank1_Acceleration_Policy(0,workset.numCells),*this);
      cudaCheckError();
    }
  }

  else {
    // Get MDField views from std::vector
    for (int i =0; i<numFields;i++){
      //val_kokkos[i]=this->val[i].get_view();
      val_kokkos[i]=this->val[i].get_static_view();
    }
    d_val=val_kokkos.template view<ExecutionSpace>();

    Kokkos::parallel_for(PHAL_GatherSolRank0_Policy(0,workset.numCells),*this);
    cudaCheckError();

    if (workset.transientTerms && this->enableTransient){ 
      // Get MDField views from std::vector
      for (int i =0; i<numFields;i++){
        //val_dot_kokkos[i]=this->val_dot[i].get_view();
        val_dot_kokkos[i]=this->val_dot[i].get_static_view();
      }
      d_val_dot=val_dot_kokkos.template view<ExecutionSpace>();

      Kokkos::parallel_for(PHAL_GatherSolRank0_Transient_Policy(0,workset.numCells),*this);  
      cudaCheckError();
    }
    if (workset.accelerationTerms && this->enableAcceleration){
      // Get MDField views from std::vector
      for (int i =0; i<numFields;i++){
        //val_dotdot_kokkos[i]=this->val_dotdot[i].get_view();
        val_dotdot_kokkos[i]=this->val_dotdot[i].get_static_view();
      }
      d_val_dotdot=val_dotdot_kokkos.template view<ExecutionSpace>();

      Kokkos::parallel_for(PHAL_GatherSolRank0_Acceleration_Policy(0,workset.numCells),*this);
      cudaCheckError();
    }
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
		      const Teuchos::RCP<Albany::Layouts>& dl) :
GatherSolutionBase<PHAL::AlbanyTraits::Jacobian, Traits>(p,dl),
numFields(GatherSolutionBase<PHAL::AlbanyTraits::Jacobian,Traits>::numFieldsBase)
{
}

template<typename Traits>
GatherSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
GatherSolution(const Teuchos::ParameterList& p) :
GatherSolutionBase<PHAL::AlbanyTraits::Jacobian, Traits>(p,p.get<Teuchos::RCP<Albany::Layouts> >("Layouts Struct")),
numFields(GatherSolutionBase<PHAL::AlbanyTraits::Jacobian,Traits>::numFieldsBase)
{
}

//********************************************************************
////Kokkos functors for Jacobian
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT 
template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
operator() (const PHAL_GatherJacRank2_Tag&, const int& cell) const{
  for (int node = 0; node < this->numNodes; ++node){
    int firstunk = neq * node + this->offset;
    for (int eq = 0; eq < numFields; eq++){
      typename PHAL::Ref<ScalarT>::type valref = (this->valTensor)(cell,node,eq/numDim,eq%numDim);
      valref=FadType(valref.size(), xT_constView(nodeID(cell,node,this->offset+eq)));
      valref.fastAccessDx(firstunk + eq) =j_coeff;
    }
  }
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
operator() (const PHAL_GatherJacRank2_Transient_Tag&, const int& cell) const{
  for (int node = 0; node < this->numNodes; ++node){
    int firstunk = neq * node + this->offset;
    for (int eq = 0; eq < numFields; eq++){
      typename PHAL::Ref<ScalarT>::type valref = (this->valTensor_dot)(cell,node,eq/numDim,eq%numDim);
      valref =FadType(valref.size(), xdotT_constView(nodeID(cell,node,this->offset+eq)));
      valref.fastAccessDx(firstunk + eq) =m_coeff;
    }
  }
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
operator() (const PHAL_GatherJacRank2_Acceleration_Tag&, const int& cell) const{
  for (int node = 0; node < this->numNodes; ++node){
    int firstunk = neq * node + this->offset;
    for (int eq = 0; eq < numFields; eq++){
      typename PHAL::Ref<ScalarT>::type valref = (this->valTensor_dotdot)(cell,node,eq/numDim,eq%numDim);
      valref=FadType(valref.size(), xdotdotT_constView(nodeID(cell,node,this->offset+eq)));
      valref.fastAccessDx(firstunk + eq) =n_coeff;
    }
  }
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
operator() (const PHAL_GatherJacRank1_Tag&, const int& cell) const{
  for (int node = 0; node < this->numNodes; node++){
    int firstunk = neq * node + this->offset;
    for (int eq = 0; eq < numFields; eq++){
      typename PHAL::Ref<ScalarT>::type valref = (this->valVec)(cell,node,eq);
      valref =FadType(valref.size(), xT_constView(nodeID(cell,node,this->offset+eq)));
      valref.fastAccessDx(firstunk + eq) =j_coeff;
    }
  }
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
operator() (const PHAL_GatherJacRank1_Transient_Tag&, const int& cell) const{
  for (int node = 0; node < this->numNodes; ++node){
    int firstunk = neq * node + this->offset;
    for (int eq = 0; eq < numFields; eq++){
      typename PHAL::Ref<ScalarT>::type valref = (this->valVec_dot)(cell,node,eq);
      valref =FadType(valref.size(), xdotT_constView(nodeID(cell,node,this->offset+eq)));
      valref.fastAccessDx(firstunk + eq) =m_coeff;
    }
  }  
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
operator() (const PHAL_GatherJacRank1_Acceleration_Tag&, const int& cell) const{
  for (int node = 0; node < this->numNodes; ++node){
    int firstunk = neq * node + this->offset;
    for (int eq = 0; eq < numFields; eq++){
      typename PHAL::Ref<ScalarT>::type valref = (this->valVec_dotdot)(cell,node,eq);
      valref =FadType(valref.size(), xdotdotT_constView(nodeID(cell,node,this->offset+eq)));
      valref.fastAccessDx(firstunk + eq) =n_coeff;
    }
  }
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
operator() (const PHAL_GatherJacRank0_Tag&, const int& cell) const{
  for (int node = 0; node < this->numNodes; ++node){
    int firstunk = neq * node + this->offset;
    for (int eq = 0; eq < numFields; eq++){
      typename PHAL::Ref<ScalarT>::type valref = d_val[eq](cell,node);
      valref =FadType(valref.size(), xT_constView(nodeID(cell,node,this->offset+eq)));
      valref.fastAccessDx(firstunk + eq) =j_coeff;
    }
  }
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
operator() (const PHAL_GatherJacRank0_Transient_Tag&, const int& cell) const{
  for (int node = 0; node < this->numNodes; ++node){
    int firstunk = neq * node + this->offset;
    for (int eq = 0; eq < numFields; eq++){
      typename PHAL::Ref<ScalarT>::type valref = d_val_dot[eq](cell,node);
      valref =FadType(valref.size(), xdotT_constView(nodeID(cell,node,this->offset+eq)));
      valref.fastAccessDx(firstunk + eq) =m_coeff;
    }
  }
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
operator() (const PHAL_GatherJacRank0_Acceleration_Tag&, const int& cell) const{
  for (int node = 0; node < this->numNodes; ++node){
    int firstunk = neq * node + this->offset;
    for (int eq = 0; eq < numFields; eq++){
      typename PHAL::Ref<ScalarT>::type valref = d_val_dotdot[eq](cell,node);
      valref = FadType(valref.size(), xdotdotT_constView(nodeID(cell,node,this->offset+eq)));
      valref.fastAccessDx(firstunk + eq) = n_coeff;
    }
  }
}

#endif

// **********************************************************************
template<typename Traits>
void GatherSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  auto nodeID = workset.wsElNodeEqID;
  Teuchos::RCP<const Tpetra_Vector> xT = workset.xT;
  Teuchos::RCP<const Tpetra_Vector> xdotT = workset.xdotT;
  Teuchos::RCP<const Tpetra_Vector> xdotdotT = workset.xdotdotT;
  Teuchos::ArrayRCP<const ST> xT_constView, xdotT_constView, xdotdotT_constView;
  xT_constView = xT->get1dView();
  if(!xdotT.is_null())
    xdotT_constView = xdotT->get1dView();
  if(!xdotdotT.is_null())
    xdotdotT_constView = xdotdotT->get1dView();

  int numDim = 0;
  if (this->tensorRank==2) numDim = this->valTensor.dimension(2); // only needed for tensor fields

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const int neq = nodeID.dimension(2);
    const std::size_t num_dof = neq * this->numNodes;

    for (std::size_t node = 0; node < this->numNodes; ++node) {
      int firstunk = neq * node + this->offset;
      for (std::size_t eq = 0; eq < numFields; eq++) {
        typename PHAL::Ref<ScalarT>::type
          valref = (this->tensorRank == 0 ? this->val[eq](cell,node) :
                    this->tensorRank == 1 ? this->valVec(cell,node,eq) :
                    this->valTensor(cell,node, eq/numDim, eq%numDim));
        valref = FadType(valref.size(), xT_constView[nodeID(cell,node,this->offset + eq)]);
        // valref.setUpdateValue(!workset.ignore_residual); Not used anymore
        valref.fastAccessDx(firstunk + eq) = workset.j_coeff;
      }
      if (workset.transientTerms && this->enableTransient) {
        for (std::size_t eq = 0; eq < numFields; eq++) {
        typename PHAL::Ref<ScalarT>::type
          valref = (this->tensorRank == 0 ? this->val_dot[eq](cell,node) :
                    this->tensorRank == 1 ? this->valVec_dot(cell,node,eq) :
                    this->valTensor_dot(cell,node, eq/numDim, eq%numDim));
        valref = FadType(valref.size(), xdotT_constView[nodeID(cell,node,this->offset + eq)]);
        valref.fastAccessDx(firstunk + eq) = workset.m_coeff;
        }
      }
      if (workset.accelerationTerms && this->enableAcceleration) {
        for (std::size_t eq = 0; eq < numFields; eq++) {
        typename PHAL::Ref<ScalarT>::type
          valref = (this->tensorRank == 0 ? this->val_dotdot[eq](cell,node) :
                    this->tensorRank == 1 ? this->valVec_dotdot(cell,node,eq) :
                    this->valTensor_dotdot(cell,node, eq/numDim, eq%numDim));
        valref = FadType(valref.size(), xdotdotT_constView[nodeID(cell,node,this->offset + eq)]);
        valref.fastAccessDx(firstunk + eq) = workset.n_coeff;
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
  neq = nodeID.dimension(2);
  j_coeff=workset.j_coeff;
  m_coeff=workset.m_coeff;
  n_coeff=workset.n_coeff;

  // Get Tpetra vector view from a specific device 
  auto xT_2d = workset.xT->template getLocalView<PHX::Device>();
  xT_constView = Kokkos::subview(xT_2d, Kokkos::ALL(), 0);
  if(!workset.xdotT.is_null()) {
    auto xdotT_2d = workset.xdotT->template getLocalView<PHX::Device>();
    xdotT_constView = Kokkos::subview(xdotT_2d, Kokkos::ALL(), 0);
  }
  if(!workset.xdotdotT.is_null()) {
    auto xdotdotT_2d = workset.xdotdotT->template getLocalView<PHX::Device>();
    xdotdotT_constView = Kokkos::subview(xdotdotT_2d, Kokkos::ALL(), 0);
  }

  if (this->tensorRank == 2) {
    numDim = this->valTensor.dimension(2);

    Kokkos::parallel_for(PHAL_GatherJacRank2_Policy(0,workset.numCells),*this);
    cudaCheckError();

    if (workset.transientTerms && this->enableTransient) {
      Kokkos::parallel_for(PHAL_GatherJacRank2_Transient_Policy(0,workset.numCells),*this);
      cudaCheckError();
    }

    if (workset.accelerationTerms && this->enableAcceleration) {
      Kokkos::parallel_for(PHAL_GatherJacRank2_Acceleration_Policy(0,workset.numCells),*this);
      cudaCheckError();
    }
  } 

  else if (this->tensorRank == 1) {
    Kokkos::parallel_for(PHAL_GatherJacRank1_Policy(0,workset.numCells),*this);
    cudaCheckError();

    if (workset.transientTerms && this->enableTransient) {
      Kokkos::parallel_for(PHAL_GatherJacRank1_Transient_Policy(0,workset.numCells),*this);
      cudaCheckError();
    }

    if (workset.accelerationTerms && this->enableAcceleration) {
      Kokkos::parallel_for(PHAL_GatherJacRank1_Acceleration_Policy(0,workset.numCells),*this);
      cudaCheckError();
    }
  }

  else {
    // Get MDField views from std::vector
    for (int i =0; i<numFields;i++) {
      //val_kokkos[i]=this->val[i].get_view();
      val_kokkos[i]=this->val[i].get_static_view();
    }
    d_val=val_kokkos.template view<ExecutionSpace>();

    Kokkos::parallel_for(PHAL_GatherJacRank0_Policy(0,workset.numCells),*this);
    cudaCheckError();

    if (workset.transientTerms && this->enableTransient) {
    // Get MDField views from std::vector
      for (int i =0; i<numFields;i++) {
        //val_dot_kokkos[i]=this->val_dot[i].get_view();
        val_dot_kokkos[i]=this->val_dot[i].get_static_view();
      }
      d_val_dot=val_dot_kokkos.template view<ExecutionSpace>();

      Kokkos::parallel_for(PHAL_GatherJacRank0_Transient_Policy(0,workset.numCells),*this);
      cudaCheckError();
    }

    if (workset.accelerationTerms && this->enableAcceleration) {
    // Get MDField views from std::vector
      for (int i =0; i<numFields;i++) {
        //val_dotdot_kokkos[i]=this->val_dotdot[i].get_view();
        val_dotdot_kokkos[i]=this->val_dotdot[i].get_static_view();
      }
      d_val_dot=val_dotdot_kokkos.template view<ExecutionSpace>();

      Kokkos::parallel_for(PHAL_GatherJacRank0_Acceleration_Policy(0,workset.numCells),*this);
      cudaCheckError();
    }
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
                              const Teuchos::RCP<Albany::Layouts>& dl) :
  GatherSolutionBase<PHAL::AlbanyTraits::Tangent, Traits>(p,dl),
  numFields(GatherSolutionBase<PHAL::AlbanyTraits::Tangent,Traits>::numFieldsBase)
{
}

template<typename Traits>
GatherSolution<PHAL::AlbanyTraits::Tangent, Traits>::
GatherSolution(const Teuchos::ParameterList& p) :
  GatherSolutionBase<PHAL::AlbanyTraits::Tangent, Traits>(p,p.get<Teuchos::RCP<Albany::Layouts> >("Layouts Struct")),
  numFields(GatherSolutionBase<PHAL::AlbanyTraits::Tangent,Traits>::numFieldsBase)
{
}

// **********************************************************************
template<typename Traits>
void GatherSolution<PHAL::AlbanyTraits::Tangent, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  auto nodeID = workset.wsElNodeEqID;
  Teuchos::RCP<const Tpetra_Vector> xT = workset.xT;
  Teuchos::RCP<const Tpetra_Vector> xdotT = workset.xdotT;
  Teuchos::RCP<const Tpetra_Vector> xdotdotT = workset.xdotdotT;
  Teuchos::RCP<const Tpetra_MultiVector> VxT = workset.VxT;
  Teuchos::RCP<const Tpetra_MultiVector> VxdotT = workset.VxdotT;
  Teuchos::RCP<const Tpetra_MultiVector> VxdotdotT = workset.VxdotdotT;
  Teuchos::RCP<const Tpetra_MultiVector> VpT = workset.VpT;

  //get const (read-only) view of xT and xdotT
  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();

  Teuchos::RCP<ParamVec> params = workset.params;
  //int num_cols_tot = workset.param_offset + workset.num_cols_p;

  int numDim = 0;
  if(this->tensorRank==2) numDim = this->valTensor.dimension(2); // only needed for tensor fields
  
  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    for (std::size_t node = 0; node < this->numNodes; ++node) {
      for (std::size_t eq = 0; eq < numFields; eq++) {
        typename PHAL::Ref<ScalarT>::type
          valref = ((this->tensorRank == 2) ? (this->valTensor)(cell,node,eq/numDim,eq%numDim) :
                    (this->tensorRank == 1) ? (this->valVec)(cell,node,eq) :
                    (this->val[eq])(cell,node));
        if (VxT != Teuchos::null && workset.j_coeff != 0.0) {
          valref = TanFadType(valref.size(), xT_constView[nodeID(cell,node,this->offset + eq)]);
          for (int k=0; k<workset.num_cols_x; k++)
            valref.fastAccessDx(k) =
              workset.j_coeff*VxT->getData(k)[nodeID(cell,node,this->offset + eq)];
        }
        else
          valref = TanFadType(xT_constView[nodeID(cell,node,this->offset + eq)]);
      }
   }


   if (workset.transientTerms && this->enableTransient) {
    Teuchos::ArrayRCP<const ST> xdotT_constView = xdotT->get1dView();
    for (std::size_t node = 0; node < this->numNodes; ++node) {
        for (std::size_t eq = 0; eq < numFields; eq++) {
        typename PHAL::Ref<ScalarT>::type
          valref = ((this->tensorRank == 2) ? (this->valTensor_dot)(cell,node,eq/numDim,eq%numDim) :
                    (this->tensorRank == 1) ? (this->valVec_dot)(cell,node,eq) :
                    (this->val_dot[eq])(cell,node));
          valref = TanFadType(valref.size(), xdotT_constView[nodeID(cell,node,this->offset + eq)]);
          if (VxdotT != Teuchos::null && workset.m_coeff != 0.0) {
            for (int k=0; k<workset.num_cols_x; k++)
              valref.fastAccessDx(k) =
                workset.m_coeff*VxdotT->getData(k)[nodeID(cell,node,this->offset + eq)];
          }
        }
      }
   }

   if (workset.accelerationTerms && this->enableAcceleration) {
    Teuchos::ArrayRCP<const ST> xdotdotT_constView = xdotdotT->get1dView();
    for (std::size_t node = 0; node < this->numNodes; ++node) {
        for (std::size_t eq = 0; eq < numFields; eq++) {
        typename PHAL::Ref<ScalarT>::type
          valref = ((this->tensorRank == 2) ? (this->valTensor_dotdot)(cell,node,eq/numDim,eq%numDim) :
                    (this->tensorRank == 1) ? (this->valVec_dotdot)(cell,node,eq) :
                    (this->val_dotdot[eq])(cell,node));

          valref = TanFadType(valref.size(), xdotdotT_constView[nodeID(cell,node,this->offset + eq)]);
          if (VxdotdotT != Teuchos::null && workset.n_coeff != 0.0) {
            for (int k=0; k<workset.num_cols_x; k++)
              valref.fastAccessDx(k) =
                workset.n_coeff*VxdotdotT->getData(k)[nodeID(cell,node,this->offset + eq)];
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
               const Teuchos::RCP<Albany::Layouts>& dl) :
  GatherSolutionBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p,dl),
  numFields(GatherSolutionBase<PHAL::AlbanyTraits::DistParamDeriv,Traits>::numFieldsBase)
{
}

template<typename Traits>
GatherSolution<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
GatherSolution(const Teuchos::ParameterList& p) :
  GatherSolutionBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p,p.get<Teuchos::RCP<Albany::Layouts> >("Layouts Struct")),
  numFields(GatherSolutionBase<PHAL::AlbanyTraits::DistParamDeriv,Traits>::numFieldsBase)
{
}

// **********************************************************************
template<typename Traits>
void GatherSolution<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  auto nodeID = workset.wsElNodeEqID;
  Teuchos::RCP<const Tpetra_Vector> xT = workset.xT;
  Teuchos::RCP<const Tpetra_Vector> xdotT = workset.xdotT;
  Teuchos::RCP<const Tpetra_Vector> xdotdotT = workset.xdotdotT;

  //get const (read-only) view of xT and xdotT
  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();

  if (this->tensorRank == 1) {
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      for (std::size_t node = 0; node < this->numNodes; ++node) {
        for (std::size_t eq = 0; eq < numFields; eq++)
          (this->valVec)(cell,node,eq) = xT_constView[nodeID(cell,node,this->offset + eq)];
      }

    if (workset.transientTerms && this->enableTransient) {
    Teuchos::ArrayRCP<const ST> xdotT_constView = xdotT->get1dView();
      for (std::size_t node = 0; node < this->numNodes; ++node) {
          for (std::size_t eq = 0; eq < numFields; eq++)
            (this->valVec_dot)(cell,node,eq) = xdotT_constView[nodeID(cell,node,this->offset + eq)];
      }
    }

    if (workset.accelerationTerms && this->enableAcceleration) {
    Teuchos::ArrayRCP<const ST> xdotdotT_constView = xdotdotT->get1dView();
      for (std::size_t node = 0; node < this->numNodes; ++node) {
          for (std::size_t eq = 0; eq < numFields; eq++)
            (this->valVec_dotdot)(cell,node,eq) = xdotdotT_constView[nodeID(cell,node,this->offset + eq)];
        }
      }
    } 
  } else
  if (this->tensorRank == 2) {
    int numDim = this->valTensor.dimension(2);
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      for (std::size_t node = 0; node < this->numNodes; ++node) {
        for (std::size_t eq = 0; eq < numFields; eq++)
          (this->valTensor)(cell,node,eq/numDim,eq%numDim) = xT_constView[nodeID(cell,node,this->offset + eq)];
      }

    if (workset.transientTerms && this->enableTransient) {
    Teuchos::ArrayRCP<const ST> xdotT_constView = xdotT->get1dView();
      for (std::size_t node = 0; node < this->numNodes; ++node) {
          for (std::size_t eq = 0; eq < numFields; eq++)
            (this->valTensor_dot)(cell,node,eq/numDim,eq%numDim) = xdotT_constView[nodeID(cell,node,this->offset + eq)];
      }
    }

    if (workset.accelerationTerms && this->enableAcceleration) {
    Teuchos::ArrayRCP<const ST> xdotdotT_constView = xdotdotT->get1dView();
      for (std::size_t node = 0; node < this->numNodes; ++node) {
          for (std::size_t eq = 0; eq < numFields; eq++)
            (this->valTensor_dotdot)(cell,node,eq/numDim,eq%numDim) = xdotdotT_constView[nodeID(cell,node,this->offset + eq)];
        }
      }
    }
  } else {
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      for (std::size_t node = 0; node < this->numNodes; ++node) {
        for (std::size_t eq = 0; eq < numFields; eq++)
          (this->val[eq])(cell,node) = xT_constView[nodeID(cell,node,this->offset + eq)];
      }
    if (workset.transientTerms && this->enableTransient) {
    Teuchos::ArrayRCP<const ST> xdotT_constView = xdotT->get1dView();
      for (std::size_t node = 0; node < this->numNodes; ++node) {
          for (std::size_t eq = 0; eq < numFields; eq++)
            (this->val_dot[eq])(cell,node) = xdotT_constView[nodeID(cell,node,this->offset + eq)];
      }
    }

    if (workset.accelerationTerms && this->enableAcceleration) {
    Teuchos::ArrayRCP<const ST> xdotdotT_constView = xdotdotT->get1dView();
      for (std::size_t node = 0; node < this->numNodes; ++node) {
          for (std::size_t eq = 0; eq < numFields; eq++)
            (this->val_dotdot[eq])(cell,node) = xdotdotT_constView[nodeID(cell,node,this->offset + eq)];
        }
      }
    }
  }
}

// **********************************************************************

// **********************************************************************
// Specialization: Stochastic Galerkin Residual
// **********************************************************************

#ifdef ALBANY_SG
template<typename Traits>
GatherSolution<PHAL::AlbanyTraits::SGResidual, Traits>::
GatherSolution(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
  GatherSolutionBase<PHAL::AlbanyTraits::SGResidual, Traits>(p,dl),
  numFields(GatherSolutionBase<PHAL::AlbanyTraits::SGResidual,Traits>::numFieldsBase)
{
}

template<typename Traits>
GatherSolution<PHAL::AlbanyTraits::SGResidual, Traits>::
GatherSolution(const Teuchos::ParameterList& p) :
  GatherSolutionBase<PHAL::AlbanyTraits::SGResidual, Traits>(p,p.get<Teuchos::RCP<Albany::Layouts> >("Layouts Struct")),
  numFields(GatherSolutionBase<PHAL::AlbanyTraits::SGResidual,Traits>::numFieldsBase)
{
}

// **********************************************************************
template<typename Traits>
void GatherSolution<PHAL::AlbanyTraits::SGResidual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  auto nodeID = workset.wsElNodeEqID;
  Teuchos::RCP<Stokhos::OrthogPolyExpansion<int,RealType> > sg_expansion =
    workset.sg_expansion;
  Teuchos::RCP<const Stokhos::EpetraVectorOrthogPoly > x =
    workset.sg_x;
  Teuchos::RCP<const Stokhos::EpetraVectorOrthogPoly > xdot =
    workset.sg_xdot;
  Teuchos::RCP<const Stokhos::EpetraVectorOrthogPoly > xdotdot =
    workset.sg_xdotdot;

  int numDim = 0;
  if(this->tensorRank==2) numDim = this->valTensor.dimension(2); // only needed for tensor fields

  int nblock = x->size();
  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    for (std::size_t node = 0; node < this->numNodes; ++node) {
      for (std::size_t eq = 0; eq < numFields; eq++) {
        typename PHAL::Ref<ScalarT>::type
          valref = ((this->tensorRank == 2) ? (this->valTensor)(cell,node,eq/numDim,eq%numDim) :
                    (this->tensorRank == 1) ? (this->valVec)(cell,node,eq) :
                    (this->val[eq])(cell,node));
        valref.reset(sg_expansion);
        valref.copyForWrite();
        for (int block=0; block<nblock; block++)
          valref.fastAccessCoeff(block) =
            (*x)[block][nodeID(cell,node,this->offset + eq)];
      }
      if (workset.transientTerms && this->enableTransient) {
        for (std::size_t eq = 0; eq < numFields; eq++) {
          typename PHAL::Ref<ScalarT>::type
            valref = ((this->tensorRank == 2) ? (this->valTensor_dot)(cell,node,eq/numDim,eq%numDim) :
                      (this->tensorRank == 1) ? (this->valVec_dot)(cell,node,eq) :
                      (this->val_dot[eq])(cell,node));
          valref.reset(sg_expansion);
          valref.copyForWrite();
          for (int block=0; block<nblock; block++)
            valref.fastAccessCoeff(block) =
              (*xdot)[block][nodeID(cell,node,this->offset + eq)];
        }
      }
      if (workset.accelerationTerms && this->enableAcceleration) {
        for (std::size_t eq = 0; eq < numFields; eq++) {
          typename PHAL::Ref<ScalarT>::type
            valref = ((this->tensorRank == 2) ? (this->valTensor_dotdot)(cell,node,eq/numDim,eq%numDim) :
                      (this->tensorRank == 1) ? (this->valVec_dotdot)(cell,node,eq) :
                      (this->val_dotdot[eq])(cell,node));
          valref.reset(sg_expansion);
          valref.copyForWrite();
          for (int block=0; block<nblock; block++)
            valref.fastAccessCoeff(block) =
              (*xdotdot)[block][nodeID(cell,node,this->offset + eq)];
        }
      }
    }
  }

}

// **********************************************************************

// **********************************************************************
// Specialization: Stochastic Galerkin Jacobian
// **********************************************************************

template<typename Traits>
GatherSolution<PHAL::AlbanyTraits::SGJacobian, Traits>::
GatherSolution(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
  GatherSolutionBase<PHAL::AlbanyTraits::SGJacobian, Traits>(p,dl),
  numFields(GatherSolutionBase<PHAL::AlbanyTraits::SGJacobian,Traits>::numFieldsBase)
{
}

template<typename Traits>
GatherSolution<PHAL::AlbanyTraits::SGJacobian, Traits>::
GatherSolution(const Teuchos::ParameterList& p) :
  GatherSolutionBase<PHAL::AlbanyTraits::SGJacobian, Traits>(p,p.get<Teuchos::RCP<Albany::Layouts> >("Layouts Struct")),
  numFields(GatherSolutionBase<PHAL::AlbanyTraits::SGJacobian,Traits>::numFieldsBase)
{
}

// **********************************************************************
template<typename Traits>
void GatherSolution<PHAL::AlbanyTraits::SGJacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  auto nodeID = workset.wsElNodeEqID;
  Teuchos::RCP<Stokhos::OrthogPolyExpansion<int,RealType> > sg_expansion =
    workset.sg_expansion;
  Teuchos::RCP<const Stokhos::EpetraVectorOrthogPoly > x =
    workset.sg_x;
  Teuchos::RCP<const Stokhos::EpetraVectorOrthogPoly > xdot =
    workset.sg_xdot;
  Teuchos::RCP<const Stokhos::EpetraVectorOrthogPoly > xdotdot =
    workset.sg_xdotdot;

  int numDim = 0;
  if(this->tensorRank==2) numDim = this->valTensor.dimension(2); // only needed for tensor fields

  int nblock = x->size();
  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    for (std::size_t node = 0; node < this->numNodes; ++node) {
      int neq = nodeID.dimension(2);
      std::size_t num_dof = neq * this->numNodes;

      for (std::size_t eq = 0; eq < numFields; eq++) {
        typename PHAL::Ref<ScalarT>::type
          valref = ((this->tensorRank == 2) ? (this->valTensor)(cell,node,eq/numDim,eq%numDim) :
                    (this->tensorRank == 1) ? (this->valVec)(cell,node,eq) :
                    (this->val[eq])(cell,node));
        valref = SGFadType(valref.size(), 0.0);
        //valref.setUpdateValue(!workset.ignore_residual);
        valref.fastAccessDx(neq * node + eq + this->offset) = workset.j_coeff;
        valref.val().reset(sg_expansion);
        valref.val().copyForWrite();
        for (int block=0; block<nblock; block++)
          valref.val().fastAccessCoeff(block) = (*x)[block][nodeID(cell,node,this->offset + eq)];
      }
      if (workset.transientTerms && this->enableTransient) {
        for (std::size_t eq = 0; eq < numFields; eq++) {
          typename PHAL::Ref<ScalarT>::type
            valref = ((this->tensorRank == 2) ? (this->valTensor_dot)(cell,node,eq/numDim,eq%numDim) :
                      (this->tensorRank == 1) ? (this->valVec_dot)(cell,node,eq) :
                      (this->val_dot[eq])(cell,node));
          valref = SGFadType(valref.size(), 0.0);
          valref.fastAccessDx(neq * node + eq + this->offset) = workset.m_coeff;
          valref.val().reset(sg_expansion);
          valref.val().copyForWrite();
          for (int block=0; block<nblock; block++)
            valref.val().fastAccessCoeff(block) = (*xdot)[block][nodeID(cell,node,this->offset + eq)];
        }
      }
      if (workset.accelerationTerms && this->enableAcceleration) {
        for (std::size_t eq = 0; eq < numFields; eq++) {
          typename PHAL::Ref<ScalarT>::type
            valref = ((this->tensorRank == 2) ? (this->valTensor_dotdot)(cell,node,eq/numDim,eq%numDim) :
                      (this->tensorRank == 1) ? (this->valVec_dotdot)(cell,node,eq) :
                      (this->val_dotdot[eq])(cell,node));
          valref = SGFadType(valref.size(), 0.0);
          valref.fastAccessDx(neq * node + eq + this->offset) = workset.n_coeff;
          valref.val().reset(sg_expansion);
          valref.val().copyForWrite();
          for (int block=0; block<nblock; block++)
            valref.val().fastAccessCoeff(block) = (*xdotdot)[block][nodeID(cell,node,this->offset + eq)];
        }
      }
    }
  }

}

// **********************************************************************

// **********************************************************************
// Specialization: Stochastic Galerkin Tangent
// **********************************************************************

template<typename Traits>
GatherSolution<PHAL::AlbanyTraits::SGTangent, Traits>::
GatherSolution(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
  GatherSolutionBase<PHAL::AlbanyTraits::SGTangent, Traits>(p,dl),
  numFields(GatherSolutionBase<PHAL::AlbanyTraits::SGTangent,Traits>::numFieldsBase)
{
}

template<typename Traits>
GatherSolution<PHAL::AlbanyTraits::SGTangent, Traits>::
GatherSolution(const Teuchos::ParameterList& p) :
  GatherSolutionBase<PHAL::AlbanyTraits::SGTangent, Traits>(p,p.get<Teuchos::RCP<Albany::Layouts> >("Layouts Struct")),
  numFields(GatherSolutionBase<PHAL::AlbanyTraits::SGTangent,Traits>::numFieldsBase)
{
}

// **********************************************************************
template<typename Traits>
void GatherSolution<PHAL::AlbanyTraits::SGTangent, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  auto nodeID = workset.wsElNodeEqID;
  Teuchos::RCP<Stokhos::OrthogPolyExpansion<int,RealType> > sg_expansion =
    workset.sg_expansion;
  Teuchos::RCP<const Stokhos::EpetraVectorOrthogPoly > x =
    workset.sg_x;
  Teuchos::RCP<const Stokhos::EpetraVectorOrthogPoly > xdot =
    workset.sg_xdot;
  Teuchos::RCP<const Stokhos::EpetraVectorOrthogPoly > xdotdot =
    workset.sg_xdotdot;
  Teuchos::RCP<const Epetra_MultiVector> Vx = workset.Vx;
  Teuchos::RCP<const Epetra_MultiVector> Vxdot = workset.Vxdot;
  Teuchos::RCP<const Epetra_MultiVector> Vxdotdot = workset.Vxdotdot;
  Teuchos::RCP<const Epetra_MultiVector> Vp = workset.Vp;
  Teuchos::RCP<ParamVec> params = workset.params;
  int num_cols_tot = workset.param_offset + workset.num_cols_p;

  int numDim = 0;
  if(this->tensorRank==2) numDim = this->valTensor.dimension(2); // only needed for tensor fields
 
  int nblock = x->size();
  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    for (std::size_t node = 0; node < this->numNodes; ++node) {
      for (std::size_t eq = 0; eq < numFields; eq++) {
        typename PHAL::Ref<ScalarT>::type
        valref = (this->tensorRank == 2 ? (this->valTensor)(cell,node,eq/numDim,eq%numDim) :
                  this->tensorRank == 1 ? (this->valVec)(cell,node,eq) :
                  (this->val[eq])(cell,node));
        valref = SGFadType(valref.size(), 0.0);
        if (Vx != Teuchos::null && workset.j_coeff != 0.0) {
          for (int k=0; k<workset.num_cols_x; k++)
            valref.fastAccessDx(k) =
              workset.j_coeff*(*Vx)[k][nodeID(cell,node,this->offset + eq)];
        }
        (valref.val()).reset(sg_expansion);
        (valref.val()).copyForWrite();
        for (int block=0; block<nblock; block++)
          (valref.val()).fastAccessCoeff(block) = (*x)[block][nodeID(cell,node,this->offset + eq)];
      }
      if (workset.transientTerms && this->enableTransient) {
        for (std::size_t eq = 0; eq < numFields; eq++) {
          typename PHAL::Ref<ScalarT>::type
            valref = ((this->tensorRank == 2) ? (this->valTensor_dot)(cell,node,eq/numDim,eq%numDim) :
                      (this->tensorRank == 1) ? (this->valVec_dot)(cell,node,eq) :
                      (this->val_dot[eq])(cell,node));
          valref = SGFadType(valref.size(), 0.0);
          if (Vxdot != Teuchos::null && workset.m_coeff != 0.0) {
            for (int k=0; k<workset.num_cols_x; k++)
              valref.fastAccessDx(k) =
                workset.m_coeff*(*Vxdot)[k][nodeID(cell,node,this->offset + eq)];
          }
          valref.val().reset(sg_expansion);
          valref.val().copyForWrite();
          for (int block=0; block<nblock; block++)
            valref.val().fastAccessCoeff(block) = (*xdot)[block][nodeID(cell,node,this->offset + eq)];
        }
      }
      if (workset.accelerationTerms && this->enableAcceleration) {
        for (std::size_t eq = 0; eq < numFields; eq++) {
          typename PHAL::Ref<ScalarT>::type
            valref = ((this->tensorRank == 2) ? (this->valTensor_dotdot)(cell,node,eq/numDim,eq%numDim) :
                      (this->tensorRank == 1) ? (this->valVec_dotdot)(cell,node,eq) :
                      (this->val_dotdot[eq])(cell,node));
          valref = SGFadType(valref.size(), 0.0);
          if (Vxdotdot != Teuchos::null && workset.n_coeff != 0.0) {
            for (int k=0; k<workset.num_cols_x; k++)
              valref.fastAccessDx(k) =
                workset.n_coeff*(*Vxdotdot)[k][nodeID(cell,node,this->offset + eq)];
          }
          valref.val().reset(sg_expansion);
          valref.val().copyForWrite();
          for (int block=0; block<nblock; block++)
            valref.val().fastAccessCoeff(block) = (*xdotdot)[block][nodeID(cell,node,this->offset + eq)];
        }
      }
    }
  }

 
}

// **********************************************************************
#endif 
#ifdef ALBANY_ENSEMBLE 

// **********************************************************************
// Specialization: Multi-point Residual
// **********************************************************************

template<typename Traits>
GatherSolution<PHAL::AlbanyTraits::MPResidual, Traits>::
GatherSolution(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
  GatherSolutionBase<PHAL::AlbanyTraits::MPResidual, Traits>(p,dl),
  numFields(GatherSolutionBase<PHAL::AlbanyTraits::MPResidual,Traits>::numFieldsBase)
{
}

template<typename Traits>
GatherSolution<PHAL::AlbanyTraits::MPResidual, Traits>::
GatherSolution(const Teuchos::ParameterList& p) :
  GatherSolutionBase<PHAL::AlbanyTraits::MPResidual, Traits>(p,p.get<Teuchos::RCP<Albany::Layouts> >("Layouts Struct")),
  numFields(GatherSolutionBase<PHAL::AlbanyTraits::MPResidual,Traits>::numFieldsBase)
{
}

// **********************************************************************
template<typename Traits>
void GatherSolution<PHAL::AlbanyTraits::MPResidual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  auto nodeID = workset.wsElNodeEqID;
  Teuchos::RCP<const Stokhos::ProductEpetraVector > x =
    workset.mp_x;
  Teuchos::RCP<const Stokhos::ProductEpetraVector > xdot =
    workset.mp_xdot;
  Teuchos::RCP<const Stokhos::ProductEpetraVector > xdotdot =
    workset.mp_xdotdot;

  int numDim = 0;
  if(this->tensorRank==2) numDim = this->valTensor.dimension(2); // only needed for tensor fields
  int nblock = x->size();
  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    for (std::size_t node = 0; node < this->numNodes; ++node) {
      for (std::size_t eq = 0; eq < numFields; eq++) {
        typename PHAL::Ref<ScalarT>::type
        valref = (this->tensorRank == 2 ? (this->valTensor)(cell,node,eq/numDim,eq%numDim) :
                  this->tensorRank == 1 ? (this->valVec)(cell,node,eq) :
                  (this->val[eq])(cell,node));
        valref.reset(nblock);
        valref.copyForWrite();
        for (int block=0; block<nblock; block++)
          valref.fastAccessCoeff(block) =
            (*x)[block][nodeID(cell,node,this->offset + eq)];
      }
      if (workset.transientTerms && this->enableTransient) {
        for (std::size_t eq = 0; eq < numFields; eq++) {
          typename PHAL::Ref<ScalarT>::type
            valref = ((this->tensorRank == 2) ? (this->valTensor_dot)(cell,node,eq/numDim,eq%numDim) :
                      (this->tensorRank == 1) ? (this->valVec_dot)(cell,node,eq) :
                      (this->val_dot[eq])(cell,node));
          valref.reset(nblock);
          valref.copyForWrite();
          for (int block=0; block<nblock; block++)
            valref.fastAccessCoeff(block) =
              (*xdot)[block][nodeID(cell,node,this->offset + eq)];
        }
      }
      if (workset.accelerationTerms && this->enableAcceleration) {
        for (std::size_t eq = 0; eq < numFields; eq++) {
          typename PHAL::Ref<ScalarT>::type
            valref = ((this->tensorRank == 2) ? (this->valTensor_dotdot)(cell,node,eq/numDim,eq%numDim) :
                      (this->tensorRank == 1) ? (this->valVec_dotdot)(cell,node,eq) :
                      (this->val_dotdot[eq])(cell,node));
          valref.reset(nblock);
          valref.copyForWrite();
          for (int block=0; block<nblock; block++)
            valref.fastAccessCoeff(block) =
              (*xdotdot)[block][nodeID(cell,node,this->offset + eq)];
        }
      }
    }
  }

}

// **********************************************************************

// **********************************************************************
// Specialization: Mulit-point Jacobian
// **********************************************************************

template<typename Traits>
GatherSolution<PHAL::AlbanyTraits::MPJacobian, Traits>::
GatherSolution(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
  GatherSolutionBase<PHAL::AlbanyTraits::MPJacobian, Traits>(p,dl),
  numFields(GatherSolutionBase<PHAL::AlbanyTraits::MPJacobian,Traits>::numFieldsBase)
{
}

template<typename Traits>
GatherSolution<PHAL::AlbanyTraits::MPJacobian, Traits>::
GatherSolution(const Teuchos::ParameterList& p) :
  GatherSolutionBase<PHAL::AlbanyTraits::MPJacobian, Traits>(p,p.get<Teuchos::RCP<Albany::Layouts> >("Layouts Struct")),
  numFields(GatherSolutionBase<PHAL::AlbanyTraits::MPJacobian,Traits>::numFieldsBase)
{
}

// **********************************************************************
template<typename Traits>
void GatherSolution<PHAL::AlbanyTraits::MPJacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  auto nodeID = workset.wsElNodeEqID;
  Teuchos::RCP<const Stokhos::ProductEpetraVector > x =
    workset.mp_x;
  Teuchos::RCP<const Stokhos::ProductEpetraVector > xdot =
    workset.mp_xdot;
  Teuchos::RCP<const Stokhos::ProductEpetraVector > xdotdot =
    workset.mp_xdotdot;

  int numDim = 0;
  if(this->tensorRank==2) numDim = this->valTensor.dimension(2); // only needed for tensor fields
  int nblock = x->size();
  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    for (std::size_t node = 0; node < this->numNodes; ++node) {
      int neq = nodeID.dimension(2);
      std::size_t num_dof = neq * this->numNodes;

      for (std::size_t eq = 0; eq < numFields; eq++) {
        typename PHAL::Ref<ScalarT>::type
        valref = (this->tensorRank == 2 ? (this->valTensor)(cell,node,eq/numDim,eq%numDim) :
                  this->tensorRank == 1 ? (this->valVec)(cell,node,eq) :
                  (this->val[eq])(cell,node));
        valref = MPFadType(valref.size(), 0.0);
        //valref.setUpdateValue(!workset.ignore_residual);
        valref.fastAccessDx(neq * node + eq + this->offset) = workset.j_coeff;
        valref.val().reset(nblock);
        valref.val().copyForWrite();
        for (int block=0; block<nblock; block++)
          valref.val().fastAccessCoeff(block) = (*x)[block][nodeID(cell,node,this->offset + eq)];
      }
      if (workset.transientTerms && this->enableTransient) {
        for (std::size_t eq = 0; eq < numFields; eq++) {
          typename PHAL::Ref<ScalarT>::type
            valref = ((this->tensorRank == 2) ? (this->valTensor_dot)(cell,node,eq/numDim,eq%numDim) :
                      (this->tensorRank == 1) ? (this->valVec_dot)(cell,node,eq) :
                      (this->val_dot[eq])(cell,node));
          valref = MPFadType(valref.size(), 0.0);
          valref.fastAccessDx(neq * node + eq + this->offset) = workset.m_coeff;
          valref.val().reset(nblock);
          valref.val().copyForWrite();
          for (int block=0; block<nblock; block++)
            valref.val().fastAccessCoeff(block) = (*xdot)[block][nodeID(cell,node,this->offset + eq)];
        }
      }
      if (workset.accelerationTerms && this->enableAcceleration) {
        for (std::size_t eq = 0; eq < numFields; eq++) {
          typename PHAL::Ref<ScalarT>::type
            valref = ((this->tensorRank == 2) ? (this->valTensor_dotdot)(cell,node,eq/numDim,eq%numDim) :
                      (this->tensorRank == 1) ? (this->valVec_dotdot)(cell,node,eq) :
                      (this->val_dotdot[eq])(cell,node));
          valref = MPFadType(valref.size(), 0.0);
          valref.fastAccessDx(neq * node + eq + this->offset) = workset.n_coeff;
          valref.val().reset(nblock);
          valref.val().copyForWrite();
          for (int block=0; block<nblock; block++)
            valref.val().fastAccessCoeff(block) = (*xdotdot)[block][nodeID(cell,node,this->offset + eq)];
        }
      }
    }
  }
}

// **********************************************************************

// **********************************************************************
// Specialization: Multi-point Galerkin Tangent
// **********************************************************************

template<typename Traits>
GatherSolution<PHAL::AlbanyTraits::MPTangent, Traits>::
GatherSolution(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
  GatherSolutionBase<PHAL::AlbanyTraits::MPTangent, Traits>(p,dl),
  numFields(GatherSolutionBase<PHAL::AlbanyTraits::MPTangent,Traits>::numFieldsBase)
{
}

template<typename Traits>
GatherSolution<PHAL::AlbanyTraits::MPTangent, Traits>::
GatherSolution(const Teuchos::ParameterList& p) :
  GatherSolutionBase<PHAL::AlbanyTraits::MPTangent, Traits>(p,p.get<Teuchos::RCP<Albany::Layouts> >("Layouts Struct")),
  numFields(GatherSolutionBase<PHAL::AlbanyTraits::MPTangent,Traits>::numFieldsBase)
{
}

// **********************************************************************
template<typename Traits>
void GatherSolution<PHAL::AlbanyTraits::MPTangent, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  auto nodeID = workset.wsElNodeEqID;
  Teuchos::RCP<const Stokhos::ProductEpetraVector > x =
    workset.mp_x;
  Teuchos::RCP<const Stokhos::ProductEpetraVector > xdot =
    workset.mp_xdot;
  Teuchos::RCP<const Stokhos::ProductEpetraVector > xdotdot =
    workset.mp_xdotdot;
  Teuchos::RCP<const Epetra_MultiVector> Vx = workset.Vx;
  Teuchos::RCP<const Epetra_MultiVector> Vxdot = workset.Vxdot;
  Teuchos::RCP<const Epetra_MultiVector> Vxdotdot = workset.Vxdotdot;
  Teuchos::RCP<const Epetra_MultiVector> Vp = workset.Vp;
  Teuchos::RCP<ParamVec> params = workset.params;
  int num_cols_tot = workset.param_offset + workset.num_cols_p;

  int numDim = 0;
  if(this->tensorRank==2) numDim = this->valTensor.dimension(2); // only needed for tensor fields
  int nblock = x->size();
  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    for (std::size_t node = 0; node < this->numNodes; ++node) {
      for (std::size_t eq = 0; eq < numFields; eq++) {
        typename PHAL::Ref<ScalarT>::type
        valref = (this->tensorRank == 2 ? (this->valTensor)(cell,node,eq/numDim,eq%numDim) :
                  this->tensorRank == 1 ? (this->valVec)(cell,node,eq) :
                  (this->val[eq])(cell,node));
        valref = MPFadType(valref.size(), 0.0);
        if (Vx != Teuchos::null && workset.j_coeff != 0.0) {
          for (int k=0; k<workset.num_cols_x; k++)
            valref.fastAccessDx(k) =
              workset.j_coeff*(*Vx)[k][nodeID(cell,node,this->offset + eq)];
        }
        valref.val().reset(nblock);
        valref.val().copyForWrite();
        for (int block=0; block<nblock; block++)
          valref.val().fastAccessCoeff(block) = (*x)[block][nodeID(cell,node,this->offset + eq)];
      }
      if (workset.transientTerms && this->enableTransient) {
        for (std::size_t eq = 0; eq < numFields; eq++) {
          typename PHAL::Ref<ScalarT>::type
            valref = ((this->tensorRank == 2) ? (this->valTensor_dot)(cell,node,eq/numDim,eq%numDim) :
                      (this->tensorRank == 1) ? (this->valVec_dot)(cell,node,eq) :
                      (this->val_dot[eq])(cell,node));
          valref = MPFadType(valref.size(), 0.0);
          if (Vxdot != Teuchos::null && workset.m_coeff != 0.0) {
            for (int k=0; k<workset.num_cols_x; k++)
              valref.fastAccessDx(k) =
                workset.m_coeff*(*Vxdot)[k][nodeID(cell,node,this->offset + eq)];
          }
          valref.val().reset(nblock);
          valref.val().copyForWrite();
          for (int block=0; block<nblock; block++)
            valref.val().fastAccessCoeff(block) = (*xdot)[block][nodeID(cell,node,this->offset + eq)];
        }
      }
      if (workset.accelerationTerms && this->enableAcceleration) {
        for (std::size_t eq = 0; eq < numFields; eq++) {
          typename PHAL::Ref<ScalarT>::type
            valref = ((this->tensorRank == 2) ? (this->valTensor_dotdot)(cell,node,eq/numDim,eq%numDim) :
                      (this->tensorRank == 1) ? (this->valVec_dotdot)(cell,node,eq) :
                      (this->val_dotdot[eq])(cell,node));
          valref = MPFadType(valref.size(), 0.0);
          if (Vxdotdot != Teuchos::null && workset.n_coeff != 0.0) {
            for (int k=0; k<workset.num_cols_x; k++)
              valref.fastAccessDx(k) =
                workset.n_coeff*(*Vxdotdot)[k][nodeID(cell,node,this->offset + eq)];
          }
          valref.val().reset(nblock);
          valref.val().copyForWrite();
          for (int block=0; block<nblock; block++)
            valref.val().fastAccessCoeff(block) = (*xdotdot)[block][nodeID(cell,node,this->offset + eq)];
        }
      }
    }
  }
}
#endif

}
