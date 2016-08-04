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

  if (p.isType<int>("Offset of First DOF"))
    offset = p.get<int>("Offset of First DOF");
  else offset = 0;


 Index=Kokkos::View <int***, PHX::Device>("Index_kokkos", dl->node_vector->dimension(0), dl->node_vector->dimension(1), dl->node_vector->dimension(2));

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
  val_kokkos.resize(numFields);
  val_dot_kokkos.resize(numFields);
  val_dotdot_kokkos.resize(numFields);
}

template<typename Traits>
GatherSolution<PHAL::AlbanyTraits::Residual, Traits>::
GatherSolution(const Teuchos::ParameterList& p) :
  GatherSolutionBase<PHAL::AlbanyTraits::Residual, Traits>(p,p.get<Teuchos::RCP<Albany::Layouts> >("Layouts Struct")),
  numFields(GatherSolutionBase<PHAL::AlbanyTraits::Residual,Traits>::numFieldsBase)
{
  val_kokkos.resize(numFields);
  val_dot_kokkos.resize(numFields);
  val_dotdot_kokkos.resize(numFields);
}
// ********************************************************************
//Kokkos functors for Residual
template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherSolution<PHAL::AlbanyTraits::Residual, Traits>::
operator() (const tensorRank_1Tag& tag, const int& i) const{
  
  for (int node = 0; node < this->numNodes; ++node) 
    for (int eq = 0; eq < numFields; eq++) 
     (this->valVec)(i,node,eq)= xT_constView[wsID_kokkos(i, node,this->offset+eq)];
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherSolution<PHAL::AlbanyTraits::Residual, Traits>::
operator() (const tensorRank_1_enableTransientTag& tag, const int& i) const{

  for (int node = 0; node < this->numNodes; ++node)
    for (int eq = 0; eq < numFields; eq++)
     (this->valVec_dot)(i,node,eq)= xdotT_constView[wsID_kokkos(i, node, this->offset+eq)];
}


template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherSolution<PHAL::AlbanyTraits::Residual, Traits>::
operator() (const tensorRank_1_enableAccelerationTag& tag, const int& i) const{

  for (int node = 0; node < this->numNodes; ++node)
    for (int eq = 0; eq < numFields; eq++)
     (this->valVec_dotdot)(i,node,eq)= xdotdotT_constView[wsID_kokkos(i, node, this->offset+eq)];
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherSolution<PHAL::AlbanyTraits::Residual, Traits>::
operator() (const tensorRank_2Tag& tag, const int& i) const{

  for (int node = 0; node < this->numNodes; ++node)
    for (int eq = 0; eq < numFields; eq++)
        (this->valTensor)(i,node,eq/numDim,eq%numDim)= xT_constView[wsID_kokkos(i, node, this->offset+eq)];
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherSolution<PHAL::AlbanyTraits::Residual, Traits>::
operator() (const tensorRank_2_enableTransientTag& tag, const int& i) const{

  for (int node = 0; node < this->numNodes; ++node)
    for (int eq = 0; eq < numFields; eq++)
       (this->valTensor_dot)(i,node,eq/numDim,eq%numDim)= xdotT_constView[wsID_kokkos (i, node, this->offset+eq)];
}


template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherSolution<PHAL::AlbanyTraits::Residual, Traits>::
operator() (const tensorRank_2_enableAccelerationTag& tag, const int& i) const{


  for (int node = 0; node < this->numNodes; ++node)
    for (int eq = 0; eq < numFields; eq++)
       (this->valTensor_dotdot)(i,node,eq/numDim,eq%numDim)= xdotdotT_constView[wsID_kokkos(i, node, this->offset+eq)];
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherSolution<PHAL::AlbanyTraits::Residual, Traits>::
operator() (const tensorRank_0Tag& tag, const int& i) const{

  for (int node = 0; node < this->numNodes; ++node)
    for (int eq = 0; eq < numFields; eq++)
        (d_val[eq](i,node))= xT_constView[wsID_kokkos(i, node, this->offset+eq)];
  
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherSolution<PHAL::AlbanyTraits::Residual, Traits>::
operator() (const tensorRank_0_enableTransientTag& tag, const int& i) const{

  for (int node = 0; node < this->numNodes; ++node)
    for (int eq = 0; eq < numFields; eq++)
       (d_val_dot[eq](i,node))= xdotT_constView[wsID_kokkos (i, node, this->offset+eq)];
}


template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherSolution<PHAL::AlbanyTraits::Residual, Traits>::
operator() (const tensorRank_0_enableAccelerationTag& tag, const int& i) const{

  for (int node = 0; node < this->numNodes; ++node)
    for (int eq = 0; eq < numFields; eq++)
       (d_val_dotdot[eq](i,node))= xdotdotT_constView[wsID_kokkos(i, node, this->offset+eq)];
}


// **********************************************************************
template<typename Traits>
void GatherSolution<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::RCP<const Tpetra_Vector> xT = workset.xT;
  Teuchos::RCP<const Tpetra_Vector> xdotT = workset.xdotT;
  Teuchos::RCP<const Tpetra_Vector> xdotdotT = workset.xdotdotT;

 //In purpose to use Kokkos functores declaration of the temporary data has been mooved to the class
  //get const (read-only) view of xT and xdotT
//  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();
//  Teuchos::ArrayRCP<const ST> xdotT_constView = xdotT->get1dView();
//  Teuchos::ArrayRCP<const ST> xdotdotT_constView = xdotdotT->get1dView();
  
  xT_constView = xT->get1dView();
  if(Teuchos::nonnull(xdotT))
    xdotT_constView = xdotT->get1dView();
  if(Teuchos::nonnull(xdotdotT))
    xdotdotT_constView = xdotdotT->get1dView();

#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  if (this->tensorRank == 1) {
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];
      for (std::size_t node = 0; node < this->numNodes; ++node) {
      const Teuchos::ArrayRCP<int>& eqID  = nodeID[node];
        for (std::size_t eq = 0; eq < numFields; eq++) 
          (this->valVec)(cell,node,eq) = xT_constView[eqID[this->offset + eq]];
        if (workset.transientTerms && this->enableTransient) {
          for (std::size_t eq = 0; eq < numFields; eq++) 
            (this->valVec_dot)(cell,node,eq) = xdotT_constView[eqID[this->offset + eq]];
        }
        if (workset.accelerationTerms && this->enableAcceleration) {
          for (std::size_t eq = 0; eq < numFields; eq++) 
            (this->valVec_dotdot)(cell,node,eq) = xdotdotT_constView[eqID[this->offset + eq]];
        }
      }
    }
  } else 
  if (this->tensorRank == 2) {
    int numDim = this->valTensor.dimension(2);
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

      for (std::size_t node = 0; node < this->numNodes; ++node) {
      const Teuchos::ArrayRCP<int>& eqID  = nodeID[node];
        for (std::size_t eq = 0; eq < numFields; eq++) 
          (this->valTensor)(cell,node,eq/numDim,eq%numDim) = xT_constView[eqID[this->offset + eq]];
        if (workset.transientTerms && this->enableTransient) {
          for (std::size_t eq = 0; eq < numFields; eq++) 
            (this->valTensor_dot)(cell,node,eq/numDim,eq%numDim) = xdotT_constView[eqID[this->offset + eq]];
        }
        if (workset.accelerationTerms && this->enableAcceleration) {
          for (std::size_t eq = 0; eq < numFields; eq++) 
            (this->valTensor_dotdot)(cell,node,eq/numDim,eq%numDim) = xdotdotT_constView[eqID[this->offset + eq]];
        }
      }
    }
  } else {
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

      for (std::size_t node = 0; node < this->numNodes; ++node) {
      const Teuchos::ArrayRCP<int>& eqID  = nodeID[node];
        for (std::size_t eq = 0; eq < numFields; eq++) 
          (this->val[eq])(cell,node) = xT_constView[eqID[this->offset + eq]];
        if (workset.transientTerms && this->enableTransient) {
          for (std::size_t eq = 0; eq < numFields; eq++) 
            (this->val_dot[eq])(cell,node) = xdotT_constView[eqID[this->offset + eq]];
        }
        if (workset.accelerationTerms && this->enableAcceleration) {
          for (std::size_t eq = 0; eq < numFields; eq++) 
            (this->val_dotdot[eq])(cell,node) = xdotdotT_constView[eqID[this->offset + eq]];
        }
      }
    }
  }
#else
#ifdef ALBANY_TIMER
  auto start = std::chrono::high_resolution_clock::now();
#endif

 wsID_kokkos=workset.wsElNodeEqID_kokkos;

   if (this->tensorRank == 2){

     numDim = this->valTensor.dimension(2);

     Kokkos::parallel_for(tensorRank_2Policy(0,workset.numCells),*this);

     if (workset.transientTerms && this->enableTransient) 
        Kokkos::parallel_for(tensorRank_2_enableTransientPolicy(0,workset.numCells),*this);    
     if (workset.accelerationTerms && this->enableAcceleration) 
        Kokkos::parallel_for(tensorRank_2_enableAccelerationPolicy(0,workset.numCells),*this);

   } else if (this->tensorRank == 1){
 
     Kokkos::parallel_for(tensorRank_1Policy(0,workset.numCells),*this);
  
     if (workset.transientTerms && this->enableTransient) 
       Kokkos::parallel_for(tensorRank_1_enableTransientPolicy(0,workset.numCells),*this);
     if (workset.accelerationTerms && this->enableAcceleration) 
        Kokkos::parallel_for(tensorRank_1_enableAccelerationPolicy(0,workset.numCells),*this);
 
   } else  {

     for (int i =0; i<numFields;i++){
     //  val_kokkos[i]=this->val[i].get_view();
       val_kokkos[i]=this->val[i].get_static_view();
     }
     d_val=val_kokkos.template view<executionSpace>();
     Kokkos::parallel_for(tensorRank_0Policy(0,workset.numCells),*this);

     if (workset.transientTerms && this->enableTransient){ 
        for (int i =0; i<numFields;i++) 
      //     val_dot_kokkos[i]=this->val_dot[i].get_view();
             val_dot_kokkos[i]=this->val_dot[i].get_static_view();
        d_val_dot=val_dot_kokkos.template view<executionSpace>();
        Kokkos::parallel_for(tensorRank_0_enableTransientPolicy(0,workset.numCells),*this);  
     }
     if (workset.accelerationTerms && this->enableAcceleration){
        for (int i =0; i<numFields;i++)
        //   val_dotdot_kokkos[i]=this->val_dotdot[i].get_view();
           val_dotdot_kokkos[i]=this->val_dotdot[i].get_static_view();
        d_val_dotdot=val_dotdot_kokkos.template view<executionSpace>();
        Kokkos::parallel_for(tensorRank_0_enableAccelerationPolicy(0,workset.numCells),*this);
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
  val_kokkosjac.resize(numFields);
  val_dot_kokkosjac.resize(numFields);
  val_dotdot_kokkosjac.resize(numFields);
}

template<typename Traits>
GatherSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
GatherSolution(const Teuchos::ParameterList& p) :
GatherSolutionBase<PHAL::AlbanyTraits::Jacobian, Traits>(p,p.get<Teuchos::RCP<Albany::Layouts> >("Layouts Struct")),
numFields(GatherSolutionBase<PHAL::AlbanyTraits::Jacobian,Traits>::numFieldsBase)
{
  val_kokkosjac.resize(numFields);
  val_dot_kokkosjac.resize(numFields);
  val_dotdot_kokkosjac.resize(numFields);
}
//********************************************************************
////Kokkos functors for Jacobian

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
operator() (const tensorRank_2Tag& tag, const int& i) const{

const int neq=wsID_kokkos.dimension(2);
//const int num_dof = neq * this->numNodes;

for (int node = 0; node < this->numNodes; ++node){
  int firstunk = neq * node + this->offset;
    for (int eq = 0; eq < numFields; eq++){
      typename PHAL::Ref<ScalarT>::type valref = (this->valTensor)(i,node,eq/numDim,eq%numDim);
      valref=FadType(valref.size(), xT_constView[wsID_kokkos(i,node,this->offset+eq)]);
      //((this->valTensor)(i,node,eq/numDim,eq%numDim)).setUpdateValue(!ignore_residual);
      valref.fastAccessDx(firstunk + eq) =j_coeff;
    }
  }
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
operator() (const tensorRank_2_enableTransientTag& tag, const int& i) const{

  const int neq=wsID_kokkos.dimension(2);
  const int num_dof = neq * this->numNodes;
  
  for (int node = 0; node < this->numNodes; ++node){
    int firstunk = neq * node + this->offset;
    for (int eq = 0; eq < numFields; eq++){
      typename PHAL::Ref<ScalarT>::type valref = (this->valTensor_dot)(i,node,eq/numDim,eq%numDim);
      valref =FadType(valref.size(), xdotT_constView[wsID_kokkos(i,node,this->offset+eq)]);
      valref.fastAccessDx(firstunk + eq) =m_coeff;
    }
  }

}


template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
operator() (const tensorRank_2_enableAccelerationTag& tag, const int& i) const{

  const int neq=wsID_kokkos.dimension(2);
  const int num_dof = neq * this->numNodes;

  for (int node = 0; node < this->numNodes; ++node){
    int firstunk = neq * node + this->offset;
    for (int eq = 0; eq < numFields; eq++){
      typename PHAL::Ref<ScalarT>::type valref = (this->valTensor_dotdot)(i,node,eq/numDim,eq%numDim);
      valref=FadType(valref.size(), xdotdotT_constView[wsID_kokkos(i,node,this->offset+eq)]);
      valref.fastAccessDx(firstunk + eq) =n_coeff;
    }
  }

}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
operator() (const tensorRank_1Tag& tag, const int& i) const{

  const int neq=wsID_kokkos.dimension(2);
  const int num_dof = neq * this->numNodes;


  for (int node = 0; node < this->numNodes; node++){
    int firstunk = neq * node + this->offset;
    for (int eq = 0; eq < numFields; eq++){
      typename PHAL::Ref<ScalarT>::type valref = this->valVec(i,node,eq);
      valref =FadType(valref.size(), xT_constView[wsID_kokkos(i,node,this->offset+eq)]);
      //((this->valVec)(i,node,eq)).setUpdateValue(!ignore_residual);
      valref.fastAccessDx(firstunk + eq) =j_coeff;
    }
  }
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
operator() (const tensorRank_1_enableTransientTag& tag, const int& i) const{

  const int neq=wsID_kokkos.dimension(2);
  const int num_dof = neq * this->numNodes;

  for (int node = 0; node < this->numNodes; ++node){
    int firstunk = neq * node + this->offset;
    for (int eq = 0; eq < numFields; eq++){
      typename PHAL::Ref<ScalarT>::type valref = (this->valVec_dot)(i,node,eq);
      valref =FadType(valref.size(), xdotT_constView[wsID_kokkos(i,node,this->offset+eq)]);
      valref.fastAccessDx(firstunk + eq) =m_coeff;
    }
  }  

}


template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
operator() (const tensorRank_1_enableAccelerationTag& tag, const int& i) const{
 
  const int neq=wsID_kokkos.dimension(2);
  const int num_dof = neq * this->numNodes;

  for (int node = 0; node < this->numNodes; ++node){
    int firstunk = neq * node + this->offset;
    for (int eq = 0; eq < numFields; eq++){
      typename PHAL::Ref<ScalarT>::type valref = (this->valVec_dotdot(i,node,eq));
      valref =FadType(valref.size(), xdotdotT_constView[wsID_kokkos(i,node,this->offset+eq)]);
      valref.fastAccessDx(firstunk + eq) =n_coeff;
    }
  }

}


template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
operator() (const tensorRank_0Tag& tag, const int& i) const{

  const int neq=wsID_kokkos.dimension(2);
  const int num_dof = neq * this->numNodes;

  for (int node = 0; node < this->numNodes; ++node){
    int firstunk = neq * node + this->offset;
    for (int eq = 0; eq < numFields; eq++){
      typename PHAL::Ref<ScalarT>::type valref = d_val[eq](i,node);
      valref =FadType(valref.size(), xT_constView[wsID_kokkos(i,node,this->offset+eq)]);
       //(d_val[eq](i,node)).setUpdateValue(!ignore_residual);
      valref.fastAccessDx(firstunk + eq) =j_coeff;
    }
  }
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
operator() (const tensorRank_0_enableTransientTag& tag, const int& i) const{

  const int neq=wsID_kokkos.dimension(2);
  const int num_dof = neq * this->numNodes;

  for (int node = 0; node < this->numNodes; ++node){
    int firstunk = neq * node + this->offset;
    for (int eq = 0; eq < numFields; eq++){
      typename PHAL::Ref<ScalarT>::type valref = (d_val_dot[eq](i,node));
      valref =FadType(valref.size(), xdotT_constView[wsID_kokkos(i,node,this->offset+eq)]);
      valref.fastAccessDx(firstunk + eq) =m_coeff;
    }
  }

}


template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
operator() (const tensorRank_0_enableAccelerationTag& tag, const int& i) const{

  const int neq=wsID_kokkos.dimension(2);
  const int num_dof = neq * this->numNodes;

  for (int node = 0; node < this->numNodes; ++node){
    int firstunk = neq * node + this->offset;
    for (int eq = 0; eq < numFields; eq++){
      typename PHAL::Ref<ScalarT>::type valref = (d_val_dotdot[eq](i,node));
      valref = FadType(valref.size(), xdotdotT_constView[wsID_kokkos(i,node,this->offset+eq)]);
      valref.fastAccessDx(firstunk + eq) = n_coeff;
    }
  }

}



// **********************************************************************
template<typename Traits>
void GatherSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  Teuchos::RCP<const Tpetra_Vector> xT = workset.xT;
  Teuchos::RCP<const Tpetra_Vector> xdotT = workset.xdotT;
  Teuchos::RCP<const Tpetra_Vector> xdotdotT = workset.xdotdotT;

  xT_constView = xT->get1dView();
  if(Teuchos::nonnull(xdotT))
    xdotT_constView = xdotT->get1dView();
  if(Teuchos::nonnull(xdotdotT))
    xdotdotT_constView = xdotdotT->get1dView();

#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  int numDim = 0;
  if (this->tensorRank==2) numDim = this->valTensor.dimension(2); // only needed for tensor fields

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];
    const int neq = nodeID[0].size();
    const std::size_t num_dof = neq * this->numNodes;

    for (std::size_t node = 0; node < this->numNodes; ++node) {
      const Teuchos::ArrayRCP<int>& eqID  = nodeID[node];
      int firstunk = neq * node + this->offset;
      for (std::size_t eq = 0; eq < numFields; eq++) {
        typename PHAL::Ref<ScalarT>::type
          valref = (this->tensorRank == 0 ? this->val[eq](cell,node) :
                    this->tensorRank == 1 ? this->valVec(cell,node,eq) :
                    this->valTensor(cell,node, eq/numDim, eq%numDim));
        valref = FadType(valref.size(), xT_constView[eqID[this->offset + eq]]);
        // valref.setUpdateValue(!workset.ignore_residual); Not used anymore
        valref.fastAccessDx(firstunk + eq) = workset.j_coeff;
      }
      if (workset.transientTerms && this->enableTransient) {
        for (std::size_t eq = 0; eq < numFields; eq++) {
        typename PHAL::Ref<ScalarT>::type
          valref = (this->tensorRank == 0 ? this->val_dot[eq](cell,node) :
                    this->tensorRank == 1 ? this->valVec_dot(cell,node,eq) :
                    this->valTensor_dot(cell,node, eq/numDim, eq%numDim));
        valref = FadType(valref.size(), xdotT_constView[eqID[this->offset + eq]]);
        valref.fastAccessDx(firstunk + eq) = workset.m_coeff;
        }
      }
      if (workset.accelerationTerms && this->enableAcceleration) {
        for (std::size_t eq = 0; eq < numFields; eq++) {
        typename PHAL::Ref<ScalarT>::type
          valref = (this->tensorRank == 0 ? this->val_dotdot[eq](cell,node) :
                    this->tensorRank == 1 ? this->valVec_dotdot(cell,node,eq) :
                    this->valTensor_dotdot(cell,node, eq/numDim, eq%numDim));
        valref = FadType(valref.size(), xdotdotT_constView[eqID[this->offset + eq]]);
        valref.fastAccessDx(firstunk + eq) = workset.n_coeff;
        }
      }
    }
  }
#else
#ifdef ALBANY_TIMER
 auto start = std::chrono::high_resolution_clock::now();
#endif
    wsID_kokkos=workset.wsElNodeEqID_kokkos;

    ignore_residual=workset.ignore_residual;
    j_coeff=workset.j_coeff;
    m_coeff=workset.m_coeff;
    n_coeff=workset.n_coeff;


   if(this->tensorRank==2) numDim = this->valTensor.dimension(2);
  
   if (this->tensorRank == 2){
 
     Kokkos::parallel_for(tensorRank_2Policy(0,workset.numCells),*this);

     if (workset.transientTerms && this->enableTransient)
        Kokkos::parallel_for(tensorRank_2_enableTransientPolicy(0,workset.numCells),*this);
     if (workset.accelerationTerms && this->enableAcceleration)
        Kokkos::parallel_for(tensorRank_2_enableAccelerationPolicy(0,workset.numCells),*this);

   } else if (this->tensorRank == 1){
     Kokkos::parallel_for(tensorRank_1Policy(0,workset.numCells),*this);

     if (workset.transientTerms && this->enableTransient){
       Kokkos::parallel_for(tensorRank_1_enableTransientPolicy(0,workset.numCells),*this);}
     if (workset.accelerationTerms && this->enableAcceleration){
        Kokkos::parallel_for(tensorRank_1_enableAccelerationPolicy(0,workset.numCells),*this);}
   } else  {
     
     for (int i =0; i<numFields;i++)
        // val_kokkosjac[i]=this->val[i].get_view();
         val_kokkosjac[i]=this->val[i].get_static_view();
     d_val=val_kokkosjac.template view<executionSpace>();
     Kokkos::parallel_for(tensorRank_0Policy(0,workset.numCells),*this);

     if (workset.transientTerms && this->enableTransient){
        for (int i =0; i<numFields;i++)
           // val_dot_kokkosjac[i]=this->val_dot[i].get_view();
           val_dot_kokkosjac[i]=this->val_dot[i].get_static_view();
        d_val_dot=val_dot_kokkosjac.template view<executionSpace>();
        Kokkos::parallel_for(tensorRank_0_enableTransientPolicy(0,workset.numCells),*this);
     }

     if (workset.accelerationTerms && this->enableAcceleration){
        for (int i =0; i<numFields;i++)
           // val_dotdot_kokkosjac[i]=this->val_dotdot[i].get_view();
           val_dotdot_kokkosjac[i]=this->val_dotdot[i].get_static_view();
        d_val_dot=val_dotdot_kokkosjac.template view<executionSpace>();
        Kokkos::parallel_for(tensorRank_0_enableAccelerationPolicy(0,workset.numCells),*this);
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
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

    for (std::size_t node = 0; node < this->numNodes; ++node) {
      const Teuchos::ArrayRCP<int>& eqID  = nodeID[node];
      for (std::size_t eq = 0; eq < numFields; eq++) {
        typename PHAL::Ref<ScalarT>::type
          valref = ((this->tensorRank == 2) ? (this->valTensor)(cell,node,eq/numDim,eq%numDim) :
                    (this->tensorRank == 1) ? (this->valVec)(cell,node,eq) :
                    (this->val[eq])(cell,node));
        if (VxT != Teuchos::null && workset.j_coeff != 0.0) {
          valref = TanFadType(valref.size(), xT_constView[eqID[this->offset + eq]]);
          for (int k=0; k<workset.num_cols_x; k++)
            valref.fastAccessDx(k) =
              workset.j_coeff*VxT->getData(k)[eqID[this->offset + eq]];
        }
        else
          valref = TanFadType(xT_constView[eqID[this->offset + eq]]);
      }
   }


   if (workset.transientTerms && this->enableTransient) {
    Teuchos::ArrayRCP<const ST> xdotT_constView = xdotT->get1dView();
    for (std::size_t node = 0; node < this->numNodes; ++node) {
      const Teuchos::ArrayRCP<int>& eqID  = nodeID[node];
        for (std::size_t eq = 0; eq < numFields; eq++) {
        typename PHAL::Ref<ScalarT>::type
          valref = ((this->tensorRank == 2) ? (this->valTensor_dot)(cell,node,eq/numDim,eq%numDim) :
                    (this->tensorRank == 1) ? (this->valVec_dot)(cell,node,eq) :
                    (this->val_dot[eq])(cell,node));
          valref = TanFadType(valref.size(), xdotT_constView[eqID[this->offset + eq]]);
          if (VxdotT != Teuchos::null && workset.m_coeff != 0.0) {
            for (int k=0; k<workset.num_cols_x; k++)
              valref.fastAccessDx(k) =
                workset.m_coeff*VxdotT->getData(k)[eqID[this->offset + eq]];
          }
        }
      }
   }

   if (workset.accelerationTerms && this->enableAcceleration) {
    Teuchos::ArrayRCP<const ST> xdotdotT_constView = xdotdotT->get1dView();
    for (std::size_t node = 0; node < this->numNodes; ++node) {
      const Teuchos::ArrayRCP<int>& eqID  = nodeID[node];
        for (std::size_t eq = 0; eq < numFields; eq++) {
        typename PHAL::Ref<ScalarT>::type
          valref = ((this->tensorRank == 2) ? (this->valTensor_dotdot)(cell,node,eq/numDim,eq%numDim) :
                    (this->tensorRank == 1) ? (this->valVec_dotdot)(cell,node,eq) :
                    (this->val_dotdot[eq])(cell,node));

          valref = TanFadType(valref.size(), xdotdotT_constView[eqID[this->offset + eq]]);
          if (VxdotdotT != Teuchos::null && workset.n_coeff != 0.0) {
            for (int k=0; k<workset.num_cols_x; k++)
              valref.fastAccessDx(k) =
                workset.n_coeff*VxdotdotT->getData(k)[eqID[this->offset + eq]];
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
  Teuchos::RCP<const Tpetra_Vector> xT = workset.xT;
  Teuchos::RCP<const Tpetra_Vector> xdotT = workset.xdotT;
  Teuchos::RCP<const Tpetra_Vector> xdotdotT = workset.xdotdotT;

  //get const (read-only) view of xT and xdotT
  Teuchos::ArrayRCP<const ST> xT_constView = xT->get1dView();

  if (this->tensorRank == 1) {
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

      for (std::size_t node = 0; node < this->numNodes; ++node) {
      const Teuchos::ArrayRCP<int>& eqID  = nodeID[node];
        for (std::size_t eq = 0; eq < numFields; eq++)
          (this->valVec)(cell,node,eq) = xT_constView[eqID[this->offset + eq]];
      }

    if (workset.transientTerms && this->enableTransient) {
    Teuchos::ArrayRCP<const ST> xdotT_constView = xdotT->get1dView();
      for (std::size_t node = 0; node < this->numNodes; ++node) {
      const Teuchos::ArrayRCP<int>& eqID  = nodeID[node];
          for (std::size_t eq = 0; eq < numFields; eq++)
            (this->valVec_dot)(cell,node,eq) = xdotT_constView[eqID[this->offset + eq]];
      }
    }

    if (workset.accelerationTerms && this->enableAcceleration) {
    Teuchos::ArrayRCP<const ST> xdotdotT_constView = xdotdotT->get1dView();
      for (std::size_t node = 0; node < this->numNodes; ++node) {
      const Teuchos::ArrayRCP<int>& eqID  = nodeID[node];
          for (std::size_t eq = 0; eq < numFields; eq++)
            (this->valVec_dotdot)(cell,node,eq) = xdotdotT_constView[eqID[this->offset + eq]];
        }
      }
    } 
  } else
  if (this->tensorRank == 2) {
    int numDim = this->valTensor.dimension(2);
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

      for (std::size_t node = 0; node < this->numNodes; ++node) {
      const Teuchos::ArrayRCP<int>& eqID  = nodeID[node];
        for (std::size_t eq = 0; eq < numFields; eq++)
          (this->valTensor)(cell,node,eq/numDim,eq%numDim) = xT_constView[eqID[this->offset + eq]];
      }

    if (workset.transientTerms && this->enableTransient) {
    Teuchos::ArrayRCP<const ST> xdotT_constView = xdotT->get1dView();
      for (std::size_t node = 0; node < this->numNodes; ++node) {
      const Teuchos::ArrayRCP<int>& eqID  = nodeID[node];
          for (std::size_t eq = 0; eq < numFields; eq++)
            (this->valTensor_dot)(cell,node,eq/numDim,eq%numDim) = xdotT_constView[eqID[this->offset + eq]];
      }
    }

    if (workset.accelerationTerms && this->enableAcceleration) {
    Teuchos::ArrayRCP<const ST> xdotdotT_constView = xdotdotT->get1dView();
      for (std::size_t node = 0; node < this->numNodes; ++node) {
      const Teuchos::ArrayRCP<int>& eqID  = nodeID[node];
          for (std::size_t eq = 0; eq < numFields; eq++)
            (this->valTensor_dotdot)(cell,node,eq/numDim,eq%numDim) = xdotdotT_constView[eqID[this->offset + eq]];
        }
      }
    }
  } else {
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

      for (std::size_t node = 0; node < this->numNodes; ++node) {
      const Teuchos::ArrayRCP<int>& eqID  = nodeID[node];
        for (std::size_t eq = 0; eq < numFields; eq++)
          (this->val[eq])(cell,node) = xT_constView[eqID[this->offset + eq]];
      }
    if (workset.transientTerms && this->enableTransient) {
    Teuchos::ArrayRCP<const ST> xdotT_constView = xdotT->get1dView();
      for (std::size_t node = 0; node < this->numNodes; ++node) {
      const Teuchos::ArrayRCP<int>& eqID  = nodeID[node];
          for (std::size_t eq = 0; eq < numFields; eq++)
            (this->val_dot[eq])(cell,node) = xdotT_constView[eqID[this->offset + eq]];
      }
    }

    if (workset.accelerationTerms && this->enableAcceleration) {
    Teuchos::ArrayRCP<const ST> xdotdotT_constView = xdotdotT->get1dView();
      for (std::size_t node = 0; node < this->numNodes; ++node) {
      const Teuchos::ArrayRCP<int>& eqID  = nodeID[node];
          for (std::size_t eq = 0; eq < numFields; eq++)
            (this->val_dotdot[eq])(cell,node) = xdotdotT_constView[eqID[this->offset + eq]];
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
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

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
            (*x)[block][nodeID[node][this->offset + eq]];
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
              (*xdot)[block][nodeID[node][this->offset + eq]];
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
              (*xdotdot)[block][nodeID[node][this->offset + eq]];
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
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

    for (std::size_t node = 0; node < this->numNodes; ++node) {
      int neq = nodeID[node].size();
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
          valref.val().fastAccessCoeff(block) = (*x)[block][nodeID[node][this->offset + eq]];
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
            valref.val().fastAccessCoeff(block) = (*xdot)[block][nodeID[node][this->offset + eq]];
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
            valref.val().fastAccessCoeff(block) = (*xdotdot)[block][nodeID[node][this->offset + eq]];
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
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

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
              workset.j_coeff*(*Vx)[k][nodeID[node][this->offset + eq]];
        }
        (valref.val()).reset(sg_expansion);
        (valref.val()).copyForWrite();
        for (int block=0; block<nblock; block++)
          (valref.val()).fastAccessCoeff(block) = (*x)[block][nodeID[node][this->offset + eq]];
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
                workset.m_coeff*(*Vxdot)[k][nodeID[node][this->offset + eq]];
          }
          valref.val().reset(sg_expansion);
          valref.val().copyForWrite();
          for (int block=0; block<nblock; block++)
            valref.val().fastAccessCoeff(block) = (*xdot)[block][nodeID[node][this->offset + eq]];
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
                workset.n_coeff*(*Vxdotdot)[k][nodeID[node][this->offset + eq]];
          }
          valref.val().reset(sg_expansion);
          valref.val().copyForWrite();
          for (int block=0; block<nblock; block++)
            valref.val().fastAccessCoeff(block) = (*xdotdot)[block][nodeID[node][this->offset + eq]];
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
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

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
            (*x)[block][nodeID[node][this->offset + eq]];
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
              (*xdot)[block][nodeID[node][this->offset + eq]];
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
              (*xdotdot)[block][nodeID[node][this->offset + eq]];
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
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

    for (std::size_t node = 0; node < this->numNodes; ++node) {
      int neq = nodeID[node].size();
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
          valref.val().fastAccessCoeff(block) = (*x)[block][nodeID[node][this->offset + eq]];
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
            valref.val().fastAccessCoeff(block) = (*xdot)[block][nodeID[node][this->offset + eq]];
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
            valref.val().fastAccessCoeff(block) = (*xdotdot)[block][nodeID[node][this->offset + eq]];
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
    const Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >& nodeID  = workset.wsElNodeEqID[cell];

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
              workset.j_coeff*(*Vx)[k][nodeID[node][this->offset + eq]];
        }
        valref.val().reset(nblock);
        valref.val().copyForWrite();
        for (int block=0; block<nblock; block++)
          valref.val().fastAccessCoeff(block) = (*x)[block][nodeID[node][this->offset + eq]];
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
                workset.m_coeff*(*Vxdot)[k][nodeID[node][this->offset + eq]];
          }
          valref.val().reset(nblock);
          valref.val().copyForWrite();
          for (int block=0; block<nblock; block++)
            valref.val().fastAccessCoeff(block) = (*xdot)[block][nodeID[node][this->offset + eq]];
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
                workset.n_coeff*(*Vxdotdot)[k][nodeID[node][this->offset + eq]];
          }
          valref.val().reset(nblock);
          valref.val().copyForWrite();
          for (int block=0; block<nblock; block++)
            valref.val().fastAccessCoeff(block) = (*xdotdot)[block][nodeID[node][this->offset + eq]];
        }
      }
    }
  }
}
#endif

}
