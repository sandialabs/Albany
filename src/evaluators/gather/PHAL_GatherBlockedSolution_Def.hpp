//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "PHAL_GatherBlockedSolution.hpp"

#include "Albany_Macros.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_Utils.hpp"

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include <vector>
#include <string>
#ifdef ALBANY_TIMER
#include <chrono>
#endif


namespace PHAL {

template<typename EvalT, typename Traits>
GatherBlockedSolutionBase<EvalT,Traits>::
GatherBlockedSolutionBase(const Teuchos::ParameterList& p,
                          const Teuchos::RCP<Albany::Layouts>& dl)
{
  auto get_rank = [&] (const PHX::DataLayout& l) -> int {
    if (l.identifier()==dl->node_scalar->identifier() ||
        l.identifier()==dl->cell_scalar2->identifier()) {
      return 0;
    } else if (l.identifier()==dl->node_vector->identifier() ||
               l.identifier()==dl->node_gradient->identifier() ||
               l.identifier()==dl->cell_gradient->identifier() ||
               l.identifier()==dl->cell_vector->identifier()) {
      return 1;
    } else if (l.identifier()==dl->node_tensor->identifier() ||
               l.identifier()==dl->cell_tensor->identifier()) {
      return 2;
    } else {
      ALBANY_ABORT("Error! Unsupported DataLayout in GatherBlockedSolutionBase.\n"
                   "  - Layout id: " + l.identifier() + "\n");
    }
  };

  auto get_loc = [&] (const PHX::DataLayout& l) -> int {
    using FL = Albany::FieldLocation;

    if (l.identifier()==dl->node_scalar->identifier() ||
        l.identifier()==dl->node_vector->identifier() ||
        l.identifier()==dl->node_gradient->identifier() ||
        l.identifier()==dl->node_tensor->identifier()) {
      return Albany::etoi(FL::Node);
    } else if (l.identifier()==dl->cell_scalar2->identifier() ||
               l.identifier()==dl->cell_vector->identifier() ||
               l.identifier()==dl->cell_gradient->identifier() ||
               l.identifier()==dl->cell_tensor->identifier()) {
      return Albany::etoi(FL::Cell);
    } else {
      ALBANY_ABORT("Error! Unsupported DataLayout in GatherBlockedSolutionBase.\n"
                   "  - Layout id: " + l.identifier() + "\n");
    }
  };
  
  auto get_ndims = [&] (const PHX::DataLayout& l) -> int {
    if (l.identifier()==dl->node_scalar->identifier() ||
        l.identifier()==dl->cell_scalar2->identifier()) {
      return 0;
    } else if (l.identifier()==dl->node_vector->identifier() ||
               l.identifier()==dl->node_gradient->identifier()) {
      return l.extent_int(2);
    } else if (l.identifier()==dl->cell_vector->identifier() ||
               l.identifier()==dl->cell_gradient->identifier()) {
      return l.extent_int(1);
    } else if (l.identifier()==dl->node_tensor->identifier()) {
      return l.extent_int(2)*l.extent_int(3);
    } else if (l.identifier()==dl->cell_tensor->identifier()) {
      return l.extent_int(1)*l.extent_int(2);
    } else {
      ALBANY_ABORT("Error! Unsupported DataLayout in GatherBlockedSolutionBase.\n"
                   "  - Layout id: " + l.identifier() + "\n");
    }
  };

  using layout_ptr = Teuchos::RCP<PHX::DataLayout>;

  std::string name, name_dot, name_dotdot;
  if (p.isParameter("X Name")) {
    name = p.get<std::string>("X Name");
    gather_x = true;
  }
  // auto xdot_names    = p.get<Teuchos::ArrayRCP<std::string>>("X_dot Names");
  // auto xdotdot_names = p.get<Teuchos::ArrayRCP<std::string>>("X_dotdot Names");

  // The layout and block information must be the same for x, xdot, and xdotdot.
  auto layout = p.get<Teuchos::Array<layout_ptr>>("Data Layout");
  block = p.get<Teuchos::RCP<Albany::Block>>("Block");
  
  if (gather_x) {
    val = PHX::MDField<ScalarT>(name, layout);
    this->addEvaluatedField(val);



    // val.reserver(num_x);
    // val_kokkos.resize(num_x);
    // // x_views.resize(num_x);
    // // ranks_x = IView("ranks_x",num_x);
    // ranks_x.resize(num_x);
    // locs_x.resize(num_x);
    // ndims_x = IView("ndims_x",num_x);
    // // locs_x  = IView("locs_x",num_x);

    // // auto h_ranks_x = Kokkos::create_mirror_view(ranks_x);
    // auto h_ndims_x = Kokkos::create_mirror_view(ndims_x);
    // // auto h_locs_x  = Kokkos::create_mirror_view(locs_x);
    // for (int i=0; i<num_x; ++i) {
    //   val.push_back(PHX::MDField<ScalarT>(x_names[i],x_layouts[i]));
    //   val_kokkos[i] = val.back().get_static_view();

    //   // h_ranks_x[i] = get_rank(x_layouts[i]);
    //   // h_locs_x[i]  = get_loc(x_layouts[i]);
    //   ranks_x[i] = get_rank(x_layouts[i]);
    //   locs_x[i]  = get_loc(x_layouts[i]);
    //   h_ndims_x[i] = get_ndims(x_layouts[i]);
    // }
    // d_val = val_kokkos.template view<ExecutionSpace>();
  }

  // if (xdot_names.size()>0) {
  //   auto xdot_layouts = p.get<Teuchos::Array<layout_ptr>>("xdot Layouts");
  //   TEUCHOS_TEST_FOR_EXCEPTION (xdot_layouts.size()==xdot_names.size(), std::runtime_error,
  //       "Error! 'xdot Layouts' and 'xdot Names' have inconsistent lengths.\n"
  //       "  - xdot Names size  : " + std::to_string(xdot_names.size()) + "\n"
  //       "  - xdot Layouts size: " + std::to_string(xdot_layouts.size()) + "\n");
  //   num_xdot = xdot_names.size();
  //   gather_xdot = true;

  //   val_dot.reserver(num_xdot);
  //   val_dot_kokkos.resize(num_xdot);
  //   ranks_xdot = IView("ranks_xdot",num_xdot);
  //   ndims_xdot = IView("ndims_xdot",num_xdot);
  //   locs_xdot  = IView("locs_xdot",num_xdot);

  //   auto h_ranks_xdot = Kokkos::create_mirror_view(ranks_xdot);
  //   auto h_ndims_xdot = Kokkos::create_mirror_view(ndims_xdot);
  //   auto h_locs_xdot  = Kokkos::create_mirror_view(locs_xdot);
  //   for (int i=0; i<num_xdot; ++i) {
  //     val_dot.push_back(PHX::MDField<ScalarT>(x_names[i],xdot_layouts[i]));
  //     this->addEvaluatedField(val_dot.back());
  //     val_dot_kokkos[i] = val_dot.back().get_static_view();

  //     h_ranks_xdot[i] = get_rank(xdot_layouts[i]);
  //     h_ndims_xdot[i] = get_ndims(xdot_layouts[i]);
  //     h_locs_xdot[i]  = get_loc(xdot_layouts[i]);
  //   }
  //   d_val_dot = val_kokkos.template view<ExecutionSpace>();
  // }

  // if (xdotdot_names.size()>0) {
  //   auto xdotdot_layouts = p.get<Teuchos::Array<layout_ptr>>("xdotdot Layouts");
  //   TEUCHOS_TEST_FOR_EXCEPTION (xdotdot_layouts.size()==xdotdot_names.size(), std::runtime_error,
  //       "Error! 'xdotdot Layouts' and 'xdotdot Names' have inconsistent lengths.\n"
  //       "  - xdotdot Names size  : " + std::to_string(xdotdot_names.size()) + "\n"
  //       "  - xdotdot Layouts size: " + std::to_string(xdotdot_layouts.size()) + "\n");
  //   num_xdotdot = xdotdot_names.size();
  //   gather_xdotdot = true;

  //   val_dotdot.reserver(num_xdotdot);
  //   val_dotdot_kokkos.resize(num_xdotdot);
  //   ranks_xdotdot = IView("ranks_xdotdot",num_xdotdot);
  //   ndims_xdotdot = IView("ndims_xdotdot",num_xdotdot);
  //   locs_xdotdot  = IView("locs_xdotdot",num_xdotdot);

  //   auto h_ranks_xdotdot = Kokkos::create_mirror_view(ranks_xdotdot);
  //   auto h_ndims_xdotdot = Kokkos::create_mirror_view(ndims_xdotdot);
  //   auto h_locs_xdotdot  = Kokkos::create_mirror_view(locs_xdotdot);
  //   for (int i=0; i<num_xdotdot; ++i) {
  //     val_dotdot.push_back(PHX::MDField<ScalarT>(x_names[i],xdotdot_layouts[i]));
  //     this->addEvaluatedField(val_dotdot.back());
  //     val_dotdot_kokkos[i] = val_dotdot.back().get_static_view();

  //     h_ranks_xdotdot[i] = get_rank(xdotdot_layouts[i]);
  //     h_ndims_xdotdot[i] = get_ndims(xdotdot_layouts[i]);
  //     h_locs_xdotdot[i]  = get_loc(xdotdot_layouts[i]);
  //   }
  //   d_val_dotdot = val_kokkos.template view<ExecutionSpace>();
  // }

  // Determine location of each component in block
  auto block_str = p.get<std::string>("Solution Block Structure");

//   // scalar
//     val.resize(solution_names.size());
//     for (int eq = 0; eq < solution_names.size(); ++eq) {
//       PHX::MDField<ScalarT,Cell,Node> f(solution_names[eq],dl->node_scalar);
//       val[eq] = f;
//       this->addEvaluatedField(val[eq]);
//     }
//     // repeat for xdot if transient is enabled
//     if (gather_xdot) {
//       const Teuchos::ArrayRCP<std::string>& names_dot =
//         p.get< Teuchos::ArrayRCP<std::string> >("Time Dependent Solution Names");

//       val_dot.resize(names_dot.size());
//       for (int eq = 0; eq < names_dot.size(); ++eq) {
//         PHX::MDField<ScalarT,Cell,Node> f(names_dot[eq],dl->node_scalar);
//         val_dot[eq] = f;
//         this->addEvaluatedField(val_dot[eq]);
//       }
//     }
//     // repeat for xdotdot if acceleration is enabled
//     if (enableAcceleration) {
//       const Teuchos::ArrayRCP<std::string>& names_dotdot =
//         p.get< Teuchos::ArrayRCP<std::string> >("Solution Acceleration Names");

//       val_dotdot.resize(names_dotdot.size());
//       for (int eq = 0; eq < names_dotdot.size(); ++eq) {
//         PHX::MDField<ScalarT,Cell,Node> f(names_dotdot[eq],dl->node_scalar);
//         val_dotdot[eq] = f;
//         this->addEvaluatedField(val_dotdot[eq]);
//       }
//     }
//     numFieldsBase = val.size();
//   } else if ( tensorRank == 1 ) {
//     // vector
//     PHX::MDField<ScalarT,Cell,Node,VecDim> f(solution_names[0],dl->node_vector);
//     valVec= f;
//     this->addEvaluatedField(valVec);
//     // repeat for xdot if transient is enabled
//     if (enableTransient) {
//       const Teuchos::ArrayRCP<std::string>& names_dot =
//         p.get< Teuchos::ArrayRCP<std::string> >("Time Dependent Solution Names");

//       PHX::MDField<ScalarT,Cell,Node,VecDim> fdot(names_dot[0],dl->node_vector);
//       valVec_dot= fdot;
//       this->addEvaluatedField(valVec_dot);
//     }
//     // repeat for xdotdot if acceleration is enabled
//     if (enableAcceleration) {
//       const Teuchos::ArrayRCP<std::string>& names_dotdot =
//         p.get< Teuchos::ArrayRCP<std::string> >("Solution Acceleration Names");

//       PHX::MDField<ScalarT,Cell,Node,VecDim> fdotdot(names_dotdot[0],dl->node_vector);
//       valVec_dotdot = fdotdot;
//       this->addEvaluatedField(valVec_dotdot);
//     }
//     numFieldsBase = dl->node_vector->extent(2);
//   } else if ( tensorRank == 2 ) {
//     // tensor
//     PHX::MDField<ScalarT,Cell,Node,VecDim,VecDim> f(solution_names[0],dl->node_tensor);
//     valTensor = f;
//     this->addEvaluatedField(valTensor);
//     // repeat for xdot if transient is enabled
//     if (enableTransient) {
//       const Teuchos::ArrayRCP<std::string>& names_dot =
//         p.get< Teuchos::ArrayRCP<std::string> >("Time Dependent Solution Names");

//       PHX::MDField<ScalarT,Cell,Node,VecDim,VecDim> fdot(names_dot[0],dl->node_tensor);
//       valTensor_dot = fdot;
//       this->addEvaluatedField(valTensor_dot);
//     }
//     // repeat for xdotdot if acceleration is enabled
//     if (enableAcceleration) {
//       const Teuchos::ArrayRCP<std::string>& names_dotdot =
//         p.get< Teuchos::ArrayRCP<std::string> >("Solution Acceleration Names");

//       PHX::MDField<ScalarT,Cell,Node,VecDim,VecDim> fdotdot(names_dotdot[0],dl->node_tensor);
//       valTensor_dotdot = fdotdot;
//       this->addEvaluatedField(valTensor_dotdot);
//     }
//     numFieldsBase = (dl->node_tensor->extent(2))*(dl->node_tensor->extent(3));
//   }

// #ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
//   if ( tensorRank == 0 ) {
//     val_kokkos.resize(numFieldsBase);
//     if (enableTransient)
//       val_dot_kokkos.resize(numFieldsBase);
//     if (enableAcceleration)
//       val_dotdot_kokkos.resize(numFieldsBase);
//   }
// #endif

//   if (p.isType<int>("Offset of First DOF"))
//     offset = p.get<int>("Offset of First DOF");
//   else offset = 0;

  this->setName("Gather Blocked Solution"+PHX::print<EvalT>() );
}

// **********************************************************************
template<typename EvalT, typename Traits>
void GatherBlockedSolutionBase<EvalT,Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& /* fm */)
{
  // if (tensorRank == 0) {
  //   for (std::size_t eq = 0; eq < numFieldsBase; ++eq)
  //     this->utils.setFieldData(val[eq],fm);
  //   if (enableTransient) {
  //     for (std::size_t eq = 0; eq < val_dot.size(); ++eq)
  //       this->utils.setFieldData(val_dot[eq],fm);
  //   }
  //   if (enableAcceleration) {
  //     for (std::size_t eq = 0; eq < val_dotdot.size(); ++eq)
  //       this->utils.setFieldData(val_dotdot[eq],fm);
  //   }
  //   numNodes = val[0].extent(1);
  // } else if (tensorRank == 1) {
  //   this->utils.setFieldData(valVec,fm);
  //   if (enableTransient) this->utils.setFieldData(valVec_dot,fm);
  //   if (enableAcceleration) this->utils.setFieldData(valVec_dotdot,fm);
  //   numNodes = valVec.extent(1);
  // } else if (tensorRank == 2) {
  //   this->utils.setFieldData(valTensor,fm);
  //   if (enableTransient) this->utils.setFieldData(valTensor_dot,fm);
  //   if (enableAcceleration) this->utils.setFieldData(valTensor_dotdot,fm);
  //   numNodes = valTensor.extent(1);
  // }
  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields(),false);
}

// **********************************************************************

// **********************************************************************
// Specialization: Residual
// **********************************************************************

template<typename Traits>
GatherBlockedSolution<PHAL::AlbanyTraits::Residual, Traits>::
GatherBlockedSolution(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl)
 : base_type(p,dl)
{
  // Nothing to do
}

template<typename Traits>
GatherBlockedSolution<PHAL::AlbanyTraits::Residual, Traits>::
GatherBlockedSolution(const Teuchos::ParameterList& p)
 : base_type(p,p.get<Teuchos::RCP<Albany::Layouts> >("Layouts Struct"))
{
  // Nothing to do
}

// ********************************************************************
// Kokkos functors for Residual
template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherBlockedSolution<PHAL::AlbanyTraits::Residual, Traits>::
operator() (const Rank0_Node_Tag&, const int& cell) const{
  for (size_t node = 0; node < this->numNodes; ++node)
    this->val(cell,node) = this->x_view(nodeID(cell, node, this->block_offset));
    // (this->valVec)(cell,node)= x_constView(nodeID(cell, node,this->offset+eq));
}

// template<typename Traits>
// KOKKOS_INLINE_FUNCTION
// void GatherBlockedSolution<PHAL::AlbanyTraits::Residual, Traits>::
// operator() (const PHAL_GatherBlockedSolRank1_Transient_Tag&, const int& cell) const{
//   for (size_t node = 0; node < this->numNodes; ++node)
//     for (int eq = 0; eq < numFields; eq++)
//       (this->valVec_dot)(cell,node,eq)= xdot_constView(nodeID(cell, node, this->offset+eq));
// }

// template<typename Traits>
// KOKKOS_INLINE_FUNCTION
// void GatherBlockedSolution<PHAL::AlbanyTraits::Residual, Traits>::
// operator() (const PHAL_GatherBlockedSolRank1_Acceleration_Tag&, const int& cell) const{
//   for (size_t node = 0; node < this->numNodes; ++node)
//     for (int eq = 0; eq < numFields; eq++)
//       (this->valVec_dotdot)(cell,node,eq)= xdotdot_constView(nodeID(cell, node, this->offset+eq));
// }

// template<typename Traits>
// KOKKOS_INLINE_FUNCTION
// void GatherBlockedSolution<PHAL::AlbanyTraits::Residual, Traits>::
// operator() (const PHAL_GatherBlockedSolRank2_Tag&, const int& cell) const{
//   for (size_t node = 0; node < this->numNodes; ++node)
//     for (int eq = 0; eq < numFields; eq++)
//       (this->valTensor)(cell,node,eq/numDim,eq%numDim)= x_constView(nodeID(cell, node, this->offset+eq));
// }

// template<typename Traits>
// KOKKOS_INLINE_FUNCTION
// void GatherBlockedSolution<PHAL::AlbanyTraits::Residual, Traits>::
// operator() (const PHAL_GatherBlockedSolRank2_Transient_Tag&, const int& cell) const{
//   for (size_t node = 0; node < this->numNodes; ++node)
//     for (int eq = 0; eq < numFields; eq++)
//       (this->valTensor_dot)(cell,node,eq/numDim,eq%numDim)= xdot_constView(nodeID(cell, node, this->offset+eq));
// }

// template<typename Traits>
// KOKKOS_INLINE_FUNCTION
// void GatherBlockedSolution<PHAL::AlbanyTraits::Residual, Traits>::
// operator() (const PHAL_GatherBlockedSolRank2_Acceleration_Tag&, const int& cell) const{
//   for (size_t node = 0; node < this->numNodes; ++node)
//     for (int eq = 0; eq < numFields; eq++)
//       (this->valTensor_dotdot)(cell,node,eq/numDim,eq%numDim)= xdotdot_constView(nodeID(cell, node, this->offset+eq));
// }

// template<typename Traits>
// KOKKOS_INLINE_FUNCTION
// void GatherBlockedSolution<PHAL::AlbanyTraits::Residual, Traits>::
// operator() (const PHAL_GatherBlockedSolRank0_Tag&, const int& cell) const{
//   for (size_t node = 0; node < this->numNodes; ++node)
//     for (int eq = 0; eq < numFields; eq++)
//       d_val[eq](cell,node)= x_constView(nodeID(cell, node, this->offset+eq));
// }

// template<typename Traits>
// KOKKOS_INLINE_FUNCTION
// void GatherBlockedSolution<PHAL::AlbanyTraits::Residual, Traits>::
// operator() (const PHAL_GatherBlockedSolRank0_Transient_Tag&, const int& cell) const{
//   for (size_t node = 0; node < this->numNodes; ++node)
//     for (int eq = 0; eq < numFields; eq++)
//       d_val_dot[eq](cell,node)= xdot_constView(nodeID(cell, node, this->offset+eq));
// }

// template<typename Traits>
// KOKKOS_INLINE_FUNCTION
// void GatherBlockedSolution<PHAL::AlbanyTraits::Residual, Traits>::
// operator() (const PHAL_GatherBlockedSolRank0_Acceleration_Tag&, const int& cell) const{
//   for (size_t node = 0; node < this->numNodes; ++node)
//     for (int eq = 0; eq < numFields; eq++)
//       d_val_dotdot[eq](cell,node)= xdotdot_constView(nodeID(cell, node, this->offset+eq));
// }

// **********************************************************************
template<typename Traits>
void GatherBlockedSolution<PHAL::AlbanyTraits::Residual, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // const auto x_prod       = Albany::getProductVector(workset.x);
  // const auto xdot    = workset.xdot;
  // const auto xdotdot = workset.xdotdot;

// #ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
//   auto nodeID = workset.wsElNodeEqID;
//   Teuchos::ArrayRCP<const ST> x_constView, xdot_constView, xdotdot_constView;
//   x_constView = Albany::getLocalData(x);
//   if(!xdot.is_null()) {
//     xdot_constView = Albany::getLocalData(xdot);
//   }
//   if(!xdotdot.is_null()) {
//     xdotdot_constView = Albany::getLocalData(xdotdot);
//   }

//   if (this->tensorRank == 1) {
//     for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
//       for (std::size_t node = 0; node < this->numNodes; ++node) {
//         for (std::size_t eq = 0; eq < numFields; eq++)
//           (this->valVec)(cell,node,eq) = x_constView[nodeID(cell,node,this->offset + eq)];
//         if (workset.transientTerms && this->enableTransient) {
//           for (std::size_t eq = 0; eq < numFields; eq++)
//             (this->valVec_dot)(cell,node,eq) = xdot_constView[nodeID(cell,node,this->offset + eq)];
//         }
//         if (workset.accelerationTerms && this->enableAcceleration) {
//           for (std::size_t eq = 0; eq < numFields; eq++)
//             (this->valVec_dotdot)(cell,node,eq) = xdotdot_constView[nodeID(cell,node,this->offset + eq)];
//         }
//       }
//     }
//   } else if (this->tensorRank == 2) {
//     int numDim = this->valTensor.extent(2);
//     for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
//       for (std::size_t node = 0; node < this->numNodes; ++node) {
//         for (std::size_t eq = 0; eq < numFields; eq++)
//           (this->valTensor)(cell,node,eq/numDim,eq%numDim) = x_constView[nodeID(cell,node,this->offset + eq)];
//         if (workset.transientTerms && this->enableTransient) {
//           for (std::size_t eq = 0; eq < numFields; eq++)
//             (this->valTensor_dot)(cell,node,eq/numDim,eq%numDim) = xdot_constView[nodeID(cell,node,this->offset + eq)];
//         }
//         if (workset.accelerationTerms && this->enableAcceleration) {
//           for (std::size_t eq = 0; eq < numFields; eq++)
//             (this->valTensor_dotdot)(cell,node,eq/numDim,eq%numDim) = xdotdot_constView[nodeID(cell,node,this->offset + eq)];
//         }
//       }
//     }
//   } else {
//     for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
//       for (std::size_t node = 0; node < this->numNodes; ++node) {
//         for (std::size_t eq = 0; eq < numFields; eq++)
//           (this->val[eq])(cell,node) = x_constView[nodeID(cell,node,this->offset + eq)];
//         if (workset.transientTerms && this->enableTransient) {
//           for (std::size_t eq = 0; eq < numFields; eq++)
//             (this->val_dot[eq])(cell,node) = xdot_constView[nodeID(cell,node,this->offset + eq)];
//         }
//         if (workset.accelerationTerms && this->enableAcceleration) {
//           for (std::size_t eq = 0; eq < numFields; eq++)
//             (this->val_dotdot[eq])(cell,node) = xdotdot_constView[nodeID(cell,node,this->offset + eq)];
//         }
//       }
//     }
//   }

// #else
#ifdef ALBANY_TIMER
  auto start = std::chrono::high_resolution_clock::now();
#endif

  for (int i=0; i<this->num_x; ++i) {
    auto& blocks = this->x_blocks[i];
    Teuchos::RCP<Thyra_Vector> x = workset.x;
    for (auto& idx : blocks->global_block_coords) {
      auto tmp = Albany::getProductVector(x);
      x = tmp->getVectorBlock(idx);
    }

    // Now we have the (non-blocked) thyra vector from which to extract the view
    this->x_view = Albany::getDeviceData(x);

    // Flatten out combinations, so we can use a simple switch rather than nested switches
    using FL = Albany::FieldLocation;
    using FR = Albany::FieldRankType;
    auto get_code = [](const int rank, const int loc) -> int {
      return rank*10 + loc;
    };
    auto get_code_enum = [&](const FR rank, const FL loc) -> int {
      return get_code(etoi(loc),etoi(rank));
    };

    auto code = get_code(this->x_ranks[i],this->x_locs[i]);
    switch (code) {
      case get_code_enum(FR::Scalar,FL::Node):
        Kokkos::parallel_for(Rank0_Node_Policy(0,workset.numCells),*this);
        cudaCheckError();
        break;
      case get_code_enum(FR::Scalar,FL::Cell):
        Kokkos::parallel_for(Rank0_Cell_Policy(0,workset.numCells),*this);
        cudaCheckError();
        break;
      case get_code_enum(FR::Vector,FL::Node):    // Fallthrough
      case get_code_enum(FR::Gradient,FL::Node):
        Kokkos::parallel_for(Rank1_Node_Policy(0,workset.numCells),*this);
        cudaCheckError();
        break;
      case get_code_enum(FR::Vector,FL::Cell):    // Fallthrough
      case get_code_enum(FR::Gradient,FL::Cell):
        Kokkos::parallel_for(Rank1_Cell_Policy(0,workset.numCells),*this);
        cudaCheckError();
        break;
      case get_code_enum(FR::Tensor,FL::Node):
        Kokkos::parallel_for(Rank2_Node_Policy(0,workset.numCells),*this);
        cudaCheckError();
        break;
      case get_code_enum(FR::Tensor,FL::Cell):
        Kokkos::parallel_for(Rank2_Cell_Policy(0,workset.numCells),*this);
        cudaCheckError();
        break;
      default:
        ALBANY_ABORT ("Error! Unexpected rank in GatherBlockedSolution<Residual>.\n"
                      "       This is an internal error. Please, contact developers.\n");
    }
  }

  // Get map for local data structures
  // nodeID = workset.wsElNodeEqID;

  // Get vector view from a specific device
  // x_constView = Albany::getDeviceData(x);
  // if(!xdot.is_null()) {
  //   xdot_constView = Albany::getDeviceData(xdot);
  // }
  // if(!xdotdot.is_null()) {
  //   xdotdot_constView = Albany::getDeviceData(xdotdot);
  // }

  // if (this->tensorRank == 2){
  //   numDim = this->valTensor.extent(2);
  //   Kokkos::parallel_for(PHAL_GatherBlockedSolRank2_Policy(0,workset.numCells),*this);
  //   cudaCheckError();

  //   if (workset.transientTerms && this->enableTransient) {
  //     Kokkos::parallel_for(PHAL_GatherBlockedSolRank2_Transient_Policy(0,workset.numCells),*this);
  //     cudaCheckError();
  //   }

  //   if (workset.accelerationTerms && this->enableAcceleration) {
  //     Kokkos::parallel_for(PHAL_GatherBlockedSolRank2_Acceleration_Policy(0,workset.numCells),*this);
  //     cudaCheckError();
  //   }
  // } else if (this->tensorRank == 1){
  //   Kokkos::parallel_for(PHAL_GatherBlockedSolRank1_Policy(0,workset.numCells),*this);
  //   cudaCheckError();

  //   if (workset.transientTerms && this->enableTransient) {
  //     Kokkos::parallel_for(PHAL_GatherBlockedSolRank1_Transient_Policy(0,workset.numCells),*this);
  //     cudaCheckError();
  //   }

  //   if (workset.accelerationTerms && this->enableAcceleration) {
  //     Kokkos::parallel_for(PHAL_GatherBlockedSolRank1_Acceleration_Policy(0,workset.numCells),*this);
  //     cudaCheckError();
  //   }
  // } else {
  //   // Get MDField views from std::vector
  //   for (int i =0; i<numFields;i++){
  //     //val_kokkos[i]=this->val[i].get_view();
  //     val_kokkos[i]=this->val[i].get_static_view();
  //   }
  //   d_val=val_kokkos.template view<ExecutionSpace>();

  //   Kokkos::parallel_for(PHAL_GatherBlockedSolRank0_Policy(0,workset.numCells),*this);
  //   cudaCheckError();

  //   if (workset.transientTerms && this->enableTransient){
  //     // Get MDField views from std::vector
  //     for (int i =0; i<numFields;i++){
  //       //val_dot_kokkos[i]=this->val_dot[i].get_view();
  //       val_dot_kokkos[i]=this->val_dot[i].get_static_view();
  //     }
  //     d_val_dot=val_dot_kokkos.template view<ExecutionSpace>();

  //     Kokkos::parallel_for(PHAL_GatherBlockedSolRank0_Transient_Policy(0,workset.numCells),*this);
  //     cudaCheckError();
  //   }
  //   if (workset.accelerationTerms && this->enableAcceleration){
  //     // Get MDField views from std::vector
  //     for (int i =0; i<numFields;i++){
  //       //val_dotdot_kokkos[i]=this->val_dotdot[i].get_view();
  //       val_dotdot_kokkos[i]=this->val_dotdot[i].get_static_view();
  //     }
  //     d_val_dotdot=val_dotdot_kokkos.template view<ExecutionSpace>();

  //     Kokkos::parallel_for(PHAL_GatherBlockedSolRank0_Acceleration_Policy(0,workset.numCells),*this);
  //     cudaCheckError();
  //   }
  // }

#ifdef ALBANY_TIMER
  PHX::Device::fence();
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
  long long millisec= std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
  std::cout<< "GaTher Solution Residual time = "  << millisec << "  "  << microseconds << std::endl;
#endif
// #endif
}

// **********************************************************************
// Specialization: Jacobian
// **********************************************************************

template<typename Traits>
GatherBlockedSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
GatherBlockedSolution(const Teuchos::ParameterList& p,
          const Teuchos::RCP<Albany::Layouts>& dl) :
GatherBlockedSolutionBase<PHAL::AlbanyTraits::Jacobian, Traits>(p,dl),
numFields(GatherBlockedSolutionBase<PHAL::AlbanyTraits::Jacobian,Traits>::numFieldsBase)
{
}

template<typename Traits>
GatherBlockedSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
GatherBlockedSolution(const Teuchos::ParameterList& p) :
GatherBlockedSolutionBase<PHAL::AlbanyTraits::Jacobian, Traits>(p,p.get<Teuchos::RCP<Albany::Layouts> >("Layouts Struct")),
numFields(GatherBlockedSolutionBase<PHAL::AlbanyTraits::Jacobian,Traits>::numFieldsBase)
{
}

//********************************************************************
////Kokkos functors for Jacobian
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherBlockedSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
operator() (const PHAL_GatherJacRank2_Tag&, const int& cell) const{
  for (size_t node = 0; node < this->numNodes; ++node){
    int firstunk = neq * node + this->offset;
    for (int eq = 0; eq < numFields; eq++){
      typename PHAL::Ref<ScalarT>::type valref = (this->valTensor)(cell,node,eq/numDim,eq%numDim);
      valref=FadType(valref.size(), x_constView(nodeID(cell,node,this->offset+eq)));
      valref.fastAccessDx(firstunk + eq) =j_coeff;
    }
  }
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherBlockedSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
operator() (const PHAL_GatherJacRank2_Transient_Tag&, const int& cell) const{
  for (size_t node = 0; node < this->numNodes; ++node){
    int firstunk = neq * node + this->offset;
    for (int eq = 0; eq < numFields; eq++){
      typename PHAL::Ref<ScalarT>::type valref = (this->valTensor_dot)(cell,node,eq/numDim,eq%numDim);
      valref =FadType(valref.size(), xdot_constView(nodeID(cell,node,this->offset+eq)));
      valref.fastAccessDx(firstunk + eq) =m_coeff;
    }
  }
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherBlockedSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
operator() (const PHAL_GatherJacRank2_Acceleration_Tag&, const int& cell) const{
  for (size_t node = 0; node < this->numNodes; ++node){
    int firstunk = neq * node + this->offset;
    for (int eq = 0; eq < numFields; eq++){
      typename PHAL::Ref<ScalarT>::type valref = (this->valTensor_dotdot)(cell,node,eq/numDim,eq%numDim);
      valref=FadType(valref.size(), xdotdot_constView(nodeID(cell,node,this->offset+eq)));
      valref.fastAccessDx(firstunk + eq) =n_coeff;
    }
  }
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherBlockedSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
operator() (const PHAL_GatherJacRank1_Tag&, const int& cell) const{
  for (size_t node = 0; node < this->numNodes; node++){
    int firstunk = neq * node + this->offset;
    for (int eq = 0; eq < numFields; eq++){
      typename PHAL::Ref<ScalarT>::type valref = (this->valVec)(cell,node,eq);
      valref =FadType(valref.size(), x_constView(nodeID(cell,node,this->offset+eq)));
      valref.fastAccessDx(firstunk + eq) =j_coeff;
    }
  }
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherBlockedSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
operator() (const PHAL_GatherJacRank1_Transient_Tag&, const int& cell) const{
  for (size_t node = 0; node < this->numNodes; ++node){
    int firstunk = neq * node + this->offset;
    for (int eq = 0; eq < numFields; eq++){
      typename PHAL::Ref<ScalarT>::type valref = (this->valVec_dot)(cell,node,eq);
      valref =FadType(valref.size(), xdot_constView(nodeID(cell,node,this->offset+eq)));
      valref.fastAccessDx(firstunk + eq) =m_coeff;
    }
  }
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherBlockedSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
operator() (const PHAL_GatherJacRank1_Acceleration_Tag&, const int& cell) const{
  for (size_t node = 0; node < this->numNodes; ++node){
    int firstunk = neq * node + this->offset;
    for (int eq = 0; eq < numFields; eq++){
      typename PHAL::Ref<ScalarT>::type valref = (this->valVec_dotdot)(cell,node,eq);
      valref =FadType(valref.size(), xdotdot_constView(nodeID(cell,node,this->offset+eq)));
      valref.fastAccessDx(firstunk + eq) =n_coeff;
    }
  }
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherBlockedSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
operator() (const PHAL_GatherJacRank0_Tag&, const int& cell) const{
  for (size_t node = 0; node < this->numNodes; ++node){
    int firstunk = neq * node + this->offset;
    for (int eq = 0; eq < numFields; eq++){
      typename PHAL::Ref<ScalarT>::type valref = d_val[eq](cell,node);
      valref =FadType(valref.size(), x_constView(nodeID(cell,node,this->offset+eq)));
      valref.fastAccessDx(firstunk + eq) =j_coeff;
    }
  }
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherBlockedSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
operator() (const PHAL_GatherJacRank0_Transient_Tag&, const int& cell) const{
  for (size_t node = 0; node < this->numNodes; ++node){
    int firstunk = neq * node + this->offset;
    for (int eq = 0; eq < numFields; eq++){
      typename PHAL::Ref<ScalarT>::type valref = d_val_dot[eq](cell,node);
      valref =FadType(valref.size(), xdot_constView(nodeID(cell,node,this->offset+eq)));
      valref.fastAccessDx(firstunk + eq) =m_coeff;
    }
  }
}

template<typename Traits>
KOKKOS_INLINE_FUNCTION
void GatherBlockedSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
operator() (const PHAL_GatherJacRank0_Acceleration_Tag&, const int& cell) const{
  for (size_t node = 0; node < this->numNodes; ++node){
    int firstunk = neq * node + this->offset;
    for (int eq = 0; eq < numFields; eq++){
      typename PHAL::Ref<ScalarT>::type valref = d_val_dotdot[eq](cell,node);
      valref = FadType(valref.size(), xdotdot_constView(nodeID(cell,node,this->offset+eq)));
      valref.fastAccessDx(firstunk + eq) = n_coeff;
    }
  }
}

#endif

// **********************************************************************
template<typename Traits>
void GatherBlockedSolution<PHAL::AlbanyTraits::Jacobian, Traits>::
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

  int numDim = 0;
  if (this->tensorRank==2) numDim = this->valTensor.extent(2); // only needed for tensor fields

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const int neq = nodeID.extent(2);

    for (std::size_t node = 0; node < this->numNodes; ++node) {
      int firstunk = neq * node + this->offset;
      for (std::size_t eq = 0; eq < numFields; eq++) {
        typename PHAL::Ref<ScalarT>::type
          valref = (this->tensorRank == 0 ? this->val[eq](cell,node) :
                    this->tensorRank == 1 ? this->valVec(cell,node,eq) :
                    this->valTensor(cell,node, eq/numDim, eq%numDim));
        valref = FadType(valref.size(), x_constView[nodeID(cell,node,this->offset + eq)]);
        valref.fastAccessDx(firstunk + eq) = workset.j_coeff;
      }
      if (workset.transientTerms && this->enableTransient) {
        for (std::size_t eq = 0; eq < numFields; eq++) {
        typename PHAL::Ref<ScalarT>::type
          valref = (this->tensorRank == 0 ? this->val_dot[eq](cell,node) :
                    this->tensorRank == 1 ? this->valVec_dot(cell,node,eq) :
                    this->valTensor_dot(cell,node, eq/numDim, eq%numDim));
        valref = FadType(valref.size(), xdot_constView[nodeID(cell,node,this->offset + eq)]);
        valref.fastAccessDx(firstunk + eq) = workset.m_coeff;
        }
      }
      if (workset.accelerationTerms && this->enableAcceleration) {
        for (std::size_t eq = 0; eq < numFields; eq++) {
        typename PHAL::Ref<ScalarT>::type
          valref = (this->tensorRank == 0 ? this->val_dotdot[eq](cell,node) :
                    this->tensorRank == 1 ? this->valVec_dotdot(cell,node,eq) :
                    this->valTensor_dotdot(cell,node, eq/numDim, eq%numDim));
        valref = FadType(valref.size(), xdotdot_constView[nodeID(cell,node,this->offset + eq)]);
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

  if (this->tensorRank == 2) {
    numDim = this->valTensor.extent(2);

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
  } else if (this->tensorRank == 1) {
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
  } else {
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
GatherBlockedSolution<PHAL::AlbanyTraits::Tangent, Traits>::
GatherBlockedSolution(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
  GatherBlockedSolutionBase<PHAL::AlbanyTraits::Tangent, Traits>(p,dl),
  numFields(GatherBlockedSolutionBase<PHAL::AlbanyTraits::Tangent,Traits>::numFieldsBase)
{
}

template<typename Traits>
GatherBlockedSolution<PHAL::AlbanyTraits::Tangent, Traits>::
GatherBlockedSolution(const Teuchos::ParameterList& p) :
  GatherBlockedSolutionBase<PHAL::AlbanyTraits::Tangent, Traits>(p,p.get<Teuchos::RCP<Albany::Layouts> >("Layouts Struct")),
  numFields(GatherBlockedSolutionBase<PHAL::AlbanyTraits::Tangent,Traits>::numFieldsBase)
{
}

// **********************************************************************
template<typename Traits>
void GatherBlockedSolution<PHAL::AlbanyTraits::Tangent, Traits>::
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
  const auto& x_constView   = Albany::getLocalData(x);
  const auto& Vx_data       = Albany::getLocalData(Vx);
  const auto& Vxdot_data    = Albany::getLocalData(Vxdot);
  const auto& Vxdotdot_data = Albany::getLocalData(Vxdotdot);

  Teuchos::RCP<ParamVec> params = workset.params;
  //int num_cols_tot = workset.param_offset + workset.num_cols_p;

  int numDim = 0;
  if(this->tensorRank==2) {
    numDim = this->valTensor.extent(2); // only needed for tensor fields
  }

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    for (std::size_t node = 0; node < this->numNodes; ++node) {
      for (std::size_t eq = 0; eq < numFields; eq++) {
        typename PHAL::Ref<ScalarT>::type
          valref = ((this->tensorRank == 2) ? (this->valTensor)(cell,node,eq/numDim,eq%numDim) :
                    (this->tensorRank == 1) ? (this->valVec)(cell,node,eq) :
                    (this->val[eq])(cell,node));
        if (Vx != Teuchos::null && workset.j_coeff != 0.0) {
          valref = TanFadType(valref.size(), x_constView[nodeID(cell,node,this->offset + eq)]);
          for (int k=0; k<workset.num_cols_x; k++)
            valref.fastAccessDx(k) =
              workset.j_coeff*Vx_data[k][nodeID(cell,node,this->offset + eq)];
        } else {
          valref = TanFadType(x_constView[nodeID(cell,node,this->offset + eq)]);
        }
      }
   }


   if (workset.transientTerms && this->enableTransient) {
    Teuchos::ArrayRCP<const ST> xdot_constView = Albany::getLocalData(xdot);
    for (std::size_t node = 0; node < this->numNodes; ++node) {
        for (std::size_t eq = 0; eq < numFields; eq++) {
        typename PHAL::Ref<ScalarT>::type
          valref = ((this->tensorRank == 2) ? (this->valTensor_dot)(cell,node,eq/numDim,eq%numDim) :
                    (this->tensorRank == 1) ? (this->valVec_dot)(cell,node,eq) :
                    (this->val_dot[eq])(cell,node));
          valref = TanFadType(valref.size(), xdot_constView[nodeID(cell,node,this->offset + eq)]);
          if (Vxdot != Teuchos::null && workset.m_coeff != 0.0) {
            for (int k=0; k<workset.num_cols_x; k++)
              valref.fastAccessDx(k) =
                workset.m_coeff*Vxdot_data[k][nodeID(cell,node,this->offset + eq)];
          }
        }
      }
   }

   if (workset.accelerationTerms && this->enableAcceleration) {
    Teuchos::ArrayRCP<const ST> xdotdot_constView = Albany::getLocalData(xdotdot);
    for (std::size_t node = 0; node < this->numNodes; ++node) {
        for (std::size_t eq = 0; eq < numFields; eq++) {
        typename PHAL::Ref<ScalarT>::type
          valref = ((this->tensorRank == 2) ? (this->valTensor_dotdot)(cell,node,eq/numDim,eq%numDim) :
                    (this->tensorRank == 1) ? (this->valVec_dotdot)(cell,node,eq) :
                    (this->val_dotdot[eq])(cell,node));

          valref = TanFadType(valref.size(), xdotdot_constView[nodeID(cell,node,this->offset + eq)]);
          if (Vxdotdot != Teuchos::null && workset.n_coeff != 0.0) {
            for (int k=0; k<workset.num_cols_x; k++)
              valref.fastAccessDx(k) =
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
GatherBlockedSolution<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
GatherBlockedSolution(const Teuchos::ParameterList& p,
               const Teuchos::RCP<Albany::Layouts>& dl) :
  GatherBlockedSolutionBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p,dl),
  numFields(GatherBlockedSolutionBase<PHAL::AlbanyTraits::DistParamDeriv,Traits>::numFieldsBase)
{
}

template<typename Traits>
GatherBlockedSolution<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
GatherBlockedSolution(const Teuchos::ParameterList& p) :
  GatherBlockedSolutionBase<PHAL::AlbanyTraits::DistParamDeriv, Traits>(p,p.get<Teuchos::RCP<Albany::Layouts> >("Layouts Struct")),
  numFields(GatherBlockedSolutionBase<PHAL::AlbanyTraits::DistParamDeriv,Traits>::numFieldsBase)
{
}

// **********************************************************************
template<typename Traits>
void GatherBlockedSolution<PHAL::AlbanyTraits::DistParamDeriv, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  auto nodeID = workset.wsElNodeEqID;
  const auto& x       = workset.x;
  const auto& xdot    = workset.xdot;
  const auto& xdotdot = workset.xdotdot;

  //get const (read-only) view of x and xdot
  const auto& x_constView = Albany::getLocalData(x);

  if (this->tensorRank == 1) {
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      for (std::size_t node = 0; node < this->numNodes; ++node) {
        for (std::size_t eq = 0; eq < numFields; eq++)
          (this->valVec)(cell,node,eq) = x_constView[nodeID(cell,node,this->offset + eq)];
      }

    if (workset.transientTerms && this->enableTransient) {
      Teuchos::ArrayRCP<const ST> xdot_constView = Albany::getLocalData(xdot);
      for (std::size_t node = 0; node < this->numNodes; ++node) {
          for (std::size_t eq = 0; eq < numFields; eq++)
            (this->valVec_dot)(cell,node,eq) = xdot_constView[nodeID(cell,node,this->offset + eq)];
      }
    }

    if (workset.accelerationTerms && this->enableAcceleration) {
      Teuchos::ArrayRCP<const ST> xdotdot_constView = Albany::getLocalData(xdotdot);
      for (std::size_t node = 0; node < this->numNodes; ++node) {
          for (std::size_t eq = 0; eq < numFields; eq++)
            (this->valVec_dotdot)(cell,node,eq) = xdotdot_constView[nodeID(cell,node,this->offset + eq)];
        }
      }
    }
  } else if (this->tensorRank == 2) {
    int numDim = this->valTensor.extent(2);
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      for (std::size_t node = 0; node < this->numNodes; ++node) {
        for (std::size_t eq = 0; eq < numFields; eq++)
          (this->valTensor)(cell,node,eq/numDim,eq%numDim) = x_constView[nodeID(cell,node,this->offset + eq)];
      }

    if (workset.transientTerms && this->enableTransient) {
      Teuchos::ArrayRCP<const ST> xdot_constView = Albany::getLocalData(xdot);
      for (std::size_t node = 0; node < this->numNodes; ++node) {
          for (std::size_t eq = 0; eq < numFields; eq++)
            (this->valTensor_dot)(cell,node,eq/numDim,eq%numDim) = xdot_constView[nodeID(cell,node,this->offset + eq)];
      }
    }

    if (workset.accelerationTerms && this->enableAcceleration) {
      Teuchos::ArrayRCP<const ST> xdotdot_constView = Albany::getLocalData(xdotdot);
      for (std::size_t node = 0; node < this->numNodes; ++node) {
          for (std::size_t eq = 0; eq < numFields; eq++)
            (this->valTensor_dotdot)(cell,node,eq/numDim,eq%numDim) = xdotdot_constView[nodeID(cell,node,this->offset + eq)];
        }
      }
    }
  } else {
    for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
      for (std::size_t node = 0; node < this->numNodes; ++node) {
        for (std::size_t eq = 0; eq < numFields; eq++)
          (this->val[eq])(cell,node) = x_constView[nodeID(cell,node,this->offset + eq)];
      }
    if (workset.transientTerms && this->enableTransient) {
      Teuchos::ArrayRCP<const ST> xdot_constView = Albany::getLocalData(xdot);
      for (std::size_t node = 0; node < this->numNodes; ++node) {
          for (std::size_t eq = 0; eq < numFields; eq++)
            (this->val_dot[eq])(cell,node) = xdot_constView[nodeID(cell,node,this->offset + eq)];
      }
    }

    if (workset.accelerationTerms && this->enableAcceleration) {
      Teuchos::ArrayRCP<const ST> xdotdot_constView = Albany::getLocalData(xdotdot);
      for (std::size_t node = 0; node < this->numNodes; ++node) {
          for (std::size_t eq = 0; eq < numFields; eq++)
            (this->val_dotdot[eq])(cell,node) = xdotdot_constView[nodeID(cell,node,this->offset + eq)];
        }
      }
    }
  }
}

// **********************************************************************
// Specialization: HessianVec
// **********************************************************************

template<typename Traits>
GatherBlockedSolution<PHAL::AlbanyTraits::HessianVec, Traits>::
GatherBlockedSolution(const Teuchos::ParameterList& p,
               const Teuchos::RCP<Albany::Layouts>& dl) :
  GatherBlockedSolutionBase<PHAL::AlbanyTraits::HessianVec, Traits>(p,dl),
  numFields(GatherBlockedSolutionBase<PHAL::AlbanyTraits::HessianVec,Traits>::numFieldsBase)
{
}

template<typename Traits>
GatherBlockedSolution<PHAL::AlbanyTraits::HessianVec, Traits>::
GatherBlockedSolution(const Teuchos::ParameterList& p) :
  GatherBlockedSolutionBase<PHAL::AlbanyTraits::HessianVec, Traits>(p,p.get<Teuchos::RCP<Albany::Layouts> >("Layouts Struct")),
  numFields(GatherBlockedSolutionBase<PHAL::AlbanyTraits::HessianVec,Traits>::numFieldsBase)
{
}

// **********************************************************************
template<typename Traits>
void GatherBlockedSolution<PHAL::AlbanyTraits::HessianVec, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  const auto& x       = workset.x;
  const auto& xdot    = workset.xdot;
  const auto& xdotdot = workset.xdotdot;

  Teuchos::RCP<const Thyra_MultiVector> direction_x = workset.hessianWorkset.direction_x;

  auto nodeID = workset.wsElNodeEqID;
  Teuchos::ArrayRCP<const ST> x_constView, xdot_constView, xdotdot_constView, direction_x_constView;
  bool g_xx_is_active = !workset.hessianWorkset.hess_vec_prod_g_xx.is_null();
  bool g_xp_is_active = !workset.hessianWorkset.hess_vec_prod_g_xp.is_null();
  bool g_px_is_active = !workset.hessianWorkset.hess_vec_prod_g_px.is_null();
  bool f_xx_is_active = !workset.hessianWorkset.hess_vec_prod_f_xx.is_null();
  bool f_xp_is_active = !workset.hessianWorkset.hess_vec_prod_f_xp.is_null();
  bool f_px_is_active = !workset.hessianWorkset.hess_vec_prod_f_px.is_null();
  x_constView = Albany::getLocalData(x);
  if(!xdot.is_null()) {
    xdot_constView = Albany::getLocalData(xdot);
  }
  if(!xdotdot.is_null()) {
    xdot_constView = Albany::getLocalData(xdot);
  }

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
        "\nError in GatherBlockedSolution<HessianVec, Traits>: "
        "direction_x is not set and the direction is active.\n");
    direction_x_constView = Albany::getLocalData(direction_x->col(0));
  }

  int numDim = 0;
  if (this->tensorRank==2) numDim = this->valTensor.extent(2); // only needed for tensor fields

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    const int neq = nodeID.extent(2);

    for (std::size_t node = 0; node < this->numNodes; ++node) {
      int firstunk = neq * node + this->offset;
      for (std::size_t eq = 0; eq < numFields; eq++) {
        typename PHAL::Ref<ScalarT>::type
          valref = (this->tensorRank == 0 ? this->val[eq](cell,node) :
                    this->tensorRank == 1 ? this->valVec(cell,node,eq) :
                    this->valTensor(cell,node, eq/numDim, eq%numDim));
        RealType xvec_val = x_constView[nodeID(cell,node,this->offset + eq)];

        valref = HessianVecFad(valref.size(), xvec_val);
        // If we differentiate w.r.t. the solution, we have to set the first
        // derivative to 1
        if (is_x_active)
          valref.fastAccessDx(firstunk + eq).val() = 1;
        // If we differentiate w.r.t. the solution direction, we have to set
        // the second derivative to the related direction value
        if (is_x_direction_active)
          valref.val().fastAccessDx(0) = direction_x_constView[nodeID(cell,node,this->offset + eq)];
      }
    }
  }

  for (std::size_t cell=0; cell < workset.numCells; ++cell ) {
    for (std::size_t node = 0; node < this->numNodes; ++node) {
      if (workset.transientTerms && this->enableTransient) {
        for (std::size_t eq = 0; eq < numFields; eq++) {
          typename PHAL::Ref<ScalarT>::type
            valref = (this->tensorRank == 0 ? this->val_dot[eq](cell,node) :
                      this->tensorRank == 1 ? this->valVec_dot(cell,node,eq) :
                      this->valTensor_dot(cell,node, eq/numDim, eq%numDim));
          valref = xdot_constView[nodeID(cell,node,this->offset + eq)];
        }
      }
      if (workset.accelerationTerms && this->enableAcceleration) {
        for (std::size_t eq = 0; eq < numFields; eq++) {
          typename PHAL::Ref<ScalarT>::type
            valref = (this->tensorRank == 0 ? this->val_dotdot[eq](cell,node) :
                      this->tensorRank == 1 ? this->valVec_dotdot(cell,node,eq) :
                      this->valTensor_dotdot(cell,node, eq/numDim, eq%numDim));
          valref = xdotdot_constView[nodeID(cell,node,this->offset + eq)];
        }
      }
    }
  }
}

} // namespace PHAL
