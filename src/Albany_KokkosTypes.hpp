//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_KOKKOS_TYPES_HPP
#define ALBANY_KOKKOS_TYPES_HPP

// Get all Albany configuration macros
#include "Albany_config.h"

#include "Albany_ScalarOrdinalTypes.hpp"

// Get Kokkos node wrapper
#include "Tpetra_KokkosCompat_ClassicNodeAPI_Wrapper.hpp"

// Get Kokkos graph and matrix
#include "Kokkos_StaticCrsGraph.hpp"
#include "KokkosSparse_CrsMatrix.hpp"

// Phalanx determines the Kokkos node we use for Tpetra types
#include "Phalanx_KokkosDeviceTypes.hpp"

// The Kokkos node is determined from the Phalanx Device
typedef Tpetra::KokkosCompat::KokkosDeviceWrapperNode<PHX::Device>  KokkosNode;

namespace Albany
{
// NOTE: Tpetra may use a different LO type (Albany uses int32, while tpetra uses int). When extracting local views/matrices,
//       be careful about this. At worst, you may need to extract pointers and reinterpret_cast them.

// kokkos 1d and 2d views to be used for on-device kernels
template<typename Scalar, typename MemoryTraits = Kokkos::MemoryUnmanaged>
using DeviceView1d = Kokkos::View<Scalar*, Kokkos::LayoutLeft, PHX::Device, MemoryTraits>;
template<typename Scalar, typename MemoryTraits = Kokkos::MemoryUnmanaged>
using DeviceView2d = Kokkos::View<Scalar**, Kokkos::LayoutLeft, PHX::Device, MemoryTraits>;

// Kokkos types for local graphs/matrices, to be used for on-device kernels
using DeviceLocalGraph  = Kokkos::StaticCrsGraph<LO, Kokkos::LayoutLeft, KokkosNode::device_type, void, size_t>;

template<typename Scalar>
using DeviceLocalMatrix = KokkosSparse::CrsMatrix<Scalar, LO, KokkosNode::device_type, void, DeviceLocalGraph::size_type>;

} // namespace Albany

#endif // ALBANY_KOKKOS_TYPES_HPP
