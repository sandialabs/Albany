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
#include "KokkosSparse_StaticCrsGraph.hpp"
#include "KokkosSparse_CrsMatrix.hpp"

// Phalanx determines the Kokkos node we use for Tpetra types
#include "Phalanx_KokkosDeviceTypes.hpp"

// To get assert macros
#include "Albany_Macros.hpp"

// The Kokkos node is determined from the Phalanx Device
typedef Tpetra::KokkosCompat::KokkosDeviceWrapperNode<PHX::Device>  KokkosNode;

namespace Albany
{

using DeviceMemSpace = PHX::Device::memory_space;
using HostMemSpace   = Kokkos::HostSpace;

template<typename DT, typename MemSpace = DeviceMemSpace>
using ViewLR = Kokkos::View<DT,Kokkos::LayoutRight,MemSpace>;
// NOTE: Tpetra may use a different LO type (Albany uses int32, while tpetra uses int). When extracting local views/matrices,
//       be careful about this. At worst, you may need to extract pointers and reinterpret_cast them.

using DevLayout = PHX::Device::array_layout;

// kokkos 1d and 2d views to be used for on-device kernels
template<typename Scalar, typename MemoryTraits = Kokkos::MemoryUnmanaged>
using DeviceView1d = Kokkos::View<Scalar*, DevLayout, PHX::Device, MemoryTraits>;
template<typename Scalar, typename MemoryTraits = Kokkos::MemoryUnmanaged>
using DeviceView2d = Kokkos::View<Scalar**, DevLayout, PHX::Device, MemoryTraits>;

// Thyra view types for underlying Tpetra kokkos views
template<typename Scalar>
using ThyraVDeviceView = Kokkos::View<Scalar*, Kokkos::LayoutLeft, PHX::Device>;
template<typename Scalar>
using ThyraMVDeviceView = Kokkos::View<Scalar**, Kokkos::LayoutLeft, PHX::Device>;

// Kokkos types for local graphs/matrices, to be used for on-device kernels
using DeviceLocalGraph  = KokkosSparse::StaticCrsGraph<LO, Kokkos::LayoutLeft, KokkosNode::device_type, void, size_t>;

template<typename Scalar>
using DeviceLocalMatrix = KokkosSparse::CrsMatrix<Scalar, LO, KokkosNode::device_type, void, DeviceLocalGraph::size_type>;

} // namespace Albany

#endif // ALBANY_KOKKOS_TYPES_HPP
