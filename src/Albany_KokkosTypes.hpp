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
#include "KokkosCompat_ClassicNodeAPI_Wrapper.hpp"

// Get Kokkos graph and matrix
#include "Kokkos_StaticCrsGraph.hpp"
#include "KokkosSparse_CrsMatrix.hpp"

// Phalanx determines the Kokkos node we use for Tpetra types
#include "Phalanx_KokkosDeviceTypes.hpp"

// To get assert macros
#include "Albany_Macros.hpp"

// The Kokkos node is determined from the Phalanx Device
typedef Kokkos::Compat::KokkosDeviceWrapperNode<PHX::Device>  KokkosNode;

namespace Albany
{

using DeviceMemSpace = PHX::Device::memory_space;
using HostMemSpace   = Kokkos::HostSpace;

template<typename DT, typename MemSpace = DeviceMemSpace>
using ViewLR = Kokkos::View<DT,Kokkos::LayoutRight,MemSpace>;
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

// A tiny tiny version of a dual view.
// No correctness check except non-null dev view in ctor.
template<typename DT>
struct DualView {
  using dev_t  = ViewLR<DT,DeviceMemSpace>;
  using host_t = typename dev_t::HostMirror;

  DualView () = default;

  // Construct DualView on the fly.
  DualView (const std::string& name,
      const size_t n0,
      const size_t n1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t n2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t n3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t n4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t n5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t n6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t n7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG)
   : d_view(name,n0,n1,n2,n3,n4,n5,n6,n7)
  {
    h_view = Kokkos::create_mirror_view (d_view);
  }
  DualView (dev_t d_view_) : d_view(d_view_) {
    ALBANY_ASSERT (d_view.data()!=nullptr, "Invalid device view.");
    h_view = Kokkos::create_mirror_view (d_view);
  }
  DualView (const DualView&) = default;
  DualView& operator= (const DualView&) = default;

  DualView& operator= (const dev_t d_view_) {
    d_view = d_view_;
    h_view = Kokkos::create_mirror_view (d_view);
  }

  const dev_t&  dev  () const { return d_view; }
  const host_t& host () const { return h_view; }

  void sync_to_host () {
    Kokkos::deep_copy(h_view,d_view);
  }
  void sync_to_dev  () {
    Kokkos::deep_copy(d_view,h_view);
  }

  int size () const { return d_view().size(); }

private:

  dev_t   d_view;
  host_t  h_view;
};


} // namespace Albany

#endif // ALBANY_KOKKOS_TYPES_HPP
