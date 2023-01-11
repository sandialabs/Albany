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

  using const_DT      = typename dev_t::traits::const_data_type;
  using nonconst_DT   = typename dev_t::traits::non_const_data_type;
  using other_DT      = typename std::conditional<std::is_same<DT,const_DT>::value,
                                                  nonconst_DT, const_DT>::type;

  using type          = DualView<DT>;
  using const_type    = DualView<const_DT>;
  using nonconst_type = DualView<nonconst_DT>;

  // So that conversion-type operator can access internals
  friend struct DualView<other_DT>;

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
   : DualView(dev_t(name,n0,n1,n2,n3,n4,n5,n6,n7))
  {
    // Nothing else to do
  }
  DualView (dev_t d_view_)
   : d_view(d_view_)
  {
    ALBANY_ASSERT (d_view.data()!=nullptr, "Invalid device view.");
    h_view = Kokkos::create_mirror_view (d_view);
  }
  DualView (const DualView&) = default;

  template<typename SrcDT>
  typename std::enable_if<
    std::is_same<SrcDT,nonconst_DT>::value,
    DualView<DT>
  >::type&
  operator= (const DualView<SrcDT>& src) {
    d_view = src.d_view;
    h_view = src.h_view;

    return *this;
  }

  template<typename SrcDT>
  typename std::enable_if<
    std::is_same<SrcDT,const_DT>::value,
    DualView<DT>
  >::type&
  operator= (const DualView<SrcDT>& src) {
    static_assert (std::is_same<DT,const_DT>::value,
        "Error! Cannot assign DualView<const T> to DualView<T>.\n");
    d_view = src.d_view;
    h_view = src.h_view;

    return *this;
  }

  // Allow implicit conversion to DualView<const DT>;
  operator const_type() const {
    const_type ct;
    ct.d_view = d_view;
    ct.h_view = h_view;
    return ct;
  }

  KOKKOS_INLINE_FUNCTION
  const dev_t&  dev  () const { return d_view; }
  KOKKOS_INLINE_FUNCTION
  const host_t& host () const { return h_view; }

  void sync_to_host () {
    Kokkos::deep_copy(h_view,d_view);
  }
  void sync_to_dev  () {
    Kokkos::deep_copy(d_view,h_view);
  }

  int size () const { return d_view.size(); }

  void reset (const dev_t& d) {
    d_view = d;
    sync_to_host();
  }

  void resize (const std::string& name,
               const size_t n0,
               const size_t n1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
               const size_t n2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
               const size_t n3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
               const size_t n4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
               const size_t n5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
               const size_t n6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
               const size_t n7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG) {
    ALBANY_ASSERT (d_view.size()==0, "Cannot resize a non-trivial DualView.");
    *this = DualView<DT>(name,n0,n1,n2,n3,n4,n5,n6,n7);
  }

private:

  dev_t   d_view;
  host_t  h_view;
};


} // namespace Albany

#endif // ALBANY_KOKKOS_TYPES_HPP
