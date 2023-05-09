//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_DUAL_DYN_RANK_VIEW_HPP
#define ALBANY_DUAL_DYN_RANK_VIEW_HPP

#include "Albany_KokkosTypes.hpp"

namespace Albany
{

// A tiny tiny version of a dual view for DynRankView.
// There is no tracking of the state of the dev and host views.
// In particular, consecutive calls to sync_to_dev will all
// perform the deep copy.
template<typename DT>
struct DualDynRankView {
  using dev_t  = Kokkos::DynRankView<DT,Kokkos::LayoutRight,DeviceMemSpace>;
  using host_t = typename dev_t::HostMirror;

  using drvtraits   = typename dev_t::drvtraits;
  using const_DT    = typename drvtraits::const_data_type;
  using nonconst_DT = typename drvtraits::non_const_data_type;
  using other_DT    = typename std::conditional<std::is_same<DT,const_DT>::value,
                                                nonconst_DT, const_DT>::type;

  using type          = DualDynRankView<DT>;
  using const_type    = DualDynRankView<const_DT>;
  using nonconst_type = DualDynRankView<nonconst_DT>;

  // So that conversion-type operator can access internals
  friend struct DualDynRankView<other_DT>;

  DualDynRankView () = default;

  // Construct DualDynRankView on the fly.
  DualDynRankView (const std::string& name,
      const size_t n0,
      const size_t n1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t n2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t n3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t n4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t n5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t n6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      const size_t n7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG)
   : DualDynRankView(dev_t(name,n0,n1,n2,n3,n4,n5,n6,n7))
  {
    // Nothing else to do
  }
  DualDynRankView (dev_t d_view_)
   : d_view(d_view_)
  {
    ALBANY_ASSERT (d_view.data()!=nullptr, "Invalid device view.");
    h_view = Kokkos::create_mirror_view (d_view);
  }
  template<typename HostT, typename =
    typename std::enable_if<not std::is_same<dev_t,host_t>::value and
                                std::is_same<HostT,host_t>::value>::type
  >
  DualDynRankView (HostT t_view_)
   : h_view(t_view_)
  {
    ALBANY_ASSERT (h_view.data()!=nullptr, "Invalid host view.");
    d_view = dev_t(h_view.label(),
        (h_view.rank()>0 ? h_view.extent(0) : KOKKOS_INVALID_INDEX),
        (h_view.rank()>1 ? h_view.extent(1) : KOKKOS_INVALID_INDEX),
        (h_view.rank()>2 ? h_view.extent(2) : KOKKOS_INVALID_INDEX),
        (h_view.rank()>3 ? h_view.extent(3) : KOKKOS_INVALID_INDEX),
        (h_view.rank()>4 ? h_view.extent(4) : KOKKOS_INVALID_INDEX),
        (h_view.rank()>5 ? h_view.extent(5) : KOKKOS_INVALID_INDEX),
        (h_view.rank()>6 ? h_view.extent(6) : KOKKOS_INVALID_INDEX),
        (h_view.rank()>7 ? h_view.extent(7) : KOKKOS_INVALID_INDEX));
  }
  DualDynRankView (const DualDynRankView&) = default;

  template<typename SrcDT>
  typename std::enable_if<
    std::is_same<SrcDT,nonconst_DT>::value,
    DualDynRankView<DT>
  >::type&
  operator= (const DualDynRankView<SrcDT>& src) {
    d_view = src.d_view;
    h_view = src.h_view;

    return *this;
  }

  template<typename SrcDT>
  typename std::enable_if<
    std::is_same<SrcDT,const_DT>::value,
    DualDynRankView<DT>
  >::type&
  operator= (const DualDynRankView<SrcDT>& src) {
    static_assert (std::is_same<DT,const_DT>::value,
        "Error! Cannot assign DualDynRankView<const T> to DualDynRankView<T>.\n");
    d_view = src.d_view;
    h_view = src.h_view;

    return *this;
  }

  // Allow implicit conversion to DualDynRankView<const DT>;
  template<typename SrcDT = DT, typename = typename std::enable_if<not std::is_same<SrcDT,const_DT>::value>::type>
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

  int rank () const { return d_view.rank(); }
  int size () const { return d_view.size(); }

  void reset (const dev_t& d) {
    ALBANY_ASSERT (d.data()!=nullptr, "Invalid device view.");
    d_view = d;
    h_view = Kokkos::create_mirror_view(d_view);
    sync_to_host();
  }

  void reset_from_host_ptr (DT* ptr,
                            const size_t n0,
                            const size_t n1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                            const size_t n2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                            const size_t n3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                            const size_t n4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                            const size_t n5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                            const size_t n6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
                            const size_t n7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG)
  {
    dev_t v_dev(ptr,n0,n1,n2,n3,n4,n5,n6,n7);
    reset (v_dev);
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
    ALBANY_ASSERT (d_view.size()==0, "Cannot resize a non-trivial DualDynRankView.");
    *this = DualDynRankView<DT>(name,n0,n1,n2,n3,n4,n5,n6,n7);
  }

  template<typename IntT>
  void dimensions (std::vector<IntT>& dims) {
    dims.resize(rank());
    for (size_t i=0; i<dims.size(); ++i) {
      dims[i] = d_view.extent(i);
    }
  }

private:

  dev_t   d_view;
  host_t  h_view;
};

} // namespace Albany

#endif // ALBANY_DUAL_DYN_RANK_VIEW_HPP
