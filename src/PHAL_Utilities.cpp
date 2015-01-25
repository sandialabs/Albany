#include "Albany_Application.hpp"
#include "PHAL_Utilities.hpp"

namespace PHAL {

template<> int getDerivativeDimensions<PHAL::AlbanyTraits::Jacobian> (
  const Albany::Application* app, const Albany::MeshSpecsStruct* ms)
{
  return app->getNumEquations() * ms->ctd.node_count;
}

template<> int getDerivativeDimensions<PHAL::AlbanyTraits::Tangent> (
  const Albany::Application* app, const Albany::MeshSpecsStruct* ms)
{
  //amb Need to figure this out. Unlike the Jacobian case, it appears that in
  // the Tangent case, it's OK to overestimate the size.
  return 32;
}

template<> int getDerivativeDimensions<PHAL::AlbanyTraits::DistParamDeriv> (
  const Albany::Application* app, const Albany::MeshSpecsStruct* ms)
{
  //amb Need to figure out.
  return app->getNumEquations() * ms->ctd.node_count;
}

template<> int getDerivativeDimensions<PHAL::AlbanyTraits::Jacobian> (
 const Albany::Application* app, const int ebi)
{
  return getDerivativeDimensions<PHAL::AlbanyTraits::Jacobian>(
    app, app->getDiscretization()->getMeshStruct()->getMeshSpecs()[ebi].get());
}

template<> int getDerivativeDimensions<PHAL::AlbanyTraits::Tangent> (
 const Albany::Application* app, const int ebi)
{
  return getDerivativeDimensions<PHAL::AlbanyTraits::Tangent>(
    app, app->getDiscretization()->getMeshStruct()->getMeshSpecs()[ebi].get());
}

template<> int getDerivativeDimensions<PHAL::AlbanyTraits::DistParamDeriv> (
 const Albany::Application* app, const int ebi)
{
  return getDerivativeDimensions<PHAL::AlbanyTraits::DistParamDeriv>(
    app, app->getDiscretization()->getMeshStruct()->getMeshSpecs()[ebi].get());
}

#define pr(m) std::cout << "amb: (phu) " << m << std::endl;
#define prc(m) std::cout << "amb: (phu) " << #m << " " << (m) << std::endl;
namespace {
template<typename ScalarT>
struct A2V {
  std::vector<ScalarT>& v;
  A2V (std::vector<ScalarT>& v) : v(v) {}
  void operator() (typename Ref<const ScalarT>::type a, const int i) {
    v[i] = a;
    pr("v[" << i << "] = " << v[i] << "\na = " << a);
  }
};

template<typename ScalarT>
struct V2A {
  const std::vector<ScalarT>& v;
  V2A (const std::vector<ScalarT>& v) : v(v) {}
  void operator() (typename Ref<ScalarT>::type a, const int i) {
    a = v[i];
    pr("v[" << i << "] = " << v[i] << "\na = " << a);
  }
};

template<typename ScalarT>
void copy (const PHX::MDField<ScalarT>& a, std::vector<ScalarT>& v) {
  v.resize(a.size());
  A2V<ScalarT> a2v(v);
  loop(a2v, a);
}

template<typename ScalarT>
void copy (const std::vector<ScalarT>& v, PHX::MDField<ScalarT>& a) {
  V2A<ScalarT> v2a(v);
  loop(v2a, a);
}

template<typename ScalarT>
void myReduceAll (
  const Teuchos_Comm& comm, const Teuchos::EReductionType reduct_type,
  std::vector<ScalarT>& v)
{
  
}

template<> void myReduceAll<double> (
  const Teuchos_Comm& comm, const Teuchos::EReductionType reduct_type,
  std::vector<double>& v)
{
  reduceAll<int, double>(comm, reduct_type, v.size(), Teuchos::ptr(&v[0]));
}
} // namespace

template<typename ScalarT>
void reduceAll (
  const Teuchos_Comm& comm, const Teuchos::EReductionType reduct_type,
  PHX::MDField<ScalarT>& a)
{
  pr("rank " << a.rank() << " size " << a.size());
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "impl in progress");
  return;
  std::vector<ScalarT> v;
  copy<ScalarT>(a, v);
  pr("reducing");
  myReduceAll<ScalarT>(comm, reduct_type, v);
  copy<ScalarT>(v, a);
}

//amb Should put this somewhere useful.
//amb This should go somewhere useful.
#ifdef ALBANY_SG_MP
# ifdef ALBANY_FADTYPE_NOTEQUAL_TANFADTYPE
#define apply_to_all_ad_types(macro)            \
  macro(RealType)                               \
  macro(FadType)                                \
  macro(TanFadType)                             \
  macro(SGType)                                 \
  macro(SGFadType)                              \
  macro(MPType)                                 \
  macro(MPFadType)
# else
#define apply_to_all_ad_types(macro)            \
  macro(RealType)                               \
  macro(FadType)                                \
  macro(SGType)                                 \
  macro(SGFadType)                              \
  macro(MPType)                                 \
  macro(MPFadType)
# endif
#else // ALBANY_SG_MP
# ifdef ALBANY_FADTYPE_NOTEQUAL_TANFADTYPE
#define apply_to_all_ad_types(macro)            \
  macro(RealType)                               \
  macro(FadType)                                \
  macro(TanFadType)
# else
#define apply_to_all_ad_types(macro)            \
  macro(RealType)                               \
  macro(FadType)
# endif
#endif // ALBANY_SG_MP

#define eti(T)                                                          \
  template void reduceAll<T> (                                          \
    const Teuchos_Comm&, const Teuchos::EReductionType, PHX::MDField<T>&);
apply_to_all_ad_types(eti)
#undef eti
#undef apply_to_all_ad_types

#undef apply_to_all_eval_types
} // namespace PHAL
