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
  return app->getTangentDerivDimension();
}

template<> int getDerivativeDimensions<PHAL::AlbanyTraits::DistParamDeriv> (
  const Albany::Application* app, const Albany::MeshSpecsStruct* ms)
{
  //Mauro: currently distributed derivatives work only with scalar parameters, to be updated.
  return ms->ctd.node_count;
}

template<> int getDerivativeDimensions<PHAL::AlbanyTraits::Jacobian> (
 const Albany::Application* app, const int ebi)
{
  Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> > mesh_specs = 
  app->returnMeshSpecs(); 
  return getDerivativeDimensions<PHAL::AlbanyTraits::Jacobian>(
    app, mesh_specs[ebi].get());
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

namespace {
template<typename ScalarT>
struct A2V {
  std::vector<ScalarT>& v;
  A2V (std::vector<ScalarT>& v) : v(v) {}
  void operator() (typename Ref<const ScalarT>::type a, const int i) {
    v[i] = a;
  }
};

template<typename ScalarT>
struct V2A {
  const std::vector<ScalarT>& v;
  V2A (const std::vector<ScalarT>& v) : v(v) {}
  void operator() (typename Ref<ScalarT>::type a, const int i) {
    a = v[i];
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
  typedef typename ScalarT::value_type ValueT;
  // Size of array to hold one Fad's derivatives.
  const int sz = v[0].size();
  // Pack into a vector of values.
  std::vector<ValueT> pack;
  for (int i = 0; i < v.size(); ++i) {
    pack.push_back(v[i].val());
    for (int j = 0; j < sz; ++j)
      pack.push_back(v[i].fastAccessDx(j));
  }
  // reduceAll the package.
  switch (reduct_type) {
  case Teuchos::REDUCE_SUM: {
    std::vector<ValueT> send(pack);
    Teuchos::reduceAll<int, ValueT>(
      comm, reduct_type, pack.size(), &send[0], &pack[0]);
  } break;
  default: TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "not impl'ed");
  }
  // Unpack.
  int slot = 0;
  for (int i = 0; i < v.size(); ++i) {
    v[i].val() = pack[slot++];
    for (int j = 0; j < sz; ++j)
      v[i].fastAccessDx(j) = pack[slot++];
  }
}

template<> void myReduceAll<RealType> (
  const Teuchos_Comm& comm, const Teuchos::EReductionType reduct_type,
  std::vector<RealType>& v)
{
  std::vector<RealType> send(v);
  Teuchos::reduceAll<int, RealType>(
    comm, reduct_type, v.size(), &send[0], &v[0]);
}
} // namespace

template<typename ScalarT>
void reduceAll (
  const Teuchos_Comm& comm, const Teuchos::EReductionType reduct_type,
  PHX::MDField<ScalarT>& a)
{
  std::vector<ScalarT> v;
  copy<ScalarT>(a, v);
  myReduceAll<ScalarT>(comm, reduct_type, v);
  copy<ScalarT>(v, a);
}

template<typename ScalarT>
void reduceAll (
  const Teuchos_Comm& comm, const Teuchos::EReductionType reduct_type,
  ScalarT& a)
{
  typedef typename ScalarT::value_type ValueT;
  const int sz = a.size();
  // Pack into a vector of values.
  std::vector<ValueT> pack;
  pack.push_back(a.val());
  for (int j = 0; j < sz; ++j)
    pack.push_back(a.fastAccessDx(j));
  // reduceAll the package.
  switch (reduct_type) {
  case Teuchos::REDUCE_SUM: {
    std::vector<ValueT> send(pack);
    Teuchos::reduceAll<int, ValueT>(
      comm, reduct_type, pack.size(), &send[0], &pack[0]);
  } break;
  default: TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "not impl'ed");
  }
  // Unpack.
  int slot = 0;
  a.val() = pack[slot++];
  for (int j = 0; j < sz; ++j)
    a.fastAccessDx(j) = pack[slot++];
}

template<>
void reduceAll<RealType> (
  const Teuchos_Comm& comm, const Teuchos::EReductionType reduct_type,
  RealType& a)
{
  RealType send = a;
  Teuchos::reduceAll<int, RealType>(
    comm, reduct_type, send, Teuchos::Ptr<RealType>(&a));
}

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
#define eti(T)                                                  \
  template void reduceAll<T> (                                  \
    const Teuchos_Comm&, const Teuchos::EReductionType, T&);
apply_to_all_ad_types(eti)
#undef eti
#undef apply_to_all_ad_types

} // namespace PHAL
