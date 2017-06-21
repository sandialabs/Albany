//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_Application.hpp"
#include "PHAL_Utilities.hpp"

namespace PHAL {

template<> int getDerivativeDimensions<PHAL::AlbanyTraits::Jacobian> (
  const Albany::Application* app, const Albany::MeshSpecsStruct* ms)
{
  const Teuchos::RCP<const Teuchos::ParameterList> pl = app->getProblemPL();
  if (Teuchos::nonnull(pl)) {
    const bool extrudedColumnCoupled = pl->isParameter("Extruded Column Coupled in 2D Response") ? pl->get<bool>("Extruded Column Coupled in 2D Response") : false;
    if(extrudedColumnCoupled)
      { //all column is coupled
        int side_node_count = ms->ctd.side[2].topology->node_count;
        int node_count = ms->ctd.node_count;
        int numLevels = app->getDiscretization()->getLayeredMeshNumbering()->numLayers+1;
        return app->getNumEquations()*(node_count + side_node_count*numLevels);
      }
  }
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
 const Albany::Application* app, const int ebi, const bool explicit_scheme)
{
  const Teuchos::RCP<const Teuchos::ParameterList> pl = app->getProblemPL();
  if (Teuchos::nonnull(pl)) {
    const std::string problemName = pl->isType<std::string>("Name") ? pl->get<std::string>("Name") : "";
    if(problemName == "FELIX Coupled FO H 3D")
    { //all column is coupled
      int side_node_count = app->getEnrichedMeshSpecs()[ebi].get()->ctd.side[2].topology->node_count;
      int node_count = app->getEnrichedMeshSpecs()[ebi].get()->ctd.node_count;
      int numLevels = app->getDiscretization()->getLayeredMeshNumbering()->numLayers+1;
      return app->getNumEquations()*(node_count + side_node_count*numLevels);
    }
#ifdef ALBANY_AERAS
    if ((problemName == "Aeras Hydrostatic")  && (explicit_scheme == true))
    {
      return 1;
    }
    if (problemName == "Aeras Shallow Water No AD 3D") {
      if (explicit_scheme == false) {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Implicit time-integration " <<
                                   "not supported for Aeras Shallow Water No AD 3D problem type!\n"); 
      }
      return 1; 
    }
#endif
   }
   return getDerivativeDimensions<PHAL::AlbanyTraits::Jacobian>(
     app, app->getEnrichedMeshSpecs()[ebi].get());
}

template<> int getDerivativeDimensions<PHAL::AlbanyTraits::Tangent> (
 const Albany::Application* app, const int ebi, const bool explicit_scheme)
{
  return getDerivativeDimensions<PHAL::AlbanyTraits::Tangent>(
    app, app->getEnrichedMeshSpecs()[ebi].get());
}

template<> int getDerivativeDimensions<PHAL::AlbanyTraits::DistParamDeriv> (
 const Albany::Application* app, const int ebi, const bool explicit_scheme)
{
  return getDerivativeDimensions<PHAL::AlbanyTraits::DistParamDeriv>(
    app, app->getEnrichedMeshSpecs()[ebi].get());
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

#ifdef ALBANY_STOKHOS
template<> void myReduceAll<MPType> (
  const Teuchos_Comm& comm, const Teuchos::EReductionType reduct_type,
  std::vector<MPType>& v)
{
  std::vector<MPType> send(v);
  Teuchos::reduceAll<int, MPType>(
    comm, reduct_type, v.size(), &send[0], &v[0]);
}
#endif
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
  ScalarT b = a;
  Teuchos::reduceAll(comm, reduct_type, 1, &a, &b);
  a = b;
}

template<typename ScalarT>
void broadcast (const Teuchos_Comm& comm, const int root_rank,
                PHX::MDField<ScalarT>& a) {
  std::vector<ScalarT> v;
  copy<ScalarT>(a, v);
  Teuchos::broadcast<int, ScalarT>(comm, root_rank, v.size(), &v[0]);
  copy<ScalarT>(v, a);
}

#ifdef ALBANY_SG
# ifdef ALBANY_ENSEMBLE
#  ifdef ALBANY_FADTYPE_NOTEQUAL_TANFADTYPE
#define apply_to_all_ad_types(macro)            \
  macro(RealType)                               \
  macro(FadType)                                \
  macro(TanFadType)                             \
  macro(SGType)                                 \
  macro(SGFadType)                              \
  macro(MPType)                                 \
  macro(MPFadType)
#  else
#define apply_to_all_ad_types(macro)            \
  macro(RealType)                               \
  macro(FadType)                                \
  macro(SGType)                                 \
  macro(SGFadType)                              \
  macro(MPType)                                 \
  macro(MPFadType)
#  endif
# else //ALBANY_ENSEMBLE
#  ifdef ALBANY_FADTYPE_NOTEQUAL_TANFADTYPE
#define apply_to_all_ad_types(macro)            \
  macro(RealType)                               \
  macro(FadType)                                \
  macro(TanFadType)                             \
  macro(SGType)                                 \
  macro(SGFadType)
#  else
#define apply_to_all_ad_types(macro)            \
  macro(RealType)                               \
  macro(FadType)                                \
  macro(SGType)                                 \
  macro(SGFadType)
#  endif
# endif //ALBANY_ENSEMBLE
#else  //ALBANY_SG
# ifdef ALBANY_ENSEMBLE
#  ifdef ALBANY_FADTYPE_NOTEQUAL_TANFADTYPE
#define apply_to_all_ad_types(macro)            \
  macro(RealType)                               \
  macro(FadType)                                \
  macro(TanFadType)                             \
  macro(MPType)                                 \
  macro(MPFadType)
#  else
#define apply_to_all_ad_types(macro)            \
  macro(RealType)                               \
  macro(FadType)                                \
  macro(MPType)                                 \
  macro(MPFadType)
#  endif
# else //ALBANY_ENSEMBLE
#  ifdef ALBANY_FADTYPE_NOTEQUAL_TANFADTYPE
#define apply_to_all_ad_types(macro)            \
  macro(RealType)                               \
  macro(FadType)                                \
  macro(TanFadType)
#  else
#define apply_to_all_ad_types(macro)            \
  macro(RealType)                               \
  macro(FadType)
#  endif
# endif //ALBANY_ENSEMBLE
#endif //ALBANY_SG

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
#define eti(T)                                                          \
  template void broadcast<T> (                                          \
    const Teuchos_Comm&, const int root_rank, PHX::MDField<T>&);
apply_to_all_ad_types(eti)
#undef eti
#undef apply_to_all_ad_types

} // namespace PHAL
