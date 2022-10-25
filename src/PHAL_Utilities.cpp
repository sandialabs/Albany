//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <stdexcept>

#include "Albany_Application.hpp"
#include "Albany_StateInfoStruct.hpp"
#include "PHAL_Utilities.hpp"

namespace {

template <typename EvalT> std::string getSFadSizeName();
template <> std::string getSFadSizeName<PHAL::AlbanyTraits::Jacobian>() {return "ALBANY_SFAD_SIZE";}
template <> std::string getSFadSizeName<PHAL::AlbanyTraits::Tangent>() {return "ALBANY_TAN_SFAD_SIZE";}
template <> std::string getSFadSizeName<PHAL::AlbanyTraits::DistParamDeriv>() {return "ALBANY_TAN_SFAD_SIZE";}
template <> std::string getSFadSizeName<PHAL::AlbanyTraits::HessianVec>() {return "ALBANY_HES_VEC_SFAD_SIZE";}

template <typename EvalT>
void checkDerivativeDimensions(const int dDims)
{
  // Check derivative dimensions against fad size
  using FadT = typename EvalT::EvaluationType::ScalarT;
  if (FadT::StorageType::is_statically_sized) {
    const int static_size = FadT::StorageType::static_size;
    if (static_size != dDims) {
      const auto sfadSizeName = getSFadSizeName<EvalT>();
      std::stringstream ss1, ss2;
      ss1 << "Derivative dimension for " << PHX::print<EvalT>() << " is " << dDims << " but "
          << sfadSizeName << " is " << static_size << "!\n";
      ss2 << " - Rebuild with " << sfadSizeName << "=" << dDims << "\n";
      if (static_size > dDims)
        *Teuchos::VerboseObjectBase::getDefaultOStream()
            << "WARNING: " << ss1.str()
            << "Continuing with this size may cause issues...\n" << ss2.str();
      else
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, ss1.str() + ss2.str());
    }
  }
}

} // namespace

namespace PHAL {

template<> int getDerivativeDimensions<PHAL::AlbanyTraits::Jacobian> (
  const Albany::Application* app, const Albany::MeshSpecsStruct* ms, bool responseEvaluation)
{
  int dDims = app->getNumEquations() * ms->ctd.node_count;
  const Teuchos::RCP<const Teuchos::ParameterList> pl = app->getProblemPL();
  if (Teuchos::nonnull(pl)) {
    const bool landIceCoupledFOH3D = !responseEvaluation && pl->get<std::string>("Name") == "LandIce Coupled FO H 3D";
    const bool extrudedColumnCoupled = (responseEvaluation && pl->isParameter("Extruded Column Coupled in 2D Response")) ?
        pl->get<bool>("Extruded Column Coupled in 2D Response") : false;
    if(landIceCoupledFOH3D || extrudedColumnCoupled)
      { //all column is coupled
        int side_node_count = ms->ctd.side[3].topology->node_count;
        int node_count = ms->ctd.node_count;
        int numLevels = app->getDiscretization()->getLayeredMeshNumbering()->numLayers+1;
        dDims = app->getNumEquations()*(node_count + side_node_count*numLevels);
      }
  }
  checkDerivativeDimensions<PHAL::AlbanyTraits::Jacobian>(dDims);
  return dDims;
}

template<> int getDerivativeDimensions<PHAL::AlbanyTraits::Tangent> (
  const Albany::Application* app, const Albany::MeshSpecsStruct* /* ms */, bool /* responseEvaluation */)
{
  const int dDims = app->getTangentDerivDimension();
  checkDerivativeDimensions<PHAL::AlbanyTraits::Tangent>(dDims);
  return dDims;
}

template<> int getDerivativeDimensions<PHAL::AlbanyTraits::DistParamDeriv> (
  const Albany::Application* /* app */, const Albany::MeshSpecsStruct* ms, bool /* responseEvaluation */)
{
  //Mauro: currently distributed derivatives work only with scalar parameters, to be updated.
  const int dDims = ms->ctd.node_count;
  checkDerivativeDimensions<PHAL::AlbanyTraits::DistParamDeriv>(dDims);
  return dDims;
}

template<> int getDerivativeDimensions<PHAL::AlbanyTraits::HessianVec> (
  const Albany::Application* app, const Albany::MeshSpecsStruct* ms, bool responseEvaluation)
{
  const int derivativeDimension_x = getDerivativeDimensions<PHAL::AlbanyTraits::Jacobian>(app, ms, responseEvaluation);
  const int derivativeDimension_p = getDerivativeDimensions<PHAL::AlbanyTraits::DistParamDeriv>(app, ms, responseEvaluation);
  const int derivativeDimension_max = derivativeDimension_x > derivativeDimension_p ? derivativeDimension_x : derivativeDimension_p;
  checkDerivativeDimensions<PHAL::AlbanyTraits::HessianVec>(derivativeDimension_max);
  return derivativeDimension_max;
}

template <typename EvalT>
int getDerivativeDimensions(const Albany::Application* app, const int ebi)
{
  return getDerivativeDimensions<EvalT>(app, app->getEnrichedMeshSpecs()[ebi].get());
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
  for (size_t i = 0; i < v.size(); ++i) {
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
  for (size_t i = 0; i < v.size(); ++i) {
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

template int getDerivativeDimensions<PHAL::AlbanyTraits::Jacobian>(
    const Albany::Application*, const int);
template int getDerivativeDimensions<PHAL::AlbanyTraits::Tangent>(
    const Albany::Application*, const int);
template int getDerivativeDimensions<PHAL::AlbanyTraits::DistParamDeriv>(
    const Albany::Application*, const int);
template int getDerivativeDimensions<PHAL::AlbanyTraits::HessianVec>(
    const Albany::Application*, const int);

#  ifdef ALBANY_FADTYPE_NOTEQUAL_TANFADTYPE
#define apply_to_all_ad_types(macro)            \
  macro(RealType)                               \
  macro(FadType)                                \
  macro(TanFadType)                             \
  macro(HessianVecFad)
#  else
#define apply_to_all_ad_types(macro)            \
  macro(RealType)                               \
  macro(FadType)                                \
  macro(HessianVecFad)
#  endif

#define eti(T)                                                              \
  template void reduceAll<T> (                                              \
    const Teuchos_Comm&, const Teuchos::EReductionType, PHX::MDField<T>&);  \
  template void reduceAll<T> (                                              \
    const Teuchos_Comm&, const Teuchos::EReductionType, T&);                \
  template void broadcast<T> (                                              \
    const Teuchos_Comm&, const int, PHX::MDField<T>&);
apply_to_all_ad_types(eti)
#undef eti
#undef apply_to_all_ad_types

} // namespace PHAL
