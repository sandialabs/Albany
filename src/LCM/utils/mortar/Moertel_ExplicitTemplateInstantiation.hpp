//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef Moertel_ExplicitTemplateInstantiation_hpp
#define Moertel_ExplicitTemplateInstantiation_hpp

#include <Kokkos_DefaultNode.hpp>
#include "Moertel_config.h"

// Use typedefs for ST, LO, GO, N at least for now
#ifdef HAVE_MOERTEL_INST_DOUBLE_INT_LONGLONGINT
typedef double        ST;
typedef int           LO;
typedef long long int GO;
#elif HAVE_MOERTEL_INST_DOUBLE_INT_LONGINT
typedef double   ST;
typedef int      LO;
typedef long int GO;
#elif HAVE_MOERTEL_INST_DOUBLE_INT_INT
typedef double ST;
typedef int    LO;
typedef int    GO;
#elif HAVE_MOERTEL_INST_FLOAT_INT_INT
typedef float ST;
typedef int   LO;
typedef int   GO;
#endif

// Shortcut macros to make template arguments consistent

/*
#define MOERTEL_TEMPLATE_STATEMENT \
template <size_t DIM,\
          class ST,\
          class LO,\
          class GO,\
          class N >

#define MOERTEL_TEMPLATE_CLASS(x) x < DIM,ST,LO,GO,N >
*/

#define MOERTEL_TEMPLATE_STATEMENT \
  template <size_t DIM, class SEGT, class FUNCT>

#define MOERTEL_TEMPLATE_CLASS(x) x<DIM, SEGT, FUNCT>

#define MOERTEL_TEMPLATE_STATEMENT_1A(x) \
  template <size_t DIM, class SEGT, class FUNCT, x>

#define MOERTEL_TEMPLATE_CLASS_1A(x, y) x<DIM, SEGT, FUNCT, y>

#define SEGMENT_TEMPLATE_STATEMENT \
  template <size_t DIM, class SEGT, class FUNCT>

#define SEGMENT_TEMPLATE_CLASS(x) x<DIM, SEGT, FUNCT>

// typedef DefaultNodeType KokkosNode;
typedef Kokkos::Compat::
    KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>
        KokkosNode;
typedef Kokkos::Compat::
    KokkosDeviceWrapperNode<Kokkos::Serial, Kokkos::HostSpace>
        N;
// typedef KokkosClassic::DefaultNode::DefaultNodeType KokkosNode;

// ETI templates
#define MOERTEL_INSTANTIATE_TEMPLATE_CLASS_ON_NAME_ORD(name, ordinal) \
  template class name<ordinal>;

#define MOERTEL_INSTANTIATE_TEMPLATE_CLASS_ON_NAME_LO_ST( \
    name, LocalOrdinal, ScalarType)                       \
  template class name<LocalOrdinal, ScalarType>;

#define MOERTEL_INSTANTIATE_TEMPLATE_CLASS_ON_NAME_LO_GO_ST( \
    name, LocalOrdinal, GlobalOrdinal, ScalarType)           \
  template class name<LocalOrdinal, GlobalOrdinal, ScalarType>;

#define MOERTEL_INSTANTIATE_TEMPLATE_CLASS_ON_NAME_ST_LO_GO_N( \
    name, ScalarType, LocalOrdinal, GlobalOrdinal, NodeType)   \
  template class name<ScalarType, LocalOrdinal, GlobalOrdinal, NodeType>;

#define MOERTEL_INSTANTIATE_NESTED_TEMPLATE_CLASS_ST_LO_GO_N(       \
    name, name2, ScalarType, LocalOrdinal, GlobalOrdinal, NodeType) \
  template class name<name2<ScalarType, LocalOrdinal, GlobalOrdinal, NodeType>>;

#ifdef HAVE_MOERTEL_INST_DOUBLE_INT_INT
#define MOERTEL_INSTANTIATE_TEMPLATE_CLASS_DII(name)     \
  MOERTEL_INSTANTIATE_TEMPLATE_CLASS_ON_NAME_ST_LO_GO_N( \
      name, double, int, int, KokkosNode)
#else
#define MOERTEL_INSTANTIATE_TEMPLATE_CLASS_DII(name)
#endif

#ifdef HAVE_MOERTEL_INST_DOUBLE_INT_LONGLONGINT
#define MOERTEL_INSTANTIATE_TEMPLATE_CLASS_DILLI(name)   \
  MOERTEL_INSTANTIATE_TEMPLATE_CLASS_ON_NAME_ST_LO_GO_N( \
      name, double, int, long long, KokkosNode)
#else
#define MOERTEL_INSTANTIATE_TEMPLATE_CLASS_DILLI(name)
#endif

#define MOERTEL_INSTANTIATE_TEMPLATE_CLASS(name) \
  MOERTEL_INSTANTIATE_TEMPLATE_CLASS_DII(name)   \
  MOERTEL_INSTANTIATE_TEMPLATE_CLASS_DILLI(name)
#endif
