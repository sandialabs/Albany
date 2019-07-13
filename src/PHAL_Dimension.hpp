//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/12/14: no Epetra!

#ifndef PHAL_DIMENSION_HPP
#define PHAL_DIMENSION_HPP

#include "Shards_Array.hpp"
#include "Phalanx_ExtentTraits.hpp"

struct Dim {};
struct VecDim {};
struct LayerDim {};
struct QuadPoint {};
struct Node {};
struct Vertex {};
struct Point {};
struct Cell {};
struct Side {};
struct Dummy {};

namespace PHX {
  template<> struct is_extent<Dim> : std::true_type {};
  template<> struct is_extent<VecDim> : std::true_type {};
  template<> struct is_extent<LayerDim> : std::true_type {};
  template<> struct is_extent<QuadPoint> : std::true_type {};
  template<> struct is_extent<Node> : std::true_type {};
  template<> struct is_extent<Vertex> : std::true_type {};
  template<> struct is_extent<Point> : std::true_type {};
  template<> struct is_extent<Cell> : std::true_type {};
  template<> struct is_extent<Side> : std::true_type {};
  template<> struct is_extent<Dummy> : std::true_type {};
}

struct DIM : public shards::ArrayDimTag {
  DIM(){};
  const char * name() const ;
  static const DIM& tag();
};

struct QP : public shards::ArrayDimTag {
  QP(){};
  const char * name() const ;
  static const QP& tag();
};

struct NODE : public shards::ArrayDimTag {
  NODE(){};
  const char * name() const ;
  static const NODE& tag();
};

struct CELL : public shards::ArrayDimTag {
  CELL(){};
  const char * name() const ;
  static const CELL& tag();
};

#endif
