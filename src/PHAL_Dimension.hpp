//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PHAL_DIMENSION_HPP
#define PHAL_DIMENSION_HPP

#include "Shards_Array.hpp"
#include "Phalanx_ExtentTraits.hpp"

struct Dim : public shards::ArrayDimTag {
  Dim(){};
  const char * name() const ;
  static const Dim& tag();
};

struct VecDim : public shards::ArrayDimTag {
  VecDim(){};
  const char * name() const ;
  static const VecDim& tag();
};

struct LayerDim : public shards::ArrayDimTag {
  LayerDim(){};
  const char * name() const ;
  static const LayerDim& tag();
};

struct QuadPoint : public shards::ArrayDimTag {
  QuadPoint(){};
  const char * name() const ;
  static const QuadPoint& tag();
};

struct Node : public shards::ArrayDimTag {
  Node(){};
  const char * name() const ;
  static const Node& tag();
};

struct Vertex : public shards::ArrayDimTag {
  Vertex(){};
  const char * name() const ;
  static const Vertex& tag();
};

struct Point : public shards::ArrayDimTag {
  Point(){};
  const char * name() const ;
  static const Point& tag();
};

struct Cell : public shards::ArrayDimTag {
  Cell(){};
  const char * name() const ;
  static const Cell& tag();
};

struct Side : public shards::ArrayDimTag {
  Side(){};
  const char * name() const;
  static const Side& tag();
};

struct Dummy : public shards::ArrayDimTag {
  Dummy(){};
  const char * name() const ;
  static const Dummy& tag();
};

namespace PHX {
  template<> std::string print<Dim>();
  template<> std::string print<VecDim>();
  template<> std::string print<LayerDim>();
  template<> std::string print<QuadPoint>();
  template<> std::string print<Node>();
  template<> std::string print<Vertex>();
  template<> std::string print<Point>();
  template<> std::string print<Cell>();
  template<> std::string print<Side>();
  template<> std::string print<Dummy>();

  template<> struct is_extent<Dim> : std::true_type {};
  template<> struct is_extent<LayerDim> : std::true_type {};
  template<> struct is_extent<VecDim> : std::true_type {};
  template<> struct is_extent<QuadPoint> : std::true_type {};
  template<> struct is_extent<Node> : std::true_type {};
  template<> struct is_extent<Vertex> : std::true_type {};
  template<> struct is_extent<Point> : std::true_type {};
  template<> struct is_extent<Cell> : std::true_type {};
  template<> struct is_extent<Side> : std::true_type {};
  template<> struct is_extent<Dummy> : std::true_type {};
}

#endif
