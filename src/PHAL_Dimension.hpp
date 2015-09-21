//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/12/14: no Epetra!

#ifndef PHAL_DIMENSION_HPP
#define PHAL_DIMENSION_HPP

#include "Shards_Array.hpp"

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
#endif
