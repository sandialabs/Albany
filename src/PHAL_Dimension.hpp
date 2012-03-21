/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


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

struct Dummy : public shards::ArrayDimTag {
  Dummy(){};
  const char * name() const ;
  static const Dummy& tag();
};
#endif
