//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "PHAL_Dimension.hpp"

namespace PHX {
  template<> std::string print<Dim>(){return "Dim";}
  template<> std::string print<VecDim>(){return "VecDim";}
  template<> std::string print<LayerDim>(){return "LayerDim";}
  template<> std::string print<QuadPoint>(){return "QuadPoint";}
  template<> std::string print<Node>(){return "Node";}
  template<> std::string print<Vertex>(){return "Vertex";}
  template<> std::string print<Point>(){return "Point";}
  template<> std::string print<Cell>(){return "Cell";}
  template<> std::string print<Side>(){return "Side";}
  template<> std::string print<Dummy>(){return "";}
}

const char * Dim::name() const
{
  static auto s = PHX::print<Dim>();
  return s.c_str() ;
}
const Dim & Dim::tag()
{ static const Dim myself ; return myself ; }

const char * VecDim::name() const
{
  static auto s = PHX::print<VecDim>();
  return s.c_str() ;
}
const VecDim & VecDim::tag()
{ static const VecDim myself ; return myself ; }

const char * LayerDim::name() const
{
  static auto s = PHX::print<LayerDim>();
  return s.c_str() ;
}
const LayerDim & LayerDim::tag()
{ static const LayerDim myself ; return myself ; }

const char * QuadPoint::name() const
{
  static auto s = PHX::print<QuadPoint>();
  return s.c_str() ;
}

const QuadPoint & QuadPoint::tag()
{ static const QuadPoint myself ; return myself ; }

const char * Node::name() const
{
  static auto s = PHX::print<Node>();
  return s.c_str() ;
}
const Node & Node::tag()
{ static const Node myself ; return myself ; }

const char * Vertex::name() const
{
  static auto s = PHX::print<Vertex>();
  return s.c_str() ;
}
const Vertex & Vertex::tag()
{ static const Vertex myself ; return myself ; }

const char * Point::name() const
{
  static auto s = PHX::print<Point>();
  return s.c_str() ;
}
const Point & Point::tag()
{ static const Point myself ; return myself ; }

const char * Cell::name() const
{
  static auto s = PHX::print<Cell>();
  return s.c_str() ;
}
const Cell & Cell::tag()
{ static const Cell myself ; return myself ; }

const char * Side::name() const
{
  static auto s = PHX::print<Side>();
  return s.c_str() ;
}
const Side & Side::tag()
{static const Side myself ; return myself; }

const char * Dummy::name() const
{
  static auto s = PHX::print<Dummy>();
  return s.c_str() ;
}
const Dummy & Dummy::tag()
{ static const Dummy myself ; return myself ; }

