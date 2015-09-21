//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/12/14: no Epetra!

#include "PHAL_Dimension.hpp"

const char * Dim::name() const
{ static const char n[] = "Dim" ; return n ; }
const Dim & Dim::tag()
{ static const Dim myself ; return myself ; }

const char * VecDim::name() const
{ static const char n[] = "VecDim" ; return n ; }
const VecDim & VecDim::tag()
{ static const VecDim myself ; return myself ; }

const char * QuadPoint::name() const
{ static const char n[] = "QuadPoint" ; return n ; }
const QuadPoint & QuadPoint::tag()
{ static const QuadPoint myself ; return myself ; }

const char * Node::name() const
{ static const char n[] = "Node" ; return n ; }
const Node & Node::tag()
{ static const Node myself ; return myself ; }

const char * Vertex::name() const
{ static const char n[] = "Vertex" ; return n ; }
const Vertex & Vertex::tag()
{ static const Vertex myself ; return myself ; }

const char * Point::name() const
{ static const char n[] = "Point" ; return n ; }
const Point & Point::tag()
{ static const Point myself ; return myself ; }

const char * Cell::name() const
{ static const char n[] = "Cell" ; return n ; }
const Cell & Cell::tag()
{ static const Cell myself ; return myself ; }

const char * Side::name() const
{ static const char n[] = "Side" ; return n; }
const Side & Side::tag()
{static const Side myself ; return myself; }

const char * Dummy::name() const
{ static const char n[] = "Dummy" ; return n ; }
const Dummy & Dummy::tag()
{ static const Dummy myself ; return myself ; }
