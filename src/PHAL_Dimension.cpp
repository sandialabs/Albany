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

const char * Face::name() const
{ static const char n[] = "Face" ; return n; }
const Face & Face::tag()
{static const Face myself ; return myself; }

const char * Dummy::name() const 
{ static const char n[] = "Dummy" ; return n ; }
const Dummy & Dummy::tag() 
{ static const Dummy myself ; return myself ; }
