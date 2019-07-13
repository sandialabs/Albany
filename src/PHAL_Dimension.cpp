//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

//IK, 9/12/14: no Epetra!

#include "PHAL_Dimension.hpp"

const char * DIM::name() const
{ static const char n[] = "DIM" ; return n ; }
const DIM & DIM::tag()
{ static const DIM myself ; return myself ; }

const char * QP::name() const
{ static const char n[] = "QP" ; return n ; }
const QP & QP::tag()
{ static const QP myself ; return myself ; }

const char * NODE::name() const
{ static const char n[] = "NODE" ; return n ; }
const NODE & NODE::tag()
{ static const NODE myself ; return myself ; }

const char * CELL::name() const
{ static const char n[] = "CELL" ; return n ; }
const CELL & CELL::tag()
{ static const CELL myself ; return myself ; }
