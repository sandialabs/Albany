//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef QCAD_STRINGFORMULAEVALUATOR_HPP
#define QCAD_STRINGFORMULAEVALUATOR_HPP

//Prototypes

//Evaluate a string expression which may contain x,y,z symbols.  Throws EvaluateException on error.
template<typename coordType>
coordType Evaluate(std::string strExpression, coordType x, coordType y, coordType z);

class EvaluateException;


#endif
