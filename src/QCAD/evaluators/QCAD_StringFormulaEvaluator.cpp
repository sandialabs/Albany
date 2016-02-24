//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <stdlib.h>

#include <string>
#include <list>
#include <sstream>
#include <iostream>
#include <exception>
#include <math.h>

#include "QCAD_StringFormulaEvaluator.hpp"

class EvaluateException: public std::exception
{
public:
  EvaluateException(std::string message) {
    msg = message;
  }
  ~EvaluateException() throw () {}

  virtual const char* what() const throw() {
    return msg.c_str();
  }

private:
  std::string msg;
};

template<typename coordType>
coordType toDbl(std::string s, coordType x, coordType y, coordType z) {
  if(s == "x") return x;
  if(s == "y") return y;
  if(s == "z") return z;
  coordType t = atof(s.c_str()); //Must be able to assign coordType with a double
  return t;
}


template<typename coordType>
std::string toStr(coordType d) {
  std::ostringstream convert; 
  convert << d;
  return convert.str();
}

void printList(std::list<std::string> l) {
  for(std::list<std::string>::iterator it = l.begin(); it != l.end(); ++it)
    std::cout << "|" << *it;
  std::cout << "|" << std::endl;
}


template<typename coordType>
coordType Evaluate(std::string strExpression, coordType x, coordType y, coordType z)
{
  bool bDebug = false;
  //std::transform(expr.begin(), expr.end(), expr.begin(), ::tolower); // i.e., expr.tolower()

  std::string expr;
  for(std::string::iterator it = strExpression.begin(); it != strExpression.end(); ++it) {
    if(*it == ' ') continue;
    expr += *it;
  }

  if(bDebug) { std::cout << "After manip: " << expr << std::endl; }

  std::list<std::string> stack;  // not a std::stack because we need list operations later
  std::string value = "";
       
  for (std::size_t i = 0; i < expr.length(); i++) {
    if(bDebug) { std::cout << "i = " << i << ":  val=" << value << ",  stack="; printList(stack); }
    std::string s = expr.substr(i, 1);

    // pick up any doublelogical operators first.
    if (i < expr.length() - 1) {
      std::string op = expr.substr(i, 2);
      if (op == "<=" || op == ">=" || op == "==") {
	stack.push_front(value);
	value = "";
	stack.push_front(op);
	i++;
	continue;
      }
    }

    char chr = s[0];
    if (!isdigit(chr) && !isalpha(chr) && chr != '.' && value != "") {
      stack.push_front(value);
      value = "";
    }
    
    if (s == "(") {
      std::string innerExp = "";
      i++; //Fetch Next Character
      int bracketCount = 0;
      for (; i < expr.length(); i++) {
	s = expr.substr(i, 1);
	if (s == "(") bracketCount++;
	if (s == ")") {
	  if (bracketCount == 0) break;
	  bracketCount--;
	}
	innerExp += s;
      }
      if(bracketCount != 0) 
	throw EvaluateException("Mismatched parenthesis");

      stack.push_front( toStr(Evaluate(innerExp,x,y,z)) );
    }
    else if (s == "+" ||
	     s == "-" ||
	     s == "*" ||
	     s == "/" ||
	     s == "<" ||
	     s == ">" ||
	     s == "^") {
      stack.push_front(s);
    }
    else if(isdigit(chr) || isalpha(chr) || chr == '.') {
      if(chr == '.' && value.find('.') != std::string::npos)
	throw EvaluateException("Invalid decimal.");
      value += s;

      if (i == (expr.length() - 1))
	stack.push_front(value);
    }
    else {
      throw EvaluateException("Invalid character.");
    }
  }
  if(bDebug) { std::cout << "Stack at end = "; printList(stack); }

  double result = 0;
  std::list<std::string>& list = stack; //now we use stack as a list, so just for code readability
  std::list<std::string>::reverse_iterator it, itm1, itp1, begin;
  begin = list.rbegin(); ++begin;
  for( it = begin; it != list.rend(); ++it) {
    if (*it == "^") {
      it++; itm1 = it; it--; it--; itp1 = it; it++;  //setup iterators
      *it = toStr( pow(toDbl(*itp1,x,y,z),toDbl(*itm1,x,y,z)) );
      list.erase(it.base()); itm1++; list.erase(itm1.base());  //erase() erases the element *AFTER* what it points to
    }
  }
  if(bDebug) { std::cout << "List after ^ block = "; printList(list); }

  begin = list.rbegin(); ++begin;
  for( it = begin; it != list.rend(); ++it) {
    if (*it == "/") {
      it++; itm1 = it; it--; it--; itp1 = it; it++;  //setup iterators
      *it = toStr( toDbl(*itp1,x,y,z) / toDbl(*itm1,x,y,z) );
      list.erase(it.base()); itm1++; list.erase(itm1.base());  //erase() erases the element *AFTER* what it points to
    }
  }
  if(bDebug) { std::cout << "List after / block = "; printList(list); }

  begin = list.rbegin(); ++begin;
  for( it = begin; it != list.rend(); ++it) {
    if (*it == "*") {
      it++; itm1 = it; it--; it--; itp1 = it; it++;  //setup iterators
      *it = toStr( toDbl(*itp1,x,y,z) * toDbl(*itm1,x,y,z) );
      list.erase(it.base()); itm1++; list.erase(itm1.base());  //erase() erases the element *AFTER* what it points to
    }
  }
  if(bDebug) { std::cout << "List after * block = "; printList(list); }

  begin = list.rbegin();
  for( it = begin; it != list.rend(); ++it) {
    if (*it == "+") {      
      if (it.base() == list.end()) { // unary plus
	it++; itm1 = it; it--; //setup iterators
	*it = toStr( toDbl(*itm1,x,y,z) );
	itm1++; list.erase(itm1.base());  //erase() erases the element *AFTER* what it points to
      }	
      else { // binary plus
	it++; itm1 = it; it--; it--; itp1 = it; it++;  //setup iterators
	*it = toStr( toDbl(*itp1,x,y,z) + toDbl(*itm1,x,y,z) );
	list.erase(it.base()); itm1++; list.erase(itm1.base());  //erase() erases the element *AFTER* what it points to
      }
    }
  }
  if(bDebug) { std::cout << "List after + block = "; printList(list); }

  begin = list.rbegin();
  for( it = begin; it != list.rend(); ++it) {
    if (*it == "-") {
      if (it.base() == list.end()) { // unary minus
	it++; itm1 = it; it--; //setup iterators
	*it = toStr( -toDbl(*itm1,x,y,z) );
	itm1++; list.erase(itm1.base());  //erase() erases the element *AFTER* what it points to
      }	
      else { // binary minus
	it++; itm1 = it; it--; it--; itp1 = it; it++;  //setup iterators
	*it = toStr( toDbl(*itp1,x,y,z) - toDbl(*itm1,x,y,z) );
	list.erase(it.base()); itm1++; list.erase(itm1.base());  //erase() erases the element *AFTER* what it points to
      }
    }
  }
  if(bDebug) { std::cout << "List after - block = "; printList(list); }

  //Logic operators: use list as a stack, popping from the back
  while (list.size() >= 3) {  
    coordType right = toDbl(list.back(), x,y,z); list.pop_back();
    std::string op = list.back(); list.pop_back();
    coordType left = toDbl(list.back(), x,y,z); list.pop_back();

    if (op == "<") result = (left < right) ? 1 : 0;
    else if (op == ">") result = (left > right) ? 1 : 0;
    else if (op == "<=") result = (left <= right) ? 1 : 0;
    else if (op == ">=") result = (left >= right) ? 1 : 0;
    else if (op == "==") result = (left == right) ? 1 : 0;

    list.push_back( toStr(result) );
  }
   
  return toDbl(list.back(),x,y,z);
}

//Explicit instantiations to fix linker errors -- hardcoded now; need to do this better (Andy/Eric?)
#include "PHAL_AlbanyTraits.hpp"

template double Evaluate<double>(std::string, double, double, double); // explicit instantiation.
template FadType Evaluate<FadType>(std::string, FadType, FadType, FadType); // explicit instantiation.
#ifdef ALBANY_FADTYPE_NOTEQUAL_TANFADTYPE
template TanFadType Evaluate<TanFadType>(std::string, TanFadType, TanFadType, TanFadType); // explicit instantiation.
#endif

//Uncomment for testing - then this file alone builds a command line calculator
/*int main(int argc, char* argv[]) {

  double result;
  double x=1, y=2, z=3;
  std::string input;

  while (true) {
    std::cout << "Please enter an expression to evaluate (q to quit) x=1,y=2,z=3: ";
    std::getline(std::cin, input);
    if( input == "q" ) break;
    try {
      result = Evaluate(input,x,y,z);
    }
    catch(std::exception& e) {
      std::cout << "Error: " << e.what() << std::endl;
      continue;
    }
    std::cout << ">> " << result << std::endl;
  }
  std::cout << "Goodbye!" << std::endl;

  //std::string test1("5+4");
  //double result1 = Evaluate(test1);
  //std::cout << test1 << " = " << result1 << std::endl;
  return 0;
}*/
  
