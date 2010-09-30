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


#include <cmath>

#include "Albany_AnalyticFunction.hpp"
#include "Teuchos_TestForException.hpp"


enum {CONSTANT=0,
      GAUSS_SIN_1D,
      GAUSS_COS_1D,
      GAUSS_1D, 
      GAUSS_2D};
namespace {
const std::vector<std::string> &names()
{
  static std::vector<std::string> list;
  static bool init=true;
  if (init) {
    init = false;
    list.push_back("Constant"); 
    list.push_back("1D Gauss-Sin"); 
    list.push_back("1D Gauss-Cos");
    list.push_back("1D Gauss");
    list.push_back("2D Gauss");
  }
  return list;
}
std::string stringify_names()
{
  const std::vector<std::string> list = names();
  std::string n;
  for (std::size_t i=0; i<list.size(); ++i) {
    n += "'"+list[i]+"', " ;
  }
  return n;
}
int name_to_id(const std::string name)
{
  const std::vector<std::string> &list = names();
  int id = find(list.begin(), list.end(), name) - list.begin();
  if (id == list.size()) id = -1;
  return id;
}
}
bool Albany::ValidIdentifier(const std::string name)
{
  const bool valid = (-1 != name_to_id(name));
  return valid;
}
void Albany::AnalyticFunction(Epetra_Vector& u, 
                       const unsigned int l,
                       const unsigned int h,
                       const double t, 
                       const double alpha, 
                       const double beta, 
                       const std::string name)
{
   const double pi=3.141592653589793;
   const int id = name_to_id(name);
   switch (id) {
   case CONSTANT : {
     for (int i=0; i< u.MyLength(); i++)  u[i] = beta; 
   } 
   break;
   case GAUSS_SIN_1D : {
     const double dx = 1.0 / (u.GlobalLength()-1);
     for (int i=0; i< u.MyLength(); i++) {
       const double x = dx * u.Map().GID(i);
       u[i] = sin(pi * x) * exp(-pi*pi*t) + 0.5*alpha*x*(1.0-x);
     }
   } 
   break;
   case GAUSS_COS_1D: {
     const double dx = 1.0 / (u.GlobalLength()-1);
     for (int i=0; i< u.MyLength(); i++) {
       const double x = dx * u.Map().GID(i);
       u[i] = 1 + cos(2*pi*x) * exp(-4*pi*pi*t) + 0.5*alpha*x*(1.0-x);
     }
   } 
   break;
   case GAUSS_1D :{
     const double wave_speed = beta ? beta : 1;
     const double dx = 2.0 / (u.GlobalLength()-2);
     for (int i=0; i< u.MyLength(); i+=2) {
       const double y = dx*u.Map().GID(i)/2;
       const double x = y + wave_speed*t;
       const double z = x - std::floor(x) - 0.5;
       u[i+0] = exp(-alpha*alpha*z*z);
       u[i+1] = exp(-alpha*alpha*z*z)/wave_speed;
     }
   } 
   break;
   case GAUSS_2D:{
     const double dx = 1.0/(l-1);
     const double dy = 1.0/(h-1);
     for (int i=0; i< u.MyLength(); i+=3) {
       const unsigned int gid = u.Map().GID(i)/3;
       const unsigned int nx = gid%l;
       const unsigned int ny = gid/l;
       const double x = nx*dx;
       const double y = ny*dy;
       const double a = (x+t)   - std::floor(x+t)   - 0.5;
       const double b = (y+t)   - std::floor(y+t)   - 0.5;
       const double s = std::sqrt(2.0);
       const double c = (x+y+t)/s - std::floor(x+y+t)/s - 0.5;
       const double p = exp(-20.0*20.0*a*a);
       const double q = exp(-20.0*20.0*b*b);
       const double r = exp(-20.0*20.0*c*c);
       u[i]   = p + q + r;
       u[i+1] = p + r;
       u[i+2] = q + r;
     }
   } 
   break;
   default : {
     TEST_FOR_EXCEPTION(1 != 0, std::logic_error, //Teuchos::Exceptions::InvalidParameter,
                        "Error! Analytic Function name should be one of: "<<std::endl
                        << stringify_names()<<std::endl
                        << "Found: `"<<name<<"'" 
                        << std::endl);
   }
   }
}
