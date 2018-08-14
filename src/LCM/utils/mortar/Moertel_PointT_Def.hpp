//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Moertel_OverlapT.hpp"
#include "Moertel_PointT.hpp"
#include "Moertel_UtilsT.hpp"

/*----------------------------------------------------------------------*
 |  ctor (public)                                            mwgee 10/05|
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT
MoertelT::MOERTEL_TEMPLATE_CLASS(
    PointT)::PointT(const int id, const double* xi, int out)
    : id_(id), outputlevel_(out)
{
  xi_[0] = xi[0];
  xi_[1] = xi[1];
  node_  = Teuchos::null;
  vals_[0].clear();
  vals_[1].clear();
  vals_[2].clear();
}

/*----------------------------------------------------------------------*
 |  dtor (public)                                            mwgee 10/05|
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT
MoertelT::MOERTEL_TEMPLATE_CLASS(PointT)::~PointT()
{
  vals_[0].clear();
  vals_[1].clear();
  vals_[2].clear();
}
/*----------------------------------------------------------------------*
 |  << operator                                              mwgee 10/05|
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT
std::ostream&
operator<<(
    std::ostream& os,
    const MoertelT::MOERTEL_TEMPLATE_CLASS(PointT) & point)
{
  point.Print();
  return (os);
}
/*----------------------------------------------------------------------*
 |  print (public)                                           mwgee 10/05|
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT
void MoertelT::MOERTEL_TEMPLATE_CLASS(PointT)::Print() const
{
  std::cout << "Point " << id_ << " xi[0]/[1] = " << xi_[0] << " / " << xi_[1]
            << std::endl;
  if (node_ != Teuchos::null) std::cout << *node_;
  return;
}

/*----------------------------------------------------------------------*
 |  store shape function values (public)                     mwgee 10/05|
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT
void MoertelT::MOERTEL_TEMPLATE_CLASS(
    PointT)::StoreFunctionValues(int place, double* val, int valdim)
{
  vals_[place].resize(valdim);
  for (int i = 0; i < valdim; ++i) vals_[place][i] = val[i];
  return;
}
