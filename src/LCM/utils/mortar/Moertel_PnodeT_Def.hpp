//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Moertel_InterfaceT.hpp"
#include "Moertel_PnodeT.hpp"

/*----------------------------------------------------------------------*
 |  ctor (public)                                            mwgee 07/05|
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT
MoertelT::MOERTEL_TEMPLATE_CLASS(ProjectedNodeT)::ProjectedNodeT(
    const MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT) & basenode,
    const double* xi,
    MoertelT::MOERTEL_TEMPLATE_CLASS(SegmentT) * pseg)
    : MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)(basenode), orthseg_(-1)
{
  pseg_ = pseg;
  if (xi) {
    xi_[0] = xi[0];
    xi_[1] = xi[1];
  } else {
    xi_[0] = 999.0;
    xi_[1] = 999.0;
  }
}

/*----------------------------------------------------------------------*
 |  ctor for orthogonal projection (public)                  mwgee 08/05|
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT
Moertel::MOERTEL_TEMPLATE_CLASS(ProjectedNodeT)::ProjectedNodeT(
    const MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT) & basenode,
    const double* xi,
    MoertelT::MOERTEL_TEMPLATE_CLASS(SegmentT) * pseg,
    int orthseg)
    : MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)(basenode), orthseg_(orthseg)
{
  pseg_ = pseg;
  if (xi) {
    xi_[0] = xi[0];
    xi_[1] = xi[1];
  } else {
    xi_[0] = 999.0;
    xi_[1] = 999.0;
  }
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                       mwgee 07/05|
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT
MoertelT::MOERTEL_TEMPLATE_CLASS(ProjectedNodeT)::ProjectedNodeT(
    MoertelT::MOERTEL_TEMPLATE_CLASS(ProjectedNodeT) & old)
    : MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)(old)
{
  pseg_    = old.pseg_;
  xi_[0]   = old.xi_[0];
  xi_[1]   = old.xi_[1];
  orthseg_ = old.orthseg_;
}

/*----------------------------------------------------------------------*
 |  dtor (public)                                            mwgee 06/05|
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT
MoertelT::MOERTEL_TEMPLATE_CLASS(ProjectedNodeT)::~ProjectedNodeT()
{
  pseg_ =
      NULL;  // this is just a 'referencing' ptr, not in charge of destroying
}

/*----------------------------------------------------------------------*
 |  print node                                               mwgee 06/05|
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT
bool NoertelT::MOERTEL_TEMPLATE_CLASS(ProjectedNodeT)::Print() const
{
  std::cout << "Projected ";
  const MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)& basenode =
      dynamic_cast<const MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)&>(*this);
  std::cout << basenode;
  if (pseg_) {
    std::cout << "is on ";
    std::cout << *pseg_;
    std::cout << "at xi[0]/[1] = " << xi_[0] << "/" << xi_[1];
  } else {
    std::cout << "on Segment !!!!!NULL!!!!! at xi[0]/[1] = " << xi_[0] << "/"
              << xi_[1];
  }
  std::cout << "orth to seg " << orthseg_ << std::endl;
  return true;
}

/*----------------------------------------------------------------------*
 |  << operator                                              mwgee 06/05|
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT
std::ostream&
operator<<(
    std::ostream& os,
    const MoertelT::MOERTEL_TEMPLATE_CLASS(ProjectedNodeT) & pnode)
{
  pnode.Print();
  return (os);
}
