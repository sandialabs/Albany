//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Moertel_InterfaceT.hpp"
#include "Moertel_NodeT.hpp"
#include "Moertel_OverlapT.hpp"
#include "Moertel_PnodeT.hpp"
#include "Moertel_ProjectorT.hpp"
#include "Moertel_SegmentT.hpp"
#include "Moertel_Segment_bilineartri.H"
#include "Moertel_UtilsT.hpp"

/*----------------------------------------------------------------------*
 |  copy a polygon of points (private)                       mwgee 10/05|
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT_1A(class IFace)
bool MoertelT::MOERTEL_TEMPLATE_CLASS_1A(OverlapT, IFace)::CopyPointPolygon(
    std::map<int, Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(PointT)>>& from,
    std::map<int, Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(PointT)>>& to)
{
  std::map<int, Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(PointT)>>::
      iterator pcurr;
  for (pcurr = from.begin(); pcurr != from.end(); ++pcurr)
    if (pcurr->second != Teuchos::null) {
      Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(PointT)> tmp =
          Teuchos::rcp(new MoertelT::MOERTEL_TEMPLATE_CLASS(PointT)(
              pcurr->second->Id(),
              pcurr->second->Xi(),
              pcurr->second->OutLevel()));
      to.insert(
          std::
              pair<int, Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(PointT)>>(
                  tmp->Id(), tmp));
    }
  return true;
}

/*----------------------------------------------------------------------*
 |  find intersection (private)                              mwgee 10/05|
 *----------------------------------------------------------------------*/

MOERTEL_TEMPLATE_STATEMENT_1A(class IFace)
bool MoertelT::MOERTEL_TEMPLATE_CLASS_1A(OverlapT, IFace)::Clip_Intersect(
    const double* N,
    const double* PE,
    const double* P0,
    const double* P1,
    double*       xi)
{
  /*
          This function intersects two lines, the line orthogonal to N passing
     through point PE, and the line P1P2  (going through pts P1 and P2). If
     there is an intersection, xi[0] returns the \xi coordinate of the
     intersection and xi[1] the \eta coordinate. Note that this function assumes
     that it is relative to the parametric coordinates of a segment.
  */

  double P1P0[2];
  P1P0[0] = P1[0] - P0[0];
  P1P0[1] = P1[1] - P0[1];

  // determine the denominator - N \cdot P1P0

  double denom = -(N[0] * P1P0[0] + N[1] * P1P0[1]);

  // if the denom is zero, then lines are parallel, no intersection

  if (fabs(denom) <= std::numeric_limits<double>::min()) return false;

  double alpha = (N[0] * (P0[0] - PE[0]) + N[1] * (P0[1] - PE[1])) / denom;

  // alpha is the line parameter of the line P0 - P1
  // if alpha is outside 0 <= alpha <= 1 there is no intersection of the
  // clip edge defined by point PE and normal N with line P1P0 in the master
  // polygon cout << "OVERLAP Clip_Intersect: alpha " << alpha << endl;

  // GAH - Should this always be fatal? In practice this test fails occasionally
  // with alpha outside this range, but very close to zero or 1. If this is
  // guarded by a point in/out test at P0 and P1 it is likely best to use the
  // version below

  if (alpha < 0.0 || 1.0 < alpha) return false;

  // Compute the coord xi of the intersecting point

  xi[0] = P0[0] + alpha * P1P0[0];
  xi[1] = P0[1] + alpha * P1P0[1];

  // cout << "OVERLAP Clip_Intersect: found intersection xi " << xi[0] << "/" <<
  // xi[1] << endl;
  return true;
}

MOERTEL_TEMPLATE_STATEMENT_1A(class IFace)
bool MoertelT::MOERTEL_TEMPLATE_CLASS_1A(OverlapT, IFace)::
    Guarded_Clip_Intersect(
        const double* N,
        const double* PE,
        const double* P0,
        const double* P1,
        double*       xi)
{
  /*
          This function intersects two lines, the line orthogonal to N passing
     through point PE, and the line P1P2  (going through pts P1 and P2). If
     there is an intersection, xi[0] returns the \xi coordinate of the
     intersection and xi[1] the \eta coordinate. Note that this function assumes
     that it is relative to the parametric coordinates of a segment.

          Note that this should only be used if one knows that P0 and P1 are on
     opposite sides of a clipping plane.
  */

  double P1P0[2];
  P1P0[0] = P1[0] - P0[0];
  P1P0[1] = P1[1] - P0[1];

  // determine the denominator - N \cdot P1P0

  double denom = -(N[0] * P1P0[0] + N[1] * P1P0[1]);

  // if the denom is zero, then lines are parallel, no intersection

  if (fabs(denom) <= std::numeric_limits<double>::min()) return false;

  double alpha = (N[0] * (P0[0] - PE[0]) + N[1] * (P0[1] - PE[1])) / denom;

  // alpha is the line parameter of the line P0 - P1
  // if alpha is outside 0 <= alpha <= 1 there is no intersection of the
  // clip edge defined by point PE and normal N with line P1P0 in the master
  // polygon cout << "OVERLAP Clip_Intersect: alpha " << alpha << endl;

  // Compute the coord xi of the intersecting point

  xi[0] = P0[0] + alpha * P1P0[0];
  xi[1] = P0[1] + alpha * P1P0[1];

  // cout << "OVERLAP Clip_Intersect: found intersection xi " << xi[0] << "/" <<
  // xi[1] << endl;
  return true;
}

/*----------------------------------------------------------------------*
 |  test point (private)                                     mwgee 10/05|
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT_1A(class IFace)
bool MoertelT::MOERTEL_TEMPLATE_CLASS_1A(OverlapT, IFace)::Clip_TestPoint(
    const double* N,
    const double* PE,
    const double* P,
    double        eps)
{
  double PPE[2];
  PPE[0] = P[0] - PE[0];
  PPE[1] = P[1] - PE[1];

  /*
     dotproduct gives a measure of the angle between N with tail at PE, and P
     with tail at PE If there is a component of P in the direction of N (greater
     than eps), return false.
  */

  double dotproduct = PPE[0] * N[0] + PPE[1] * N[1];
  // cout << "OVERLAP Clip_TestPoint: dotproduct " << dotproduct << endl;

  if (dotproduct > eps) {
    return false;
  } else {
    return true;
  }
}

/*----------------------------------------------------------------------*
 |  find parameterization alpha for point on line (private)  mwgee 10/05|
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT_1A(class IFace)
double MoertelT::MOERTEL_TEMPLATE_CLASS_1A(OverlapT, IFace)::
    Clip_ParameterPointOnLine(
        const double* P0,
        const double* P1,
        const double* P)
{
  double dist1 =
      sqrt((P[0] - P0[0]) * (P[0] - P0[0]) + (P[1] - P0[1]) * (P[1] - P0[1]));
  double dist2 = sqrt(
      (P1[0] - P0[0]) * (P1[0] - P0[0]) + (P1[1] - P0[1]) * (P1[1] - P0[1]));
  if (dist2 < 1.0e-10) {
    std::stringstream oss;
    oss << "***ERR*** MOERTEL::Overlap::Clip_ParameterPointOnLine:\n"
        << "***ERR*** edge length too small, division by near zero\n"
        << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
    throw ReportError(oss);
  }
  return (dist1 / dist2);
}

/*----------------------------------------------------------------------*
 |  add segment (private)                                    mwgee 10/05|
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT_1A(class IFace)
bool MoertelT::MOERTEL_TEMPLATE_CLASS_1A(OverlapT, IFace)::AddSegment(
    int id,
    MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT) * seg)
{
  Teuchos::RCP<MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)> tmp =
      Teuchos::rcp(seg);
  s_.insert(
      std::pair<int, Teuchos::RCP<MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)>>(
          id, tmp));
  return true;
}

/*----------------------------------------------------------------------*
 |  add point (private)                                      mwgee 10/05|
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT_1A(class IFace)
bool MoertelT::MOERTEL_TEMPLATE_CLASS_1A(OverlapT, IFace)::AddPointtoPolygon(
    const int     id,
    const double* P)
{
  // check whether this point is already in there
  std::map<int, Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(PointT)>>::
      iterator curr = p_.find(id);
  // it's there
  if (curr != p_.end()) {
    curr->second->SetXi(P);

  } else {
    //	std::cout << "OVERLAP Clip_AddPointtoPolygon: added point " << id
    //         << " xi=" << P[0] << "/" << P[1] << std::endl;

    Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(PointT)> p = Teuchos::rcp(
        new MoertelT::MOERTEL_TEMPLATE_CLASS(PointT)(id, P, OutLevel()));
    p_.insert(
        std::pair<int, Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(PointT)>>(
            id, p));
  }

  return true;
}

/*----------------------------------------------------------------------*
 |  add point (private)                                      mwgee 10/05|
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT_1A(class IFace)
bool MoertelT::MOERTEL_TEMPLATE_CLASS_1A(OverlapT, IFace)::AddPointtoPolygon(
    std::map<int, Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(PointT)>>& p,
    const int                                                              id,
    const double*                                                          P)
{
  Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(PointT)> point = Teuchos::rcp(
      new MoertelT::MOERTEL_TEMPLATE_CLASS(PointT)(id, P, OutLevel()));
  p.insert(
      std::pair<int, Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(PointT)>>(
          id, point));

  return true;
}

/*----------------------------------------------------------------------*
 |  remove point (private)                                      mwgee 10/05|
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT_1A(class IFace)
bool MoertelT::MOERTEL_TEMPLATE_CLASS_1A(
    OverlapT,
    IFace)::RemovePointfromPolygon(const int id, const double* P)
{
  // check whether this point is in there
  std::map<int, Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(PointT)>>::
      iterator curr = p_.find(id);
  if (curr != p_.end()) {
    curr->second = Teuchos::null;
    p_.erase(id);
    // cout << "OVERLAP Clip_RemovePointfromPolygon: removed point " << id <<
    // endl;
    return true;
  } else {
    // cout << "OVERLAP Clip_RemovePointfromPolygon: do nothing\n";
    return false;
  }
}

/*----------------------------------------------------------------------*
 |  get point view (private)                                 mwgee 10/05|
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT_1A(class IFace)
void MoertelT::MOERTEL_TEMPLATE_CLASS_1A(OverlapT, IFace)::PointView(
    std::vector<Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(PointT)>>& points)
{
  // allocate vector of ptrs
  points.resize(SizePointPolygon());

  // get the point views
  int count = 0;
  std::map<int, Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(PointT)>>::
      iterator pcurr;
  for (pcurr = p_.begin(); pcurr != p_.end(); ++pcurr) {
    points[count] = pcurr->second;
    ++count;
  }
  if (count != SizePointPolygon()) {
    std::stringstream oss;
    oss << "***ERR*** MOERTEL::Overlap::PointView:\n"
        << "***ERR*** number of point wrong\n"
        << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
    throw ReportError(oss);
  }
  return;
}

/*----------------------------------------------------------------------*
 |  get segment view (protected)                             mwgee 11/05|
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT_1A(class IFace)
void MoertelT::MOERTEL_TEMPLATE_CLASS_1A(OverlapT, IFace)::SegmentView(
    std::vector<Teuchos::RCP<MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)>>& segs)
{
  // allocate vector of ptrs
  segs.resize(Nseg());

  // get the segment views
  int count = 0;
  std::map<int, Teuchos::RCP<MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)>>::
      iterator curr;
  for (curr = s_.begin(); curr != s_.end(); ++curr) {
    segs[count] = curr->second;
    ++count;
  }
  if (count != Nseg()) {
    std::stringstream oss;
    oss << "***ERR*** MOERTEL::Overlap::SegmentView:\n"
        << "***ERR*** number of segments wrong\n"
        << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
    throw ReportError(oss);
  }
  return;
}

/*----------------------------------------------------------------------*
 |  get point view (private)                                 mwgee 10/05|
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT_1A(class IFace)
void MoertelT::MOERTEL_TEMPLATE_CLASS_1A(OverlapT, IFace)::PointView(
    std::map<int, Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(PointT)>>& p,
    std::vector<Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(PointT)>>& points)
{
  // allocate vector of ptrs
  int np = p.size();
  points.resize(np);

  // get the point views
  int count = 0;
  std::map<int, Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(PointT)>>::
      iterator pcurr;
  for (pcurr = p.begin(); pcurr != p.end(); ++pcurr) {
    points[count] = pcurr->second;
    ++count;
  }
  if (count != np) {
    std::stringstream oss;
    oss << "***ERR*** MOERTEL::Overlap::PointView:\n"
        << "***ERR*** number of point wrong\n"
        << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
    throw ReportError(oss);
  }
  return;
}

/*----------------------------------------------------------------------*
 |  get point view (private)                                 mwgee 11/05|
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT_1A(class IFace)
void MoertelT::MOERTEL_TEMPLATE_CLASS_1A(OverlapT, IFace)::PointView(
    std::vector<MoertelT::MOERTEL_TEMPLATE_CLASS(PointT) *>& p,
    const int*                                               nodeids,
    const int                                                np)
{
  p.resize(np);

  for (int i = 0; i < np; ++i) {
    std::map<int, Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(PointT)>>::
        iterator pcurr = p_.find(nodeids[i]);
    if (pcurr == p_.end()) {
      std::stringstream oss;
      oss << "***ERR*** MOERTEL::Overlap::PointView:\n"
          << "***ERR*** cannot find point " << nodeids[i] << "\n"
          << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
      throw ReportError(oss);
    }
    p[i] = pcurr->second.get();
  }
  return;
}

/*----------------------------------------------------------------------*
 |  perform a quick search (private)                         mwgee 10/05|
 *----------------------------------------------------------------------*/

/*
        Note that this algorithm is based on selecting a minimal bounding sphere
   for the master seg, of diameter mdiam, and slave seg diameter sdiam. It then
   calculates the minimum distance between any two nodes between the slave and
   master segments, minlength. It returns false if minlength is greater than 1.5
   times (mdiam + sdiam); false if the segments cannot overlap.

        If there is a gap that one is integrating across, this algorithm may not
   be so good as one really wants to know if the master and slave segs can "see"
   each other.

        GAH
*/

MOERTEL_TEMPLATE_STATEMENT_1A(class IFace)
bool MoertelT::MOERTEL_TEMPLATE_CLASS_1A(OverlapT, IFace)::QuickOverlapTest()
{
  MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)** snode = sseg_.Nodes();
  MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)** mnode = mseg_.Nodes();
  const int nsnode                                = sseg_.Nnode();
  const int nmnode                                = mseg_.Nnode();
  // compute closest distance between 2 nodes of sseg and mseg
  double dmin[3];
  double minlength;
  for (int i = 0; i < 3; ++i)
    dmin[i] = snode[0]->XCoords()[i] - mnode[0]->XCoords()[i];
  minlength = MoertelT::length(dmin, 3);
  for (int slave = 1; slave < nsnode; ++slave)
    for (int master = 0; master < nmnode; ++master) {
      for (int i = 0; i < 3; ++i)
        dmin[i] = snode[slave]->XCoords()[i] - mnode[master]->XCoords()[i];
      double length = MoertelT::length(dmin, 3);
      if (length < minlength) minlength = length;
    }

  // compute max element edge
  double mdiam;
  double sdiam;
  if (nmnode == 4) {
    double d1[3];
    d1[0] = mnode[0]->XCoords()[0] - mnode[2]->XCoords()[0];
    d1[1] = mnode[0]->XCoords()[1] - mnode[2]->XCoords()[1];
    d1[2] = mnode[0]->XCoords()[2] - mnode[2]->XCoords()[2];
    double d2[3];
    d2[0]          = mnode[1]->XCoords()[0] - mnode[3]->XCoords()[0];
    d2[1]          = mnode[1]->XCoords()[1] - mnode[3]->XCoords()[1];
    d2[2]          = mnode[1]->XCoords()[2] - mnode[3]->XCoords()[2];
    double length1 = MoertelT::length(d1, 3);
    double length2 = MoertelT::length(d2, 3);
    if (length1 >= length2)
      mdiam = length1;
    else
      mdiam = length2;
  } else {
    double d1[3];
    d1[0] = mnode[0]->XCoords()[0] - mnode[1]->XCoords()[0];
    d1[1] = mnode[0]->XCoords()[1] - mnode[1]->XCoords()[1];
    d1[2] = mnode[0]->XCoords()[2] - mnode[1]->XCoords()[2];
    double d2[3];
    d2[0] = mnode[0]->XCoords()[0] - mnode[2]->XCoords()[0];
    d2[1] = mnode[0]->XCoords()[1] - mnode[2]->XCoords()[1];
    d2[2] = mnode[0]->XCoords()[2] - mnode[2]->XCoords()[2];
    double d3[3];
    d3[0]          = mnode[1]->XCoords()[0] - mnode[2]->XCoords()[0];
    d3[1]          = mnode[1]->XCoords()[1] - mnode[2]->XCoords()[1];
    d3[2]          = mnode[1]->XCoords()[2] - mnode[2]->XCoords()[2];
    double length1 = MoertelT::length(d1, 3);
    double length2 = MoertelT::length(d2, 3);
    double length3 = MoertelT::length(d3, 3);
    if (length1 >= length2)
      mdiam = length1;
    else
      mdiam = length2;
    if (mdiam >= length3)
      ;
    else
      mdiam = length3;
  }
  if (nsnode == 4) {
    double d1[3];
    d1[0] = snode[0]->XCoords()[0] - snode[2]->XCoords()[0];
    d1[1] = snode[0]->XCoords()[1] - snode[2]->XCoords()[1];
    d1[2] = snode[0]->XCoords()[2] - snode[2]->XCoords()[2];
    double d2[3];
    d2[0]          = snode[1]->XCoords()[0] - snode[3]->XCoords()[0];
    d2[1]          = snode[1]->XCoords()[1] - snode[3]->XCoords()[1];
    d2[2]          = snode[1]->XCoords()[2] - snode[3]->XCoords()[2];
    double length1 = MoertelT::length(d1, 3);
    double length2 = MoertelT::length(d2, 3);
    if (length1 >= length2)
      sdiam = length1;
    else
      sdiam = length2;
  } else {
    double d1[3];
    d1[0] = snode[0]->XCoords()[0] - snode[1]->XCoords()[0];
    d1[1] = snode[0]->XCoords()[1] - snode[1]->XCoords()[1];
    d1[2] = snode[0]->XCoords()[2] - snode[1]->XCoords()[2];
    double d2[3];
    d2[0] = snode[0]->XCoords()[0] - snode[2]->XCoords()[0];
    d2[1] = snode[0]->XCoords()[1] - snode[2]->XCoords()[1];
    d2[2] = snode[0]->XCoords()[2] - snode[2]->XCoords()[2];
    double d3[3];
    d3[0]          = snode[1]->XCoords()[0] - snode[2]->XCoords()[0];
    d3[1]          = snode[1]->XCoords()[1] - snode[2]->XCoords()[1];
    d3[2]          = snode[1]->XCoords()[2] - snode[2]->XCoords()[2];
    double length1 = MoertelT::length(d1, 3);
    double length2 = MoertelT::length(d2, 3);
    double length3 = MoertelT::length(d3, 3);
    if (length1 >= length2)
      sdiam = length1;
    else
      sdiam = length2;
    if (sdiam >= length3)
      ;
    else
      sdiam = length3;
  }

  // std::cerr << "minlength " << minlength << " sdiam " << sdiam << " mdiam "
  // << mdiam;

  // Max distance between mseg and sseg for contact purposes

  if (minlength > Rough_Search_Radius * (sdiam + mdiam)) {
    // std::cerr << " test NOT passed\n";
    return false;
  } else {
    // std::cerr << " test passed\n";
    return true;
  }

#if 0  // the old overlap test works only if projection works
  // we need the projection of the master element on the slave element 
  // to do this test, otherwise we assume we passed it
  if (!havemxi_) 
    return true;
  
  // find the max and min coords of master points
  double mmax[2]; 
  mmax[0] = mxi_[0][0];
  mmax[1] = mxi_[0][1];
  double mmin[2]; 
  mmin[0] = mxi_[0][0];
  mmin[1] = mxi_[0][1];
  for (int point=1; point<mseg_.Nnode(); ++point)
    for (int dim=0; dim<2; ++dim)
    {
      if (mmax[dim] < mxi_[point][dim]) 
        mmax[dim] = mxi_[point][dim];
      if (mmin[dim] > mxi_[point][dim]) 
        mmin[dim] = mxi_[point][dim];
    }


  bool isin = false;
  // check xi0 direction
  if (  (0.0 <= mmin[0] && mmin[0] <= 1.0) ||
        (0.0 <= mmax[0] && mmax[0] <= 1.0) ||
        (mmin[0] <= 0.0 && 1.0 <= mmax[0])  )
  isin = true;
  
  if (!isin) 
  {
    //cout << "OVERLAP: " << sseg_.Id() << " and " << mseg_.Id() << " do NOT overlap in QuickOverlapTest\n";
    return false;
  }

  // check xi1 direction
  if (  (0.0 <= mmin[1] && mmin[1] <= 1.0) ||
        (0.0 <= mmax[1] && mmax[1] <= 1.0) ||
        (mmin[1] <= 0.0 && 1.0 <= mmax[1])  )
  isin = true;
  
  if (!isin)
  {
    //cout << "OVERLAP: " << sseg_.Id() << " and " << mseg_.Id() << " do NOT overlap in QuickOverlapTest\n";
    return false;
  }
#endif
  return true;
}

/*----------------------------------------------------------------------*
 |  perform a quick search (protected)                        mwgee 4/06|
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT_1A(class IFace)
bool MoertelT::MOERTEL_TEMPLATE_CLASS_1A(OverlapT, IFace)::Centroid(
    double xi[],
    const std::vector<Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(PointT)>>&
              points,
    const int np)
{
  xi[0] = xi[1] = 0.0;
  double A      = 0.0;
  for (int i = 0; i < np - 1; ++i) {
    const double* xi_i   = points[i]->Xi();
    const double* xi_ip1 = points[i + 1]->Xi();
    A += xi_ip1[0] * xi_i[1] - xi_i[0] * xi_ip1[1];
    xi[0] +=
        (xi_i[0] + xi_ip1[0]) * (xi_ip1[0] * xi_i[1] - xi_i[0] * xi_ip1[1]);
    xi[1] +=
        (xi_i[1] + xi_ip1[1]) * (xi_ip1[0] * xi_i[1] - xi_i[0] * xi_ip1[1]);
  }
  const double* xi_i   = points[np - 1]->Xi();
  const double* xi_ip1 = points[0]->Xi();
  A += xi_ip1[0] * xi_i[1] - xi_i[0] * xi_ip1[1];

  // bail - if we do not have a polygon
  if (fabs(A) <= std::numeric_limits<double>::min()) return false;

  xi[0] += (xi_i[0] + xi_ip1[0]) * (xi_ip1[0] * xi_i[1] - xi_i[0] * xi_ip1[1]);
  xi[1] += (xi_i[1] + xi_ip1[1]) * (xi_ip1[0] * xi_i[1] - xi_i[0] * xi_ip1[1]);
  xi[0] /= (3.0 * A);
  xi[1] /= (3.0 * A);
  return true;
}
