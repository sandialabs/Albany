//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef MOERTEL_PROJECTORT_HPP
#define MOERTEL_PROJECTORT_HPP

#include <ctime>
#include <iomanip>
#include <iostream>

// ----------   User Defined Includes   ----------

/*!
\brief MOERTEL: namespace of the Moertel package

The Moertel package depends on \ref Epetra, \ref EpetraExt, \ref Teuchos,
\ref Amesos, \ref ML and \ref AztecOO:<br>
Use at least the following lines in the configure of Trilinos:<br>
\code
--enable-moertel
--enable-epetra
--enable-epetraext
--enable-teuchos
--enable-ml
--enable-aztecoo --enable-aztecoo-teuchos
--enable-amesos
\endcode

*/
namespace MoertelT {

// forward declarations
MOERTEL_TEMPLATE_STATEMENT
class InterfaceT;

SEGMENT_TEMPLATE_STATEMENT
class SegmentT;

MOERTEL_TEMPLATE_STATEMENT
class NodeT;

/*!
\class Projector

\brief <b> A class to perform projections of nodes onto opposing segments in 2D
and 3D </b>

This class performs all necessary projections of nodes onto opposing segment
surfaces in 2D and 3D applying 2 different kinds of projection techniques.<br>
In 2D problems, the user has a choice of projecting nodes onto opposing segment
surfaces either orthogonal to that segment surface or along a previously
constructed C0-continuous normal field of the slave side.<br> In both cases
finding the projection of a Node on a Segment in terms of the segment's local
coordinates of the projection point is a nonlinear operation. A local Newton
iteration is involved and a dense solve of a 2x2 system is necessary within the
Newton iteration.<br>

When projecting along the C0-continuous normal field, the field of normals is
defined over the slave side discretization and is an interpolation of (weighted
averaged) nodal normals of the slave side. Projections in both directions are
performed along that same normal field making it necessary to have different
methods for projecting slave to mortar and vice versa.

In 3D projection is always performed along a previously constructed outward
field of nodal normals. The projection of a point in 3D along a field onto a 2D
surface is always a nonlinear iteration and a Newton method is applied here
involving a dense 3x3 solve in each Newton step.

These projections make up for a pretty good share of the overall computational
cost of the mortar method though convergence in the Newton iterations is usually
excellent.

\author Glen Hansen (gahanse@sandia.gov)

*/
MOERTEL_TEMPLATE_STATEMENT
class ProjectorT
{
 public:
  // @{ \name Constructors and destructor

  /*!
  \brief Constructor

  Constructs an instance of this class.<br>
  Note that this is \b not a collective call as projections are performed in
  parallel by individual processes.

  \param twoD : True if problem is 2D, false if problem is 3D
  \param outlevel : Level of output information written to stdout ( 0 - 10 )
  */
  explicit ProjectorT(bool twoD, int outlevel);

  /*!
  \brief Destructor
  */
  virtual ~ProjectorT();

  //@}
  // @{ \name Public members

  /*!
  \brief Return the level of output written to stdout ( 0 - 10 )

  */
  int
  OutLevel()
  {
    return outputlevel_;
  }

  /*!
  \brief Return whether this instance was constructed for 2D or 3D projections

  */
  bool
  IsTwoDimensional()
  {
    return twoD_;
  }

  //@}
  // @{ \name 2D and 3D projection methods

  /*!
  \brief Project a Node onto a Segment along the Node 's normal

  Used to project a Node from the slave side onto a Segment on the mortar side

  This method will compute the coordinates of a projection of a Node in the
  local coordinate system of a Segment. The projection point will not
  necessarily fall inside the Segment. However, if the projection point is far
  outside the segment's boundaries, problems with the internal nonlinear
  iteration might occur and a warning is issued when convergence can not be
  achieved in a limited number of iterations.

  \param node (in): Node to project
  \param seg (in) : Segment to project on
  \param xi (out) : Local coordinates if projection of Node in Segment 's
  coordinate System \param gap (out) : Gap between node and projection along
  projection vector.
  */
  bool
  ProjectNodetoSegment_NodalNormal(
      MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT) & node,
      MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT) & seg,
      double  xi[],
      double& gap);

  /*!
  \brief Project a Node onto a Segment along the interpolated outward normal
  field of the Segment

  Used to project a Node from the mortar side onto a Segment on the slave side

  This method will compute the coordinates of a projection of a Node in the
  local coordinate system of a Segment. The projection point will not
  necessarily fall inside the Segment. However, if the projection point is far
  outside the segment's boundaries, problems with the internal nonlinear
  iteration might occur and a warning is issued when convergence can not be
  achieved in a limited number of iterations.

  \param node (in): Node to project
  \param seg (in) : Segment to project on
  \param xi (out) : Local coordinates if projection of Node in Segment 's
  coordinate System \param gap (out) : Gap between node and projection along
  projection vector.
  */
  bool
  ProjectNodetoSegment_SegmentNormal(
      MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT) & node,
      MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT) & seg,
      double  xi[],
      double& gap);

  //@}
  // @{ \name Additional 2D projection methods

  /*!
  \brief Project a Node onto a Segment orthogonal to the Segment (2D problems
  only)

  Used to project a Node from the mortar side onto a Segment on the slave side

  This method will compute the coordinates of a projection of a Node in the
  local coordinate system of a Segment. The projection point will not
  necessarily fall inside the Segment. However, if the projection point is far
  outside the segment's boundaries, problems with the internal nonlinear
  iteration might occur and a warning is issued when convergence can not be
  achieved in a limited number of iterations.

  \param node (in): Node to project
  \param seg (in) : Segment to project on
  \param xi (out) : Local coordinates if projection of Node in Segment 's
  coordinate System \param gap (out) : Gap between node and projection along
  projection vector.
  */
  bool
  ProjectNodetoSegment_SegmentOrthogonal(
      MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT) & node,
      MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT) & seg,
      double  xi[],
      double& gap);

  /*!
  \brief Project a Node onto a Segment orthogonal another Segment (2D problems
  only)

  Used to project a Node from the slave side onto a Segment on the mortar side
  orthogonal to some slave Segment

  This method will compute the coordinates of a projection of a Node in the
  local coordinate system of a Segment. The projection point will not
  necessarily fall inside the Segment. However, if the projection point is far
  outside the segment's boundaries, problems with the internal nonlinear
  iteration might occur and a warning is issued when convergence can not be
  achieved in a limited number of iterations.

  \param node (in): Node to project
  \param seg (in) : Segment to project on
  \param xi (out) : Local coordinates if projection of Node in Segment 's
  coordinate System \param gap (out) : Gap between node and projection along
  projection vector. \param sseg (in) : Segment to project orthogonal to
  */
  bool
  ProjectNodetoSegment_Orthogonal_to_Slave(
      MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT) & snode,
      MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT) & seg,
      double  xi[],
      double& gap,
      MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT) & sseg);

  //@}

 private:
  // don't want = operator
  ProjectorT
  operator=(const MoertelT::MOERTEL_TEMPLATE_CLASS(ProjectorT) & old);
  // don't want copy-ctor
  ProjectorT(MoertelT::MOERTEL_TEMPLATE_CLASS(ProjectorT) & old);

  //====2D projection methods

  // evaluate F and gradF functions for ProjectNodetoSegment_NodalNormal in 2D
  double
  evaluate_F_2D_NodalNormal(
      MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT) & node,
      MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT) & seg,
      double  eta,
      double& gap);

  double
  evaluate_gradF_2D_NodalNormal(
      MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT) & node,
      MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT) & seg,
      double eta);

  // evaluate F and gradF functions for ProjectNodetoSegment_SegmentNormal in 2D
  double
  evaluate_F_2D_SegmentNormal(
      MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT) & node,
      MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT) & seg,
      double  eta,
      double& gap);

  double
  evaluate_gradF_2D_SegmentNormal(
      MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT) & node,
      MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT) & seg,
      double eta);

  // evaluate F and gradF functions for ProjectNodetoSegment_SegmentOrthogonal
  // in 2D
  double
  evaluate_F_2D_SegmentOrthogonal(
      MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT) & node,
      MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT) & seg,
      double  eta,
      double& gap);

  double
  evaluate_gradF_2D_SegmentOrthogonal(
      MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT) & node,
      MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT) & seg,
      double eta);

  // evalauate F and gradF functions for
  // ProjectNodetoSegment_Orthogonal_to_Slave in 2D
  double
  evaluate_F_2D_SegmentOrthogonal_to_g(
      MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT) & node,
      MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT) & seg,
      double  eta,
      double& gap,
      double* g);

  double
  evaluate_gradF_2D_SegmentOrthogonal_to_g(
      MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT) & node,
      MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT) & seg,
      double  eta,
      double* g);

  //====3D projection methods

  // evaluate F and gradF functions for ProjectNodetoSegment_NodalNormal in 3D
  bool
  evaluate_FgradF_3D_NodalNormal(
      double* F,
      double  dF[][3],
      const MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT) & node,
      MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT) & seg,
      double* eta,
      double  alpha,
      double& gap);
  // evaluate F and gradF functions for ProjectNodetoSegment_SegmentNormal in 3D
  bool
  evaluate_FgradF_3D_SegmentNormal(
      double* F,
      double  dF[][3],
      const MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT) & node,
      MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT) & seg,
      double* eta,
      double  alpha,
      double& gap);

 private:
  bool twoD_;         // dimension of the projection, true if 2-dimensional
  int  outputlevel_;  // amount of output to be written (0-10)
};

}  // namespace MoertelT

#ifndef HAVE_MOERTEL_EXPLICIT_INSTANTIATION
#include "Moertel_Projector3DT_Def.hpp"
#include "Moertel_ProjectorT_Def.hpp"
#endif

#endif  // MOERTEL_PROJECTOR_H
