//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef MOERTEL_PROJECTNODET_HPP
#define MOERTEL_PROJECTNODET_HPP

#include <ctime>
#include <iomanip>
#include <iostream>

#include "Moertel_NodeT.hpp"

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

MOERTEL_TEMPLATE_STATEMENT
class SegmentT;

/*!
\class ProjectedNode

\brief <b> A class to handle the projection of a node onto some segment </b>

The \ref MOERTEL::ProjectedNode class supports the ostream& operator <<

\author Glen Hansen (gahanse@sandia.gov)

*/
MOERTEL_TEMPLATE_STATEMENT
class ProjectedNodeT : public MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)
{
 public:
  // @{ \name Constructors and destructor

  /*!
  \brief Constructor

  Constructs an instance of this class.<br>
  Note that this is \b not a collective call as nodes shall only have one owning
  process.

  \param basenode : the node this class is the projection of
  \param xi : local coordinates of the projection in the segment its projected
  onto \param pseg : Segment this projection is located in
  */
  explicit ProjectedNodeT(
      const MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT) & basenode,
      const double* xi,
      MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT) * pseg);

  /*!
  \brief Constructor (case of orthogonal projection only)

  Constructs an instance of this class.<br>
  Note that this is \b not a collective call as nodes shall only have one owning
  process.

  \param basenode : the node this class is the projection of
  \param xi : local coordinates of the projection in the segment its projected
  onto \param pseg : Segment this projection is located in \param orthseg : id
  of segment this projection is orthogonal to which might be different from pseg
  */
  explicit ProjectedNodeT(
      const MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT) & basenode,
      const double* xi,
      MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT) * pseg,
      int orthseg);

  /*!
  \brief Copy-Constructor

  */
  ProjectedNodeT(MoertelT::MOERTEL_TEMPLATE_CLASS(ProjectedNodeT) & old);

  /*!
  \brief Destructor

  */
  virtual ~ProjectedNodeT();

  //@}
  // @{ \name Public members

  /*!
  \brief Print this ProjectedNode and its Node

  */
  bool
  Print() const;

  /*!
  \brief Return view of the local coordinates of the projection in the segment

  */
  double*
  Xi()
  {
    return xi_;
  }

  /*!
  \brief Return pointer to segment this projection is in

  */
  MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT) * Segment() { return pseg_; }

  /*!
  \brief Return id of segment this projection is orthogonal to (might be
  different from \ref Segment() )

  */
  int
  OrthoSegment()
  {
    return orthseg_;
  }

  //@}

 protected:
  // don't want = operator
  ProjectedNodeT
  operator=(const MoertelT::MOERTEL_TEMPLATE_CLASS(ProjectedNodeT) & old);

 protected:
  double xi_[2];  // local coordinates of this projected node on pseg_;
  MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT) *
      pseg_;     // segment this projected node is on
  int orthseg_;  // id of segment this projection is orthogonal to
                 // (used only in orth. projection, otherwise -1)
};

}  // namespace MoertelT

// << operator
MOERTEL_TEMPLATE_STATEMENT
std::ostream&
operator<<(
    std::ostream& os,
    const Moertel::MOERTEL_TEMPLATE_CLASS(ProjectedNodeT) & pnode);

#ifndef HAVE_MOERTEL_EXPLICIT_INSTANTIATION
#include "Moertel_PnodeT_Def.hpp"
#endif

#endif  // MOERTEL_PROJECTNODE_H
