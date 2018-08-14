//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef MOERTEL_POINTT_HPP
#define MOERTEL_POINTT_HPP

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

/*!
\class Point

\brief <b> A light weight version of a node </b>

This class defines a point on a segment. It is a light weight version of a node.
It is used in the integration of 2D interfaces where the mortar side is
imprinted to the slave side. The overlap between a mortar and a slave segment
leads to a polygon defined by points on the slave segment. The polygon is then
discretized by triangle finite elements (eventually adding more points) to
perform the integration on the polygon region. A point might therefore become a
node of the polygon discretization and therefore has capabilities to store a
Node class.

The \ref MOERTEL::Point class supports the ostream& operator <<

\author Glen Hansen (gahanse@sandia.gov)

*/
MOERTEL_TEMPLATE_STATEMENT
class PointT
{
 public:
  // @{ \name Constructors and destructors

  /*!
  \brief Constructor

  Constructs an instance of this class.<br>
  Note that this is \b not a collective call as points shall only have one
  owning process.

  \param id : A unique positive point id.
  \param xi : Coordinates of point in a segment (2D)
  \param out : Level of output information written to stdout ( 0 - 10 )
  */
  PointT(const int id, const double* xi, int out);

  /*!
  \brief Destructor
  */
  virtual ~PointT();

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
  \brief Print this node to stdout

  */
  void
  Print() const;

  /*!
  \brief Return id of this point

  */
  inline int
  Id()
  {
    return id_;
  }

  /*!
  \brief Return view of segment local coordinates of this point (2D)

  */
  inline const double*
  Xi()
  {
    return &xi_[0];
  }

  /*!
  \brief Return view of global coordinates of this point (3D)

  If this point holds a Node it will return a pointer to the
  global 3D coordinates of that Node. If it does not hold a Node, it will
  return NULL

  */
  inline const double*
  XCoords()
  {
    if (node_ != Teuchos::null)
      return node_->XCoords();
    else
      return NULL;
  }

  /*!
  \brief Return view of Node

  If this point holds a Node it will return a pointer to the Node.
  If it does not hold a Node, it will return Teuchos::null

  */
  inline Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)>
  Node()
  {
    return node_;
  }

  /*!
  \brief Set segment local coordinates of this point (2D) in a segment

  */
  bool
  SetXi(const double* xi)
  {
    xi_[0] = xi[0];
    xi_[1] = xi[1];
    return true;
  }

  /*!
  \brief Set a Node to this point

  The Point takes ownership of the Node and will destroy it

  */
  bool SetNode(MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT) * node)
  {
    node_ = Teuchos::rcp(node);
    return true;
  }

  /*!
  \brief Store finite element function values at the Point 's coordinate \ref
  Xi()

  \param place : Place in internal data structure where to store function
  values.<br> place=0 is used to store trace space function values of the slave
  segment.<br> place=1 is used to store Lagrange multiplier space function
  values of the slave segment.<br> place=2 is used to store trace space function
  values of the master segment.<br> \param val : Vector of length valdim holding
  function values \param valdim : Dimension of val

  */
  void
  StoreFunctionValues(int place, double* val, int valdim);

  /*!
  \brief Return view of function values stored in this Point

  Returns a view of the function values that were stored in this Point using
  \ref StoreFunctionValues

  */
  std::vector<double>*
  FunctionValues()
  {
    return vals_;
  }

  //@}

 private:
  // don't want = operator
  PointT
  operator=(const PointT& old);
  // don't want copy-ctor
  PointT(MoertelT::MOERTEL_TEMPLATE_CLASS(PointT) & old);

 private:
  int    id_;  // id of this point
  int    outputlevel_;
  double xi_[2];  // local coords in some slave elements coord system
  Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)>
                      node_;  // a node at this point (contains real world coords)
  std::vector<double> vals_[3];  // [0] values of shape function 0 from sseg
                                 // [1] values of shape function 1 from sseg
                                 // [2] values of shape function 0 from mseg
};                               // class Point

}  // namespace MoertelT

// << operator
MOERTEL_TEMPLATE_STATEMENT
std::ostream&
operator<<(
    std::ostream& os,
    const MoertelT::MOERTEL_TEMPLATE_CLASS(PointT) & point);

#ifndef HAVE_MOERTEL_EXPLICIT_INSTANTIATION
#include "Moertel_PointT_Def.hpp"
#endif

#endif  // MOERTEL_POINT_H
