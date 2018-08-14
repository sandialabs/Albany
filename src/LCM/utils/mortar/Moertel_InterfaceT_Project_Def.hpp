//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Moertel_InterfaceT.hpp"
#include "Moertel_PnodeT.hpp"
#include "Moertel_ProjectorT.hpp"
#include "Moertel_UtilsT.hpp"

/*----------------------------------------------------------------------*
 |  build averaged normals                                              |
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT
bool MoertelT::MOERTEL_TEMPLATE_CLASS(InterfaceT)::BuildNormals()
{
  //-------------------------------------------------------------------
  // interface needs to be complete
  if (!IsComplete()) {
    if (gcomm_->getRank() == 0)
      std::cout << "***ERR*** MoertelT::Interface::Project:\n"
                << "***ERR*** Complete() not called on interface " << Id_
                << "\n"
                << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__
                << "\n";
    return false;
  }

  //-------------------------------------------------------------------
  // send all procs not member of this interface's intra-comm out of here
  if (lcomm_ == Teuchos::null) return true;

  //-------------------------------------------------------------------
  // interface segments need to have at least one function on each side
  std::map<int, Teuchos::RCP<MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)>>::
      iterator curr;
  for (int side = 0; side < 2; ++side)
    for (curr = rseg_[side].begin(); curr != rseg_[side].end(); ++curr) {
      if (curr->second->Nfunctions() < 1) {
        std::cout << "***ERR*** MoertelT::Interface::Project:\n"
                  << "***ERR*** interface " << Id_ << ", mortar side\n"
                  << "***ERR*** segment " << curr->second->Id()
                  << " needs at least 1 function set\n"
                  << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__
                  << "\n";
        return false;
      }
    }

  //-------------------------------------------------------------------
  // build nodal normals on both sides
  std::map<int, Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)>>::iterator
      ncurr;

  for (int side = 0; side < 2; ++side)

    for (ncurr = rnode_[side].begin(); ncurr != rnode_[side].end(); ++ncurr) {
#if 0
      std::cout << "side " << side << ": " << *(ncurr->second);
#endif
      ncurr->second->BuildAveragedNormal();
#if 0
      std::cout << "side " << side << ": " << *(ncurr->second);
#endif

    }  // for(ncurr ...)

  return true;
}

/*----------------------------------------------------------------------*
 |  build averaged normals and make projection of nodes                 |
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT
bool MoertelT::MOERTEL_TEMPLATE_CLASS(InterfaceT)::Project()
{
  bool ok = false;

  //-------------------------------------------------------------------
  // interface needs to be complete
  if (!IsComplete()) {
    if (gcomm_->getRank() == 0)
      std::cout << "***ERR*** MoertelT::Interface::Project:\n"
                << "***ERR*** Complete() not called on interface " << Id_
                << "\n"
                << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__
                << "\n";
    return false;
  }

  //-------------------------------------------------------------------
  // send all procs not member of this interface's intra-comm out of here
  if (lcomm_ == Teuchos::null) return true;

  //-------------------------------------------------------------------
  // interface segments need to have at least one function on each side
  std::map<int, Teuchos::RCP<MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)>>::
      iterator curr;
  for (int side = 0; side < 2; ++side)
    for (curr = rseg_[side].begin(); curr != rseg_[side].end(); ++curr)
      if (curr->second->Nfunctions() < 1) {
        std::cout << "***ERR*** MoertelT::Interface::Project:\n"
                  << "***ERR*** interface " << Id_ << ", mortar side\n"
                  << "***ERR*** segment " << curr->second->Id()
                  << " needs at least 1 function set\n"
                  << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__
                  << "\n";
        return false;
      }

  //-------------------------------------------------------------------
  // build nodal normals on both sides
  std::map<int, Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)>>::iterator
      ncurr;
  for (int side = 0; side < 2; ++side)
    for (ncurr = rnode_[side].begin(); ncurr != rnode_[side].end(); ++ncurr) {
#if 0
      std::cout << "side " << side << ": " << *(ncurr->second);
#endif
      ncurr->second->BuildAveragedNormal();
#if 0
      std::cout << "side " << side << ": " << *(ncurr->second);
#endif
    }

  //-------------------------------------------------------------------
  // check the type of projection to be used and project nodes
  // projection along the normal field:
  // uses the slave side's interpolated normal field to
  // project slave nodes onto the master surfaces.
  // Then projects master nodes along the same normal field on slave
  // surfaces. Overrides the normals in the master nodes by the negative
  // of the normal field of their projection point
  if (GetProjectionType() ==
      MoertelT::MOERTEL_TEMPLATE_CLASS(InterfaceT)::proj_continousnormalfield) {
    ok = ProjectNodes_NormalField();
    if (!ok) return false;
  } else if (
      GetProjectionType() ==
      MoertelT::MOERTEL_TEMPLATE_CLASS(InterfaceT)::proj_orthogonal) {
    ok = ProjectNodes_Orthogonal();
    if (!ok) return false;
  } else {
    std::cout << "***ERR*** MoertelT::Interface::Project:\n"
              << "***ERR*** interface " << Id() << "\n"
              << "***ERR*** unknown type of nodal projection\n"
              << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
    return false;
  }
  return true;
}

/*----------------------------------------------------------------------*
 | do projection of nodes on master and slave side                      |
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT
bool MoertelT::MOERTEL_TEMPLATE_CLASS(InterfaceT)::ProjectNodes_NormalField()
{
  if (!IsComplete()) {
    std::stringstream oss;
    oss << "***ERR*** MoertelT::Interface::ProjectNodes_NormalField:\n"
        << "***ERR*** Complete() not called on interface " << Id() << "\n"
        << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
    throw MoertelT::ReportError(oss);
  }
  if (lcomm_ == Teuchos::null) return true;

  // project the slave nodes onto the master surface along the slave normal
  // field
  ProjectNodes_SlavetoMaster_NormalField();

  // project the master nodes ontp the slave surface along slave's normal field
  ProjectNodes_MastertoSlave_NormalField();

  return true;
}

/*----------------------------------------------------------------------*
 | project the slave nodes onto master segments along slave normal field|
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT
bool MoertelT::MOERTEL_TEMPLATE_CLASS(
    InterfaceT)::ProjectNodes_SlavetoMaster_NormalField()
{
  if (!IsComplete()) {
    std::stringstream oss;
    oss << "***ERR*** "
           "MoertelT::Interface::ProjectNodes_SlavetoMaster_NormalField:\n"
        << "***ERR*** Complete() not called on interface " << Id() << "\n"
        << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
    throw MoertelT::ReportError(oss);
  }
  if (lcomm_ == Teuchos::null) return true;

  int mside = MortarSide();
  int sside = OtherSide(mside);

  // iterate over all nodes of the slave side and project those belonging to me
  std::map<int, Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)>>::iterator
      scurr;
  for (scurr = rnode_[sside].begin(); scurr != rnode_[sside].end(); ++scurr) {
    Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)> snode = scurr->second;
    if (NodePID(snode->Id()) != lcomm_->getRank()) continue;

    const double*                                         sx = snode->XCoords();
    double                                                mindist = 1.0e+20;
    Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)> closenode =
        Teuchos::null;

    // find a node on the master side, that is closest to me
    std::map<int, Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)>>::
        iterator mcurr;
    for (mcurr = rnode_[mside].begin(); mcurr != rnode_[mside].end(); ++mcurr) {
      Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)> mnode =
          mcurr->second;
      const double* mx = mnode->XCoords();

      // build distance | mnode->XCoords() - snode->XCoords() |
      double dist = 0.0;
      for (int i = 0; i < 3; ++i) dist += (mx[i] - sx[i]) * (mx[i] - sx[i]);
      dist = sqrt(dist);
      if (dist <= mindist) {
        mindist   = dist;
        closenode = mnode;
      }
      std::cout << "snode " << snode->Id() << " mnode " << mnode->Id()
                << " mindist " << mindist << " dist " << dist << std::endl;
    }
    if (closenode == Teuchos::null) {
      std::stringstream oss;
      oss << "***ERR*** "
             "MoertelT::Interface::ProjectNodes_SlavetoMaster_NormalField:\n"
          << "***ERR*** Weired: for slave node " << snode->Id()
          << " no closest master node found\n"
          << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
      throw MoertelT::ReportError(oss);
    }

#if 0
    std::cout << "snode     " << *snode;
    std::cout << "closenode " << *closenode;
#endif

    // get segments attached to closest node cnode
    int nseg                                          = closenode->Nseg();
    MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)** segs = closenode->Segments();

    // create a projection-iterator
    MoertelT::MOERTEL_TEMPLATE_CLASS(ProjectorT)
        projector(IsOneDimensional(), OutLevel());

    // finding a good geometric projection is somehow
    // critical. We work with some tolerance here and pick the 'best'
    // out of all acceptable projections made
    // loop these segments and project onto them along snode's normal vector
    double       bestdist[2];
    double       gap, bestgap = 0.0;
    const double tol = 0.2;
    bestdist[0] = bestdist[1]                           = 1.0e+20;
    MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)* bestseg = NULL;
    for (int i = 0; i < nseg; ++i) {
      // project the slave node onto that master segment
      double xi[2];
      xi[0] = xi[1] = 0.0;
      projector.ProjectNodetoSegment_NodalNormal(*snode, *(segs[i]), xi, gap);

      // check whether xi is better then previous projections
      if (IsOneDimensional())  // 2D case
      {
        if (abs(xi[0]) < abs(bestdist[0])) {
          bestdist[0] = xi[0];
          bestdist[1] = xi[1];
          bestseg     = segs[i];
          bestgap     = gap;
        }
      } else  // 3D case
      {
        double third = 1. / 3.;
        // it's 'inside' with some tolerance
        if (xi[0] <= 1. + tol && xi[1] <= abs(1. - xi[0]) + tol &&
            xi[0] >= 0. - tol && xi[1] >= 0. - tol) {
          // it's better in both directions
          if (sqrt((xi[0] - third) * (xi[0] - third)) <
                  sqrt((bestdist[0] - third) * (bestdist[0] - third)) &&
              sqrt((xi[1] - third) * (xi[1] - third)) <
                  sqrt((bestdist[1] - third) * (bestdist[1] - third))) {
            bestdist[0] = xi[0];
            bestdist[1] = xi[1];
            bestseg     = segs[i];
            bestgap     = gap;
          }
          // it's better in one direction and 'in' in the other
          else if (
              (sqrt((xi[0] - third) * (xi[0] - third)) <
                   sqrt((bestdist[0] - third) * (bestdist[0] - third)) &&
               xi[1] <= abs(1. - xi[0]) + tol && xi[1] >= 0. - tol) ||
              (sqrt((xi[1] - third) * (xi[1] - third)) <
                   sqrt((bestdist[1] - third) * (bestdist[1] - third)) &&
               xi[0] <= 1. + tol && xi[0] >= 0. - tol)) {
            bestdist[0] = xi[0];
            bestdist[1] = xi[1];
            bestseg     = segs[i];
            bestgap     = gap;
          }
        }
      }
    }  // for (int i=0; i<nseg; ++i)

    // check whether the bestseg and bestdist are inside the segment
    // (with some tolerance of 20%)
    bool ok = false;
    if (IsOneDimensional()) {
      if (abs(bestdist[0]) < 1.2) ok = true;
    } else {
      if (bestdist[0] <= 1. + tol &&
          bestdist[1] <= abs(1. - bestdist[0]) + tol &&
          bestdist[0] >= 0. - tol && bestdist[1] >= 0. - tol)
        ok = true;
    }

    if (ok)  // the projection is good
    {
      // create a projected node and store it in snode
      MoertelT::MOERTEL_TEMPLATE_CLASS(ProjectedNodeT)* pnode =
          new MoertelT::MOERTEL_TEMPLATE_CLASS(ProjectedNodeT)(
              *snode, bestdist, bestseg);
      snode->SetProjectedNode(pnode);
      snode->SetGap(bestgap);
    } else {
      if (OutLevel() > 6)
        std::cout << "MoertelT: ***WRN***: Projection s->m: Node "
                  << snode->Id() << " does not have projection\n";
      snode->SetProjectedNode(NULL);
    }
  }  // for (scurr=rnode_[sside].begin(); scurr!=rnode_[sside].end(); ++scurr)
  lcomm_->barrier();

  // loop all slave nodes again and make the projections redundant
  std::vector<double> bcast(4 * rnode_[sside].size());
  for (int proc = 0; proc < lcomm_->getSize(); ++proc) {
    int blength = 0;
    if (proc == lcomm_->getRank()) {
      for (scurr = rnode_[sside].begin(); scurr != rnode_[sside].end();
           ++scurr) {
        Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)> snode =
            scurr->second;
        if (proc != NodePID(snode->Id()))
          continue;  // I cannot have a projection on a node not owned by me
        Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(ProjectedNodeT)> pnode =
            snode->GetProjectedNode();
        if (pnode == Teuchos::null)
          continue;  // this node does not have a projection
        const double* xi = pnode->Xi();
        bcast[blength]   = (double)pnode->Id();
        ++blength;
        if (pnode->Segment())
          bcast[blength] = (double)pnode->Segment()->Id();
        else
          bcast[blength] = -0.1;  // indicating this node does not have
                                  // projection but lagrange multipliers
        ++blength;
        bcast[blength] = xi[0];
        ++blength;
        bcast[blength] = xi[1];
        ++blength;
        bcast[blength] = pnode->Gap();
        ++blength;
      }
      if (blength > (int)(4 * rnode_[sside].size())) {
        std::stringstream oss;
        oss << "***ERR*** "
               "MoertelT::Interface::ProjectNodes_SlavetoMaster_NormalField:\n"
            << "***ERR*** Overflow in communication buffer occured\n"
            << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
        throw MoertelT::ReportError(oss);
      }
    }
    Teuchos::broadcast<LO, int>(*lcomm_, proc, 1, &blength);
    Teuchos::broadcast<LO, double>(*lcomm_, proc, blength, &bcast[0]);
    if (proc != lcomm_->getRank()) {
      int i;
      for (i = 0; i < blength;) {
        int nid = (int)bcast[i];
        ++i;
        double sid = bcast[i];
        ++i;
        double* xi = &bcast[i];
        ++i;
        ++i;
        double gap = bcast[i];
        ++i;
        Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)> snode =
            GetNodeView(nid);
        Teuchos::RCP<MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)> seg =
            Teuchos::null;
        if (sid != -0.1) seg = GetSegmentView((int)sid);
        if (snode == Teuchos::null) {
          std::stringstream oss;
          oss << "***ERR*** "
                 "MoertelT::Interface::ProjectNodes_SlavetoMaster_NormalField:"
                 "\n"
              << "***ERR*** Cannot get view of node\n"
              << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
          throw MoertelT::ReportError(oss);
        }
        MoertelT::MOERTEL_TEMPLATE_CLASS(ProjectedNodeT)* pnode =
            new MoertelT::MOERTEL_TEMPLATE_CLASS(ProjectedNodeT)(
                *snode, xi, seg.get());
        snode->SetProjectedNode(pnode);
        snode->SetGap(gap);
      }
      if (i != blength) {
        std::stringstream oss;
        oss << "***ERR*** "
               "MoertelT::Interface::ProjectNodes_SlavetoMaster_NormalField:\n"
            << "***ERR*** Mismatch in dimension of recv buffer\n"
            << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
        throw MoertelT::ReportError(oss);
      }
    }
  }  // for (int proc=0; proc<lComm()->NumProc(); ++proc)
  bcast.clear();
  lcomm_->barrier();

#if 1
  // Postprocess the projections
  // The slave side of the interface might be larger then the master side
  // of the interface so not all slave nodes have a projection.
  // For those slave nodes without a projection attached to a slave segment
  // which overlaps with the master side, Lagrange multipliers have to be
  // introduced. This is done by checking all nodes without a projection
  // whether they are attached to some slave segment on which another node
  // HAS a projection. If this case is found, a pseudo ProjectedNode is
  // introduced for that node.
  for (scurr = rnode_[sside].begin(); scurr != rnode_[sside].end(); ++scurr) {
    Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)> snode = scurr->second;

    // don't do anything on nodes that already have a projection
    if (snode->GetProjectedNode() != Teuchos::null) continue;

    // get segments adjacent to this node
    int nseg                                          = snode->Nseg();
    MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)** segs = snode->Segments();

    // loop segments and check for other nodes with projection
    bool foundit = false;
    for (int i = 0; i < nseg; ++i) {
      int nnode                                       = segs[i]->Nnode();
      MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)** nodes = segs[i]->Nodes();
      for (int j = 0; j < nnode; ++j)
        if (nodes[j]->GetProjectedNode() != Teuchos::null)
          if (nodes[j]->GetProjectedNode()->Segment()) {
            foundit = true;
            break;
          }
      if (foundit) break;
    }

    if (foundit) {
#if 1
      std::cout << "Node without projection:\n" << *snode;
      std::cout << "...get's lagrange multipliers\n\n";
#endif
      MoertelT::MOERTEL_TEMPLATE_CLASS(ProjectedNodeT)* pnode =
          new MoertelT::MOERTEL_TEMPLATE_CLASS(ProjectedNodeT)(
              *snode, NULL, NULL);
      snode->SetProjectedNode(pnode);
      snode->SetGap(0.);
    }
  }  // for (scurr=rnode_[sside].begin(); scurr!=rnode_[sside].end(); ++scurr)
#endif

  return true;
}

/*----------------------------------------------------------------------*
 | project nodes master to slave along slave cont. normal field         |
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT
bool MoertelT::MOERTEL_TEMPLATE_CLASS(
    InterfaceT)::ProjectNodes_MastertoSlave_NormalField()
{
  if (!IsComplete()) {
    std::stringstream oss;
    oss << "***ERR*** "
           "MoertelT::Interface::ProjectNodes_MastertoSlave_NormalField:\n"
        << "***ERR*** Complete() not called on interface " << Id() << "\n"
        << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
    throw MoertelT::ReportError(oss);
  }
  if (lcomm_ == Teuchos::null) return true;

  int mside = MortarSide();
  int sside = OtherSide(mside);

  // iterate over all nodes of the master side and project those belonging to me
  std::map<int, Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)>>::iterator
      mcurr;
  for (mcurr = rnode_[mside].begin(); mcurr != rnode_[mside].end(); ++mcurr) {
    Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)> mnode = mcurr->second;
    if (NodePID(mnode->Id()) != lcomm_->getRank()) continue;

    const double*                   mx        = mnode->XCoords();
    double                          mindist   = 1.0e+20;
    Teuchos::RCP<MoertelT::(NodeT)> closenode = Teuchos::null;

    // find a node on the slave side that is closest to me
    std::map<int, Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)>>::
        iterator scurr;
    for (scurr = rnode_[sside].begin(); scurr != rnode_[sside].end(); ++scurr) {
      Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)> snode =
          scurr->second;
      const double* sx = snode->XCoords();

      // build distance | snode->XCoords() - mnode->XCoords() |
      double dist = 0.0;
      for (int i = 0; i < 3; ++i) dist += (mx[i] - sx[i]) * (mx[i] - sx[i]);
      dist = sqrt(dist);
      if (dist < mindist) {
        mindist   = dist;
        closenode = snode;
      }
    }
    if (closenode == Teuchos::null) {
      std::stringstream oss;
      oss << "***ERR*** "
             "MoertelT::Interface::ProjectNodes_MastertoSlave_NormalField:\n"
          << "***ERR*** Weired: for master node " << mnode->Id()
          << " no closest master node found\n"
          << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
      throw MoertelT::ReportError(oss);
    }

#if 0
    std::cout << "snode     " << *mnode;
    std::cout << "closenode " << *closenode;
#endif

    // get segments attached to closest node closenode
    int nseg                                          = closenode->Nseg();
    MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)** segs = closenode->Segments();

    // create a projection operator
    MoertelT::MOERTEL_TEMPLATE_CLASS(ProjectorT)
        projector(IsOneDimensional(), OutLevel());

    // loop these segments and find best projection
    double       bestdist[2];
    const double tol     = 0.2;
    double       bestgap = 0.0, gap;
    bestdist[0] = bestdist[1]                           = 1.0e+20;
    MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)* bestseg = NULL;
    for (int i = 0; i < nseg; ++i) {
      // project the master node on the slave segment along the segments
      // interpolated normal field
      double xi[2];
      xi[0] = xi[1] = 0.0;
      projector.ProjectNodetoSegment_SegmentNormal(*mnode, *(segs[i]), xi, gap);

      // check whether xi is better then previous projections
      if (IsOneDimensional()) {
        if (abs(xi[0]) < abs(bestdist[0])) {
          bestdist[0] = xi[0];
          bestdist[1] = xi[1];
          bestseg     = segs[i];
          bestgap     = gap;
        }
      } else {
        double third = 1. / 3.;
        // it's 'inside' with some tolerance
        if (xi[0] <= 1. + tol && xi[1] <= abs(1. - xi[0]) + tol &&
            xi[0] >= 0. - tol && xi[1] >= 0. - tol) {
          // it's better in both directions
          if (sqrt((xi[0] - third) * (xi[0] - third)) <
                  sqrt((bestdist[0] - third) * (bestdist[0] - third)) &&
              sqrt((xi[1] - third) * (xi[1] - third)) <
                  sqrt((bestdist[1] - third) * (bestdist[1] - third))) {
            bestdist[0] = xi[0];
            bestdist[1] = xi[1];
            bestseg     = segs[i];
            bestgap     = gap;
          }
          // it's better in one direction and 'in' in the other
          else if (
              (sqrt((xi[0] - third) * (xi[0] - third)) <
                   sqrt((bestdist[0] - third) * (bestdist[0] - third)) &&
               xi[1] <= abs(1. - xi[0]) + tol && xi[1] >= 0. - tol) ||
              (sqrt((xi[1] - third) * (xi[1] - third)) <
                   sqrt((bestdist[1] - third) * (bestdist[1] - third)) &&
               xi[0] <= 1. + tol && xi[0] >= 0. - tol)) {
            bestdist[0] = xi[0];
            bestdist[1] = xi[1];
            bestseg     = segs[i];
            bestgap     = gap;
          }
        }
      }
    }  // for (int i=0; i<nseg; ++i)

    // check whether the bestseg/bestdist are inside that segment
    // (with some tolerance of 20%)
    bool ok = false;
    if (IsOneDimensional()) {
      if (abs(bestdist[0]) < 1.1) ok = true;
    } else {
      if (bestdist[0] <= 1. + tol &&
          bestdist[1] <= abs(1. - bestdist[0]) + tol &&
          bestdist[0] >= 0. - tol && bestdist[1] >= 0. - tol)
        ok = true;
    }

    if (ok)  // the projection is good
    {
      // build the interpolated normal and overwrite the mnode normal with -n
      int nsnode                                       = bestseg->Nnode();
      MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)** snodes = bestseg->Nodes();
      std::vector<double> val(nsnode);
      bestseg->EvaluateFunction(0, bestdist, &val[0], nsnode, NULL);
      double NN[3];
      NN[0] = NN[1] = NN[2] = 0.0;
      for (int i = 0; i < nsnode; ++i) {
        const double* Normal = snodes[i]->Normal();
        for (int j = 0; j < 3; ++j) NN[j] -= val[i] * Normal[j];
      }
      val.clear();
      mnode->SetN(NN);

      // create projected node and store it in mnode
      MoertelT::MOERTEL_TEMPLATE_CLASS(ProjectedNodeT)* pnode =
          new MoertelT::MOERTEL_TEMPLATE_CLASS(ProjectedNodeT)(
              *mnode, bestdist, bestseg);
      mnode->SetProjectedNode(pnode);
      mnode->SetGap(bestgap);
    } else  // this mnode does not have a valid projection
    {
      if (OutLevel() > 6)
        std::cout << "MoertelT: ***WRN***: Projection m->s: Node "
                  << mnode->Id() << " does not have projection\n";
      mnode->SetProjectedNode(NULL);
    }
  }  // for (scurr=rnode_[mside].begin(); scurr!=rnode_[mside].end(); ++scurr)

  // loop all master nodes again and make the projection and the new normal
  // redundant
  int                 bsize = 7 * rnode_[mside].size();
  std::vector<double> bcast(bsize);
  for (int proc = 0; proc < lcomm_->getSize(); ++proc) {
    int blength = 0;
    if (proc == lcomm_->getRank()) {
      for (mcurr = rnode_[mside].begin(); mcurr != rnode_[mside].end();
           ++mcurr) {
        Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)> mnode =
            mcurr->second;
        if (proc != NodePID(mnode->Id()))
          continue;  // cannot have a projection on a node i don't own
        Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(ProjectedNodeT)> pnode =
            mnode->GetProjectedNode();
        if (pnode == Teuchos::null)
          continue;  // this node does not have a projection
        const double* xi     = pnode->Xi();
        const double* Normal = mnode->Normal();
        bcast[blength]       = (double)pnode->Id();
        ++blength;
        if (pnode->Segment())
          bcast[blength] = (double)pnode->Segment()->Id();
        else
          bcast[blength] = -0.1;
        ++blength;
        bcast[blength] = xi[0];
        ++blength;
        bcast[blength] = xi[1];
        ++blength;
        bcast[blength] = Normal[0];
        ++blength;
        bcast[blength] = Normal[1];
        ++blength;
        bcast[blength] = Normal[2];
        ++blength;
        bcast[blength] = pnode->Gap();
        ++blength;
      }
      if (blength > bsize) {
        std::stringstream oss;
        oss << "***ERR*** "
               "MoertelT::Interface::ProjectNodes_MastertoSlave_NormalField:\n"
            << "***ERR*** Overflow in communication buffer occured\n"
            << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
        throw MoertelT::ReportError(oss);
      }
    }
    Teuchos::broadcast<LO, int>(*lcomm_, proc, 1, &blength);
    Teuchos::broadcast<LO, double>(*lcomm_, proc, blength, &bcast[0]);
    if (proc != lcomm_->getRank()) {
      int i;
      for (i = 0; i < blength;) {
        int nid = (int)bcast[i];
        ++i;
        double sid = bcast[i];
        ++i;
        double* xi = &bcast[i];
        ++i;
        ++i;
        double* n = &bcast[i];
        ++i;
        ++i;
        ++i;
        double gap = bcast[i];
        ++i;
        Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)> mnode =
            GetNodeView(nid);
        Teuchos::RCP<MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)> seg =
            Teuchos::null;
        if (sid != -0.1) seg = GetSegmentView((int)sid);
        if (mnode == Teuchos::null) {
          std::stringstream oss;
          oss << "***ERR*** "
                 "MoertelT::Interface::ProjectNodes_MastertoSlave_NormalField:"
                 "\n"
              << "***ERR*** Cannot get view of node\n"
              << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
          throw MoertelT::ReportError(oss);
        }
        mnode->SetN(n);
        MoertelT::MOERTEL_TEMPLATE_CLASS(ProjectedNodeT)* pnode =
            new MoertelT::MOERTEL_TEMPLATE_CLASS(ProjectedNodeT)(
                *mnode, xi, seg.get());
        mnode->SetProjectedNode(pnode);
        mnode->SetGap(gap);
      }
      if (i != blength) {
        std::stringstream oss;
        oss << "***ERR*** "
               "MoertelT::Interface::ProjectNodes_MastertoSlave_NormalField:\n"
            << "***ERR*** Mismatch in dimension of recv buffer\n"
            << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
        throw MoertelT::ReportError(oss);
      }
    }
  }  // for (int proc=0; proc<lComm()->NumProc(); ++proc)
  bcast.clear();

  return true;
}

/*----------------------------------------------------------------------*
 | do projection of nodes on master and slave side                      |
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT
bool MoertelT::MOERTEL_TEMPLATE_CLASS(InterfaceT)::ProjectNodes_Orthogonal()
{
  if (!IsComplete()) {
    std::stringstream oss;
    oss << "***ERR*** MoertelT::Interface::ProjectNodes_Orthogonal:\n"
        << "***ERR*** Complete() not called on interface " << Id() << "\n"
        << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
    throw MoertelT::ReportError(oss);
  }
  if (lcomm_ == Teuchos::null) return true;

  // project the master nodes onto the slave surface orthogonaly
  ProjectNodes_MastertoSlave_Orthogonal();

  // project the slave nodes onto the master surface orthogonal to adjacent
  // slave segment
  ProjectNodes_SlavetoMaster_Orthogonal();

  return true;
}

/*----------------------------------------------------------------------*
 | project nodes master to slave along slave orthogonal         |
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT
bool MoertelT::MOERTEL_TEMPLATE_CLASS(
    InterfaceT)::ProjectNodes_MastertoSlave_Orthogonal()
{
  if (!IsComplete()) {
    std::stringstream oss;
    oss << "***ERR*** "
           "MoertelT::Interface::ProjectNodes_MastertoSlave_Orthogonal:\n"
        << "***ERR*** Complete() not called on interface " << Id() << "\n"
        << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
    throw MoertelT::ReportError(oss);
  }
  if (lcomm_ == Teuchos::null) return true;

  int mside = MortarSide();
  int sside = OtherSide(mside);

  // iterate over all master nodes and project those belonging to me
  std::map<int, Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)>>::iterator
      mcurr;
  for (mcurr = rnode_[mside].begin(); mcurr != rnode_[mside].end(); ++mcurr) {
    Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)> mnode = mcurr->second;
    if (NodePID(mnode->Id()) != lcomm_->getRank()) continue;

    const double*                                         mx = mnode->XCoords();
    double                                                mindist = 1.0e+20;
    Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)> closenode =
        Teuchos::null;

    // find a node on the slave side that is closest to me
    std::map<int, Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)>>::
        iterator scurr;
    for (scurr = rnode_[sside].begin(); scurr != rnode_[sside].end(); ++scurr) {
      Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)> snode =
          scurr->second;
      const double* sx = snode->XCoords();

      // build distance | snode->XCoords() - mnode->XCoords() |
      double dist = 0.0;
      for (int i = 0; i < 3; ++i) dist += (mx[i] - sx[i]) * (mx[i] - sx[i]);
      dist = sqrt(dist);
      if (dist < mindist) {
        mindist   = dist;
        closenode = snode;
      }
    }
    if (closenode == Teuchos::null) {
      std::stringstream oss;
      oss << "***ERR*** "
             "MoertelT::Interface::ProjectNodes_MastertoSlave_Orthogonal:\n"
          << "***ERR*** Weired: for master node " << mnode->Id()
          << " no closest master node found\n"
          << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
      throw MoertelT::ReportError(oss);
    }

    // get segments attached to closest node closenode
    int nseg                                          = closenode->Nseg();
    MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)** segs = closenode->Segments();

    // create a projection operator
    MoertelT::MOERTEL_TEMPLATE_CLASS(ProjectorT)
        projector(IsOneDimensional(), OutLevel());

    // loop these segments and find best projection
    double bestdist[2];
    double bestgap = 0.0, gap;
    bestdist[0] = bestdist[1]                           = 1.0e+20;
    MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)* bestseg = NULL;
    for (int i = 0; i < nseg; ++i) {
      // project the master node orthogonally on the slave segment
      double xi[2];
      xi[0] = xi[1] = 0.0;
      projector.ProjectNodetoSegment_SegmentOrthogonal(
          *mnode, *(segs[i]), xi, gap);

      // check whether xi is better than previous projection
      if (IsOneDimensional()) {
        if (abs(xi[0]) < abs(bestdist[0])) {
          bestdist[0] = xi[0];
          bestdist[1] = xi[1];
          bestseg     = segs[i];
          bestgap     = gap;
        }
      } else {
        std::stringstream oss;
        oss << "***ERR*** "
               "MoertelT::Interface::ProjectNodes_MastertoSlave_Orthogonal:\n"
            << "***ERR*** not impl. for 3D\n"
            << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
        throw MoertelT::ReportError(oss);
      }

    }  // for (int i=0; i<nseg; ++i)

    // check whether this best projection is good
    bool ok = false;
    if (IsOneDimensional()) {
      if (abs(bestdist[0]) < 1.01) ok = true;
    } else {
      std::stringstream oss;
      oss << "***ERR*** "
             "MoertelT::Interface::ProjectNodes_MastertoSlave_Orthogonal:\n"
          << "***ERR*** not impl. for 3D\n"
          << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
      throw MoertelT::ReportError(oss);
    }

    if (ok)  // the projection is good
    {
      // create a projected node and store it in mnode
      MoertelT::MOERTEL_TEMPLATE_CLASS(ProjectedNodeT)* pnode =
          new MoertelT::MOERTEL_TEMPLATE_CLASS(ProjectedNodeT)(
              *mnode, bestdist, bestseg);
      mnode->SetProjectedNode(pnode);
      mnode->SetGap(bestgap);
    } else  // this mnode does not have a valid projection
    {
      if (OutLevel() > 6)
        std::cout << "MoertelT: ***WRN***: Node " << mnode->Id()
                  << " does not have projection\n\n";
      // mnode->SetProjectedNode(NULL);
    }
  }  // for (mcurr=rnode_[mside].begin(); mcurr!=rnode_[mside].end(); ++mcurr)

  // loop all master nodes again and make projection redundant
  int                 bsize = 4 * rnode_[mside].size();
  std::vector<double> bcast(bsize);
  for (int proc = 0; proc < lcomm_->getSize(); ++proc) {
    int blength = 0;
    if (proc == lcomm_->getRank()) {
      for (mcurr = rnode_[mside].begin(); mcurr != rnode_[mside].end();
           ++mcurr) {
        Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)> mnode =
            mcurr->second;
        if (proc != NodePID(mnode->Id()))
          continue;  // cannot have a projection on a node i don't own
        Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(ProjectedNodeT)> pnode =
            mnode->GetProjectedNode();
        if (pnode == Teuchos::null)
          continue;  // this node does not have a projection
        const double* xi = pnode->Xi();
        bcast[blength]   = (double)pnode->Id();
        ++blength;
        bcast[blength] = (double)pnode->Segment()->Id();
        ++blength;
        bcast[blength] = xi[0];
        ++blength;
        bcast[blength] = xi[1];
        ++blength;
        bcast[blength] = pnode->Gap();
        ++blength;
      }  // for (mcurr=rnode_[mside].begin(); mcurr!=rnode_[mside].end();
         // ++mcurr)
      if (blength > bsize) {
        std::stringstream oss;
        oss << "***ERR*** "
               "MoertelT::Interface::ProjectNodes_MastertoSlave_Orthogonal:\n"
            << "***ERR*** Overflow in communication buffer occured\n"
            << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
        throw MoertelT::ReportError(oss);
      }
    }  // if (proc==lComm()->MyPID())
    Teuchos::broadcast<LO, int>(*lcomm_, proc, 1, &blength);
    Teuchos::broadcast<LO, double>(*lcomm_, proc, blength, &bcast[0]);
    if (proc != lcomm_->getRank()) {
      int i;
      for (i = 0; i < blength;) {
        int nid = (int)bcast[i];
        ++i;
        int sid = (int)bcast[i];
        ++i;
        double* xi = &bcast[i];
        ++i;
        ++i;
        double gap = bcast[i];
        ++i;
        Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)> mnode =
            GetNodeView(nid);
        Teuchos::RCP<MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)> seg =
            GetSegmentView(sid);
        if (mnode == Teuchos::null || seg == Teuchos::null) {
          std::stringstream oss;
          oss << "***ERR*** "
                 "MoertelT::Interface::ProjectNodes_MastertoSlave_Orthogonal:\n"
              << "***ERR*** Cannot get view of node or segment\n"
              << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
          throw MoertelT::ReportError(oss);
        }
        MoertelT::MOERTEL_TEMPLATE_CLASS(ProjectedNodeT)* pnode =
            new MoertelT::MOERTEL_TEMPLATE_CLASS(ProjectedNodeT)(
                *mnode, xi, seg.get());
        mnode->SetProjectedNode(pnode);
        mnode->SetGap(gap);
      }
      if (i != blength) {
        std::stringstream oss;
        oss << "***ERR*** "
               "MoertelT::Interface::ProjectNodes_MastertoSlave_Orthogonal:\n"
            << "***ERR*** Mismatch in dimension of recv buffer: " << i
            << " != " << blength << "\n"
            << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
        throw MoertelT::ReportError(oss);
      }
    }  // if (proc!=lComm()->MyPID())
  }    // for (int proc=0; proc<lComm()->NumProc(); ++proc)
  bcast.clear();

  return true;
}

/*----------------------------------------------------------------------*
 | project the slave nodes onto master segments orthogonal              |
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT
bool MoertelT::MOERTEL_TEMPLATE_CLASS(
    InterfaceT)::ProjectNodes_SlavetoMaster_Orthogonal()
{
  if (!IsComplete()) {
    std::stringstream oss;
    oss << "***ERR*** "
           "MoertelT::Interface::ProjectNodes_SlavetoMaster_Orthogonal:\n"
        << "***ERR*** Complete() not called on interface " << Id() << "\n"
        << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
    throw MoertelT::ReportError(oss);
  }
  if (lcomm_ == Teuchos::null) return true;

  int mside = MortarSide();
  int sside = OtherSide(mside);

  // iterate over all nodes of the slave side and project those belonging to me
  std::map<int, Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)>>::iterator
      scurr;
  for (scurr = rnode_[sside].begin(); scurr != rnode_[sside].end(); ++scurr) {
    Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)> snode = scurr->second;

#if 0
    std::cout << "now projecting\n " << *snode;
#endif

    if (NodePID(snode->Id()) != lcomm_->getRank()) continue;

    const double*                                         sx = snode->XCoords();
    double                                                mindist = 1.0e+20;
    Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)> closenode =
        Teuchos::null;

    // find a node on the master side, that is closest to me
    std::map<int, Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)>>::
        iterator mcurr;
    for (mcurr = rnode_[mside].begin(); mcurr != rnode_[mside].end(); ++mcurr) {
      Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)> mnode =
          mcurr->second;
      const double* mx = mnode->XCoords();

      // build distance | mnode->XCoords() - snode->XCoords() |
      double dist = 0.0;
      for (int i = 0; i < 3; ++i) dist += (mx[i] - sx[i]) * (mx[i] - sx[i]);
      dist = sqrt(dist);
      if (dist <= mindist) {
        mindist   = dist;
        closenode = mnode;
      }
      // std::cout << "snode " << snode->Id() << " mnode " << mnode->Id() << "
      // mindist " << mindist  << " dist " << dist << std::endl;
    }
    if (closenode == Teuchos::null) {
      std::stringstream oss;
      oss << "***ERR*** "
             "MoertelT::Interface::ProjectNodes_SlavetoMaster_Orthogonal:\n"
          << "***ERR*** Weired: for slave node " << snode->Id()
          << " no closest master node found\n"
          << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
      throw MoertelT::ReportError(oss);
    }

    // get segments attached to closest node cnode
    int nmseg                                          = closenode->Nseg();
    MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)** msegs = closenode->Segments();

    // create a projection-iterator
    MoertelT::MOERTEL_TEMPLATE_CLASS(ProjectorT)
        projector(IsOneDimensional(), OutLevel());

    // loop segments and find all projections onto them
    int nsseg                                          = snode->Nseg();
    MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)** ssegs = snode->Segments();
    for (int i = 0; i < nmseg; ++i) {
      // loop all segments that are adjacent to the slave node
      for (int j = 0; j < nsseg; ++j) {
        // project the slave node onto that master segment
        double xi[2];
        xi[0] = xi[1] = 0.0;
        double gap;
        projector.ProjectNodetoSegment_Orthogonal_to_Slave(
            *snode, *(msegs[i]), xi, gap, *(ssegs[j]));

        // check whether this projection is good
        bool ok = false;
        if (IsOneDimensional()) {
          if (abs(xi[0]) < 1.01) ok = true;
        } else {
          std::stringstream oss;
          oss << "***ERR*** "
                 "MoertelT::Interface::ProjectNodes_SlavetoMaster_Orthogonal:\n"
              << "***ERR*** not impl. for 3D\n"
              << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
          throw MoertelT::ReportError(oss);
        }

        if (ok)  // the projection is good
        {
          // create a projected node and store it in snode
          MoertelT::MOERTEL_TEMPLATE_CLASS(ProjectedNodeT)* pnode =
              new MoertelT::MOERTEL_TEMPLATE_CLASS(ProjectedNodeT)(
                  *snode, xi, msegs[i], ssegs[j]->Id());
          snode->SetProjectedNode(pnode);
          snode->SetGap(gap);
#if 0
          std::cout << " snode id: " << pnode->Id()
               << " projects on mseg: " << msegs[i]->Id()
               << " orth to sseg " << ssegs[j]->Id() << std::endl;
#endif
        }
      }  // for (int j=0; j<nsseg; ++j)
    }    // for (int i=0; i<nmseg; ++i)
  }  // for (scurr=rnode_[sside].begin(); scurr!=rnode_[sside].end(); ++scurr)

  // loop all slave nodes again and make projections redundant
  if (lcomm_->getSize() > 1) {
    std::vector<double> bcast(10 * rnode_[sside].size());
    for (int proc = 0; proc < lcomm_->getSize(); ++proc) {
      int blength = 0;
      if (proc == lcomm_->getRank()) {
        for (scurr = rnode_[sside].begin(); scurr != rnode_[sside].end();
             ++scurr) {
          Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)> snode =
              scurr->second;
          if (proc != NodePID(snode->Id()))
            continue;  // cannot have a projection on a node i don't own
          int npnode = 0;
          Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(ProjectedNodeT)>*
              pnode = snode->GetProjectedNode(npnode);
          if (!pnode) continue;  // no projection on this one
          bcast[blength] = (double)snode->Id();
          ++blength;
          bcast[blength] = (double)npnode;
          ++blength;
          for (int j = 0; j < npnode; ++j) {
            bcast[blength] = (double)pnode[j]->Segment()->Id();
            ++blength;
            const double* xi = pnode[j]->Xi();
            bcast[blength]   = xi[0];
            ++blength;
            bcast[blength] = xi[1];
            ++blength;
            bcast[blength] = pnode[j]->OrthoSegment();
            ++blength;
            bcast[blength] = pnode[j]->Gap();
            ++blength;
          }
          if ((int)bcast.size() < blength + 20) bcast.resize(bcast.size() + 40);
        }  // for (mcurr=rnode_[mside].begin(); mcurr!=rnode_[mside].end();
           // ++mcurr)
        if (blength >= (int)bcast.size()) {
          std::stringstream oss;
          oss << "***ERR*** "
                 "MoertelT::Interface::ProjectNodes_SlavetoMaster_Orthogonal:\n"
              << "***ERR*** Overflow in communication buffer occured\n"
              << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
          throw MoertelT::ReportError(oss);
        }
      }  // if (proc==lComm()->MyPID())

      Teuchos::broadcast<LO, int>(*lcomm_, proc, 1, &blength);
      if (proc != lcomm_->getRank()) bcast.resize(blength);
      Teuchos::broadcast<LO, double>(*lcomm_, proc, blength, &bcast[0]);

      if (proc != lcomm_->getRank()) {
        int i;
        for (i = 0; i < blength;) {
          int nid = (int)bcast[i];
          ++i;
          Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)> snode =
              GetNodeView(nid);
          int npnode = (int)bcast[i];
          ++i;
          for (int j = 0; j < npnode; ++j) {
            int sid = (int)bcast[i];
            ++i;
            double* xi = &bcast[i];
            ++i;
            ++i;
            int orthseg = (int)bcast[i];
            ++i;
            double gap = bcast[i];
            ++i;
            Teuchos::RCP<MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)> seg =
                GetSegmentView(sid);
            MoertelT::MOERTEL_TEMPLATE_CLASS(ProjectedNodeT)* pnode =
                new MoertelT::MOERTEL_TEMPLATE_CLASS(ProjectedNodeT)(
                    *snode, xi, seg.get(), orthseg);
            snode->SetProjectedNode(pnode);
            snode->SetGap(gap);
          }
        }
        if (i != blength) {
          std::stringstream oss;
          oss << "***ERR*** "
                 "MoertelT::Interface::ProjectNodes_SlavetoMaster_Orthogonal:\n"
              << "***ERR*** Mismatch in dimension of recv buffer: " << i
              << " != " << blength << "\n"
              << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
          throw MoertelT::ReportError(oss);
        }
      }  // if (proc!=lComm()->MyPID())
    }    // for (int proc=0; proc<lComm()->NumProc(); ++proc)
    bcast.clear();
  }  // if (lComm()->NumProc()>1)

#if 1
  // Postprocess the projections
  // The slave side of the interface might be larger then the master side
  // of the interface so not all slave nodes have a projection.
  // For those slave nodes without a projection attached to a slave segment
  // which overlaps with the master side, Lagrange multipliers have to be
  // introduced. This is done by checking all nodes without a projection
  // whether they are attached to some slave segment on which another node
  // HAS a projection. If this case is found, a pseudo ProjectedNode is
  // introduced for that node.
  for (scurr = rnode_[sside].begin(); scurr != rnode_[sside].end(); ++scurr) {
    Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)> snode = scurr->second;
    // do only my own nodes

    // don't do anything on nodes that already have a projection
    if (snode->GetProjectedNode() != Teuchos::null) continue;

    // get segments adjacent to this node
    int nseg                                          = snode->Nseg();
    MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)** segs = snode->Segments();
    // loop segments and check for other nodes with projection
    bool foundit = false;
    for (int i = 0; i < nseg; ++i) {
      int nnode                                       = segs[i]->Nnode();
      MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)** nodes = segs[i]->Nodes();
      for (int j = 0; j < nnode; ++j)
        if (nodes[j]->GetProjectedNode() != Teuchos::null)
          if (nodes[j]->GetProjectedNode()->Segment()) {
            foundit = true;
            break;
          }
      if (foundit) break;
    }
    if (foundit) {
#if 0
      std::cout << "Node without projection:\n" << *snode;        
      std::cout << "...get's lagrange multipliers\n\n";
#endif
      MoertelT::MOERTEL_TEMPLATE_CLASS(ProjectedNodeT)* pnode =
          new MoertelT::MOERTEL_TEMPLATE_CLASS(ProjectedNodeT)(
              *snode, NULL, NULL);
      snode->SetProjectedNode(pnode);
      snode->SetGap(0.);
    }
  }
#endif

  return true;
}
