//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Moertel_ExplicitTemplateInstantiation.hpp"

#include "Moertel_InterfaceT.hpp"
#include "Moertel_NodeT.hpp"
#include "Moertel_PnodeT.hpp"
#include "Moertel_UtilsT.hpp"

/*----------------------------------------------------------------------*
 |  ctor (public)                                            mwgee 06/05|
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT
MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)::NodeT(
    int           Id,
    const double* x,
    int           ndof,
    const int*    dof,
    bool          isonboundary,
    int           out)
    : Id_(Id),
      outputlevel_(out),
      iscorner_(false),
      isonboundary_(isonboundary),
      Drow_(Teuchos::null),
      Mrow_(Teuchos::null),
      Mmodrow_(Teuchos::null)
{
  seg_.resize(0);
  segptr_.resize(0);

  for (int i = 0; i < 3; ++i) {
    x_[i] = x[i];
    n_[i] = 0.0;
  }

  dof_.resize(ndof);
  for (int i = 0; i < ndof; ++i) dof_[i] = dof[i];

  LMdof_.resize(0);
  pnode_.resize(0);
  supportedby_.clear();
  gap_ = 0.;
}

/*----------------------------------------------------------------------*
 |  ctor (public)                                            mwgee 07/05|
 |  This constructor should not be used by the user, it is used         |
 |  used internally                                                     |
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT
MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)::NodeT(int out)
    : Id_(-1),
      outputlevel_(out),
      iscorner_(false),
      isonboundary_(false),
      Drow_(Teuchos::null),
      Mrow_(Teuchos::null),
      Mmodrow_(Teuchos::null)
{
  seg_.resize(0);
  segptr_.resize(0);

  dof_.clear();
  LMdof_.resize(0);

  for (int i = 0; i < 3; ++i) {
    n_[i] = 0.0;
    x_[i] = 0.0;
  }

  pnode_.resize(0);
  gap_ = 0.;
}

/*----------------------------------------------------------------------*
 |  copy-ctor (public)                                       mwgee 06/05|
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT
MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)::NodeT(
    const MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT) & old)
    : supportedby_(old.supportedby_)
{
  Id_           = old.Id();
  outputlevel_  = old.outputlevel_;
  iscorner_     = old.iscorner_;
  isonboundary_ = old.isonboundary_;
  gap_          = old.gap_;

  for (int i = 0; i < 3; ++i) {
    x_[i] = old.x_[i];
    n_[i] = old.n_[i];
  }

  if (old.dof_.size())
    dof_ = old.dof_;
  else
    dof_.resize(0);

  if (old.LMdof_.size())
    LMdof_ = old.LMdof_;
  else
    LMdof_.resize(0);

  if (old.seg_.size()) {
    seg_.resize(old.seg_.size());
    seg_ = old.seg_;
  } else
    seg_.resize(0);

  if (old.segptr_.size()) {
    segptr_ = old.segptr_;
  } else
    segptr_.resize(0);

  pnode_.resize(old.pnode_.size());
  for (int i = 0; i < (int)pnode_.size(); ++i)
    if (old.pnode_[i].get() != NULL) {
      pnode_[i] = Teuchos::rcp(new MoertelT::MOERTEL_TEMPLATE_CLASS(
          ProjectedNodeT)(*(old.pnode_[i])));
    }

  if (old.Drow_ != Teuchos::null) {
    std::map<int, double>* tmp = new std::map<int, double>(*(old.Drow_));
    Drow_                      = Teuchos::rcp(tmp);
  } else
    Drow_ = Teuchos::null;

  if (old.Mrow_ != Teuchos::null) {
    std::map<int, double>* tmp = new std::map<int, double>(*(old.Mrow_));
    Mrow_                      = Teuchos::rcp(tmp);
  } else
    Mrow_ = Teuchos::null;
}

/*----------------------------------------------------------------------*
 | pack data in this node into a vector                      mwgee 07/05|
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT
double* MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)::Pack(int* size)
{
  // *size = *size + Id_ + x_[3] + n_[3] + dof_.size() + ndof_*sizeof(double) +
  // seg_.size() + nseg_*sizeof(double) + iscorner_ + isonboundary_ + gap_
  *size         = 1 + 1 + 3 + 3 + 1 + dof_.size() + 1 + seg_.size() + 1 + 1 + 1;
  double* pack  = new double[*size];
  int     count = 0;

  pack[count++] = (double)(*size);
  pack[count++] = (double)Id_;
  for (int i = 0; i < 3; ++i) pack[count++] = x_[i];
  for (int i = 0; i < 3; ++i) pack[count++] = n_[i];
  pack[count++] = (double)dof_.size();
  for (int i = 0; i < (int)dof_.size(); ++i) pack[count++] = (double)dof_[i];
  pack[count++] = (double)seg_.size();
  for (int i = 0; i < (int)seg_.size(); ++i) pack[count++] = (double)seg_[i];
  pack[count++] = (double)(iscorner_);
  pack[count++] = (double)(isonboundary_);
  pack[count++] = gap_;

  if (count != *size) {
    std::stringstream oss;
    oss << "***ERR*** MoertelT::NodeT::Pack:\n"
        << "***ERR*** mismatch in packing size\n"
        << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
    throw ReportError(oss);
  }

  return pack;
}

/*----------------------------------------------------------------------*
 | unpack data from a vector in this class                   mwgee 07/05|
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT
bool MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)::UnPack(double* pack)
{
  int count = 0;
  int size  = (int)pack[count++];
  Id_       = (int)pack[count++];
  for (int i = 0; i < 3; ++i) x_[i] = pack[count++];
  for (int i = 0; i < 3; ++i) n_[i] = pack[count++];
  dof_.resize((int)pack[count++]);
  for (int i = 0; i < (int)dof_.size(); ++i) dof_[i] = (int)pack[count++];
  seg_.resize((int)pack[count++]);
  for (int i = 0; i < (int)seg_.size(); ++i) seg_[i] = (int)pack[count++];
  iscorner_     = (bool)pack[count++];
  isonboundary_ = (bool)pack[count++];
  gap_          = pack[count++];

  if (count != size) {
    std::stringstream oss;
    oss << "***ERR*** MoertelT::NodeT::UnPack:\n"
        << "***ERR*** mismatch in packing size\n"
        << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
    throw ReportError(oss);
  }

  return true;
}

/*----------------------------------------------------------------------*
 |  dtor (public)                                            mwgee 06/05|
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT
MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)::~NodeT()
{
  dof_.clear();
  LMdof_.clear();
  seg_.clear();
  segptr_.clear();
  pnode_.clear();
  Drow_    = Teuchos::null;
  Mrow_    = Teuchos::null;
  Mmodrow_ = Teuchos::null;
  supportedby_.clear();
}

/*----------------------------------------------------------------------*
 |  Reset() (public)                                           gah 07/10 |
 | Clear internal state in preparation for re-integration                |
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT
void MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)::Reset()
{
  // Clear up any projected node info
  pnode_.clear();

  // Delete internal state of M and D integration
  LMdof_.clear();
  Drow_    = Teuchos::null;
  Mrow_    = Teuchos::null;
  Mmodrow_ = Teuchos::null;
  supportedby_.clear();
}

/*----------------------------------------------------------------------*
 |  << operator                                              mwgee 06/05|
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT
std::ostream&
operator<<(
    std::ostream& os,
    const MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT) & node)
{
  node.Print();
  return (os);
}

/*----------------------------------------------------------------------*
 |  print node                                               mwgee 06/05|
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT
bool MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)::Print() const
{
  std::cout << "NodeT " << std::setw(6) << Id_ << "\tCoords ";
  for (int i = 0; i < 3; ++i) std::cout << std::setw(12) << x_[i] << " ";

  std::cout << "Normal ";
  for (int i = 0; i < 3; ++i) std::cout << std::setw(12) << n_[i] << " ";

  std::cout << "#Dofs " << dof_.size() << " Dofs ";
  for (int i = 0; i < (int)dof_.size(); ++i) std::cout << dof_[i] << " ";

  std::cout << "#LMDofs " << LMdof_.size();
  if (LMdof_.size()) {
    std::cout << " LMDofs ";
    for (int i = 0; i < (int)LMdof_.size(); ++i) std::cout << LMdof_[i] << " ";
  }

  if (IsCorner()) std::cout << " is shared among 1D interfaces";

  if (IsOnBoundary()) {
    std::cout << " is boundary of 2D-interface";
    std::cout << ", member of " << NSupportSet() << " support sets";
  }

  std::cout << std::endl;
  return true;
}

/*----------------------------------------------------------------------*
 |  set lagrange multiplier dof id                           mwgee 07/05|
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT
bool MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)::SetLagrangeMultiplierId(int LMId)
{
  // first check whether this dof has been set before
  // if so, do nothing
  for (int i = 0; i < (int)LMdof_.size(); ++i)
    if (LMdof_[i] == LMId) return true;

  // resize the vector to tak the new dof
  LMdof_.resize(LMdof_.size() + 1);

  // put in the new dof
  LMdof_[LMdof_.size() - 1] = LMId;
  return true;
}

/*----------------------------------------------------------------------*
 |  add a segment id to my adjacency list                    mwgee 06/05|
 | (checking whether already present)                                   |
 | WRN: This is NOT a collective call!                                  |
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT
bool MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)::AddSegment(int sid)
{
  if (seg_.size()) {
    // search whether sid already exists in seg_
    for (int i = 0; i < (int)seg_.size(); ++i)
      if (sid == seg_[i]) return true;

    // resize seg_
    seg_.resize(seg_.size() + 1);

    // add new sid to seg_
    seg_[seg_.size() - 1] = sid;
    return true;
  } else {
    seg_.resize(seg_.size() + 1);
    seg_[seg_.size() - 1] = sid;
  }
  return true;
}

/*----------------------------------------------------------------------*
 |                                                           mwgee 07/05|
 | construct ptrs to redundant segments from my segment id list         |
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT
bool MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)::GetPtrstoSegments(
    MoertelT::MOERTEL_TEMPLATE_CLASS(InterfaceT) & interface)
{
  if (!interface.IsComplete()) return false;
  if (!interface.lComm()) return true;
  if (!seg_.size()) return false;

  // vector segptr_ might already exist, build new
  segptr_.resize(seg_.size());

  for (int i = 0; i < (int)seg_.size(); ++i) {
    segptr_[i] = interface.GetSegmentView(seg_[i]).get();
    if (!segptr_[i]) {
      std::stringstream oss;
      oss << "***ERR*** MoertelT::NodeT::GetPtrstoSegments:\n"
          << "***ERR*** Interface " << interface.Id()
          << ": GetSegmentView failed\n"
          << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
      throw ReportError(oss);
    }
  }
  return true;
}

MOERTEL_TEMPLATE_STATEMENT
bool MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)::GetPtrstoSegments(
    MoertelT::MOERTEL_TEMPLATE_CLASS(InterfaceT) & iface)
{
  if (!iface.IsComplete()) return false;
  if (iface.lComm() == Teuchos::null) return true;
  if (!seg_.size()) return false;

  // vector segptr_ might already exist, build new
  segptr_.resize(seg_.size());

  for (int i = 0; i < (int)seg_.size(); ++i) {
    segptr_[i] = iface.GetSegmentView(seg_[i]).get();
    if (!segptr_[i]) {
      std::stringstream oss;
      oss << "***ERR*** MoertelT::NodeT::GetPtrstoSegments:\n"
          << "***ERR*** Interface " << iface.Id() << ": GetSegmentView failed\n"
          << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
      throw ReportError(oss);
    }
  }
  return true;
}

/*----------------------------------------------------------------------*
 |  build nodal normal                                       mwgee 07/05|
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT
bool MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)::BuildAveragedNormal()
{
  // get segments adjacent to me
  int  nseg = Nseg();
  int* sid  = SegmentIds();

  for (int i = 0; i < 3; ++i) n_[i] = 0.0;
  double weight = 0.0;

#if 0
  std::cout << "Building normal for node\n" << *this;
#endif

  for (int i = 0; i < nseg; ++i) {
    MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)* seg = segptr_[i];

#if 0
	std::cout << "Now averaging from Segment\n" << *seg;
	std::cout << "Finished writing segment data." << std::endl;
#endif

    if (!seg) {
      std::stringstream oss;
      oss << "***ERR*** MoertelT::NodeT::BuildAveragedNormal:\n"
          << "***ERR*** NodeT " << Id() << ": Segment " << sid[i]
          << " not found -> fatal\n"
          << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
      throw ReportError(oss);
    }
    double* n   = seg->BuildNormalAtNode(Id());
    double  wgt = seg->Area();

    // add weighted normal to n_
    for (int i = 0; i < 3; ++i) n_[i] += wgt * n[i];

    // add weight to total weight
    weight += wgt;

    delete[] n;
    n = NULL;
  }  // for (int i=0; i<nseg; ++i)

  double length = sqrt(n_[0] * n_[0] + n_[1] * n_[1] + n_[2] * n_[2]);
  for (int i = 0; i < 3; ++i) n_[i] /= length;

#if 0
  std::cout << "NodeT " << Id() << ":"
       << " normal is " << std::setw(15) << n_[0] 
       << "   "<< std::setw(15) << n_[1] << "   " << std::setw(15) << n_[2] << std::endl;
#endif

  return true;
}

/*----------------------------------------------------------------------*
 |  set a projected node                                     mwgee 07/05|
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT
bool MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)::SetProjectedNode(
    MoertelT::MOERTEL_TEMPLATE_CLASS(ProjectedNodeT) * pnode)
{
  pnode_.resize(pnode_.size() + 1);
  pnode_[pnode_.size() - 1] = Teuchos::rcp(pnode);
  return true;
}

/*----------------------------------------------------------------------*
 |  get projected nodes                                      mwgee 07/05|
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT
Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(ProjectedNodeT)>*
    MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)::GetProjectedNode(int& length)
{
  length = pnode_.size();
  if (length)
    return &(pnode_[0]);
  else
    return NULL;
}

/*----------------------------------------------------------------------*
 |  get projected node                                       mwgee 07/05|
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT
Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(ProjectedNodeT)>
    MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)::GetProjectedNode()
{
  int length = pnode_.size();
  if (length)
    return pnode_[0];
  else
    return Teuchos::null;
}

/*----------------------------------------------------------------------*
 |  add a value to the Drow_ map                             mwgee 11/05|
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT
void MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)::AddDValue(double val, int col)
{
  if (Drow_ == Teuchos::null) Drow_ = Teuchos::rcp(new std::map<int, double>());

  std::map<int, double>* Dmap = Drow_.get();

  (*Dmap)[col] += val;

  return;
}

/*----------------------------------------------------------------------*
 |  add a value to the Drow_ map                             mwgee 11/05|
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT
void MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)::AddMValue(double val, int col)
{
  if (Mrow_ == Teuchos::null) Mrow_ = Teuchos::rcp(new std::map<int, double>());

  std::map<int, double>* Mmap = Mrow_.get();

  (*Mmap)[col] += val;

  return;
}

/*----------------------------------------------------------------------*
 |  add a value to the Drow_ map                             mwgee 02/06|
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT
void MoertelT::MOERTEL_TEMPLATE_CLASS(
    NodeT)::AddMmodValue(int row, double val, int col)
{
  if (Mmodrow_ == Teuchos::null)
    Mmodrow_ = Teuchos::rcp(new std::vector<std::map<int, double>>(Ndof()));

  if ((int)Mmodrow_->size() <= row) Mmodrow_->resize(row + 1);

  std::map<int, double>& Mmap = (*Mmodrow_)[row];

  Mmap[col] += val;

  return;
}

// ETI
#ifdef HAVE_MOERTEL_INST_DOUBLE_INT_INT
template bool
MoertelT::NodeT<3, double, int, int, KokkosNode>::
    GetPtrstoSegments<double, int, int, KokkosNode>(
        MoertelT::InterfaceT<double, int, int, KokkosNode>& interface);
template bool
MoertelT::NodeT<2, double, int, int, KokkosNode>::
    GetPtrstoSegments<double, int, int, KokkosNode>(
        MoertelT::InterfaceT<double, int, int, KokkosNode>& interface);
#endif
#ifdef HAVE_MOERTEL_INST_DOUBLE_INT_LONGLONGINT
template bool
MoertelT::NodeT<3, double, int, long long, KokkosNode>::
    GetPtrstoSegments<double, int, long long, KokkosNode>(
        MoertelT::InterfaceT<double, int, long long, KokkosNode>& interface);
template bool
MoertelT::NodeT<2, double, int, long long, KokkosNode>::
    GetPtrstoSegments<double, int, long long, KokkosNode>(
        MoertelT::InterfaceT<double, int, long long, KokkosNode>& interface);
#endif
