//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Moertel_ExplicitTemplateInstantiation.hpp"

#include "Moertel_InterfaceT.hpp"
#include "Moertel_SegmentT.hpp"
#include "Moertel_UtilsT.hpp"

/*----------------------------------------------------------------------*
 |  ctor (public)                                            mwgee 06/05|
 |  id               (in)  a unique segment id                          |
 |  nnode            (in)  number of nodes on this segment              |
 |  nodeId           (in)  unique node ids of nodes on this segment     |
 |                         nodeIds have to be sorted on input such that |
 |                         1D case: end nodes of segments first going   |
 |                                  through segment in mathematical     |
 |                                  positive sense. This is             |
 |                                  important to compute the direction  |
 |                                  of the outward normal of the segment|
 |                         2D case: corner nodes of segment first in    |
 |                                  counterclockwise order              |
 |                                  This is                             |
 |                                  important to compute the direction  |
 |                                  of the outward normal of the segment|
 *----------------------------------------------------------------------*/
SEGMENT_TEMPLATE_STATEMENT
MoertelT::SEGMENT_TEMPLATE_CLASS(
    SegmentT)::Segment(int id, int nnode, int* nodeId, int outlevel)
    : Id_(id),
      outputlevel_(outlevel),
      stype_(MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)::seg_none)
{
  nodeId_.resize(nnode);
  for (int i = 0; i < nnode; ++i) nodeId_[i] = nodeId[i];
  nodeptr_.resize(0);
}

SEGMENT_TEMPLATE_STATEMENT
MoertelT::SEGMENT_TEMPLATE_CLASS(
    SegmentT)::Segment(int id, const std::vector<int>& nodev, int outlevel)
    : Id_(id),
      outputlevel_(outlevel),
      stype_(MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)::seg_none),
      nodeId_(nodev)
{
  nodeptr_.resize(0);
}

/*----------------------------------------------------------------------*
 | base class constructor                                    mwgee 07/05|
 *----------------------------------------------------------------------*/
SEGMENT_TEMPLATE_STATEMENT
MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)::Segment(int outlevel)
    : Id_(-1),
      outputlevel_(outlevel),
      stype_(MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)::seg_none)
{
  nodeId_.resize(0);
  nodeptr_.resize(0);
}

/*----------------------------------------------------------------------*
 | base class copy ctor                                      mwgee 07/05|
 *----------------------------------------------------------------------*/
SEGMENT_TEMPLATE_STATEMENT
MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)::Segment(
    MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT) & old)
{
  Id_          = old.Id_;
  outputlevel_ = old.outputlevel_;
  stype_       = old.stype_;
  nodeId_      = old.nodeId_;
  nodeptr_     = old.nodeptr_;

  // copy the functions
  // this is not a deep copy but we simply copy the refcountptr
  std::map<int, Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(FunctionT)>>::
      iterator curr;
  for (curr = old.functions_.begin(); curr != old.functions_.end(); ++curr) {
    if (curr->second == Teuchos::null) {
      std::stringstream oss;
      oss << "***ERR*** MOERTEL::Segment::BaseClone(MOERTEL::Segment& old):\n"
          << "***ERR*** function id " << curr->first << " is null\n"
          << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
      throw ReportError(oss);
    }
    Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(FunctionT)> newfunc =
        curr->second;
    functions_.insert(
        std::pair<
            int,
            Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(FunctionT)>>(
            curr->first, newfunc));
  }
}

/*----------------------------------------------------------------------*
 | base class destructor                                     mwgee 07/05|
 *----------------------------------------------------------------------*/
SEGMENT_TEMPLATE_STATEMENT
MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)::~Segment()
{
  nodeId_.clear();
  nodeptr_.clear();
  functions_.clear();
}

/*----------------------------------------------------------------------*
 |  print segment                                            mwgee 06/05|
 *----------------------------------------------------------------------*/
SEGMENT_TEMPLATE_STATEMENT
bool MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)::Print() const
{
  std::cout << "Segment " << std::setw(6) << Id_;
  if (stype_ == MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)::seg_Linear1D)
    std::cout << " Typ Linear1D   ";
  if (stype_ == MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)::seg_BiLinearQuad)
    std::cout << " Typ BiLinearQuad ";
  if (stype_ == MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)::seg_BiLinearTri)
    std::cout << " Typ BiLinearTri";
  if (stype_ == MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)::seg_none)
    std::cout << " Typ NONE       ";
  std::cout << " #Nodes " << nodeId_.size() << " Nodes: ";
  for (int i = 0; i < (int)nodeId_.size(); ++i)
    std::cout << std::setw(6) << nodeId_[i] << "  ";
  std::cout << "  #Functions " << functions_.size() << "  Types: ";
  std::map<int, Teuchos::RCP<MoertelT::SEGMENT_TEMPLATE_CLASS(FunctionT)>>::
      const_iterator curr;
  for (curr = functions_.begin(); curr != functions_.end(); ++curr)
    std::cout << curr->second->Type() << "  ";
  std::cout << std::endl;
  return true;
}

/*----------------------------------------------------------------------*
 | attach a certain shape function to this segment           mwgee 06/05|
 | the user is not supposed to destroy func!                            |
 | the user can set func to several segments!                           |
 *----------------------------------------------------------------------*/
SEGMENT_TEMPLATE_STATEMENT
bool MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)::SetFunction(
    int id,
    MoertelT::SEGMENT_TEMPLATE_CLASS(FunctionT) * func)
{
  if (id < 0) {
    std::cout << "***ERR*** MOERTEL::Segment::SetFunction:\n"
              << "***ERR*** id = " << id << " < 0 (out of range)\n"
              << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
    return false;
  }
  if (!func) {
    std::cout << "***ERR*** MOERTEL::Segment::SetFunction:\n"
              << "***ERR*** func = NULL on input\n"
              << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
    return false;
  }

  // check for existing function with this id and evtentually overwrite
  std::map<int, Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(FunctionT)>>::
      iterator curr = functions_.find(id);
  if (curr != functions_.end()) {
    curr->second = Teuchos::null;
    curr->second = Teuchos::rcp(func->Clone());
    return true;
  }
  Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(FunctionT)> newfunc =
      Teuchos::rcp(func->Clone());
  functions_.insert(
      std::pair<int, Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(FunctionT)>>(
          id, newfunc));
  return true;
}

#if 0
/*----------------------------------------------------------------------*
 | attach a certain shape function to this segment           mwgee 11/05|
 | the user is not supposed to destroy func!                            |
 | the user can set func to several segments!                           |
 *----------------------------------------------------------------------*/
bool MOERTEL::Segment::SetFunction(int id, Teuchos::RCP<MOERTEL::Function> func)
{ 
  if (id<0)
  {
	std::cout << "***ERR*** MOERTEL::Segment::SetFunction:\n"
         << "***ERR*** id = " << id << " < 0 (out of range)\n"
         << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
    return false;
  }
  if (func==Teuchos::null)
  {
	std::cout << "***ERR*** MOERTEL::Segment::SetFunction:\n"
         << "***ERR*** func = NULL on input\n"
         << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
    return false;
  }
  
  // check for existing function with this id and evtentually overwrite
  std::map<int,Teuchos::RCP<MOERTEL::Function> >::iterator curr = functions_.find(id);
  if (curr != functions_.end())
  {
    curr->second = func;
    return true;
  }
  Teuchos::RCP<MOERTEL::Function> newfunc = func;
  functions_.insert(pair<int,Teuchos::RCP<MOERTEL::Function> >(id,newfunc));
  return true;
}
#endif

/*----------------------------------------------------------------------*
 | evaluate the shape function id at the point xi            mwgee 06/05|
 | id     (in)   id of the function to evaluate                         |
 | xi     (in)   natural coordinates -1<xi<1 where to eval the function |
 | val    (out)  function values, if NULL on input, no evaluation       |
 | valdim (in)   dimension of val                                       |
 | deriv  (out)  derivatives of functions at xi, if NULL on input,      |
 |               no evaluation                                          |
 *----------------------------------------------------------------------*/
SEGMENT_TEMPLATE_STATEMENT
bool MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)::EvaluateFunction(
    int           id,
    const double* xi,
    double*       val,
    int           valdim,
    double*       deriv)
{
  std::map<int, Teuchos::RCP<MoertelT::SEGMENT_TEMPLATE_CLASS(FunctionT)>>::
      iterator curr = functions_.find(id);
  if (curr == functions_.end()) {
    std::stringstream oss;
    oss << "***ERR*** "
           "MoertelT::MOERTEL_TEMPLATE_CLASS(SegmentT)::EvaluateFunction:\n"
        << "***ERR*** function id " << id << " does not exist on this segment\n"
        << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
    throw ReportError(oss);
  }
  curr->second->EvaluateFunction(*this, xi, val, valdim, deriv);

  return true;
}

/*----------------------------------------------------------------------*
 |  << operator                                              mwgee 06/05|
 *----------------------------------------------------------------------*/
SEGMENT_TEMPLATE_STATEMENT
std::ostream&
operator<<(
    std::ostream& os,
    const MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT) & seg)
{
  seg.Print();
  return os;
}

/*----------------------------------------------------------------------*
 | get local numbering id for global node id on this segment mwgee 07/05|
 | return -1 of nid is not adjacent to this segment                     |
 *----------------------------------------------------------------------*/
SEGMENT_TEMPLATE_STATEMENT
int MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)::GetLocalNodeId(int nid)
{
  int lid = -1;
  for (int i = 0; i < Nnode(); ++i)
    if (nodeId_[i] == nid) {
      lid = i;
      break;
    }
  if (lid < 0) {
    std::stringstream oss;
    oss << "***ERR*** MOERTEL::Segment::GetLocalNodeId:\n"
        << "***ERR*** cannot find node " << nid << " in segment " << this->Id()
        << " list of nodes\n"
        << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
    throw ReportError(oss);
  }
  return lid;
}

/*----------------------------------------------------------------------*
 | build an outward normal at a node adjacent to this        mwgee 07/05|
 *----------------------------------------------------------------------*/
SEGMENT_TEMPLATE_STATEMENT
double* MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)::BuildNormalAtNode(int nid)
{
  // find this node in my list of nodes and get local numbering for it
  int lid = GetLocalNodeId(nid);

  // depending on what type of segment I am get local coordinates
  double xi[2];
  LocalCoordinatesOfNode(lid, xi);

  // build an outward unit normal at xi and return it
  return BuildNormal(xi);
}

/*----------------------------------------------------------------------*
 |                                                           mwgee 07/05|
 | construct ptrs to redundant nodes from my node id list               |
 *----------------------------------------------------------------------*/
SEGMENT_TEMPLATE_STATEMENT
bool MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)::GetPtrstoNodes(
    MoertelT::SEGMENT_TEMPLATE_CLASS(InterfaceT) & interface)
{
  if (!interface.IsComplete()) return false;
  if (!interface.lComm()) return true;
  if (!nodeId_.size()) return false;

  // vector nodeptr_ might already exist, recreate it
  nodeptr_.clear();
  nodeptr_.resize(nodeId_.size());

  for (int i = 0; i < (int)nodeId_.size(); ++i) {
    nodeptr_[i] = interface.GetNodeView(nodeId_[i]).get();
    if (!nodeptr_[i]) {
      std::stringstream oss;
      oss << "***ERR*** MOERTEL::Segment::GetPtrstoNodes:\n"
          << "***ERR*** interface " << interface.Id() << " GetNodeView failed\n"
          << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
      throw ReportError(oss);
    }
  }
  return true;
}

SEGMENT_TEMPLATE_STATEMENT
bool MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)::GetPtrstoNodes(
    MoertelT::InterfaceT<ST, LO, GO, N>& interface)
{
  if (!interface.IsComplete()) return false;
  if (interface.lComm() == Teuchos::null) return true;
  if (!nodeId_.size()) return false;

  // vector nodeptr_ might already exist, recreate it
  nodeptr_.clear();
  nodeptr_.resize(nodeId_.size());

  for (int i = 0; i < (int)nodeId_.size(); ++i) {
    nodeptr_[i] = interface.GetNodeView(nodeId_[i]).get();
    if (!nodeptr_[i]) {
      std::stringstream oss;
      oss << "***ERR*** MOERTEL::Segment::GetPtrstoNodes:\n"
          << "***ERR*** interface " << interface.Id() << " GetNodeView failed\n"
          << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
      throw ReportError(oss);
    }
  }
  return true;
}

/*----------------------------------------------------------------------*
 |                                                           mwgee 10/05|
 | construct ptrs to nodes from vector                                  |
 *----------------------------------------------------------------------*/
SEGMENT_TEMPLATE_STATEMENT
bool MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)::GetPtrstoNodes(
    std::vector<MoertelT::SEGMENT_TEMPLATE_CLASS(NodeT) *>& nodes)
{
  if (!nodeId_.size()) return false;

  // vector nodeptr_ might already exist, recreate it
  nodeptr_.clear();
  nodeptr_.resize(nodeId_.size());

  for (int i = 0; i < (int)nodeId_.size(); ++i) {
    bool foundit = true;
    for (int j = 0; j < (int)nodes.size(); ++j)
      if (nodes[j]->Id() == nodeId_[i]) {
        foundit     = true;
        nodeptr_[i] = nodes[j];
        break;
      }
    if (!foundit) {
      std::stringstream oss;
      oss << "***ERR*** MOERTEL::Segment::GetPtrstoNodes:\n"
          << "***ERR*** cannot find node " << nodeId_[i] << " in vector\n"
          << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
      throw ReportError(oss);
    }
  }
  return true;
}

/*----------------------------------------------------------------------*
 |                                                           mwgee 07/05|
 | return type of function with certain id                              |
 *----------------------------------------------------------------------*/
SEGMENT_TEMPLATE_STATEMENT
MoertelT::SEGMENT_TEMPLATE_CLASS(FunctionT)::FunctionType
    MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)::FunctionType(int id)
{
  // find the function with id id
  std::map<int, Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(FunctionT)>>::
      iterator curr = functions_.find(id);
  if (curr == functions_.end())
    return MoertelT::MOERTEL_TEMPLATE_CLASS(FunctionT)::func_none;
  else
    return curr->second->Type();
}

// Template arguments
int*
MoertelT::Linear1DSeg::Pack(int* size)
{
  // note: first there has to be the size and second there has to be the type
  // *size = *size  + stype_ + Id_ + nodeId_.size() + nnode*sizeof(int) +
  // Nfunctions() + 2*Nfunctions()*sizeof(int)
  *size     = 1 + 1 + 1 + 1 + nodeId_.size() + 1 + 2 * Nfunctions();
  int* pack = new int[*size];

  int count = 0;

  pack[count++] = *size;
  pack[count++] = (int)stype_;
  pack[count++] = Id_;
  pack[count++] = nodeId_.size();
  for (int i = 0; i < (int)nodeId_.size(); ++i) pack[count++] = nodeId_[i];
  pack[count++] = Nfunctions();

  std::map<int, Teuchos::RCP<MOERTEL::Function>>::iterator curr;
  for (curr = functions_.begin(); curr != functions_.end(); ++curr) {
    pack[count++] = curr->first;
    pack[count++] = curr->second->Type();
  }

  if (count != *size) {
    std::stringstream oss;
    oss << "***ERR*** MOERTEL::Segment_Linear1D::Pack:\n"
        << "***ERR*** mismatch in packing size\n"
        << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
    throw ReportError(oss);
  }

  return pack;
}

int*
MoertelT::BiLinearTriSeg::Pack(int* size)
{
  // note: first there has to be the size and second there has to be the type
  // *size = *size  + stype_ + Id_ + nodeId_.size() + nnode*sizeof(int) +
  // Nfunctions() + 2*Nfunctions()*sizeof(int)
  *size     = 1 + 1 + 1 + 1 + nodeId_.size() + 1 + 2 * Nfunctions();
  int* pack = new int[*size];

  int count = 0;

  pack[count++] = *size;
  pack[count++] = (int)stype_;
  pack[count++] = Id_;
  pack[count++] = nodeId_.size();
  for (int i = 0; i < (int)nodeId_.size(); ++i) pack[count++] = nodeId_[i];
  pack[count++] = Nfunctions();

  std::map<int, Teuchos::RCP<MOERTEL::Function>>::iterator curr;
  for (curr = functions_.begin(); curr != functions_.end(); ++curr) {
    pack[count++] = curr->first;
    pack[count++] = curr->second->Type();
  }

  if (count != *size) {
    std::stringstream oss;
    oss << "***ERR*** MOERTEL::Segment_BiLinearTri::Pack:\n"
        << "***ERR*** mismatch in packing size\n"
        << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
    throw ReportError(oss);
  }

  return pack;
}

int*
MoertelT::BiLinearQuadSeg::Pack(int* size)
{
  // note: first there has to be the size and second there has to be the type
  // *size = *size  + stype_ + Id_ + nodeId_.size() + nnode*sizeof(int) +
  // Nfunctions() + 2*Nfunctions()*sizeof(int)
  *size = 1 + 1 + 1 + 1 + nodeId_.size() + 1 + 2 * Nfunctions();

  int* pack  = new int[*size];
  int  count = 0;

  pack[count++] = *size;
  pack[count++] = (int)stype_;
  pack[count++] = Id_;
  pack[count++] = nodeId_.size();
  for (int i = 0; i < (int)nodeId_.size(); ++i) pack[count++] = nodeId_[i];
  pack[count++] = Nfunctions();
  std::map<int, Teuchos::RCP<MOERTEL::Function>>::iterator curr;
  for (curr = functions_.begin(); curr != functions_.end(); ++curr) {
    pack[count++] = curr->first;
    pack[count++] = curr->second->Type();
  }

  if (count != *size) {
    std::stringstream oss;
    oss << "***ERR*** MOERTEL::Segment_BiLinearQuad::Pack:\n"
        << "***ERR*** mismatch in packing size\n"
        << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
    throw ReportError(oss);
  }

  return pack;
}

bool
MoertelT::Linear1DSeg::UnPack(int* pack)
{
  // note: first there has to be the size and second there has to be the type
  int count = 0;
  int size  = pack[count++];
  stype_    = (MOERTEL::Segment::SegmentType)pack[count++];
  Id_       = pack[count++];
  nodeId_.resize(pack[count++]);
  for (int i = 0; i < (int)nodeId_.size(); ++i) nodeId_[i] = pack[count++];

  int nfunc = pack[count++];

  for (int i = 0; i < nfunc; ++i) {
    int                id   = pack[count++];
    int                type = pack[count++];
    MOERTEL::Function* func = MOERTEL::AllocateFunction(
        (MOERTEL::Function::FunctionType)type, OutLevel());
    Teuchos::RCP<MOERTEL::Function> rcptrfunc = Teuchos::rcp(func);
    functions_.insert(
        std::pair<int, Teuchos::RCP<MOERTEL::Function>>(id, rcptrfunc));
  }

  if (count != size) {
    std::stringstream oss;
    oss << "***ERR*** MOERTEL::Segment_Linear1D::UnPack:\n"
        << "***ERR*** mismatch in packing size\n"
        << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
    throw ReportError(oss);
  }

  return true;
}

bool
MoertelT::BiLinearTriSeg::UnPack(int* pack)
{
  // note: first there has to be the size and second there has to be the type
  int count = 0;
  int size  = pack[count++];
  stype_    = (MOERTEL::Segment::SegmentType)pack[count++];
  Id_       = pack[count++];
  nodeId_.resize(pack[count++]);
  for (int i = 0; i < (int)nodeId_.size(); ++i) nodeId_[i] = pack[count++];

  int nfunc = pack[count++];

  for (int i = 0; i < nfunc; ++i) {
    int                id   = pack[count++];
    int                type = pack[count++];
    MOERTEL::Function* func = MOERTEL::AllocateFunction(
        (MOERTEL::Function::FunctionType)type, OutLevel());
    Teuchos::RCP<MOERTEL::Function> rcptrfunc = Teuchos::rcp(func);
    functions_.insert(
        std::pair<int, Teuchos::RCP<MOERTEL::Function>>(id, rcptrfunc));
  }

  if (count != size) {
    std::stringstream oss;
    oss << "***ERR*** MOERTEL::Segment_BiLinearTri::UnPack:\n"
        << "***ERR*** mismatch in packing size\n"
        << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
    throw ReportError(oss);
  }

  return true;
}

bool
MoertelT::BiLinearQuadSeg::UnPack(int* pack)
{
  // note: first there has to be the size and second there has to be the type
  int count = 0;
  int size  = pack[count++];
  stype_    = (MOERTEL::Segment::SegmentType)pack[count++];
  Id_       = pack[count++];
  nodeId_.resize(pack[count++]);
  for (int i = 0; i < (int)nodeId_.size(); ++i) nodeId_[i] = pack[count++];
  int nfunc = pack[count++];
  for (int i = 0; i < nfunc; ++i) {
    int                id   = pack[count++];
    int                type = pack[count++];
    MOERTEL::Function* func = MOERTEL::AllocateFunction(
        (MOERTEL::Function::FunctionType)type, OutLevel());
    Teuchos::RCP<MOERTEL::Function> rcptrfunc = Teuchos::rcp(func);
    functions_.insert(
        std::pair<int, Teuchos::RCP<MOERTEL::Function>>(id, rcptrfunc));
  }

  if (count != size) {
    std::stringstream oss;
    oss << "***ERR*** MOERTEL::Segment_BiLinearQuad::UnPack:\n"
        << "***ERR*** mismatch in packing size\n"
        << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
    throw ReportError(oss);
  }

  return true;
}

bool
MoertelT::Linear1DSeg::LocalCoordinatesOfNode(int lid, double* xi)
{
  if (lid == 0)
    xi[0] = -1.0;
  else if (lid == 1)
    xi[0] = 1.0;
  else {
    std::stringstream oss;
    oss << "***ERR*** MOERTEL::Segment_Linear1D::LocalCoordinatesOfNode:\n"
        << "***ERR*** Segment " << Id() << ": node number " << lid
        << " out of range\n"
        << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
    throw ReportError(oss);
  }
  return true;
}

bool
MoertelT::BiLinearTriSeg::LocalCoordinatesOfNode(int lid, double* xi)
{
  if (lid == 0) {
    xi[0] = 0.0;
    xi[1] = 0.0;
  } else if (lid == 1) {
    xi[0] = 1.0;
    xi[1] = 0.0;
  } else if (lid == 2) {
    xi[0] = 0.0;
    xi[1] = 1.0;
  } else {
    std::stringstream oss;
    oss << "***ERR*** MOERTEL::Segment_BiLinearTri::LocalCoordinatesOfNode:\n"
        << "***ERR*** local node number " << lid << " out of range (0..2)\n"
        << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
    throw ReportError(oss);
  }
  return true;
}

bool
MoertelT::BiLinearQuadSeg::LocalCoordinatesOfNode(int lid, double* xi)
{
  if (lid == 0) {
    xi[0] = -1.;
    xi[1] = -1.;
  } else if (lid == 1) {
    xi[0] = 1.;
    xi[1] = -1.;
  } else if (lid == 2) {
    xi[0] = 1.;
    xi[1] = 1.;
  } else if (lid == 3) {
    xi[0] = -1.;
    xi[1] = 1.;
  } else {
    std::stringstream oss;
    oss << "***ERR*** MOERTEL::Segment_BiLinearQuad::LocalCoordinatesOfNode:\n"
        << "***ERR*** local node number " << lid << " out of range (0..3)\n"
        << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
    throw ReportError(oss);
  }
  return true;
}

double*
MoertelT::Linear1DSeg::BuildNormal(double* xi)
{
  // build the metric vectors at this local coordinates xi
  double g[3];
  for (int i = 0; i < 3; ++i) g[i] = 0.0;
  Metric(xi, g, NULL);

  // in 3D, the outward normal is g1 cross g2, in 2D, the normal is
  // n1 = g2 and n2 = -g1
  double* n     = new double[3];
  n[0]          = g[1];
  n[1]          = -g[0];
  n[2]          = 0.0;
  double length = sqrt(n[0] * n[0] + n[1] * n[1]);
  n[0] /= length;
  n[1] /= length;
  return n;
}

double*
MoertelT::BiLinearTriSeg::BuildNormal(double* xi)
{
  // linear triangles are planar, so we don't care were exactly to build the
  // normal

  // build base vectors
  double g1[3];
  double g2[3];
  for (int i = 0; i < 3; ++i) {
    g1[i] = Nodes()[1]->XCoords()[i] - Nodes()[0]->XCoords()[i];
    g2[i] = Nodes()[2]->XCoords()[i] - Nodes()[0]->XCoords()[i];
  }

  // build normal as their cross product
  double* n = new double[3];

  MOERTEL::cross(n, g1, g2);

  return n;
}

double*
MoertelT::BiLinearQuadSeg::BuildNormal(double* xi)
{
  // A bilinear quad in 3D can be warped, so it does matter where
  // to build the normal
  double G[3][3];
  Metric(xi, NULL, G);

  // the normal is G[2]
  double* n = new double[3];

  for (int i = 0; i < 3; ++i) n[i] = G[2][i];

  return n;
}

double
MoertelT::Linear1DSeg::Metric(double* xi, double g[], double G[][3])
{
  // get nodal coordinates
  const double* x[2];
  x[0] = nodeptr_[0]->XCoords();
  x[1] = nodeptr_[1]->XCoords();

  // get shape functions
  double val[2];
  double deriv[2];
  functions_[0]->EvaluateFunction(*this, xi, val, 2, deriv);

  double  glocal[2];
  double* gl;
  if (g)
    gl = g;
  else
    gl = glocal;

  // build covariant basis vector g = partial x / partial theta sup i
  for (int dim = 0; dim < 2; ++dim) {
    gl[dim] = 0.0;
    for (int node = 0; node < 2; ++node) gl[dim] += deriv[node] * x[node][dim];
  }

  // build metric tensor G sub ij = g sub i dot g sub j
  // in this 1D case, it's a scalar
  if (G) {
    G[0][0] = 0;
    for (int i = 0; i < 2; ++i) G[0][0] += gl[i] * gl[i];
  }

  // FIXME: look at file shell8/s8_tvmr.c & shell8/s8_static_keug.c
  // build dA as g1 cross g2
  // in this linear 1D case, it's just the length of g
  double dl = sqrt(gl[0] * gl[0] + gl[1] * gl[1]);

  return dl;
}

double
MoertelT::BiLinearTriSeg::Metric(double* xi, double g[], double G[][3])
{
  std::stringstream oss;
  oss << "***ERR*** MOERTEL::Segment_BiLinearTri::Metric:\n"
      << "***ERR*** not impl.\n"
      << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
  throw ReportError(oss);
  return 0.0;
}

double
MoertelT::BiLinearQuadSeg::Metric(double* xi, double g[], double G[][3])
{
  // get nodal coords;
  const double* x[4];
  for (int i = 0; i < 4; ++i) x[i] = nodeptr_[i]->XCoords();

  // get shape functions and derivatives at xi
  double val[4];
  double deriv[8];
  EvaluateFunction(0, xi, val, 4, deriv);

  // Build kovariant metric G1 and G2 = partial x / partial theta sup i
  for (int i = 0; i < 2; ++i)
    for (int dim = 0; dim < 3; ++dim) {
      G[i][dim] = 0.0;
      for (int node = 0; node < 4; ++node)
        G[i][dim] += deriv[node * 2 + i] * x[node][dim];
    }

  // build G3 as cross product of G1 x G2
  MOERTEL::cross(G[2], G[0], G[1]);

  // dA at this point is length of G[3] or |G1 x G2|
  double dA = MOERTEL::length(G[2], 3);
  return dA;
}

double
MoertelT::Linear1DSeg::Area()
{
  // get nodal coordinates
  const double* x[2];
  x[0] = nodeptr_[0]->XCoords();
  x[1] = nodeptr_[1]->XCoords();

  // build vector from x[0] to x[1]
  double tangent[2];
  tangent[0] = x[1][0] - x[0][0];
  tangent[1] = x[1][1] - x[0][1];

  double length = sqrt(tangent[0] * tangent[0] + tangent[1] * tangent[1]);
  return length;
}

double
MoertelT::BiLinearTriSeg::Area()
{
  double xi[2];
  xi[0] = xi[1] = 0.0;

  double* n = BuildNormal(xi);

  double a = 0.0;
  for (int i = 0; i < 3; ++i) a += n[i] * n[i];

  delete[] n;

  return (sqrt(a) / 2.0);
}

double
MoertelT::BiLinearQuadSeg::Area()
{
  double coord[4][2];
  double sqrtthreeinv = 1. / (sqrt(3.));
  coord[0][0]         = -sqrtthreeinv;
  coord[0][1]         = -sqrtthreeinv;
  coord[1][0]         = sqrtthreeinv;
  coord[1][1]         = -sqrtthreeinv;
  coord[2][0]         = sqrtthreeinv;
  coord[2][1]         = sqrtthreeinv;
  coord[3][0]         = -sqrtthreeinv;
  coord[3][1]         = sqrtthreeinv;
  double A            = 0.0;

  // create an integrator to get the gaussian points
  for (int gp = 0; gp < 4; ++gp) {
    double G[3][3];
    A += Metric(coord[gp], NULL, G);
  }
  return A;
}

#if 0
/*----------------------------------------------------------------------*
 |                                                           mwgee 07/05|
 | get ptr to function with id id                                       |
 *----------------------------------------------------------------------*/
MOERTEL::Function* MOERTEL::Segment::GetFunction(int id)
{ 
  // find the function with id id
  std::map<int,Teuchos::RCP<MOERTEL::Function> >::iterator curr = functions_.find(id);
  if (curr==functions_.end()) 
    return NULL;
  else
    return curr->second;
}
#endif

// ETI
#ifdef HAVE_MOERTEL_INST_DOUBLE_INT_INT
template bool MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)::
    GetPtrstoNodes<double, int, int, KokkosNode>(
        MoertelT::InterfaceT<double, int, int, KokkosNode>& interface);
#endif
#ifdef HAVE_MOERTEL_INST_DOUBLE_INT_LONGLONGINT
template bool MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)::
    GetPtrstoNodes<double, int, long long, KokkosNode>(
        MoertelT::InterfaceT<double, int, long long, KokkosNode>& interface);
#endif
