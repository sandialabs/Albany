//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <ctime>
#include <vector>

#include "Moertel_IntegratorT.hpp"
#include "Moertel_InterfaceT.hpp"
#include "Moertel_OverlapT.hpp"
#include "Moertel_PnodeT.hpp"
#include "Moertel_ProjectorT.hpp"
#include "Moertel_SegmentT.hpp"
#include "Moertel_UtilsT.hpp"

const double CONSTRAINT_MATRIX_ZERO = 1.0e-11;

/*----------------------------------------------------------------------*
  |  make mortar integration of this interface (3D problem)           |
 *----------------------------------------------------------------------*/
template <class ST, class LO, class GO, class N>
bool
MoertelT::InterfaceT<3, ST, LO, GO, N>::Mortar_Integrate(
    Teuchos::RCP<Teuchos::ParameterList> intparams)
{
  bool ok    = false;
  intparams_ = intparams;

  //-------------------------------------------------------------------
  // time this process
  Teuchos::Time time("Mortar_Integrate");
  time.start(true);

  //-------------------------------------------------------------------
  if (IsOneDimensional()) {
    if (gcomm_->getRank() == 0)
      std::cout << "***ERR*** MoertelT::InterfaceT::Mortar_Integrate:\n"
                << "***ERR*** This is not a 3D problem, we're in the wrong "
                   "method here!!!\n"
                << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__
                << "\n";

    return false;
  }

  //-------------------------------------------------------------------
  // interface needs to be complete
  if (!IsComplete()) {
    if (gcomm_->getRank() == 0)
      std::cout << "***ERR*** MoertelT::InterfaceT::Mortar_Integrate:\n"
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
  // interface needs to have a mortar side assigned
  if (MortarSide() == -1) {
    if (gcomm_->getRank() == 0)
      std::cout << "***ERR*** MoertelT::InterfaceT::Mortar_Integrate:\n"
                << "***ERR*** mortar side was not assigned on interface " << Id_
                << "\n"
                << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__
                << "\n";

    return false;
  }

  //-------------------------------------------------------------------
  // interface segments need to have at least one function on the mortar side
  // and two functions on the slave side
  int mside = MortarSide();
  int sside = OtherSide(mside);
  std::map<int, Teuchos::RCP<MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)>>::
      iterator scurr;

  for (scurr = seg_[mside].begin(); scurr != seg_[mside].end(); ++scurr)
    if (scurr->second->Nfunctions() < 1) {
      std::cout << "***ERR*** MoertelT::InterfaceT::Mortar_Integrate:\n"
                << "***ERR*** interface " << Id_ << ", mortar side\n"
                << "***ERR*** segment " << scurr->second->Id()
                << " needs at least 1 function set\n"
                << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__
                << "\n";
      return false;
    }

  for (scurr = seg_[sside].begin(); scurr != seg_[sside].end(); ++scurr)
    if (scurr->second->Nfunctions() < 2) {
      std::cout << "***ERR*** MoertelT::InterfaceT::Mortar_Integrate:\n"
                << "***ERR*** interface " << Id_ << ", slave side\n"
                << "***ERR*** segment " << scurr->second->Id()
                << " needs at least 2 function set\n"
                << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__
                << "\n";
      return false;
    }

  //-------------------------------------------------------------------
  // do the integration of the master and slave side
  ok = Integrate_3D();

  if (!ok) return false;

  //-------------------------------------------------------------------
  // set the flag that this interface has been successfully integrated
  isIntegrated_ = true;

  //-------------------------------------------------------------------
  // time this process
  if (OutLevel() > 5) {
    std::cout << "MoertelT::Interface " << Id() << ": Integration on proc "
              << gcomm_->getRank() << " finished in "
              << time.totalElapsedTime(true) << " sec\n";
    fflush(stdout);
  }

  //-------------------------------------------------------------------
  return true;
}

/*----------------------------------------------------------------------*
  |  make mortar integration of master/slave side in 3D (2D interface)   |
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT
bool MoertelT::MOERTEL_TEMPLATE_CLASS(InterfaceT)::Integrate_3D()
{
  if (!IsComplete()) {
    if (gcomm_->getRank() == 0)
      std::cout << "***ERR*** MoertelT::InterfaceT::Integrate_3D:\n"
                << "***ERR*** Complete() not called on interface " << Id_
                << "\n"
                << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__
                << "\n";

    return false;
  }

  if (lcomm_ == Teuchos::null) return true;

  // get the sides
  int mside = MortarSide();
  int sside = OtherSide(mside);

  // loop over all segments of slave side
  std::map<int, Teuchos::RCP<MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)>>::
      iterator scurr;

  for (scurr = rseg_[sside].begin(); scurr != rseg_[sside].end(); ++scurr) {
    // the segment to be integrated
    Teuchos::RCP<MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)> actsseg =
        scurr->second;

#if 0
    std::cout << "\n\nActive sseg id " << actsseg->Id() << "\n";
#endif

    // check whether I own at least one of the nodes on this slave segment
    const int nnode                                 = actsseg->Nnode();
    MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)** nodes = actsseg->Nodes();
    bool foundone                                   = false;

    for (int i = 0; i < nnode; ++i)
      if (NodePID(nodes[i]->Id()) == lcomm_->getRank()) {
        foundone = true;
        break;
      }

    // if none of the nodes belongs to me, do nothing on this segment
    if (!foundone) continue;

    // time this process
    // Teuchos::Time time(*lComm());
    // time.ResetStartTime();

    // loop over all segments on the master side
    std::map<int, Teuchos::RCP<MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)>>::
        iterator mcurr;

    for (mcurr = rseg_[mside].begin(); mcurr != rseg_[mside].end(); ++mcurr) {
      Teuchos::RCP<MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)> actmseg =
          mcurr->second;
#if 0
      std::cout << "Active mseg id " << actmseg->Id() << std::endl;
#endif

      // if there is an overlap, integrate the pair
      // (whether there is an overlap or not will be checked inside)
      Integrate_3D_Section(*actsseg, *actmseg);

    }  // for (mcurr=rseg_[mside].begin(); mcurr!=rseg_[mside].end(); ++mcurr)

    // std::cout << "time for this slave segment: " << time.ElapsedTime() <<
    // std::endl;

  }  // for (scurr=rseg_[sside].begin(); scurr!=rseg_[sside].end(); ++scurr)

  return true;
}

/*----------------------------------------------------------------------*
  | integrate the master/slave side's contribution from the overlap      |
  | of 2 segments (3D version) IF there is an overlap                    |
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT
bool MoertelT::MOERTEL_TEMPLATE_CLASS(InterfaceT)::Integrate_3D_Section(
    MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT) & sseg,
    MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT) & mseg)
{
  if ((sseg.Type() !=
           MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)::seg_BiLinearTri &&
       sseg.Type() !=
           MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)::seg_BiLinearQuad) ||
      (mseg.Type() !=
           MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)::seg_BiLinearTri &&
       mseg.Type() !=
           MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)::seg_BiLinearQuad)) {
    std::stringstream oss;
    oss << "***ERR*** MoertelT::InterfaceT::Integrate_3D_Section:\n"
        << "***ERR*** Integration of other then bilinear triangles/quads not "
           "implemented\n"
        << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
    throw MoertelT::ReportError(oss);
  }

  // find whether we want exact values at gaussian points
  bool exactvalues = intparams_->get("exact values at gauss points", true);

  // first determine whether there is an overlap between sseg and mseg
  // for this purpose, the 'overlapper' class is used
  // It also builds a triangulation of the overlap polygon if there is any
  MoertelT::OverlapT<MoertelT::MOERTEL_TEMPLATE_CLASS(InterfaceT)> overlap(
      sseg, mseg, *this, exactvalues, OutLevel());

  // determine the overlap triangulation if any
  bool ok = overlap.ComputeOverlap();

  if (!ok) return true;  // There's no overlap

  // # new segments the overlap polygon was discretized with
  int nseg = overlap.Nseg();
  // view of these segments
  std::vector<Teuchos::RCP<MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)>> segs;
  overlap.SegmentView(segs);

  // integrator object
  int ngp = intparams_->get("number gaussian points 2D", 12);
  MoertelT::MOERTEL_TEMPLATE_CLASS(IntegratorT)
      integrator(ngp, IsOneDimensional(), OutLevel());

  // loop segments and integrate them
  for (int s = 0; s < nseg; ++s) {
    Teuchos::RCP<MoertelT::SEGMENT_TEMPLATE_CLASS(SegmentT)> actseg = segs[s];

    // integrate master and slave part of this segment
    Teuchos::SerialDenseMatrix<LO, ST>* Ddense = NULL;
    Teuchos::SerialDenseMatrix<LO, ST>* Mdense = NULL;
    bool                                ok     = integrator.Integrate(
        actseg, sseg, mseg, &Ddense, &Mdense, overlap, 1.0e-04, exactvalues);
    if (!ok) continue;

    // assemble temporarily into the nodes
    integrator.Assemble(*this, sseg, *Ddense);
    integrator.Assemble(*this, sseg, mseg, *Mdense);

    if (Ddense) delete Ddense;

    if (Mdense) delete Mdense;

  }  // for (int s=0; s<nseg; ++s)

  segs.clear();

  return true;
}

/*----------------------------------------------------------------------*
  |  assemble integration of master/slave side in 3D (2D interface)      |
 *----------------------------------------------------------------------*/
MOERTEL_TEMPLATE_STATEMENT
bool MoertelT::MOERTEL_TEMPLATE_CLASS(InterfaceT)::Assemble_3D(
    Tpetra::CrsMatrix<ST, LO, GO, N>& D,
    Tpetra::CrsMatrix<ST, LO, GO, N>& M)
{
  if (!IsComplete()) {
    std::cout << "***ERR*** MoertelT::InterfaceT::Assemble_3D:\n"
              << "***ERR*** Complete() not called on interface " << Id_ << "\n"
              << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
    return false;
  }

  if (lcomm_ == Teuchos::null) return true;

  // get the sides
  int mside = MortarSide();
  int sside = OtherSide(mside);

  //-------------------------------------------------------------------
  // loop over all slave nodes
  std::map<int, Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)>>::iterator
      curr;

  for (curr = rnode_[sside].begin(); curr != rnode_[sside].end(); ++curr) {
    // loop only my own nodes
    if (NodePID(curr->second->Id()) != lcomm_->getRank()) continue;

    // get std::maps D and M and Mmod from node
    Teuchos::RCP<std::map<int, double>> Drow = curr->second->GetD();
    Teuchos::RCP<std::map<int, double>> Mrow = curr->second->GetM();
    Teuchos::RCP<std::vector<std::map<int, double>>> Mmod =
        curr->second->GetMmod();

    // if there's no D or M there's nothing to do
    if (Drow == Teuchos::null && Mrow == Teuchos::null) continue;

    Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)> rowsnode =
        curr->second;
    int        snlmdof = rowsnode->Nlmdof();
    const int* slmdof  = rowsnode->LMDof();
    // std::cout << "Current row snode: " << rowsnode->Id() << std::endl;

    std::map<int, double>::iterator rowcurr;

    //-------------------------------------------------------------------
    //-------------------------------------------------------------------
    // assemble the Drow
    if (Drow != Teuchos::null) {
      for (rowcurr = Drow->begin(); rowcurr != Drow->end(); ++rowcurr) {
        int    colnode = rowcurr->first;
        double val     = rowcurr->second;

        if (abs(val) < CONSTRAINT_MATRIX_ZERO) continue;

        // std::cout << "Current col snode: " << colnode << std::endl;

        // get the colsnode
        Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)> colsnode =
            GetNodeView(colnode);

        if (colsnode == Teuchos::null) {
          std::cout << "***ERR*** MoertelT::InterfaceT::Assemble_3D:\n"
                    << "***ERR*** interface " << Id_
                    << ": cannot get view of node " << colnode << "\n"
                    << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__
                    << "\n";
          return false;
        }

        // get the primal dofs
        int        sndof = colsnode->Ndof();
        const int* sdof  = colsnode->Dof();

        if (snlmdof != sndof) {
          std::cout
              << "***ERR*** MoertelT::InterfaceT::Assemble_3D:\n"
              << "***ERR*** interface " << Id_
              << ": mismatch in # lagrange multipliers and primal variables\n"
              << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
          return false;
        }

        for (int i = 0; i < snlmdof; ++i) {
          int row = slmdof[i];
          GO  col = sdof[i];
          // std::cout << "Inserting D row/col:" << row << "/" << col << " val "
          // << val << std::endl;
          int err = D.sumIntoGlobalValues(row, 1, &val, &col);

          if (err) D.insertGlobalValues(row, 1, &val, &col);

          if (err < 0) {
            std::cout << "***ERR*** MoertelT::InterfaceT::Assemble_3D:\n"
                      << "***ERR*** interface " << Id_
                      << ": Tpetra_CrsMatrix::InsertGlobalValues returned "
                      << err << "\n"
                      << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__
                      << "\n";
            return false;
          }

          if (err && OutLevel() > 0) {
            std::cout
                << "MoertelT: ***WRN*** MoertelT::InterfaceT::Assemble_3D:\n"
                << "MoertelT: ***WRN*** interface " << Id_
                << ": Tpetra_CrsMatrix::InsertGlobalValues returned " << err
                << "\n"
                << "MoertelT: ***WRN*** indicating that initial guess for "
                   "memory of D too small\n"
                << "MoertelT: ***WRN*** file/line: " << __FILE__ << "/"
                << __LINE__ << "\n";
          }
        }  // for (int i=0; i<snlmdof; ++i)
      }    // for (rowcurr=Drow->begin(); rowcurr!=Drow->end(); ++rowcurr)
    }

    //-------------------------------------------------------------------
    //-------------------------------------------------------------------
    // assemble the Mrow
    if (Mrow != Teuchos::null) {
      for (rowcurr = Mrow->begin(); rowcurr != Mrow->end(); ++rowcurr) {
        int    colnode = rowcurr->first;
        double val     = rowcurr->second;

        if (abs(val) < CONSTRAINT_MATRIX_ZERO) continue;

        // std::cout << "Current colmnode: " << colnode << std::endl;

        // get the colmnode
        Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)> colmnode =
            GetNodeView(colnode);

        if (colmnode == Teuchos::null) {
          std::cout << "***ERR*** MoertelT::InterfaceT::Assemble_3D:\n"
                    << "***ERR*** interface " << Id_
                    << ": cannot get view of node " << colnode << "\n"
                    << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__
                    << "\n";
          return false;
        }

        // get the primal dofs
        int        mndof = colmnode->Ndof();
        const int* mdof  = colmnode->Dof();

        if (snlmdof != mndof) {
          std::cout
              << "***ERR*** MoertelT::InterfaceT::Assemble_3D:\n"
              << "***ERR*** interface " << Id_
              << ": mismatch in # lagrange multipliers and primal variables\n"
              << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
          return false;
        }

        for (int i = 0; i < snlmdof; ++i) {
          int row = slmdof[i];
          GO  col = mdof[i];
          // std::cout << "Inserting M row/col:" << row << "/" << col << " val "
          // << val << std::endl;
          int err = M.sumIntoGlobalValues(row, 1, &val, &col);

          if (err) M.insertGlobalValues(row, 1, &val, &col);

          if (err < 0) {
            std::cout << "***ERR*** MoertelT::InterfaceT::Assemble_3D:\n"
                      << "***ERR*** interface " << Id_
                      << ": Tpetra_CrsMatrix::InsertGlobalValues returned "
                      << err << "\n"
                      << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__
                      << "\n";
            return false;
          }

          if (err && OutLevel() > 0) {
            std::cout
                << "MoertelT: ***WRN*** MoertelT::InterfaceT::Assemble_3D:\n"
                << "MoertelT: ***WRN*** interface " << Id_
                << ": Tpetra_CrsMatrix::InsertGlobalValues returned " << err
                << "\n"
                << "MoertelT: ***WRN*** indicating that initial guess for "
                   "memory of M too small\n"
                << "MoertelT: ***WRN*** file/line: " << __FILE__ << "/"
                << __LINE__ << "\n";
          }
        }  // for (int i=0; i<snlmdof; ++i)
      }    // for (rowcurr=Mrow->begin(); rowcurr!=Mrow->end(); ++rowcurr)
    }

    // assemble the Mmod block if there is any
    if (Mmod != Teuchos::null) {
      // loop over he rows of the Mmod block
      for (int lrow = 0; lrow < (int)Mmod->size(); ++lrow) {
        //        std::map<int,double>& Mmodrow = (*Mmod)[lrow];
        int row = slmdof[lrow];

        // loop over the columns in that row
        // FIXMEL: should this be Mmodrow (above)?
        for (rowcurr = Mrow->begin(); rowcurr != Mrow->end(); ++rowcurr) {
          GO     col = rowcurr->first;
          double val = rowcurr->second;

          if (abs(val) < CONSTRAINT_MATRIX_ZERO) continue;

          // std::cout << "Inserting M row/col:" << row << "/" << col << " val "
          // << val << std::endl;
          int err = M.sumIntoGlobalValues(row, 1, &val, &col);

          if (err) M.insertGlobalValues(row, 1, &val, &col);

          if (err < 0) {
            std::cout << "***ERR*** MoertelT::InterfaceT::Assemble_3D:\n"
                      << "***ERR*** interface " << Id_
                      << ": Tpetra_CrsMatrix::InsertGlobalValues returned "
                      << err << "\n"
                      << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__
                      << "\n";
            return false;
          }

          if (err && OutLevel() > 0) {
            std::cout
                << "MoertelT: ***WRN*** MoertelT::InterfaceT::Assemble_3D:\n"
                << "MoertelT: ***WRN*** interface " << Id_
                << ": Tpetra_CrsMatrix::InsertGlobalValues returned " << err
                << "\n"
                << "MoertelT: ***WRN*** indicating that initial guess for "
                   "memory of M too small\n"
                << "MoertelT: ***WRN*** file/line: " << __FILE__ << "/"
                << __LINE__ << "\n";
          }

        }  // for (rowcurr=Mrow->begin(); rowcurr!=Mrow->end(); ++rowcurr)

      }  // for (int lrow=0; lrow<(int)Mmod->size(); ++lrow)
    }

  }  // for (curr=rnode_[sside].begin(); curr!=rnode_[sside].end(); ++curr)

  //-------------------------------------------------------------------
  //-------------------------------------------------------------------
  // In case this interface was parallel we might have missed something upto
  // here. Boundary terms of D and M, Mmod are assembled non-local (that is to
  // close inner-interface nodes). If these inner-interface nodes belong
  // to a different proc values were not assembled.
  // Loop snodes again an check and communicate these entries
  if (lcomm_->getSize() != 1) {
    // note that we miss the communication of Mmod yet

    // allocate a sendbuffer for D and M
    int                 countD = 0;
    int                 countM = 0;
    std::vector<int>    colD_s(countD);
    std::vector<double> valD_s(countD);
    std::vector<int>    colM_s(countM);
    std::vector<double> valM_s(countM);

    for (curr = rnode_[sside].begin(); curr != rnode_[sside].end(); ++curr) {
      // we've done all my own nodes already
      if (NodePID(curr->second->Id()) == lcomm_->getRank()) continue;

      // check whether we have M or D values here
      // get maps D and M from node
      Teuchos::RCP<std::map<int, double>> Drow = curr->second->GetD();
      Teuchos::RCP<std::map<int, double>> Mrow = curr->second->GetM();

      // if there's no D/M there's nothing to do
      if (Drow == Teuchos::null && Mrow == Teuchos::null) continue;

      // std::cout << "lProc " << lComm()->MyPID() << " Node " <<
      // curr->second->Id() << " unassembled\n";

      // fill the D sendbuffer
      if (Drow != Teuchos::null) {
        // resize the sendbuffers
        colD_s.resize(colD_s.size() + Drow->size() + 2);
        valD_s.resize(valD_s.size() + Drow->size() + 2);
        // Add node Id and size
        colD_s[countD]     = curr->second->Id();
        valD_s[countD]     = 0.0;
        colD_s[countD + 1] = (int)Drow->size();
        valD_s[countD + 1] = 0.0;
        countD += 2;
        // loop D
        std::map<int, double>::iterator rowcurr;

        for (rowcurr = Drow->begin(); rowcurr != Drow->end(); ++rowcurr) {
          colD_s[countD] = rowcurr->first;
          valD_s[countD] = rowcurr->second;
          ++countD;
        }
      }

      // std::cout << "lProc " << lComm()->MyPID() << " Node " <<
      // curr->second->Id() << " countD " << countD << std::endl;

      // fill the M sendbuffer
      if (Mrow != Teuchos::null) {
        // resize the sendbuffers
        colM_s.resize(colM_s.size() + Mrow->size() + 2);
        valM_s.resize(valM_s.size() + Mrow->size() + 2);
        // Add node id and size
        colM_s[countM]     = curr->second->Id();
        valM_s[countM]     = 0.0;
        colM_s[countM + 1] = Mrow->size();
        valM_s[countM + 1] = 0.0;
        countM += 2;
        // loop M
        std::map<int, double>::iterator rowcurr;

        for (rowcurr = Mrow->begin(); rowcurr != Mrow->end(); ++rowcurr) {
          colM_s[countM] = rowcurr->first;
          valM_s[countM] = rowcurr->second;
          ++countM;
        }
      }

      // std::cout << "lProc " << lComm()->MyPID() << " Node " <<
      // curr->second->Id() << " countM " << countM << std::endl;
    }  // for (curr=rnode_[sside].begin(); curr!=rnode_[sside].end(); ++curr)

    // loop all processes in lComm and communicate and assemble
    for (int proc = 0; proc < lcomm_->getSize(); ++proc) {
      // send sizes
      int countDr = countD;
      int countMr = countM;
      Teuchos::broadcast<LO, int>(*lcomm_, proc, 1, &countDr);
      Teuchos::broadcast<LO, int>(*lcomm_, proc, 1, &countMr);
      // allocate receive buffers
      std::vector<int>    colD_r(countDr);
      std::vector<double> valD_r(countDr);
      std::vector<int>    colM_r(countMr);
      std::vector<double> valM_r(countMr);

      // send data
      if (proc == lcomm_->getRank()) {
        for (int i = 0; i < countDr; ++i) {
          colD_r[i] = colD_s[i];
          valD_r[i] = valD_s[i];
        }

        for (int i = 0; i < countMr; ++i) {
          colM_r[i] = colM_s[i];
          valM_r[i] = valM_s[i];
        }
      }
      if (countDr > 0) {
        Teuchos::broadcast<LO, int>(*lcomm_, proc, countDr, &colD_r[0]);
        Teuchos::broadcast<LO, double>(*lcomm_, proc, countDr, &valD_r[0]);
      }

      if (countMr > 0) {
        Teuchos::broadcast<LO, int>(*lcomm_, proc, countMr, &colM_r[0]);
        Teuchos::broadcast<LO, double>(*lcomm_, proc, countMr, &valM_r[0]);
      }

      // Assemble (remote procs only)
      if (proc != lcomm_->getRank()) {
        // --------------------------------------------------- Assemble D
        for (int i = 0; i < countDr;) {
          int nodeid = colD_r[i];
          int size   = colD_r[i + 1];
          i += 2;

          // find whether I am owner of this node
          if (NodePID(nodeid) == lcomm_->getRank()) {
            // get the node
            Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)> snode =
                GetNodeView(nodeid);

            if (snode == Teuchos::null) {
              std::stringstream oss;
              oss << "***ERR*** MoertelT::InterfaceT::Assemble_3D:\n"
                  << "***ERR*** Cannot find view of node " << nodeid
                  << std::endl
                  << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__
                  << "\n";
              throw MoertelT::ReportError(oss);
            }

            // get lagrange multipliers
            int        nslmdof = snode->Nlmdof();
            const int* slmdof  = snode->LMDof();

            // loop colD_r/valD_r and assemble
            for (int j = 0; j < size; ++j) {
              int    colsnode = colD_r[i + j];
              double val      = valD_r[i + j];

              if (abs(val) < CONSTRAINT_MATRIX_ZERO) continue;

              // get view of column node and primal dofs
              Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)> colnode =
                  GetNodeView(colsnode);

              if (colnode == Teuchos::null) {
                std::stringstream oss;
                oss << "***ERR*** MoertelT::InterfaceT::Assemble_3D:\n"
                    << "***ERR*** Cannot find view of node " << colsnode
                    << std::endl
                    << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__
                    << "\n";
                throw MoertelT::ReportError(oss);
              }

              int        nsdof = colnode->Ndof();
              const int* sdof  = colnode->Dof();

              if (nsdof != nslmdof) {
                std::stringstream oss;
                oss << "***ERR*** MoertelT::InterfaceT::Assemble_3D:\n"
                    << "***ERR*** Mismatch in # primal dofs and Lagrange "
                       "multipliers\n"
                    << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__
                    << "\n";
                throw MoertelT::ReportError(oss);
              }

              for (int k = 0; k < nslmdof; ++k) {
                int row = slmdof[k];
                GO  col = sdof[k];
                // std::cout << "Proc " << lComm()->MyPID() << " inserting D
                // row/col:" << row << "/" << col << " val " << val << std::endl;
                int err = D.sumIntoGlobalValues(row, 1, &val, &col);

                if (err) D.insertGlobalValues(row, 1, &val, &col);

                if (err < 0) {
                  std::stringstream oss;
                  oss << "***ERR*** MoertelT::InterfaceT::Assemble_3D:\n"
                      << "***ERR*** Serious error=" << err << " in assembly\n"
                      << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__
                      << "\n";
                  throw MoertelT::ReportError(oss);
                }

                if (err && OutLevel() > 0) {
                  std::cout
                      << "MoertelT: ***WRN*** "
                         "MoertelT::InterfaceT::Assemble_3D:\n"
                      << "MoertelT: ***WRN*** interface " << Id()
                      << ": Tpetra_CrsMatrix::InsertGlobalValues returned "
                      << err << "\n"
                      << "MoertelT: ***WRN*** indicating that initial guess "
                         "for memory of D too small\n"
                      << "MoertelT: ***WRN*** file/line: " << __FILE__ << "/"
                      << __LINE__ << "\n";
                }
              }  // for (int k=0; k<nslmdof; ++k)
            }    // for (int j=0; j<size; ++j)

            i += size;
          }

          else  // I am not owner of this node, skip it
            i += size;
        }  // for (int i=0; i<countDr;)

        // --------------------------------------------------- Assemble M
        for (int i = 0; i < countMr;) {
          int nodeid = colM_r[i];
          int size   = colM_r[i + 1];
          i += 2;

          // find whether I am owner of this node
          if (NodePID(nodeid) == lcomm_->getRank()) {
            // get the node
            Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)> snode =
                GetNodeView(nodeid);

            if (snode == Teuchos::null) {
              std::stringstream oss;
              oss << "***ERR*** MoertelT::InterfaceT::Assemble_3D:\n"
                  << "***ERR*** Cannot find view of node " << nodeid
                  << std::endl
                  << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__
                  << "\n";
              throw MoertelT::ReportError(oss);
            }

            // get the lagrange multipliers
            int        nslmdof = snode->Nlmdof();
            const int* slmdof  = snode->LMDof();

            // loop colM_r/valM_r and assemble
            for (int j = 0; j < size; ++j) {
              int    colmnode = colM_r[i + j];
              double val      = valM_r[i + j];

              if (abs(val) < CONSTRAINT_MATRIX_ZERO) continue;

              // get view of column node and primal dofs
              Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)> colnode =
                  GetNodeView(colmnode);

              if (colnode == Teuchos::null) {
                std::stringstream oss;
                oss << "***ERR*** MoertelT::InterfaceT::Assemble_3D:\n"
                    << "***ERR*** Cannot find view of node " << colmnode
                    << std::endl
                    << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__
                    << "\n";
                throw MoertelT::ReportError(oss);
              }

              int        nmdof = colnode->Ndof();
              const int* mdof  = colnode->Dof();

              if (nmdof != nslmdof) {
                std::stringstream oss;
                oss << "***ERR*** MoertelT::InterfaceT::Assemble_3D:\n"
                    << "***ERR*** Mismatch in # primal dofs and Lagrange "
                       "multipliers\n"
                    << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__
                    << "\n";
                throw MoertelT::ReportError(oss);
              }

              for (int k = 0; k < nslmdof; ++k) {
                int row = slmdof[k];
                GO  col = mdof[k];
                // std::cout << "Proc " << lComm()->MyPID() << " inserting M
                // row/col:" << row << "/" << col << " val " << val << std::endl;
                int err = M.sumIntoGlobalValues(row, 1, &val, &col);

                if (err) M.insertGlobalValues(row, 1, &val, &col);

                if (err < 0) {
                  std::stringstream oss;
                  oss << "***ERR*** MoertelT::InterfaceT::Assemble_3D:\n"
                      << "***ERR*** Serious error=" << err << " in assembly\n"
                      << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__
                      << "\n";
                  throw MoertelT::ReportError(oss);
                }

                if (err && OutLevel() > 0) {
                  std::cout
                      << "MoertelT: ***WRN*** "
                         "MoertelT::InterfaceT::Assemble_3D:\n"
                      << "MoertelT: ***WRN*** interface " << Id()
                      << ": Tpetra_CrsMatrix::InsertGlobalValues returned "
                      << err << "\n"
                      << "MoertelT: ***WRN*** indicating that initial guess "
                         "for memory of M too small\n"
                      << "MoertelT: ***WRN*** file/line: " << __FILE__ << "/"
                      << __LINE__ << "\n";
                }
              }  // for (int k=0; k<nslmdof; ++k)
            }    // for (int j=0; j<size; ++j)

            i += size;
          }

          else  // I am not owner of this node, skip it
            i += size;
        }  // for (int i=0; i<countMr;)
      }    // if (proc!=lComm()->MyPID())

      colD_r.clear();
      valD_r.clear();
      colM_r.clear();
      valM_r.clear();
    }  // for (int proc=0; proc<lComm()->NumProc(); ++proc)

    colD_s.clear();
    valD_s.clear();
    colM_s.clear();
    valM_s.clear();
  }  // if (lComm()->NumProc()!=1)

  return true;
}

/*----------------------------------------------------------------------*
  |  assemble integration of master/slave side into the residual vector (JFNK)
  |
  |
  | Here, each global node (rnode_) is visited on the slave interface side.
  | Each node stores a local D, M, and Mmod.
 *----------------------------------------------------------------------*/
//#define PDANDM
MOERTEL_TEMPLATE_STATEMENT
bool MoertelT::MOERTEL_TEMPLATE_CLASS(InterfaceT)::AssembleResidualVector()
{
  if (!IsComplete()) {
    std::cout << "***ERR*** MoertelT::InterfaceT::AssembleJFNKVec:\n"
              << "***ERR*** Complete() not called on interface " << Id_ << "\n"
              << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
    return false;
  }

  if (lcomm_ == Teuchos::null) return true;

  // get the sides
  int mside = MortarSide();
  int sside = OtherSide(mside);

  //-------------------------------------------------------------------
  // loop over all slave nodes
  std::map<int, Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)>>::iterator
      curr;

#ifdef PDANDM  // Save and print the D and M for debugging
  int              size = rnode_[sside].size();
  int              cnt  = 0;
  std::vector<int> dtable(size);

  for (curr = rnode_[sside].begin(); curr != rnode_[sside].end(); ++curr)

    dtable[cnt++] = curr->second->Id();

  Tpetra::Map<LO, GO, N> Dmap(
      Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
      &dtable[0],
      size,
      0,
      lcomm_);
  Tpetra::CrsMatrix<ST, LO, GO, N> Dmat(Dmap, 4);
  Tpetra::CrsMatrix<ST, LO, GO, N> Mmat(Dmap, 4);

#endif

  for (curr = rnode_[sside].begin(); curr != rnode_[sside].end(); ++curr) {
    // loop only my own nodes
    if (NodePID(curr->second->Id()) != lcomm_->getRank()) continue;

    //  std::cout << curr->second->Id() << std::endl;

    // get maps D and M and Mmod from node
    Teuchos::RCP<std::map<int, double>> Drow = curr->second->GetD();
    Teuchos::RCP<std::map<int, double>> Mrow = curr->second->GetM();
    Teuchos::RCP<std::vector<std::map<int, double>>> Mmod =
        curr->second->GetMmod();

    // if there's no D or M there's nothing to do
    if (Drow == Teuchos::null && Mrow == Teuchos::null) continue;

    Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)> rowsnode =
        curr->second;
    int        snlmdof = rowsnode->Nlmdof();
    const int* slmdof  = rowsnode->LMDof();
    // std::cout << "Current row snode: " << rowsnode->Id() << std::endl;

    std::map<int, double>::iterator rowcurr;

    //-------------------------------------------------------------------
    //-------------------------------------------------------------------
    // assemble the Drow
    if (Drow != Teuchos::null) {
      for (rowcurr = Drow->begin(); rowcurr != Drow->end(); ++rowcurr) {
        int    colnode = rowcurr->first;
        double val     = rowcurr->second;

        if (abs(val) <
            CONSTRAINT_MATRIX_ZERO)  // this entry of D is effectively zero
          continue;
#ifdef PDANDM  // Save the row, col, and value
        Dmat.insertGlobalValues(curr->second->Id(), 1, &val, &colnode);
#endif

        // std::cout << "Current col snode: " << colnode << std::endl;

        // get the colsnode
        Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)> colsnode =
            GetNodeView(colnode);

        if (colsnode == Teuchos::null) {
          std::cout << "***ERR*** MoertelT::InterfaceT::AssembleJFNKVec:\n"
                    << "***ERR*** interface " << Id_
                    << ": cannot get view of node " << colnode << "\n"
                    << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__
                    << "\n";
          return false;
        }

        // get the primal dofs
        int        sndof = colsnode->Ndof();
        const int* sdof  = colsnode->Dof();

        if (snlmdof != sndof) {
          std::cout
              << "***ERR*** MoertelT::InterfaceT::AssembleJFNKVec:\n"
              << "***ERR*** interface " << Id_
              << ": mismatch in # lagrange multipliers and primal variables\n"
              << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
          return false;
        }

        for (int i = 0; i < snlmdof; ++i) {
          if (!sel->EvaluateLM(
                  rowsnode, i))  // Continue if this LM is not active
            continue;

          int row = slmdof[i];
          int col = sdof[i];
          //           std::cout << "Inserting D row/col:" << row << "/" << col
          //           << " val " << val << std::endl;

          /*
             int err = D.SumIntoGlobalValues(row,1,&val,&col);
             if (err)
             err = D.InsertGlobalValues(row,1,&val,&col);
             */

          // Assemble D times soln
          // Row of D determines row in rhs
          // col of D determines row of soln

          sel->AssembleNodeVal(row, col, val);

        }  // for (int i=0; i<snlmdof; ++i)
      }    // for (rowcurr=Drow->begin(); rowcurr!=Drow->end(); ++rowcurr)
    }

    //-------------------------------------------------------------------
    //-------------------------------------------------------------------
    // assemble the Mrow
    if (Mrow != Teuchos::null) {
      for (rowcurr = Mrow->begin(); rowcurr != Mrow->end(); ++rowcurr) {
        int    colnode = rowcurr->first;
        double val     = rowcurr->second;

        if (abs(val) < CONSTRAINT_MATRIX_ZERO) continue;

          // std::cout << "Current colmnode: " << colnode << std::endl;
          //
#ifdef PDANDM  // Save the row, col, and value
        Mmat.insertGlobalValues(curr->second->Id(), 1, &val, &colnode);
#endif

        // get the colmnode
        Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)> colmnode =
            GetNodeView(colnode);

        if (colmnode == Teuchos::null) {
          std::cout << "***ERR*** MoertelT::InterfaceT::AssembleJFNKVec:\n"
                    << "***ERR*** interface " << Id_
                    << ": cannot get view of node " << colnode << "\n"
                    << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__
                    << "\n";
          return false;
        }

        // get the primal dofs
        int        mndof = colmnode->Ndof();
        const int* mdof  = colmnode->Dof();

        if (snlmdof != mndof) {
          std::cout
              << "***ERR*** MoertelT::InterfaceT::AssembleJFNKVec:\n"
              << "***ERR*** interface " << Id_
              << ": mismatch in # lagrange multipliers and primal variables\n"
              << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__ << "\n";
          return false;
        }

        for (int i = 0; i < snlmdof; ++i) {
          if (!sel->EvaluateLM(rowsnode, i))  // true if this LM is active
            continue;

          int row = slmdof[i];
          int col = mdof[i];
          // std::cout << "Inserting M row/col:" << row << "/" << col << " val "
          // << val << std::endl;

          // Assemble M times soln
          // Row of M determines row in rhs
          // col of M determines row of soln

          sel->AssembleNodeVal(row, col, val);

          /*
             int err = M.SumIntoGlobalValues(row,1,&val,&col);
             if (err)
             err = M.InsertGlobalValues(row,1,&val,&col);
             */

        }  // for (int i=0; i<snlmdof; ++i)
      }    // for (rowcurr=Mrow->begin(); rowcurr!=Mrow->end(); ++rowcurr)
    }

    // assemble the Mmod block if there is any
    if (Mmod != Teuchos::null) {
      // loop over the rows of the Mmod block

      std::stringstream oss;
      oss << "Mmod has entries in it" << std::endl;
      throw MoertelT::ReportError(oss);

      for (int lrow = 0; lrow < (int)Mmod->size(); ++lrow) {
        //        std::map<int,double>& Mmodrow = (*Mmod)[lrow];
        //
        int row = slmdof[lrow];

        // loop over the columns in that row
        // FIXMEL: should this be Mmodrow (above)?

        for (rowcurr = Mrow->begin(); rowcurr != Mrow->end(); ++rowcurr) {
          if (!sel->EvaluateLM(rowsnode, lrow))  // true if this LM is active
            continue;

          int    col = rowcurr->first;
          double val = rowcurr->second;

          if (abs(val) < CONSTRAINT_MATRIX_ZERO) continue;

          // Assemble D times soln
          // Row of D determines row in rhs
          // col of D determines row of soln

          sel->AssembleNodeVal(row, col, val);

          /*
             int err = M.SumIntoGlobalValues(row,1,&val,&col);
             if (err)
             err = M.InsertGlobalValues(row,1,&val,&col);
             */

        }  // for (rowcurr=Mrow->begin(); rowcurr!=Mrow->end(); ++rowcurr)

      }  // for (int lrow=0; lrow<(int)Mmod->size(); ++lrow)
    }

    sel->AccumulateRHS(rowsnode);

  }  // for (curr=rnode_[sside].begin(); curr!=rnode_[sside].end(); ++curr)

  // Master side - Is this even needed??? GAH

  /*
     for (curr=rnode_[mside].begin(); curr!=rnode_[mside].end(); ++curr){

     Teuchos::RCP<MOERTEL::Node> rowsnode = curr->second;

     sel->AccumulateMRHS(rowsnode);

     }
     */

  //-------------------------------------------------------------------
  //-------------------------------------------------------------------
  // In case this interface was parallel we might have missed something upto
  // here. Boundary terms of D and M, Mmod are assembled non-local (that is to
  // close inner-interface nodes). If these inner-interface nodes belong
  // to a different proc values were not assembled.
  // Loop snodes again an check and communicate these entries

  if (lcomm_->getSize() != 1) {
    // note that we miss the communication of Mmod yet

    // allocate a sendbuffer for D and M
    int                 countD = 0;
    int                 countM = 0;
    std::vector<int>    colD_s(countD);
    std::vector<double> valD_s(countD);
    std::vector<int>    colM_s(countM);
    std::vector<double> valM_s(countM);

    for (curr = rnode_[sside].begin(); curr != rnode_[sside].end(); ++curr) {
      // we've done all my own nodes already
      if (NodePID(curr->second->Id()) == lcomm_->getRank()) continue;

      // check whether we have M or D values here
      // get maps D and M from node
      Teuchos::RCP<std::map<int, double>> Drow = curr->second->GetD();
      Teuchos::RCP<std::map<int, double>> Mrow = curr->second->GetM();

      // if there's no D/M there's nothing to do
      if (Drow == Teuchos::null && Mrow == Teuchos::null) continue;

      // std::cout << "lProc " << lComm()->MyPID() << " Node " <<
      // curr->second->Id() << " unassembled\n";

      // fill the D sendbuffer
      if (Drow != Teuchos::null) {
        // resize the sendbuffers
        colD_s.resize(colD_s.size() + Drow->size() + 2);
        valD_s.resize(valD_s.size() + Drow->size() + 2);
        // Add node Id and size
        colD_s[countD]     = curr->second->Id();
        valD_s[countD]     = 0.0;
        colD_s[countD + 1] = (int)Drow->size();
        valD_s[countD + 1] = 0.0;
        countD += 2;
        // loop D
        std::map<int, double>::iterator rowcurr;

        for (rowcurr = Drow->begin(); rowcurr != Drow->end(); ++rowcurr) {
          colD_s[countD] = rowcurr->first;
          valD_s[countD] = rowcurr->second;
          ++countD;
        }
      }

      // std::cout << "lProc " << lComm()->MyPID() << " Node " <<
      // curr->second->Id() << " countD " << countD << std::endl;

      // fill the M sendbuffer
      if (Mrow != Teuchos::null) {
        // resize the sendbuffers
        colM_s.resize(colM_s.size() + Mrow->size() + 2);
        valM_s.resize(valM_s.size() + Mrow->size() + 2);
        // Add node id and size
        colM_s[countM]     = curr->second->Id();
        valM_s[countM]     = 0.0;
        colM_s[countM + 1] = Mrow->size();
        valM_s[countM + 1] = 0.0;
        countM += 2;
        // loop M
        std::map<int, double>::iterator rowcurr;

        for (rowcurr = Mrow->begin(); rowcurr != Mrow->end(); ++rowcurr) {
          colM_s[countM] = rowcurr->first;
          valM_s[countM] = rowcurr->second;
          ++countM;
        }
      }

      // std::cout << "lProc " << lComm()->MyPID() << " Node " <<
      // curr->second->Id() << " countM " << countM << std::endl;
    }  // for (curr=rnode_[sside].begin(); curr!=rnode_[sside].end(); ++curr)

    // loop all processes in lComm and communicate and assemble
    for (int proc = 0; proc < lcomm_->getSize(); ++proc) {
      // send sizes
      int countDr = countD;
      int countMr = countM;
      Teuchos::broadcast<LO, int>(*lcomm_, proc, 1, &countDr);
      Teuchos::broadcast<LO, int>(*lcomm_, proc, 1, &countMr);
      // allocate receive buffers
      std::vector<int>    colD_r(countDr);
      std::vector<double> valD_r(countDr);
      std::vector<int>    colM_r(countMr);
      std::vector<double> valM_r(countMr);

      // send data
      if (proc == lcomm_->getRank()) {
        for (int i = 0; i < countDr; ++i) {
          colD_r[i] = colD_s[i];
          valD_r[i] = valD_s[i];
        }

        for (int i = 0; i < countMr; ++i) {
          colM_r[i] = colM_s[i];
          valM_r[i] = valM_s[i];
        }
      }

      Teuchos::broadcast<LO, int>(*lcomm_, proc, countDr, &colD_r[0]);
      Teuchos::broadcast<LO, double>(*lcomm_, proc, countDr, &valD_r[0]);
      Teuchos::broadcast<LO, int>(*lcomm_, proc, countDr, &colM_r[0]);
      Teuchos::broadcast<LO, double>(*lcomm_, proc, countDr, &valM_r[0]);

      // Assemble (remote procs only)
      if (proc != lcomm_->getRank()) {
        // --------------------------------------------------- Assemble D
        for (int i = 0; i < countDr;) {
          int nodeid = colD_r[i];
          int size   = colD_r[i + 1];
          i += 2;

          // find whether I am owner of this node
          if (NodePID(nodeid) == lcomm_->getRank()) {
            // get the node
            Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)> snode =
                GetNodeView(nodeid);

            if (snode == Teuchos::null) {
              std::stringstream oss;
              oss << "***ERR*** MoertelT::InterfaceT::AssembleJFNKVec:\n"
                  << "***ERR*** Cannot find view of node " << nodeid
                  << std::endl
                  << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__
                  << "\n";
              throw MoertelT::ReportError(oss);
            }

            // get lagrange multipliers
            int        nslmdof = snode->Nlmdof();
            const int* slmdof  = snode->LMDof();

            // loop colD_r/valD_r and assemble
            for (int j = 0; j < size; ++j) {
              int    colsnode = colD_r[i + j];
              double val      = valD_r[i + j];

              if (abs(val) < CONSTRAINT_MATRIX_ZERO) continue;

              // get view of column node and primal dofs
              Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)> colnode =
                  GetNodeView(colsnode);

              if (colnode == Teuchos::null) {
                std::stringstream oss;
                oss << "***ERR*** MoertelT::InterfaceT::AssembleJFNKVec:\n"
                    << "***ERR*** Cannot find view of node " << colsnode
                    << std::endl
                    << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__
                    << "\n";
                throw MoertelT::ReportError(oss);
              }

              int        nsdof = colnode->Ndof();
              const int* sdof  = colnode->Dof();

              if (nsdof != nslmdof) {
                std::stringstream oss;
                oss << "***ERR*** MoertelT::InterfaceT::AssembleJFNKVec:\n"
                    << "***ERR*** Mismatch in # primal dofs and lagrange "
                       "mutlipliers\n"
                    << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__
                    << "\n";
                throw MoertelT::ReportError(oss);
              }

              for (int k = 0; k < nslmdof; ++k) {
                if (!sel->EvaluateLM(snode, k))  // true if this LM is active
                  continue;

                int row = slmdof[k];
                int col = sdof[k];
                // std::cout << "Proc " << lComm()->MyPID() << " inserting D
                // row/col:" << row << "/" << col << " val " << val << std::endl;

                // Assemble D times soln
                // Row of D determines row in rhs
                // col of D determines row of soln

                sel->AssembleNodeVal(row, col, val);

                /*
                   int err = D.SumIntoGlobalValues(row,1,&val,&col);
                   if (err)
                   err = D.InsertGlobalValues(row,1,&val,&col);
                   */

              }  // for (int k=0; k<nslmdof; ++k)
            }    // for (int j=0; j<size; ++j)

            i += size;
          }

          else  // I am not owner of this node, skip it
            i += size;
        }  // for (int i=0; i<countDr;)

        // --------------------------------------------------- Assemble M
        for (int i = 0; i < countMr;) {
          int nodeid = colM_r[i];
          int size   = colM_r[i + 1];
          i += 2;

          // find whether I am owner of this node
          if (NodePID(nodeid) == lcomm_->getRank()) {
            // get the node
            Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)> snode =
                GetNodeView(nodeid);

            if (snode == Teuchos::null) {
              std::stringstream oss;
              oss << "***ERR*** MoertelT::InterfaceT::AssembleJFNKVec:\n"
                  << "***ERR*** Cannot find view of node " << nodeid
                  << std::endl
                  << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__
                  << "\n";
              throw MoertelT::ReportError(oss);
            }

            // get the lagrange multipliers
            int        nslmdof = snode->Nlmdof();
            const int* slmdof  = snode->LMDof();

            // loop colM_r/valM_r and assemble
            for (int j = 0; j < size; ++j) {
              int    colmnode = colM_r[i + j];
              double val      = valM_r[i + j];

              if (abs(val) < CONSTRAINT_MATRIX_ZERO) continue;

              // get view of column node and primal dofs
              Teuchos::RCP<MoertelT::MOERTEL_TEMPLATE_CLASS(NodeT)> colnode =
                  GetNodeView(colmnode);

              if (colnode == Teuchos::null) {
                std::stringstream oss;
                oss << "***ERR*** MoertelT::InterfaceT::AssembleJFNKVec:\n"
                    << "***ERR*** Cannot find view of node " << colmnode
                    << std::endl
                    << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__
                    << "\n";
                throw MoertelT::ReportError(oss);
              }

              int        nmdof = colnode->Ndof();
              const int* mdof  = colnode->Dof();

              if (nmdof != nslmdof) {
                std::stringstream oss;
                oss << "***ERR*** MoertelT::InterfaceT::AssembleJFNKVec:\n"
                    << "***ERR*** Mismatch in # primal dofs and lagrange "
                       "mutlipliers\n"
                    << "***ERR*** file/line: " << __FILE__ << "/" << __LINE__
                    << "\n";
                throw MoertelT::ReportError(oss);
              }

              for (int k = 0; k < nslmdof; ++k) {
                int row = slmdof[k];
                int col = mdof[k];
                // std::cout << "Proc " << lComm()->MyPID() << " inserting M
                // row/col:" << row << "/" << col << " val " << val << std::endl;

                // Assemble M times soln
                // Row of M determines row in rhs
                // col of M determines row of soln

                sel->AssembleNodeVal(row, col, val);

                /*
                   int err = M.SumIntoGlobalValues(row,1,&val,&col);
                   if (err)
                   err = M.InsertGlobalValues(row,1,&val,&col);
                   */

              }  // for (int k=0; k<nslmdof; ++k)
            }    // for (int j=0; j<size; ++j)

            i += size;
          }

          else  // I am not owner of this node, skip it
            i += size;
        }  // for (int i=0; i<countMr;)
      }    // if (proc!=lComm()->MyPID())

      colD_r.clear();
      valD_r.clear();
      colM_r.clear();
      valM_r.clear();
    }  // for (int proc=0; proc<lComm()->NumProc(); ++proc)

    colD_s.clear();
    valD_s.clear();
    colM_s.clear();
    valM_s.clear();
  }  // if (lComm()->NumProc()!=1)

#ifdef PDANDM
  //  Dmat.Print(std::cout);
  //  Mmat.Print(std::cout);
  Dmat.describe(std::cout);
  Mmat.describe(std::cout);
  throw "Done";
#endif

  return true;
}
