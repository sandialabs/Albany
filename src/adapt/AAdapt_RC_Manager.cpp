//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <Intrepid2_MiniTensor.h>
#include <Phalanx_FieldManager.hpp>
#include <Teuchos_CommHelpers.hpp>

#include "AAdapt_AdaptiveSolutionManagerT.hpp"
#include "AAdapt_RC_DataTypes.hpp"
#include "AAdapt_RC_DataTypes_impl.hpp"
#include "AAdapt_RC_Reader.hpp"
#include "AAdapt_RC_Writer.hpp"
#include "AAdapt_RC_Projector_impl.hpp"
#include "AAdapt_RC_Manager.hpp"

#ifdef AMBDEBUG
#define amb_test_projector
//#define amb_do_check
#define pr(msg) lpr(0,msg)
//#define pr(msg) std::cerr << "amb: (rc) " << msg << std::endl;
//#define prc(msg) pr(#msg << " | " << (msg))
#else
#define pr(msg)
#endif

#define loop(a, i, dim)                                                 \
  for (PHX::MDField<RealType>::size_type i = 0; i < a.dimension(dim); ++i)

namespace AAdapt {
namespace rc {

// Data for internal use attached to a Field.
struct Manager::Field::Data {
  Transformation::Enum transformation;
  // Nodal data g. g has up to two components.
  Teuchos::RCP<Tpetra_MultiVector> mv[2];
};

std::string Manager::Field::get_g_name (const int g_field_idx) const {
  std::stringstream ss;
  ss << this->name << "_" << g_field_idx;
  return ss.str();
}

namespace {
// f.dimension(0) in general can be larger than mda.dimension(0) because of the
// way workset data vs bucket data are allocated.
void read (const Albany::MDArray& mda, PHX::MDField<RealType>& f) {
  switch (f.rank()) {
  case 2:
    loop(mda, cell, 0) loop(f, qp, 1)
      f(cell, qp) = mda(cell, qp);
    break;
  case 3:
    loop(mda, cell, 0) loop(f, qp, 1) loop(f, i0, 2)
      f(cell, qp, i0) = mda(cell, qp, i0);
    break;
  case 4:
    loop(mda, cell, 0) loop(f, qp, 1) loop(f, i0, 2) loop(f, i1, 3)
      f(cell, qp, i0, i1) = mda(cell, qp, i0, i1);
    break;
  default:
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                               "dims.size() \notin {2,3,4}.");
  }
}

template<typename MDArray>
void write (Albany::MDArray& mda, const MDArray& f) {
  switch (f.rank()) {
  case 2:
    loop(mda, cell, 0) loop(f, qp, 1)
      mda(cell, qp) = f(cell, qp);
    break;
  case 3:
    loop(mda, cell, 0) loop(f, qp, 1) loop(f, i0, 2)
      mda(cell, qp, i0) = f(cell, qp, i0);
    break;
  case 4:
    loop(mda, cell, 0) loop(f, qp, 1) loop(f, i0, 2) loop(f, i1, 3)
      mda(cell, qp, i0, i1) = f(cell, qp, i0, i1);
    break;
  default:
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                               "dims.size() \notin {2,3,4}.");
  }
}

#ifdef amb_do_check
class Checker {
private:
  typedef Intrepid2::Tensor<RealType> Tensor;
  int wi_, cell_, qp_, node_;
  void display (const std::string& name, const Tensor& a,
                const std::string& msg) {
    std::stringstream ss;
    const int rank = Teuchos::DefaultComm<int>::getComm()->getRank();
    ss << "amb: Checker: On rank " << rank << " with (wi, cell, qp, node) = ("
       << wi_ << ", " << cell_ << ", " << qp_ << ", " << node_ << "), " << name
       << " gave the following message: " << msg << std::endl << name
       << " = [" << a << "];" << std::endl;
    std::cout << ss.str();
  }
  bool equal (const RealType a, const RealType b) {
    return a == b ||
      std::abs(a - b) < (1e3 * std::numeric_limits<RealType>::epsilon() *
                         std::max(std::abs(a), std::abs(b)));
  }
public:
  Checker (int wi, int cell, int qp, int node = 0)
    : wi_(wi), cell_(cell), qp_(qp), node_(node) {}
#define loopa(i, dim) for (Intrepid2::Index i = 0; i < a.get_dimension(); ++i)
  bool ok_numbers (const std::string& name, const Tensor& a) {
    loopa(i, 0) loopa(j, 1) {
      const bool is_inf = std::isinf(a(i,j)), is_nan = std::isnan(a(i,j));
      if (is_nan || is_nan) {
        display(name, a, is_inf ? "inf" : "nan");
        return false;
      }
    }
    return true;
  }
  bool first () const
    { return wi_ == 0 && cell_ == 0 && qp_ == 0 && node_ == 0; }
  bool is_rotation (const std::string& name, const Tensor& a) {
    const double det = Intrepid2::det(a);
    if (std::abs(det - 1) >= 1e-10) {
      std::stringstream ss;
      ss << "det = " << det;
      display(name, a, ss.str());
      return false;
    }
    return true;
  }
  bool is_symmetric (const std::string& name, const Tensor& a) {
    if ((a.get_dimension() > 1 && !equal(a(0,1), a(1,0))) || 
        (a.get_dimension() > 2 &&
         (!equal(a(0,2), a(2,0)) || !equal(a(1,2), a(2,1))))) {
      display(name, a, "not symmetric");
      return false;
    }
    return true;
  }
  bool is_antisymmetric (const std::string& name, const Tensor& a) {
    if (!equal(a(0,0), 0) ||
        (a.get_dimension() > 1 &&
         (!equal(a(0,1), -a(1,0)) || !equal(a(1,1), 0))) ||
        (a.get_dimension() > 2 &&
         (!equal(a(0,2), -a(2,0)) || !equal(a(1,2), -a(2,1)) ||
          !equal(a(2,2), 0)))) {
      display(name, a, "not antisymmetric");
      return false;
    }
    return true;
  }
#undef loopa
};
#else // amb_do_check
class Checker {
  typedef Intrepid2::Tensor<RealType> Tensor;
public:
  Checker (int wi, int cell, int qp, int node = 0) {}
  bool first () const { return false; }
#define empty(s) bool s (const std::string&, const Tensor&) { return true; }
  empty(ok_numbers) empty(is_rotation) empty(is_symmetric)
  empty(is_antisymmetric)
#undef empty
};    
#endif // amb_do_check

Intrepid2::Tensor<RealType>&
symmetrize (Intrepid2::Tensor<RealType>& a) {
  const Intrepid2::Index dim = a.get_dimension();
  if (dim > 1) {
    a(0,1) = a(1,0) = 0.5*(a(0,1) + a(1,0));
    if (dim > 2) {
      a(0,2) = a(2,0) = 0.5*(a(0,2) + a(2,0));
      a(1,2) = a(2,1) = 0.5*(a(1,2) + a(2,1));
    }
  }
  return a;
}

struct Direction { enum Enum { g2G, G2g }; };

void calc_right_polar_LieR_LieS_G2g (
  const Intrepid2::Tensor<RealType>& F, Intrepid2::Tensor<RealType> RS[2],
  Checker& c)
{
  c.ok_numbers("F", F);
  { std::pair< Intrepid2::Tensor<RealType>, Intrepid2::Tensor<RealType> >
      RSpair = Intrepid2::polar_right(F);
    RS[0] = RSpair.first; RS[1] = RSpair.second; }
  if (c.first() ||
      ! (c.ok_numbers("R", RS[0]) && c.ok_numbers("S", RS[1]) &&
         c.is_rotation("R", RS[0]) && c.is_symmetric("S", RS[1])))
    pr("F = [\n" << F << "];\nR = [\n" << RS[0] << "];\nS = [\n"
       << RS[1] << "];");    
  RS[0] = Intrepid2::log_rotation(RS[0]);
  c.ok_numbers("r", RS[0]); c.is_antisymmetric("r", RS[0]);
  RS[1] = Intrepid2::log_sym(RS[1]);
  symmetrize(RS[1]);
  c.ok_numbers("s", RS[1]); c.is_symmetric("s write", RS[1]);
  if (c.first()) pr("r =[\n" << RS[0] << "];\ns =[\n" << RS[1] << "];");
}

void calc_right_polar_LieR_LieS_g2G (
  Intrepid2::Tensor<RealType>& R, Intrepid2::Tensor<RealType>& S, Checker& c)
{
  c.ok_numbers("r", R); c.is_antisymmetric("r", R);
  c.ok_numbers("s", S); c.is_symmetric("s read", S);
  if (c.first()) pr("r = [\n" << R << "];\ns = [\n" << S << "];");
  // Math.
  R = Intrepid2::exp_skew_symmetric(R);
  c.ok_numbers("R", R); c.is_rotation("R", R);
  S = Intrepid2::exp(S);
  symmetrize(S);
  c.ok_numbers("S", S); c.is_symmetric("S", S);
  R = Intrepid2::dot(R, S);
  c.ok_numbers("F", R);
  if (c.first()) pr("F = [\n" << R << "];");
}

void transformStateArray (const unsigned int wi, const Direction::Enum dir,
                          const Transformation::Enum transformation,
                          Albany::MDArray& mda1, Albany::MDArray& mda2) {
  switch (transformation) {
  case Transformation::none: {
    if (wi == 0) pr("none " << (dir == Direction::g2G));
    if (dir == Direction::G2g) {
      // Copy from the provisional to the primary field.
      write(mda1, mda2);
    } else {
      // In the g -> G direction, the values are already in the primary field,
      // so there's nothing to do.
    }
  } break;
  case Transformation::right_polar_LieR_LieS: {
    if (wi == 0) pr("right_polar_LieR_LieS " << Direction::g2G);
    loop(mda1, cell, 0) loop(mda1, qp, 1) {
      Checker c(wi, cell, qp);
      if (dir == Direction::G2g) {
        // Copy mda2 (provisional) -> local.
        Intrepid2::Tensor<RealType> F(mda1.dimension(2));
        loop(mda2, i, 2) loop(mda2, j, 3) F(i, j) = mda2(cell, qp, i, j);
        Intrepid2::Tensor<RealType> RS[2];
        calc_right_polar_LieR_LieS_G2g(F, RS, c);
        // Copy local -> mda1, mda2.
        loop(mda1, i, 2) loop(mda1, j, 3) {
          mda1(cell, qp, i, j) = RS[0](i, j);
          mda2(cell, qp, i, j) = RS[1](i, j);
        }
      } else {
        // Copy mda1,2 -> local.
        Intrepid2::Tensor<RealType> R(mda1.dimension(2)), S(mda2.dimension(2));
        loop(mda1, i, 2) loop(mda1, j, 3) {
          R(i, j) = mda1(cell, qp, i, j);
          S(i, j) = mda2(cell, qp, i, j);
        }
        calc_right_polar_LieR_LieS_g2G(R, S, c);
        // Copy local -> mda1. mda2 is unused after g -> G.
        loop(mda1, i, 2) loop(mda1, j, 3) mda1(cell, qp, i, j) = R(i, j);
      }
    }
  } break;
  }    
}

class Projector {
  typedef PHX::MDField<RealType,Cell,Node,QuadPoint> BasisField;
  typedef BasisField::size_type size_type;

  Teuchos::RCP<const Tpetra_Map> node_map_, ol_node_map_;
  Teuchos::RCP<Tpetra_CrsMatrix> M_;
  Teuchos::RCP<Tpetra_Export> export_;
  Teuchos::RCP<Tpetra_Operator> P_;
  // M_ persists over multiple state field manager evaluations if the mesh is
  // not adapted after every LOCA step. Indicate whether this part of M_ has
  // already been filled.
  std::vector<bool> filled_;

public:
  Projector () {}
  void init(const Teuchos::RCP<const Tpetra_Map>& node_map,
            const Teuchos::RCP<const Tpetra_Map>& ol_node_map);
  void fillMassMatrix(const PHAL::Workset& workset, const BasisField& bf,
                      const BasisField& wbf);
  void fillRhs(const PHX::MDField<RealType>& f_G_qp, Manager::Field& f,
               const PHAL::Workset& workset, const BasisField& wbf);
  void project(Manager::Field& f);
  void interp(const Manager::Field& f, const PHAL::Workset& workset,
              const BasisField& bf, Albany::MDArray& mda1,
              Albany::MDArray& mda2);
  // For testing.
  const Teuchos::RCP<const Tpetra_Map>& get_node_map () const
    { return node_map_; }
  const Teuchos::RCP<const Tpetra_Map>& get_ol_node_map () const
    { return ol_node_map_; }
private:
  bool is_filled(int wi);
};

void Projector::
init (const Teuchos::RCP<const Tpetra_Map>& node_map,
      const Teuchos::RCP<const Tpetra_Map>& ol_node_map) {
  pr("Projector::init nodes " << node_map->getGlobalNumElements() << " "
     << ol_node_map->getGlobalNumElements());
  node_map_ = node_map;
  ol_node_map_ = ol_node_map;
  const int max_num_entries = 27; // Enough for first-order hex.
  M_ = Teuchos::rcp(
    new Tpetra_CrsMatrix(ol_node_map_, ol_node_map_, max_num_entries));
  export_ = Teuchos::null;
  P_ = Teuchos::null;
  filled_.clear();
}

void Projector::
fillMassMatrix (const PHAL::Workset& workset, const BasisField& bf,
                const BasisField& wbf) {
  if (is_filled(workset.wsIndex)) return;
  filled_[workset.wsIndex] = true;

  const size_type num_node = bf.dimension(1), num_qp = bf.dimension(2);
  for (unsigned int cell = 0; cell < workset.numCells; ++cell)
    for (size_type rnode = 0; rnode < num_node; ++rnode) {
      const GO row = workset.wsElNodeID[cell][rnode];
      Teuchos::Array<GO> cols;
      Teuchos::Array<ST> vals;
      for (size_type cnode = 0; cnode < num_node; ++cnode) {
        const GO col = workset.wsElNodeID[cell][cnode];
        cols.push_back(col);
        ST v = 0;
        for (size_type qp = 0; qp < num_qp; ++qp)
          v += wbf(cell, rnode, qp) * bf(cell, cnode, qp);
        vals.push_back(v);
      }
      M_->insertGlobalValues(row, cols, vals);
    }
}

void Projector::
fillRhs (const PHX::MDField<RealType>& f_G_qp, Manager::Field& f,
         const PHAL::Workset& workset, const BasisField& wbf) {
  const int
    rank = f.layout->rank() - 2,
    num_node = wbf.dimension(1), num_qp = wbf.dimension(2),
    ndim = rank >= 1 ? f_G_qp.dimension(2) : 1;

  if (f.data_->mv[0].is_null()) {
    const int ncol = rank == 0 ? 1 : rank == 1 ? ndim : ndim*ndim;
    for (int fi = 0; fi < f.num_g_fields; ++fi)
      f.data_->mv[fi] = Teuchos::rcp(
        new Tpetra_MultiVector(ol_node_map_, ncol, true));
  }
    
  const Transformation::Enum transformation = f.data_->transformation;
  for (int cell = 0; cell < (int) workset.numCells; ++cell)
    for (int node = 0; node < num_node; ++node) {
      const GO row = workset.wsElNodeID[cell][node];
      for (int qp = 0; qp < num_qp; ++qp) {
        Checker c(workset.wsIndex, cell, qp);
        switch (rank) {
        case 0:
        case 1:
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "!impl");
          break;
        case 2: {
          switch (transformation) {
          case Transformation::none: {
            for (int i = 0, col = 0; i < ndim; ++i)
              for (int j = 0; j < ndim; ++j, ++col)
                f.data_->mv[0]->sumIntoGlobalValue(
                  row, col, f_G_qp(cell, qp, i, j) * wbf(cell, node, qp));
          } break;
          case Transformation::right_polar_LieR_LieS: {
            Intrepid2::Tensor<RealType> F(ndim);
            loop(f_G_qp, i, 2) loop(f_G_qp, j, 3)
              F(i, j) = f_G_qp(cell, qp, i, j);
            Intrepid2::Tensor<RealType> RS[2];
            calc_right_polar_LieR_LieS_G2g(F, RS, c);
            for (int fi = 0; fi < f.num_g_fields; ++fi) {
              for (int i = 0, col = 0; i < ndim; ++i)
                for (int j = 0; j < ndim; ++j, ++col)
                  f.data_->mv[fi]->sumIntoGlobalValue(
                    row, col, RS[fi](i, j) * wbf(cell, node, qp));
            }
          } break;
          }
        } break;
        default:
          std::stringstream ss;
          ss << "invalid rank: " << f.name << " with rank " << rank;
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, ss.str());
        }
      }
    }
}

void Projector::project (Manager::Field& f) {
  if ( ! M_->isFillComplete()) {
    // Export M_ so it has nonoverlapping rows and cols.
    M_->fillComplete();
    export_ = Teuchos::rcp(new Tpetra_Export(ol_node_map_, node_map_));
    Teuchos::RCP<Tpetra_CrsMatrix> M = Teuchos::rcp(
      new Tpetra_CrsMatrix(node_map_, M_->getGlobalMaxNumRowEntries()));
    M->doExport(*M_, *export_, Tpetra::ADD);
    M->fillComplete();
    M_ = M;
  }
  Teuchos::RCP<Tpetra_MultiVector> x[2];
  for (int fi = 0; fi < f.num_g_fields; ++fi) {
    const int nrhs = f.data_->mv[fi]->getNumVectors();
    // Export the rhs to the same row map.
    Teuchos::RCP<Tpetra_MultiVector>
      b = Teuchos::rcp(new Tpetra_MultiVector(M_->getRangeMap(), nrhs, true));
    b->doExport(*f.data_->mv[fi], *export_, Tpetra::ADD);
    // Create x[fi] in M_ x[fi] = b[fi]. As a side effect, initialize P_ if
    // necessary.
    Teuchos::ParameterList pl;
    pl.set("Block Size", 1); // Could be nrhs.
    pl.set("Maximum Iterations", 1000);
    pl.set("Convergence Tolerance", 1e-12);
    pl.set("Output Frequency", 10);
    pl.set("Output Style", 1);
    pl.set("Verbosity", 0);//33);
    x[fi] = solve(M_, P_, b, pl); // in AAdapt_RC_Projector_impl
    // Import (reverse mode) to the overlapping MV.
    f.data_->mv[fi]->putScalar(0);
    f.data_->mv[fi]->doImport(*x[fi], *export_, Tpetra::ADD);
#if 0
    amb::write_CrsMatrix("M", *M_);
    amb::write_MultiVector("b", *b);
    amb::write_MultiVector("x", *x[fi]);
    amb::write_MultiVector("mv", *f.data_->mv[fi]);
#endif
  }
}

void Projector::
interp (const Manager::Field& f, const PHAL::Workset& workset,
        const BasisField& bf, Albany::MDArray& mda1,
        Albany::MDArray& mda2) {
  const int
    rank = f.layout->rank() - 2,
    num_node = bf.dimension(1), num_qp = bf.dimension(2),
    ndim = rank >= 1 ? mda1.dimension(2) : 1;

  Albany::MDArray* mdas[2]; mdas[0] = &mda1; mdas[1] = &mda2;
  const int nmv = f.num_g_fields;

  for (int cell = 0; cell < (int) workset.numCells; ++cell)
    for (int qp = 0; qp < num_qp; ++qp) {
      switch (rank) {
      case 0:
      case 1:
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "!impl");
        break;
      case 2: {
        for (int i = 0; i < ndim; ++i)
          for (int j = 0; j < ndim; ++j)
            mda1(cell, qp, i, j) = 0;
        for (int node = 0; node < num_node; ++node) {
          const GO grow = workset.wsElNodeID[cell][node];
          const LO row = ol_node_map_->getLocalElement(grow);
          for (int i = 0, col = 0; i < ndim; ++i)
            for (int j = 0; j < ndim; ++j, ++col)
              for (int fi = 0; fi < nmv; ++fi)
                (*mdas[fi])(cell, qp, i, j) +=
                  f.data_->mv[fi]->getVector(col)->get1dView()[row] *
                  bf(cell, node, qp);
        }
      } break;
      default:
        std::stringstream ss;
        ss << "invalid rank: " << f.name << " with rank " << rank;
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, ss.str());
      }
    }
#if 0
  pr("bf:");
  for (int cell = 0; cell < (int) workset.numCells; ++cell)
    for (int qp = 0; qp < num_qp; ++qp) {
      for (int node = 0; node < num_node; ++node)
        std::cout << " " << bf(cell, node, qp);
      std::cout << "\n";
    }
#endif
}

bool Projector::is_filled (int wi) {
  if (filled_.size() <= wi)
    filled_.insert(filled_.end(), wi - filled_.size() + 1, false);
  return filled_[wi];
}

namespace testing {
class ProjectorTester {
  struct Impl;
  Teuchos::RCP<Impl> d;
public:
  ProjectorTester();
  void init(const Teuchos::RCP<const Tpetra_Map>& node_map,
            const Teuchos::RCP<const Tpetra_Map>& ol_node_map);
  void eval(const PHAL::Workset& workset,
            const Manager::BasisField& bf, const Manager::BasisField& wbf,
            const PHX::MDField<RealType, Cell, Vertex, Dim>& coord_vert,
            const PHX::MDField<RealType, Cell, QuadPoint, Dim>& coord_qp);
  void fillRhs(const PHAL::Workset& workset, const Manager::BasisField& wbf,
               const PHX::MDField<RealType, Cell, Vertex, Dim>& coord_vert,
               const PHX::MDField<RealType, Cell, QuadPoint, Dim>& coord_qp);
  void project();
  void interp(const PHAL::Workset& workset, const Manager::BasisField& bf,
              const PHX::MDField<RealType, Cell, Vertex, Dim>& coord_vert,
              const PHX::MDField<RealType, Cell, QuadPoint, Dim>& coord_qp);
  void finish(const PHAL::Workset& workset);
};

void testProjector(
  const Projector& pc, const PHAL::Workset& workset,
  const Manager::BasisField& bf, const Manager::BasisField& wbf,
  const PHX::MDField<RealType, Cell, Vertex, Dim>& coord_vert,
  const PHX::MDField<RealType, Cell, QuadPoint, Dim>& coord_qp);
} // namespace testing
} // namespace

struct Manager::Impl {
  Teuchos::RCP<AdaptiveSolutionManagerT> sol_mgr_;
  Teuchos::RCP<Albany::StateManager> state_mgr_;
  Teuchos::RCP<Tpetra_Vector> x_;
  Teuchos::RCP<Projector> proj_;
#ifdef amb_test_projector
  Teuchos::RCP<testing::ProjectorTester> proj_tester_;
#endif

private:
  typedef unsigned int WsIdx;
  typedef std::pair< std::string, Teuchos::RCP<Field> > Pair;
  typedef std::map< std::string, Teuchos::RCP<Field> > Map;

  Map field_map_;
  std::vector< Teuchos::RCP<Field> > fields_;
  bool building_sfm_, transform_;
  std::vector<short> is_g_;

public:
  Impl (const Teuchos::RCP<Albany::StateManager>& state_mgr,
        const bool use_projection, const bool do_transform)
    : state_mgr_(state_mgr)
  { init(use_projection, do_transform); }
  
  void registerField (
    const std::string& name, const Teuchos::RCP<PHX::DataLayout>& dl,
    const Init::Enum init_G, Transformation::Enum transformation,
    const Teuchos::RCP<Teuchos::ParameterList>& p)
  {
    if ( ! transform_) transformation = Transformation::none;

    const std::string name_rc = decorate(name);
    p->set<std::string>(name_rc + " Name", name_rc);
    p->set< Teuchos::RCP<PHX::DataLayout> >(name_rc + " Data Layout", dl);

    Map::iterator it = field_map_.find(name_rc);
    if (it != field_map_.end()) return;

    Teuchos::RCP<Field> f = Teuchos::rcp(new Field());
    fields_.push_back(f);
    f->name = name;
    f->layout = dl;
    f->num_g_fields = transformation == Transformation::none ? 1 : 2;
    f->data_ = Teuchos::rcp(new Field::Data());
    f->data_->transformation = transformation;

    field_map_.insert(Pair(name_rc, f));

    // Depending on the state variable, different quantities need to be read
    // and written. In all cases, we need two fields.
    //   Holds G and g1.
    registerStateVariable(name_rc, f->layout, init_G);
    //   Holds provisional G and, if needed, g2. If g2 is not needed, then
    // this provisional field is a waste of space and also incurs wasted work
    // in the QP transfer. However, I would need LOCA::AdaptiveSolver to
    // always correctly say, before printSolution is called, whether the mesh
    // will be adapted to avoid this extra storage and work. Maybe in the
    // future.
    registerStateVariable(name_rc + "_1", f->layout, Init::zero);
  }

  void beginAdapt () {
    // Transform G -> g and write to the primary or, depending on state, primary
    // and provisional fields.
    pr("beginAdapt: write final");
    if (proj_.is_null())
      for (Map::const_iterator it = field_map_.begin(); it != field_map_.end();
           ++it)
        for (WsIdx wi = 0; wi < is_g_.size(); ++wi)
          transformStateArray(it->first, wi, Direction::G2g);
    else {
      for (Map::iterator it = field_map_.begin(); it != field_map_.end();
           ++it)
        proj_->project(*it->second);
    }
  }

  void endAdapt (const Teuchos::RCP<const Tpetra_Map>& node_map,
                 const Teuchos::RCP<const Tpetra_Map>& ol_node_map) {
    pr("is_g_.size() was " << is_g_.size() << " and now will be "
       << state_mgr_->getStateArrays().elemStateArrays.size());
    init_g(state_mgr_->getStateArrays().elemStateArrays.size(), true);
    if (Teuchos::nonnull(proj_)) {
      proj_->init(node_map, ol_node_map);
      for (Map::iterator it = field_map_.begin(); it != field_map_.end();
           ++it) {
        Field& f = *it->second;
        for (int i = 0; i < f.num_g_fields; ++i)
          f.data_->mv[i] = Teuchos::rcp(
            new Tpetra_MultiVector(ol_node_map, f.data_->mv[i]->getNumVectors(),
                                   true));
      }
#ifdef amb_test_projector
      proj_tester_->init(node_map, ol_node_map);
#endif
    }
  }

  void initProjector (const Teuchos::RCP<const Tpetra_Map>& node_map,
                      const Teuchos::RCP<const Tpetra_Map>& ol_node_map) {
    if (Teuchos::nonnull(proj_)) {
      proj_->init(node_map, ol_node_map);
#ifdef amb_test_projector
      proj_tester_->init(node_map, ol_node_map);
#endif
    }
  }

  void interpQpField (PHX::MDField<RealType>& f_G_qp,
                      const PHAL::Workset& workset, const BasisField& bf) {
    if (proj_.is_null()) return;
    if (is_g_.empty()) {
      // Special case at startup.
      init_g(state_mgr_->getStateArrays().elemStateArrays.size(), false);
      return;
    }
    if ( ! is_g(workset.wsIndex)) return;
    // Interpolate g at NP to g at QP.
    const std::string name_rc = f_G_qp.fieldTag().name();
    const Teuchos::RCP<Field>& f = field_map_[name_rc];
    proj_->interp(*f, workset, bf, getMDArray(name_rc, workset.wsIndex),
                  getMDArray(name_rc + "_1", workset.wsIndex));
    // Transform g -> G at QP.
    transformStateArray(name_rc, workset.wsIndex, Direction::g2G);
    set_G(workset.wsIndex);
    // If this is the last workset, we're done interpolating, so release the
    // memory.
    if (workset.wsIndex == is_g_.size() - 1)
      for (int i = 0; i < f->num_g_fields; ++i)
        f->data_->mv[i] = Teuchos::null;
  }

  void readQpField (PHX::MDField<RealType>& f, const PHAL::Workset& workset) {
    if (workset.wsIndex == 0) pr("readQpField " << f.fieldTag().name());
    // At startup, is_g_.size() is 0. We also initialized fields to their G, not
    // g, values.
    if (is_g_.empty())
      init_g(state_mgr_->getStateArrays().elemStateArrays.size(), false);
    if (proj_.is_null()) {
      if (is_g(workset.wsIndex)) {
        // If this is the first read after an RCU, transform g -> G.
        transformStateArray(f.fieldTag().name(), workset.wsIndex,
                            Direction::g2G);
        set_G(workset.wsIndex);
      }
    } else {
      // The most obvious reason this exception could be thrown is because
      // EvalT=Jacobian is run before Residual, which I think should not happen.
      TEUCHOS_TEST_FOR_EXCEPTION(
        is_g(workset.wsIndex), std::logic_error,
        "If usingProjection(), then readQpField should always see G, not g.");
    }
    // Read from the primary field.
    read(getMDArray(f.fieldTag().name(), workset.wsIndex), f);
#if 0
    { pr("mda1:");
      const int rank = 2, num_node = f.dimension(1),
        num_qp = f.dimension(2), ndim = 3;
      const Albany::MDArray&
        mda1 = getMDArray(f.fieldTag().name(), workset.wsIndex);
      for (int cell = 0; cell < (int) workset.numCells; ++cell)
        for (int qp = 0; qp < num_qp; ++qp) {
          for (int i = 0; i < ndim; ++i)
            for (int j = 0; j < ndim; ++j)
              std::cout << " " << mda1(cell, qp, i, j);
          std::cout << "\n";
        }
    }
#endif
  }

  void writeQpField (const PHX::MDField<RealType>& f,
                     const PHAL::Workset& workset, const BasisField& wbf) {
    if (workset.wsIndex == 0)
      pr("writeQpField (provisional) " << f.fieldTag().name());
    const std::string name_rc = decorate(f.fieldTag().name());
    if (proj_.is_null()) {
      // Write to the provisional field.
      write(getMDArray(name_rc + "_1", workset.wsIndex), f);
    } else proj_->fillRhs(f, *field_map_[name_rc], workset, wbf);
  }

  Manager::Field::iterator fieldsBegin () { return fields_.begin(); }
  Manager::Field::iterator fieldsEnd () { return fields_.end(); }

  void set_building_sfm (const bool value) {
    building_sfm_ = value;
  }
  bool building_sfm () const { return building_sfm_; }

  void set_evaluating_sfm (const bool before) {
    if (before && Teuchos::nonnull(proj_)) {
      // Zero the nodal values in prep for fillRhs.
      for (Field::iterator it = fields_.begin(); it != fields_.end(); ++it)
        for (int i = 0; i < (*it)->num_g_fields; ++i)
          if (Teuchos::nonnull((*it)->data_->mv[i]))
            (*it)->data_->mv[i]->putScalar(0);
    }
  }

  Transformation::Enum get_transformation (const std::string& name_rc) {
    return field_map_[name_rc]->data_->transformation;
  }

  int numWorksets () const { return is_g_.size(); }

private:
  void init (const bool use_projection, const bool do_transform) {
    transform_ = do_transform;
    building_sfm_ = false;
    if (use_projection) {
      proj_ = Teuchos::rcp(new Projector());
#ifdef amb_test_projector
      if (proj_tester_.is_null())
        proj_tester_ = Teuchos::rcp(new testing::ProjectorTester());
#endif
    }
  }

  void registerStateVariable (
    const std::string& name, const Teuchos::RCP<PHX::DataLayout>& dl,
    const Init::Enum init)
  {
    state_mgr_->registerStateVariable(
      name, dl, "", init == Init::zero ? "scalar" : "identity", 0,
      false, false);
  }

  Albany::MDArray& getMDArray (
    const std::string& name, const WsIdx wi, const bool is_const=true)
  {
    Albany::StateArray& esa = state_mgr_->getStateArrays().elemStateArrays[wi];
    Albany::StateArray::iterator it = esa.find(name);
    TEUCHOS_TEST_FOR_EXCEPTION(
      it == esa.end(), std::logic_error, "elemStateArrays is missing " + name);
    return it->second;
  }

  void init_g (const int n, const bool is_g) {
    is_g_.clear();
    is_g_.resize(n, is_g ? 0 : fields_.size());
  }
  bool is_g (const int ws_idx) const { return is_g_[ws_idx] < fields_.size(); }
  void set_G (const int ws_idx) { ++is_g_[ws_idx]; }

  void transformStateArray (const std::string& name_rc, const WsIdx wi,
                            const Direction::Enum dir) {
    if (wi == 0) pr("transform " << name_rc);
    // Name decoration coordinates with registerField's calls to
    // registerStateVariable.
    const Transformation::Enum transformation = get_transformation(name_rc);
    AAdapt::rc::transformStateArray(
      wi, dir, transformation, getMDArray(name_rc, wi),
      getMDArray(name_rc + "_1", wi));
  }
};

Teuchos::RCP<Manager> Manager::
create (const Teuchos::RCP<Albany::StateManager>& state_mgr,
        Teuchos::ParameterList& problem_params) {
  if ( ! problem_params.isSublist("Adaptation")) return Teuchos::null;
  Teuchos::ParameterList&
    adapt_params = problem_params.sublist("Adaptation", true);
  if (adapt_params.get<bool>("Reference Configuration: Update", false)) {
    const bool use_projection = adapt_params.get<bool>(
      "Reference Configuration: Project", false);
    const bool do_transform = adapt_params.get<bool>(
      "Reference Configuration: Transform", false);
    return Teuchos::rcp(new Manager(state_mgr, use_projection, do_transform));
  }
  else return Teuchos::null;
}

void Manager::setSolutionManager(
  const Teuchos::RCP<AdaptiveSolutionManagerT>& sol_mgr)
{ impl_->sol_mgr_ = sol_mgr; }

void Manager::
getValidParameters (Teuchos::RCP<Teuchos::ParameterList>& valid_pl) {
  valid_pl->set<bool>("Reference Configuration: Update", false,
                      "Send coordinates + solution to SCOREC.");
}

void Manager::init_x_if_not (const Teuchos::RCP<const Tpetra_Map>& map) {
  if (Teuchos::nonnull(impl_->x_)) return;
  impl_->x_ = Teuchos::rcp(new Tpetra_Vector(map));
  impl_->x_->putScalar(0);
}

static void update_x (
  const Teuchos::ArrayRCP<double>& x, const Teuchos::ArrayRCP<const double>& s,
  const Teuchos::RCP<Albany::AbstractDiscretization>& disc)
{
  const int spdim = disc->getNumDim(), neq = disc->getNumEq();
  for (int i = 0; i < x.size(); i += neq)
    for (int j = 0; j < spdim; ++j)
      x[i+j] += s[i+j];  
}

void Manager::update_x (const Tpetra_Vector& soln_nol) {
  // By convention (e.g., in MechanicsProblem), the displacement DOFs are before
  // any other DOFs.
  const Teuchos::ArrayRCP<double>& x = impl_->x_->get1dViewNonConst();
  const Teuchos::ArrayRCP<const double>& s = soln_nol.get1dView();
  AAdapt::rc::update_x(x, s, impl_->state_mgr_->getDiscretization());
}

Teuchos::RCP<const Tpetra_Vector> Manager::
add_x (const Teuchos::RCP<const Tpetra_Vector>& a) const {
  Teuchos::RCP<Tpetra_Vector>
    c = Teuchos::rcp(new Tpetra_Vector(*a, Teuchos::Copy));
  const Teuchos::ArrayRCP<double>& x = c->get1dViewNonConst();
  const Teuchos::ArrayRCP<const double>& s = impl_->x_->get1dView();
  AAdapt::rc::update_x(x, s, impl_->state_mgr_->getDiscretization());
  return c;
}

Teuchos::RCP<Tpetra_Vector>& Manager::get_x () { return impl_->x_; }

template<typename EvalT>
void Manager::createEvaluators (
  PHX::FieldManager<PHAL::AlbanyTraits>& fm,
  const Teuchos::RCP<Albany::Layouts>&)
{
  fm.registerEvaluator<EvalT>(
    Teuchos::rcp(
      new Reader<EvalT, PHAL::AlbanyTraits>(Teuchos::rcp(this, false))));
}

template<>
void Manager::createEvaluators<PHAL::AlbanyTraits::Residual> (
  PHX::FieldManager<PHAL::AlbanyTraits>& fm,
  const Teuchos::RCP<Albany::Layouts>& dl)
{
  typedef PHAL::AlbanyTraits::Residual Residual;
  fm.registerEvaluator<Residual>(
    Teuchos::rcp(
      new Reader<Residual, PHAL::AlbanyTraits>(Teuchos::rcp(this, false), dl)));
  if (impl_->building_sfm()) {
    Teuchos::RCP< Writer<Residual, PHAL::AlbanyTraits> > writer = Teuchos::rcp(
      new Writer<Residual, PHAL::AlbanyTraits>(Teuchos::rcp(this, false), dl));
    fm.registerEvaluator<Residual>(writer);
    fm.requireField<Residual>(*writer->getNoOutputTag());
  }
}
  
void Manager::registerField (
  const std::string& name, const Teuchos::RCP<PHX::DataLayout>& dl,
  const Init::Enum init, const Transformation::Enum transformation,
  const Teuchos::RCP<Teuchos::ParameterList>& p)
{ impl_->registerField(name, dl, init, transformation, p); }

// The asymmetry in naming scheme (interp + read vs just write) emerges from the
// asymmetry in reading and writing. Every EvalT needs access to the RealType
// states for use in computations. In addition, Reader<Residual> needs to do
// interp if projection is used. In contrast, only Writer<Residual> is used, so
// it does both the (optional) projection and write.
void Manager::
beginQpInterp () { /* Do nothing. */ }
void Manager::
interpQpField (PHX::MDField<RealType>& f, const PHAL::Workset& workset,
               const BasisField& bf) {
  impl_->interpQpField(f, workset, bf);
}
void Manager::endQpInterp () { /* Do nothing. */ }

void Manager::
readQpField (PHX::MDField<RealType>& f, const PHAL::Workset& workset)
{ impl_->readQpField(f, workset); }

void Manager::
beginQpWrite (const PHAL::Workset& workset, const BasisField& bf,
              const BasisField& wbf) {
  if (impl_->proj_.is_null()) return;
  impl_->proj_->fillMassMatrix(workset, bf, wbf);
}
void Manager::
writeQpField (const PHX::MDField<RealType>& f, const PHAL::Workset& workset,
              const BasisField& wbf) {
  impl_->writeQpField(f, workset, wbf);
}
void Manager::endQpWrite () { /* Do nothing. */ }

void Manager::
testProjector (
  const PHAL::Workset& workset, const BasisField& bf, const BasisField& wbf,
  const PHX::MDField<RealType,Cell,Vertex,Dim>& coord_vert,
  const PHX::MDField<RealType,Cell,QuadPoint,Dim>& coord_qp)
{
#ifdef amb_test_projector
  impl_->proj_tester_->eval(workset, bf, wbf, coord_vert, coord_qp);
#endif
}

const Teuchos::RCP<Tpetra_MultiVector>& Manager::
getNodalField (const Field& f, const int g_idx, const bool overlapped) const {
  TEUCHOS_TEST_FOR_EXCEPTION( ! overlapped, std::logic_error,
                             "must be overlapped");
  return f.data_->mv[g_idx];
}

Manager::Field::iterator Manager::fieldsBegin ()
{ return impl_->fieldsBegin(); }
Manager::Field::iterator Manager::fieldsEnd ()
{ return impl_->fieldsEnd(); }

void Manager::beginBuildingSfm () { impl_->set_building_sfm(true); }
void Manager::endBuildingSfm () { impl_->set_building_sfm(false); }

void Manager::beginEvaluatingSfm () { impl_->set_evaluating_sfm(true); }
void Manager::endEvaluatingSfm () { impl_->set_evaluating_sfm(false); }

void Manager::beginAdapt () { impl_->beginAdapt(); }
void Manager::endAdapt (const Teuchos::RCP<const Tpetra_Map>& node_map,
                        const Teuchos::RCP<const Tpetra_Map>& ol_node_map)
{ impl_->endAdapt(node_map, ol_node_map); }
void Manager::initProjector (const Teuchos::RCP<const Tpetra_Map>& node_map,
                             const Teuchos::RCP<const Tpetra_Map>& ol_node_map)
{ impl_->initProjector(node_map, ol_node_map); }

bool Manager::usingProjection () const
{ return Teuchos::nonnull(impl_->proj_); }

Manager::Manager (const Teuchos::RCP<Albany::StateManager>& state_mgr,
                  const bool use_projection, const bool do_transform)
  : impl_(Teuchos::rcp(new Impl(state_mgr, use_projection, do_transform)))
{}

#define eti_fn(EvalT)                                   \
  template void Manager::createEvaluators<EvalT>(       \
    PHX::FieldManager<PHAL::AlbanyTraits>& fm,          \
    const Teuchos::RCP<Albany::Layouts>& dl);
aadapt_rc_apply_to_all_eval_types(eti_fn)
#undef eti_fn

namespace {
namespace testing {
typedef Intrepid2::Tensor<RealType> Tensor;

// Some deformation gradient tensors with det(F) > 0 for use in testing.
/*
   pr('{'); for (i = 1:3)
   pr('{'); for (j = 1:3) pr('%22.15e',F(i,j)); if (j < 3) pr(', '); end; end
   pr('}'); if (i < 3) pr(','); else pr('}'); end; pr('\n');
   end
*/
static const double Fs[3][3][3] = {
  {{-7.382752820294219e-01, -1.759182226321058e+00,  1.417301043170359e+00},
   { 7.999093048231801e-01,  5.295155264305610e-01, -3.075207765325406e-02},
   { 6.283454283198379e-02,  4.117063384659416e-01, -1.243061703605918e-01}},
  {{ 4.929646496030746e-01, -1.672547330507927e+00,  1.374629761307942e-01},
   { 9.785301515971359e-01,  8.608882413324722e-01,  6.315167262108045e-01},
   {-5.339914726510328e-01, -1.559378791976819e+00,  1.242404824706601e-01}},
  {{ 1.968477583454205e+00,  1.805729439108956e+00, -2.759426722073080e-01},
   { 7.787416415696722e-01, -5.361220317998502e-03,  1.838993634875665e-01},
   {-1.072168271881842e-02,  3.771872253769205e-01, -9.553540517889956e-01}}};

// Some sample functions. Only the constant and linear ones should be
// interpolated exactly.
double eval_f (const double x, const double y, const double z, int ivec) {
  static const double R = 0.15, H = 0.005;
  switch (ivec + 1) {
  case 1: return 2;
  case 2: return 1.5*x + 2*y + 3*z;
  case 3: return x*x + y*y + z;
  case 4: return x*x*x - x*x*y + x*y*y - y*y*y;
  case 5: return cos(2*M_PI*x/R) + sin(2*M_PI*y/R) + z;
  case 6: return x*x*x*x;
  case 7: return x*x - y*y + x*y + z;
  case 8: return x*x;
  case 9: return x*x*x;
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Error: unhandled argument in evalf() in AAdapt_RC_Manager.cpp" << std::endl);
}

// Axis-aligned bounding box on the vertices.
void getBoundingBox (const PHX::MDField<RealType, Cell, Vertex, Dim>& vs,
                     RealType lo[3], RealType hi[3]) {
  bool first = true;
  for (int cell = 0; cell < vs.dimension(0); ++cell)
    for (int iv = 0; iv < vs.dimension(1); ++iv) {
      for (int id = 0; id < vs.dimension(2); ++id) {
        const RealType v = vs(cell, iv, id);
        if (first) lo[id] = hi[id] = v;
        else {
          lo[id] = std::min(lo[id], v);
          hi[id] = std::max(hi[id], v);
        }
      }
      first = false;
    }
}

// F field.
Intrepid2::Tensor<RealType> eval_F (const RealType p[3]) {
#define in01(u) (0 <= (u) && (u) <= 1)
  TEUCHOS_ASSERT(in01(p[0]) && in01(p[1]) && in01(p[2]));
#undef in01
#define lpij for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j)
  Tensor r(3), s(3);
  lpij r(i,j) = s(i,j) = 0;
  for (int k = 0; k < 3; ++k) {
    Tensor F(3);
    lpij F(i,j) = Fs[k][i][j];
    std::pair<Tensor, Tensor> RS = Intrepid2::polar_right(F);
    RS.first = Intrepid2::log_rotation(RS.first);
    RS.second = Intrepid2::log_sym(RS.second);
    symmetrize(RS.second);
    // Right now, we are not doing anything to handle phase wrapping in r =
    // logm(R). That means that projection with Lie transformation does not work
    // in general right now. But test the correctness of the projector in the
    // case that no phase wrap occurs. Here I do that by using only one F's r.
    if (k == 0) r += p[k]*RS.first;
    s += p[k]*RS.second;
  }
  const Tensor R = Intrepid2::exp_skew_symmetric(r);
  Tensor S = Intrepid2::exp(s); symmetrize(S);
  return Intrepid2::dot(R, S);
#undef lpij
}

// The following methods test whether q == interp(M \ b(q)). q is a linear
// function of space and that lives on the integration points. M is the mass
// matrix. b(q) is the integral over each element. q* = M \ b(q) is the L_2
// projection onto the nodal points. interp(q*) is the interpolation back to the
// integration points.
//   This first method runs from start to finish but works for only one workset
// and in serial. I'll probably remove this one at some point.
//   ProjectorTester works in all cases.
void testProjector (
  const Projector& pc, const PHAL::Workset& workset,
  const Manager::BasisField& bf, const Manager::BasisField& wbf,
  const PHX::MDField<RealType, Cell, Vertex, Dim>& coord_vert,
  const PHX::MDField<RealType, Cell, QuadPoint, Dim>& coord_qp)
{
  // Works only in the case of one workset.
  TEUCHOS_ASSERT(workset.wsIndex == 0);
  const unsigned int wi = 0;

  // Set up the data containers.
  typedef PHX::MDALayout<Cell, QuadPoint, Dim, Dim> Layout;
  Teuchos::RCP<Layout> layout = Teuchos::rcp(
    new Layout(workset.numCells, coord_qp.dimension(1),
               coord_qp.dimension(2), coord_qp.dimension(2)));
  PHX::MDField<RealType> f_mdf = PHX::MDField<RealType>("f_mdf", layout);
  f_mdf.setFieldData(
    PHX::KokkosViewFactory<RealType, PHX::Device>::buildView(f_mdf.fieldTag()));
    
  std::vector<Albany::MDArray> mda;
  std::vector<double> mda_data[2];
  for (int i = 0; i < 2; ++i) {
    typedef Albany::MDArray::size_type size_t;
    mda_data[i].resize(f_mdf.dimension(0) * f_mdf.dimension(1) *
                       f_mdf.dimension(2) * f_mdf.dimension(3));
    shards::Array<RealType, shards::NaturalOrder, Cell, QuadPoint, Dim, Dim> a;
    a.assign(&mda_data[i][0], (size_t) f_mdf.dimension(0),
             (size_t) f_mdf.dimension(1), (size_t) f_mdf.dimension(2),
             (size_t) f_mdf.dimension(3));
    mda.push_back(a);
  }

  Projector p;
  p.init(pc.get_node_map(), pc.get_ol_node_map());

  // M.
  p.fillMassMatrix(workset, bf, wbf);
    
  for (int test = 0; test < 2; ++test) {
    Manager::Field f;
    f.name = "";
    f.layout = layout;
    f.data_ = Teuchos::rcp(new Manager::Field::Data());

    // b.
    if (test == 0) {
      f.data_->transformation = Transformation::none;
      f.num_g_fields = 1;
      loop(f_mdf, cell, 0) loop(f_mdf, qp, 1)
        for (int i = 0, k = 0; i < f_mdf.dimension(2); ++i)
          for (int j = 0; j < f_mdf.dimension(3); ++j, ++k)
            f_mdf(cell, qp, i, j) = eval_f(
              coord_qp(cell, qp, 0), coord_qp(cell, qp, 1),
              coord_qp(cell, qp, 2), k);
    } else {
      f.data_->transformation = Transformation::right_polar_LieR_LieS;
      f.num_g_fields = 2;
      RealType lo[3], hi[3];
      getBoundingBox(coord_vert, lo, hi);
      loop(f_mdf, cell, 0) loop(f_mdf, qp, 1) {
        RealType p[3];
        for (int k = 0; k < 3; ++k)
          p[k] = (coord_qp(cell, qp, k) - lo[k])/(hi[k] - lo[k]);
        const Intrepid2::Tensor<RealType> F = eval_F(p);
        loop(f_mdf, i, 2) loop(f_mdf, j, 3) f_mdf(cell, qp, i, j) = F(i, j);
      }
    }
    p.fillRhs(f_mdf, f, workset, wbf);

    // Solve M x = b.
    p.project(f);

    if (test == 0) { // Compare with true values at NP.
      const int ncol = 9, nverts = pc.get_node_map()->getGlobalNumElements();
      std::vector<RealType> f_true(ncol * nverts); {
        std::vector<bool> evaled(nverts, false);
        loop(f_mdf, cell, 0) loop(coord_vert, node, 1) {
          const GO gid = workset.wsElNodeID[cell][node];
          const LO lid = pc.get_node_map()->getLocalElement(gid);
          if ( ! evaled[lid]) {
            for (int k = 0; k < ncol; ++k)
              f_true[ncol*lid + k] = eval_f(
                coord_vert(cell, node, 0), coord_vert(cell, node, 1),
                coord_vert(cell, node, 2), k);
            evaled[lid] = true;
          }
        }
      }
      double err1[9], errmax[9], scale[9];
      for (int k = 0; k < ncol; ++k) { err1[k] = errmax[k] = scale[k] = 0; }
      for (int iv = 0; iv < nverts; ++iv)
        for (int k = 0; k < ncol; ++k) {
          const double d = std::abs(
            f.data_->mv[0]->getVector(k)->getData()[iv] - f_true[ncol*iv + k]);
        err1[k] += d;
        errmax[k] = std::max(errmax[k], d);
        scale[k] = std::max(scale[k], std::abs(f_true[ncol*iv + k]));
      }
      printf("err np (test %d):", test);
      const int n = f_mdf.dimension(0) * f_mdf.dimension(1);
      for (int k = 0; k < 9; ++k)
        printf(" %1.2e %1.2e (%1.2e)", err1[k]/(n*scale[k]), errmax[k]/scale[k],
               scale[k]);
      std::cout << "\n";
    }

    // Interpolate to IP.
    p.interp(f, workset, bf, mda[0], mda[1]);
    transformStateArray(wi, Direction::g2G, f.data_->transformation, mda[0],
                        mda[1]);

    { // Compare with true values at IP.
      double err1[9], errmax[9], scale[9];
      for (int k = 0; k < 9; ++k) { err1[k] = errmax[k] = scale[k] = 0; }
      loop(f_mdf, cell, 0) loop(f_mdf, qp, 1)
        for (int i = 0, k = 0; i < f_mdf.dimension(2); ++i)
          for (int j = 0; j < f_mdf.dimension(3); ++j, ++k) {
            const double d = std::abs(mda[0]((int) cell, (int) qp, i, j) -
                                      f_mdf(cell, qp, i, j));
            err1[k] += d;
            errmax[k] = std::max(errmax[k], d);
            scale[k] = std::max(scale[k], std::abs(f_mdf(cell, qp, i, j)));
          }
      printf("err ip (test %d):", test);
      const int n = f_mdf.dimension(0) * f_mdf.dimension(1);
      for (int k = 0; k < 9; ++k)
        printf(" %1.2e %1.2e (%1.2e)", err1[k]/(n*scale[k]), errmax[k]/scale[k],
               scale[k]);
      std::cout << "\n";
    }
  }
}

struct ProjectorTester::Impl {
  enum { ntests = 2 };
  bool projected, finished;
  Projector p;
  struct Point {
    RealType x[3];
    bool operator< (const Point& p) const {
      for (int i = 0; i < 3; ++i) {
        if (x[i] < p.x[i]) return true;
        if (x[i] > p.x[i]) return false;
      }
      return false;
    }
  };
  struct FValues { RealType f[9]; };
  typedef std::map<Point, FValues> Map;
  struct TestData {
    Manager::Field f;
    Map f_true_qp, f_interp_qp;
  };
  TestData td[ntests];
};

ProjectorTester::ProjectorTester () {
  pr("ProjectorTester::ProjectorTester");
  d = Teuchos::rcp(new Impl());
  for (int test = 0; test < Impl::ntests; ++test) {
    Impl::TestData& td = d->td[test];
    Manager::Field& f = td.f;
    f.name = "";
    f.data_ = Teuchos::rcp(new Manager::Field::Data());
    if (test == 0) {
      f.data_->transformation = Transformation::none;
      f.num_g_fields = 1;
    } else {
      f.data_->transformation = Transformation::right_polar_LieR_LieS;
      f.num_g_fields = 2;
    }
  }
}

void ProjectorTester::
init (const Teuchos::RCP<const Tpetra_Map>& node_map,
      const Teuchos::RCP<const Tpetra_Map>& ol_node_map) {
  d->p.init(node_map, ol_node_map);
  d->projected = d->finished = false;
}

// Figure out what needs to be done given the current state.
void ProjectorTester::
eval (const PHAL::Workset& workset,
      const Manager::BasisField& bf, const Manager::BasisField& wbf,
      const PHX::MDField<RealType, Cell, Vertex, Dim>& coord_vert,
      const PHX::MDField<RealType, Cell, QuadPoint, Dim>& coord_qp) {
  if (d->finished) return;
  const int num_qp = coord_qp.dimension(1), num_dim = coord_qp.dimension(2);
  if (workset.numCells > 0 && num_qp > 0) {
    Impl::Point p;
    for (int i = 0; i < 3; ++i) p.x[i] = coord_qp(0, 0, i);
    Impl::Map::const_iterator it = d->td[0].f_true_qp.find(p);
    if (it == d->td[0].f_true_qp.end()) {
      d->p.fillMassMatrix(workset, bf, wbf);
      fillRhs(workset, wbf, coord_vert, coord_qp);
    } else {
      if ( ! d->projected) {
        project();
        d->projected = true;
      }
      it = d->td[0].f_interp_qp.find(p);
      if (it == d->td[0].f_interp_qp.end())
        interp(workset, bf, coord_vert, coord_qp);
      else {
        finish(workset);
        d->finished = true;
      }
    }
  }
}

void ProjectorTester::
fillRhs (const PHAL::Workset& workset, const Manager::BasisField& wbf,
         const PHX::MDField<RealType, Cell, Vertex, Dim>& coord_vert,
         const PHX::MDField<RealType, Cell, QuadPoint, Dim>& coord_qp) {
  const int num_qp = coord_qp.dimension(1), num_dim = coord_qp.dimension(2);
  
  typedef PHX::MDALayout<Cell, QuadPoint, Dim, Dim> Layout;
  Teuchos::RCP<Layout> layout = Teuchos::rcp(
    new Layout(workset.numCells, num_qp, num_dim, num_dim));
  PHX::MDField<RealType> f_mdf = PHX::MDField<RealType>("f_mdf", layout);
  f_mdf.setFieldData(
    PHX::KokkosViewFactory<RealType, PHX::Device>::buildView(f_mdf.fieldTag()));

  for (int test = 0; test < Impl::ntests; ++test) {
    Impl::TestData& td = d->td[test];
    Manager::Field& f = td.f;
    f.layout = layout;
    
    // Fill f_mdf and f_true_qp.
    loop(f_mdf, cell, 0) loop(f_mdf, qp, 1) {
      Impl::Point p;
      for (int i = 0; i < 3; ++i) p.x[i] = coord_qp(cell, qp, i);
      Impl::FValues fv;
      if (test == 0)
        for (int k = 0; k < 9; ++k) fv.f[k] = eval_f(p.x[0], p.x[1], p.x[2], k);
      else {
        // I don't have a bounding box, so come up with something reasonable.
        RealType alpha[3] = {0, 0, 0};
        alpha[0] = (100 + p.x[0])/200;
        const Intrepid2::Tensor<RealType> F = eval_F(alpha);
        loop(f_mdf, i, 2) loop(f_mdf, j, 3) fv.f[num_dim*i + j] = F(i,j);
      }
      td.f_true_qp[p] = fv;
      loop(f_mdf, i, 2) loop(f_mdf, j, 3)
        f_mdf(cell, qp, i, j) = fv.f[num_dim*i + j];
    }

    d->p.fillRhs(f_mdf, f, workset, wbf);
  }
}

void ProjectorTester::project () {
  for (int test = 0; test < Impl::ntests; ++test) d->p.project(d->td[test].f);
}

void ProjectorTester::
interp (const PHAL::Workset& workset, const Manager::BasisField& bf,
        const PHX::MDField<RealType, Cell, Vertex, Dim>& coord_vert,
        const PHX::MDField<RealType, Cell, QuadPoint, Dim>& coord_qp) {
  const int num_qp = coord_qp.dimension(1), num_dim = coord_qp.dimension(2);

  // Quick exit if we've already done this workset.
  if (workset.numCells > 0 && num_qp > 0) {
    Impl::Point p;
    for (int i = 0; i < 3; ++i) p.x[i] = coord_qp(0, 0, i);
    const Impl::Map::const_iterator it = d->td[0].f_interp_qp.find(p);
    if (it != d->td[0].f_interp_qp.end()) return;
  }

  std::vector<Albany::MDArray> mda;
  std::vector<double> mda_data[2];
  for (int i = 0; i < 2; ++i) {
    typedef Albany::MDArray::size_type size_t;
    mda_data[i].resize(workset.numCells * num_qp * num_dim * num_dim);
    shards::Array<RealType, shards::NaturalOrder, Cell, QuadPoint, Dim, Dim> a;
    a.assign(&mda_data[i][0], workset.numCells, num_qp, num_dim, num_dim);
    mda.push_back(a);
  }

  for (int test = 0; test < Impl::ntests; ++test) {
    Impl::TestData& td = d->td[test];
    Manager::Field& f = td.f;
    // Interpolate to IP.
    d->p.interp(f, workset, bf, mda[0], mda[1]);
    transformStateArray(workset.wsIndex, Direction::g2G,
                        f.data_->transformation, mda[0], mda[1]);
    // Record for later comparison.
    loop(mda[0], cell, 0) loop(mda[0], qp, 1) {
      Impl::Point p;
      for (int i = 0; i < 3; ++i) p.x[i] = coord_qp(cell, qp, i);
      Impl::FValues fv;
      loop(mda[0], i, 2) loop(mda[0], j, 3)
        fv.f[num_dim*i + j] = mda[0](cell, qp, i, j);
      td.f_interp_qp[p] = fv;
    }
  }
}

void ProjectorTester::finish (const PHAL::Workset& workset) {
  for (int test = 0; test < Impl::ntests; ++test) {
    Impl::TestData& td = d->td[test];
    // Compare with true values at IP.
    double err1[9], errmax[9], scale[9];
    for (int k = 0; k < 9; ++k) { err1[k] = errmax[k] = scale[k] = 0; }
    for (Impl::Map::const_iterator it = td.f_true_qp.begin();
         it != td.f_true_qp.end(); ++it) {
      const Impl::Point& p = it->first;
      const Impl::FValues& fv_true = it->second;
      const Impl::Map::const_iterator it_interp = td.f_interp_qp.find(p);
      if (it_interp == td.f_interp_qp.end()) {
        pr("ProjectorTester::finish(): Failed to find matching f_interp_qp.");
        pr(p.x[0] << " " << p.x[1] << " " << p.x[2]);
        break;
      }
      const Impl::FValues& fv_interp = it_interp->second;
      for (int k = 0; k < 9; ++k) {
        const double d = std::abs(fv_true.f[k] - fv_interp.f[k]);
        err1[k] += d;
        errmax[k] = std::max(errmax[k], d);
        scale[k] = std::max(scale[k], std::abs(fv_true.f[k]));
      }
    }

    double gerr1[9], gerrmax[9], gscale[9];
    int gn;
    const Teuchos::RCP<const Teuchos::Comm<int> >
      comm = Teuchos::DefaultComm<int>::getComm();
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 9, err1, gerr1);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 9, errmax, gerrmax);
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 9, scale, gscale);
    const int n = td.f_true_qp.size();
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &n, &gn);

    if (comm->getRank() == 0) {
      printf("err ip (test %d):", test);
      for (int k = 0; k < 9; ++k)
        printf(" %1.2e %1.2e (%1.2e)", gerr1[k]/(gn*gscale[k]),
               gerrmax[k]/gscale[k], gscale[k]);
      std::cout << "\n";
    }

    // Reset for next one.
    td.f_true_qp.clear();
    td.f_interp_qp.clear();
    td.f.data_->mv[0] = td.f.data_->mv[1] = Teuchos::null;
  }
}
} // namespace testing
} // namespace
} // namespace rc
} // namespace AAdapt

#undef loop
