//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <Intrepid_MiniTensor.h>
#include <Phalanx_FieldManager.hpp>
#include "AAdapt_AdaptiveSolutionManagerT.hpp"
#include "AAdapt_RC_DataTypes.hpp"
#include "AAdapt_RC_DataTypes_impl.hpp"
#include "AAdapt_RC_Reader.hpp"
#include "AAdapt_RC_Writer.hpp"
#include "AAdapt_RC_Projector_impl.hpp"
#include "AAdapt_RC_Manager.hpp"

#define pr(msg) std::cout << "amb: (rc) " << msg << std::endl;
//#define pr(msg)
#define amb_do_transform
//#define amb_do_check

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

void read (const Albany::MDArray& mda, PHX::MDField<RealType>& f) {
  switch (f.rank()) {
  case 2:
    loop(f, cell, 0) loop(f, qp, 1)
      f(cell, qp) = mda(cell, qp);
    break;
  case 3:
    loop(f, cell, 0) loop(f, qp, 1) loop(f, i0, 2)
      f(cell, qp, i0) = mda(cell, qp, i0);
    break;
  case 4:
    loop(f, cell, 0) loop(f, qp, 1) loop(f, i0, 2) loop(f, i1, 3)
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
    loop(f, cell, 0) loop(f, qp, 1)
      mda(cell, qp) = f(cell, qp);
    break;
  case 3:
    loop(f, cell, 0) loop(f, qp, 1) loop(f, i0, 2)
      mda(cell, qp, i0) = f(cell, qp, i0);
    break;
  case 4:
    loop(f, cell, 0) loop(f, qp, 1) loop(f, i0, 2) loop(f, i1, 3) {
      if (f(cell, qp, i0, i1) != 0)
        mda(cell, qp, i0, i1) = f(cell, qp, i0, i1);
    }
    break;
  default:
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                               "dims.size() \notin {2,3,4}.");
  }
}

#ifdef amb_do_check
class Checker {
private:
  typedef Intrepid::Tensor<RealType> Tensor;
  int wi_, cell_, qp_;
  void display (const std::string& name, const Tensor& a,
                const std::string& msg) {
    std::stringstream ss;
    const int rank = Teuchos::DefaultComm<int>::getComm()->getRank();
    ss << "amb: Checker: On rank " << rank << " with (wi, cell, qp) = ("
       << wi_ << ", " << cell_ << ", " << qp_ << "), " << name
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
  Checker (int wi, int cell, int qp) : wi_(wi), cell_(cell), qp_(qp) {}
#define loopa(i, dim) for (Intrepid::Index i = 0; i < a.get_dimension(); ++i)
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
  bool first () const { return wi_ == 0 && cell_ == 0 && qp_ == 0; }
  bool is_rotation (const std::string& name, const Tensor& a) {
    const double det = Intrepid::det(a);
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
  typedef Intrepid::Tensor<RealType> Tensor;
public:
  Checker (int wi, int cell, int qp) {}
  bool first () const { return false; }
#define empty(s) bool s (const std::string&, const Tensor&) { return true; }
  empty(ok_numbers) empty(is_rotation) empty(is_symmetric)
  empty(is_antisymmetric)
#undef empty
};    
#endif // amb_do_check

static Intrepid::Tensor<RealType>&
symmetrize (Intrepid::Tensor<RealType>& a) {
  const Intrepid::Index dim = a.get_dimension();
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
  const Intrepid::Tensor<RealType>& F,
  std::pair< Intrepid::Tensor<RealType>, Intrepid::Tensor<RealType> >& RS,
  Checker& c)
{
  c.ok_numbers("F", F);
  RS = Intrepid::polar_right(F);
  if (c.first())
    pr("F = [\n" << F << "];\nR = [\n" << RS.first << "];\nS = [\n"
       << RS.second << "];");
  c.ok_numbers("R", RS.first); c.ok_numbers("S", RS.second);
  c.is_rotation("R", RS.first); c.is_symmetric("S", RS.second);
  RS.first = Intrepid::log_rotation(RS.first);
  c.ok_numbers("r", RS.first); c.is_antisymmetric("r", RS.first);
  RS.second = Intrepid::log_sym(RS.second);
  symmetrize(RS.second);
  c.ok_numbers("s", RS.second); c.is_symmetric("s write", RS.second);
  if (c.first()) pr("r =\n" << RS.first << " s =\n" << RS.second);
}

void calc_right_polar_LieR_LieS_g2G (
  Intrepid::Tensor<RealType>& R, Intrepid::Tensor<RealType>& S, Checker& c)
{
  c.ok_numbers("r", R); c.is_antisymmetric("r", R);
  c.ok_numbers("s", S); c.is_symmetric("s read", S);
  if (c.first()) pr("r = [\n" << R << "];\ns = [\n" << S << "];");
  // Math.
  R = Intrepid::exp_skew_symmetric(R);
  c.ok_numbers("R", R); c.is_rotation("R", R);
  S = Intrepid::exp(S);
  symmetrize(S);
  c.ok_numbers("S", S); c.is_symmetric("S", S);
  R = Intrepid::dot(R, S);
  c.ok_numbers("F", R);
  if (c.first()) pr("F = [\n" << R << "];");
}

class Projector {
  typedef PHX::MDField<RealType,Cell,Node,QuadPoint> BasisField;
  typedef BasisField::size_type size_type;

  Teuchos::RCP<const Tpetra_Map> node_map_, ol_node_map_;
  Teuchos::RCP<Tpetra_CrsMatrix> M_;
  // M_ persists over multiple state field manager evaluations if the mesh is
  // not adapted after every LOCA step. Indicate whether this part of M_ has
  // already been filled.
  std::vector<bool> filled_;

  bool is_filled (int wi) {
    if (filled_.size() <= wi)
      filled_.insert(filled_.end(), wi - filled_.size() + 1, false);
    return filled_[wi];
  }

public:
  Projector () { pr("Projector ctor\n"); }

  void init (const Teuchos::RCP<const Tpetra_Map>& node_map,
             const Teuchos::RCP<const Tpetra_Map>& ol_node_map) {
    pr("Projector::init");
    node_map_ = node_map.create_weak();
    ol_node_map_ = ol_node_map.create_weak();
    const size_t max_num_entries = 27; // Enough for first-order hex.
    M_ = Teuchos::rcp(
      new Tpetra_CrsMatrix(ol_node_map_, ol_node_map_, max_num_entries));
    filled_.clear();
  }

  void fillMassMatrix (const PHAL::Workset& workset, const BasisField& bf,
                       const BasisField& wbf) {
    if (is_filled(workset.wsIndex)) return;
    filled_[workset.wsIndex] = true;
    if (workset.wsIndex == 0) pr("Projector::fillMassMatrix");

    const size_type num_nodes = bf.dimension(1), num_qp = bf.dimension(2);
    for (unsigned int cell = 0; cell < workset.numCells; ++cell)
      for (size_type rnode = 0; rnode < num_nodes; ++rnode) {
        const GO row = workset.wsElNodeID[cell][rnode];
        Teuchos::Array<GO> cols;
        Teuchos::Array<ST> vals;
        for (size_type cnode = 0; cnode < num_nodes; ++cnode) {
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

  void fillRhs (const PHX::MDField<RealType>& f_G_qp, Manager::Field& f,
                const PHAL::Workset& workset, const BasisField& bf,
                const BasisField& wbf) {
    if (workset.wsIndex == 0) pr("Projector::fillRhs " << f.name);

  }

  void project (Manager::Field& f) {
    pr("Projector::project " << f.name);
    if ( ! M_->isFillComplete()) {
      pr("Projector::project: doing fillComplete stuff");
      M_->fillComplete();
      // Remap.

    }

  }

  void interp (const Manager::Field& f, const PHAL::Workset& workset,
               const BasisField& bf, const BasisField& wbf,
               Albany::MDArray& mda1, Albany::MDArray& mda2) {
    if (workset.wsIndex == 0) pr("Projector::interp " << f.name);

  }
};

} // namespace

struct Manager::Impl {
  Teuchos::RCP<AdaptiveSolutionManagerT> sol_mgr_;
  Teuchos::RCP<Tpetra_Vector> x_;
  Teuchos::RCP<Projector> proj_;

private:
  typedef unsigned int WsIdx;
  typedef std::pair< std::string, Teuchos::RCP<Field> > Pair;
  typedef std::map< std::string, Teuchos::RCP<Field> > Map;

  Teuchos::RCP<Albany::StateManager> state_mgr_;
  Map field_map_;
  std::vector< Teuchos::RCP<Field> > fields_;
  bool building_sfm_;
  std::vector<bool> is_g_;

public:
  Impl (const Teuchos::RCP<Albany::StateManager>& state_mgr,
        const bool use_projection)
    : state_mgr_(state_mgr)
  { init(use_projection); }

  void registerField (
    const std::string& name, const Teuchos::RCP<PHX::DataLayout>& dl,
    const Init::Enum init_G, /*const*/ Transformation::Enum transformation,
    const Teuchos::RCP<Teuchos::ParameterList>& p)
  {
#ifndef amb_do_transform
    transformation = Transformation::none;
#endif

    const std::string name_rc = decorate(name);
    p->set<std::string>(name_rc + " Name", name_rc);
    p->set< Teuchos::RCP<PHX::DataLayout> >(name_rc + " Data Layout", dl);

    Map::iterator it = field_map_.find(name_rc);
    if (it != field_map_.end()) return;

    Teuchos::RCP<Field> f = Teuchos::rcp(new Field());
    fields_.push_back(f);
    f->name = name;
    f->layout = dl;
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
      for (Map::const_iterator it = field_map_.begin();
           it != field_map_.end(); ++it)
        for (WsIdx wi = 0; wi < is_g_.size(); ++wi)
          transformStateArray(it->first, wi, Direction::G2g);
    else
      for (Map::iterator it = field_map_.begin(); it != field_map_.end();
           ++it)
        proj_->project(*it->second);
  }

  void endAdapt (const Teuchos::RCP<const Tpetra_Map>& node_map,
                 const Teuchos::RCP<const Tpetra_Map>& ol_node_map) {
    pr("is_g_.size() was " << is_g_.size() << " and now will be "
       << state_mgr_->getStateArrays().elemStateArrays.size());
    is_g_.clear();
    is_g_.resize(state_mgr_->getStateArrays().elemStateArrays.size(), true);
    if (Teuchos::nonnull(proj_)) proj_->init(node_map, ol_node_map);
  }

  void initProjector (const Teuchos::RCP<const Tpetra_Map>& node_map,
                      const Teuchos::RCP<const Tpetra_Map>& ol_node_map) {
    if (Teuchos::nonnull(proj_)) proj_->init(node_map, ol_node_map);
  }

  void interpQpField (
    PHX::MDField<RealType>& f_G_qp, const PHAL::Workset& workset,
    const BasisField& bf, const BasisField& wbf)
  {
    if (proj_.is_null()) return;
    if (is_g_.empty()) {
      // Special case at startup.
      is_g_.resize(state_mgr_->getStateArrays().elemStateArrays.size(), false);
      return;
    }
    if ( ! is_g_[workset.wsIndex]) return;
    // Interpolate g at NP to g at QP.
    const std::string name_rc = f_G_qp.fieldTag().name();
    const Teuchos::RCP<Field>& f = field_map_[name_rc];
    proj_->interp(*f, workset, bf, wbf, getMDArray(name_rc, workset.wsIndex),
                  getMDArray(name_rc + "_1", workset.wsIndex));
    // Transform g -> G at QP.
    transformStateArray(name_rc, workset.wsIndex, Direction::g2G);
    is_g_[workset.wsIndex] = false;
    // If this is the last workset, we're done interpolating, so release the
    // memory.
    if (workset.wsIndex == is_g_.size() - 1)
      for (int i = 0; i < f->num_g_fields; ++i)
        f->data_->mv[i] = Teuchos::null;
  }

  void readQpField (PHX::MDField<RealType>& f,
                    const PHAL::Workset& workset) {
    if (workset.wsIndex == 0) pr("readQpField " << f.fieldTag().name());
    // At startup, is_g_.size() is 0. We also initialized fields to their G, not
    // g, values.
    if (is_g_.empty())
      is_g_.resize(state_mgr_->getStateArrays().elemStateArrays.size(), false);
    if (proj_.is_null()) {
      if (is_g_[workset.wsIndex]) {
        // If this is the first read after an RCU, transform g -> G.
        transformStateArray(f.fieldTag().name(), workset.wsIndex,
                            Direction::g2G);
        is_g_[workset.wsIndex] = false;
      }
    } else {
      // The most obvious reason this exception could be thrown is because
      // EvalT=Jacobian is run before Residual, which I think should not happen.
      TEUCHOS_TEST_FOR_EXCEPTION(
        is_g_[workset.wsIndex], std::logic_error,
        "If usingProjection(), then readQpField should always see G, not g.");
    }
    // Read from the primary field.
    read(getMDArray(f.fieldTag().name(), workset.wsIndex), f);
  }
  void writeQpField (
    const PHX::MDField<RealType>& f, const PHAL::Workset& workset,
    const BasisField& bf, const BasisField& wbf)
  {
    if (workset.wsIndex == 0)
      pr("writeQpField (provisional) " << f.fieldTag().name());
    const std::string name_rc = decorate(f.fieldTag().name());
    if (proj_.is_null()) {
      // Write to the provisional field.
      write(getMDArray(name_rc + "_1", workset.wsIndex), f);
    } else proj_->fillRhs(f, *field_map_[name_rc], workset, bf, wbf);
  }

  Manager::Field::iterator fieldsBegin () { return fields_.begin(); }
  Manager::Field::iterator fieldsEnd () { return fields_.end(); }

  void set_building_sfm (const bool value) {
    building_sfm_ = value;
  }
  bool building_sfm () const { return building_sfm_; }

  Transformation::Enum get_transformation (const std::string& name_rc) {
    return field_map_[name_rc]->data_->transformation;
  }

private:
  void init (const bool use_projection) {
    building_sfm_ = false;
    if (use_projection) proj_ = Teuchos::rcp(new Projector());
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

  void transformStateArray (const std::string& name_rc, const WsIdx wi,
                            const Direction::Enum dir) {
    if (wi == 0) pr("transform " << name_rc);
    // Name decoration coordinates with registerField's calls to
    // registerStateVariable.
    const Transformation::Enum transformation = get_transformation(name_rc);
    switch (transformation) {
    case Transformation::none: {
      if (wi == 0) pr("none " << (dir == Direction::g2G));
      if (dir == Direction::G2g) {
        // Copy from the provisional to the primary field.
        Albany::MDArray& mda1 = getMDArray(name_rc, wi);
        Albany::MDArray& mda2 = getMDArray(name_rc + "_1", wi);
        write(mda1, mda2);
      } else {
        // In the g -> G direction, the values are already in the primary field,
        // so there's nothing to do.
      }
    } break;
    case Transformation::right_polar_LieR_LieS: {
      if (wi == 0) pr("right_polar_LieR_LieS " << (dir == Direction::g2G));
      Albany::MDArray& mda1 = getMDArray(name_rc, wi);
      Albany::MDArray& mda2 = getMDArray(name_rc + "_1", wi);
      loop(mda1, cell, 0) loop(mda1, qp, 1) {
        Checker c(wi, cell, qp);
        if (dir == Direction::G2g) {
          // Copy mda2 (provisional) -> local.
          Intrepid::Tensor<RealType> F(mda1.dimension(2));
          loop(mda2, i, 2) loop(mda2, j, 3) F(i, j) = mda2(cell, qp, i, j);
          std::pair< Intrepid::Tensor<RealType>, Intrepid::Tensor<RealType> >
            RS;
          calc_right_polar_LieR_LieS_G2g(F, RS, c);
          // Copy local -> mda1, mda2.
          loop(mda1, i, 2) loop(mda1, j, 3) {
            mda1(cell, qp, i, j) = RS.first(i, j);
            mda2(cell, qp, i, j) = RS.second(i, j);
          }
        } else {
          // Copy mda1,2 -> local.
          Intrepid::Tensor<RealType> R(mda1.dimension(2)), S(mda2.dimension(2));
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
    return Teuchos::rcp(new Manager(state_mgr, use_projection));
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

void Manager::update_x (const Tpetra_Vector& soln_nol) {
  impl_->x_->update(1, soln_nol, 1);
}

Teuchos::RCP<const Tpetra_Vector> Manager::
add_x (const Teuchos::RCP<const Tpetra_Vector>& a) const {
  Teuchos::RCP<Tpetra_Vector>
    c = Teuchos::rcp(new Tpetra_Vector(*a, Teuchos::Copy));
  c->update(1, *impl_->x_, 1);
  return c;
}

Teuchos::RCP<const Tpetra_Vector> Manager::
add_x_ol (const Teuchos::RCP<const Tpetra_Vector>& a_ol) const {
  Teuchos::RCP<Tpetra_Vector>
    c = Teuchos::rcp(new Tpetra_Vector(a_ol->getMap()));
  c->doImport(*impl_->x_, *impl_->sol_mgr_->get_importerT(), Tpetra::INSERT);
  c->update(1, *a_ol, 1);
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
beginQpInterp (const PHAL::Workset& workset, const BasisField& bf,
               const BasisField& wbf) { /* Do nothing. */ }
void Manager::
interpQpField (PHX::MDField<RealType>& f, const PHAL::Workset& workset,
               const BasisField& bf, const BasisField& wbf) {
  impl_->interpQpField(f, workset, bf, wbf);
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
              const BasisField& bf, const BasisField& wbf) {
  impl_->writeQpField(f, workset, bf, wbf);
}
void Manager::endQpWrite () { /* Do nothing. */ }

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

void Manager::beginAdapt () { impl_->beginAdapt(); }
void Manager::endAdapt (const Teuchos::RCP<const Tpetra_Map>& node_map,
                        const Teuchos::RCP<const Tpetra_Map>& ol_node_map)
{ impl_->endAdapt(node_map, ol_node_map); }
void Manager::initProjector (const Teuchos::RCP<const Tpetra_Map>& node_map,
                             const Teuchos::RCP<const Tpetra_Map>& ol_node_map)
{ impl_->initProjector(node_map, ol_node_map); }

bool Manager::usingProjection () const {
  return Teuchos::nonnull(impl_->proj_);
}

Manager::Manager (const Teuchos::RCP<Albany::StateManager>& state_mgr,
                  const bool use_projection)
  : impl_(Teuchos::rcp(new Impl(state_mgr, use_projection)))
{}

#define eti_fn(EvalT)                                   \
  template void Manager::createEvaluators<EvalT>(       \
    PHX::FieldManager<PHAL::AlbanyTraits>& fm,          \
    const Teuchos::RCP<Albany::Layouts>& dl);
aadapt_rc_apply_to_all_eval_types(eti_fn)
#undef eti_fn

} // namespace rc
} // namespace AAdapt

#undef loop
