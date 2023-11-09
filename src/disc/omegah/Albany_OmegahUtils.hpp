#ifndef ALBANY_OMEGAH_UTILS_HPP
#define ALBANY_OMEGAH_UTILS_HPP

#include <Omega_h_element.hpp>
#include <Omega_h_array.hpp>
#include <Omega_h_defines.hpp>

#include <exception>

namespace Albany {

template<typename T>
Omega_h::HostRead<T> hostRead (const Omega_h::Read<T>& a) {
  return Omega_h::HostRead(a);
}

template<typename T>
Omega_h::HostWrite<T> hostWrite (const Omega_h::Write<T>& a) {
  return Omega_h::HostWrite(a);
}

template<typename T>
Omega_h::HostWrite<T> hostWrite (const int size, const std::string& name = "") {
  return Omega_h::HostWrite<T>(size,name);
}

inline
std::string e2str (const Topo_type topo) {
  return Omega_h::dimensional_singular_name(topo);
}

inline
std::string e2str (const Omega_h_Family family) {
  std::string s;
  switch (family) {
    case OMEGA_H_SIMPLEX:   s = "simplex";    break;
    case OMEGA_H_HYPERCUBE: s = "hypercube";  break;
    case OMEGA_H_MIXED:     s = "mixed";      break;
    default:
      s = "UNSUPPORTED";
  }
  return s;
}

inline Topo_type str2topo (const std::string& s) {
  Topo_type topo;
  static Topo_type valid_topos [8] = {
    Topo_type::vertex,
    Topo_type::edge,
    Topo_type::triangle,
    Topo_type::quadrilateral,
    Topo_type::tetrahedron,
    Topo_type::hexahedron,
    Topo_type::wedge,
    Topo_type::pyramid
  };
  
  bool found = false;
  for (auto t: valid_topos) {
    if (e2str(t)==s) {
      found = true;
      topo = t;
    }
  }

  if (not found) {
    throw std::runtime_error ("Unrecognized topo string '" + s + "'\n");
  }
  return topo;
}

inline int topo_dim (const Topo_type t) {
  int dim = -1;
  switch (t) {
    case Topo_type::vertex:        dim = 0;   break;
    case Topo_type::edge:          dim = 1;   break;
    case Topo_type::triangle:      dim = 2;   break;
    case Topo_type::quadrilateral: dim = 2;   break;
    case Topo_type::tetrahedron:   dim = 3;   break;
    case Topo_type::hexahedron:    dim = 3;   break;
    case Topo_type::wedge:         dim = 3;   break;
    case Topo_type::pyramid:       dim = 3;   break;
    default:
      throw std::runtime_error("Error! Unrecognized/unsupported topology.\n");
  }
  return dim;
}

inline Topo_type get_side_topo (const Topo_type t)
{
  Topo_type st = static_cast<Topo_type>(-1);
  switch (t) {
    case Topo_type::edge:          st = Topo_type::vertex;        break;
    case Topo_type::triangle:      st = Topo_type::edge;          break;
    case Topo_type::quadrilateral: st = Topo_type::edge;          break;
    case Topo_type::tetrahedron:   st = Topo_type::triangle;      break;
    case Topo_type::hexahedron:    st = Topo_type::quadrilateral; break;
    default:
      throw std::runtime_error(
          "Error! Unable to retrieve the side topology.\n"
          "  - input topo: " + e2str(t) + "\n");
  }
  return st;
}

} // namespace Albany

#endif // ALBANY_OMEGAH_UTILS_HPP
