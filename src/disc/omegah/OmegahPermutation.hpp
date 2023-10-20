//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_OmegahPermutation_HPP
#define ALBANY_OmegahPermutation_HPP

#include <Omega_h_element.hpp> //topological_singular_name
#include <fstream>

namespace OmegahPermutation {

struct OmegahTetFaceVtx {
  int idx[4][3];
  OmegahTetFaceVtx() {
    const int elem_dim = 3;
    const int bdry_dim = 2;
    const int vtx_dim = 0;
    for(int bdry=0; bdry<Omega_h::simplex_degree(elem_dim,bdry_dim); bdry++) {
      for(int vert=0; vert<Omega_h::simplex_degree(bdry_dim,vtx_dim); vert++) {
        idx[bdry][vert] = Omega_h::simplex_down_template(elem_dim, bdry_dim, bdry, vert);
      }
    }
  }
};

struct ShardsTetFaceVtx {
  int idx[4][3];
  ShardsTetFaceVtx() {
    shards::CellTopology tetTopo(shards::getCellTopologyData< shards::Tetrahedron<4> >());
    const int elem_dim = 3;
    const int bdry_dim = 2;
    const int vtx_dim = 0;
    for(int bdry=0; bdry<Omega_h::simplex_degree(elem_dim,bdry_dim); bdry++) {
      for(int vert=0; vert<Omega_h::simplex_degree(bdry_dim,vtx_dim); vert++) {
        idx[bdry][vert] = tetTopo.getNodeMap(bdry_dim,bdry,vert);
      }
    }
  }
};

struct TetTriVtx {
  int perm[4][3];
  void print(std::string name) {
    std::stringstream ss;
    ss << "\ntet: " << name << "(face,vert)=idx \n";
    for (int i=0; i<4; i++) {
      ss << "(" << i << ", *)=" << " ";
      for (int j=0; j<3; j++) {
        ss << perm[i][j] << " ";
      }
      ss << "\n";
    }
    std::cerr << ss.str();
  }
};

template<typename Source, typename Dest>
TetTriVtx getPerm_tet(Source& source, Dest& dest, const int src2destTetFace[4]) {
  const int elem_dim = 3;
  const int bdry_dim = 2;
  const int vtx_dim = 0;

  //compute the shards to omegah permutation
  TetTriVtx src2destTetTriVtx;
  shards::CellTopology triTopo(shards::getCellTopologyData< shards::Triangle<3> >());
  auto topoData = triTopo.getCellTopologyData();
  for(int bdry=0; bdry<Omega_h::simplex_degree(elem_dim,bdry_dim); bdry++) {
    auto src2destBdry = src2destTetFace[bdry];
    const unsigned perm = shards::findPermutation(topoData, dest.idx[src2destBdry],source.idx[bdry]);
    if(!(perm>=0 && perm<topoData->permutation_count)) {
      fprintf(stderr, "perm %d\n", perm);
    }
    assert(perm>=0 && perm<topoData->permutation_count);
    for(int vert=0; vert<Omega_h::simplex_degree(bdry_dim,vtx_dim); vert++) {
      src2destTetTriVtx.perm[src2destBdry][vert] = triTopo.getNodePermutationInverse(perm,vert);
    }
  }
  return src2destTetTriVtx;
}

struct OmegahTriVtx {
  int idx[3];
  OmegahTriVtx() {
    const int elem_dim = 2;
    const int vtx_dim = 0;
    const int ignored = -1;
    for(int vert=0; vert<Omega_h::simplex_degree(elem_dim, vtx_dim); vert++) {
      idx[vert] = Omega_h::simplex_down_template(elem_dim, vtx_dim, vert, ignored);
    }
  }
};

struct ShardsTriVtx {
  int idx[3];
  ShardsTriVtx() {
    shards::CellTopology triTopo(shards::getCellTopologyData< shards::Triangle<3> >());
    const int elem_dim = 2;
    const int bdry_dim = 0;
    const int vtx_dim = 0;
    for(size_t vert=0; vert<triTopo.getSubcellCount(vtx_dim); vert++) {
      idx[vert] = triTopo.getNodeMap(elem_dim, bdry_dim, vert);
    }
  }
};

struct TriVtx {
  int perm[3];
  void print(std::string name) const {
    std::stringstream ss;
    ss << "\ntri: " << name << "vert(0,1,2)= ";
    for (int i=0; i<3; i++) {
      ss << perm[i] << " ";
    }
    ss << "\n";
    std::cerr << ss.str();
  }
};

template<typename Source, typename Dest>
TriVtx getPerm_tri(Source& source, Dest& dest) {
  const int elm_dim = 2;
  const int vtx_dim = 0;

  //compute the shards to omegah permutation
  TriVtx src2dest;
  shards::CellTopology triTopo(shards::getCellTopologyData< shards::Triangle<3> >());
  auto perm = shards::findPermutation(triTopo.getCellTopologyData(),dest.idx,source.idx);
  for(int vert=0; vert<Omega_h::simplex_degree(elm_dim,vtx_dim); vert++) {
    src2dest.perm[vert] = triTopo.getNodePermutationInverse(perm,vert);
  }
  return src2dest;
}


struct Omegah2ShardsPerm {
  const std::string name = "Omega_h-to-Shards";

  /////// Triangles ///////// {
  /* triVtx[i] contains the Shards vertex index for the i'th vertex of an Omegah
   * triangle
   */
  const TriVtx triVtx;
  TriVtx getOmegah2ShardsPerm_tri() {
    const OmegahTriVtx ohTriVtx;
    const ShardsTriVtx shTriVtx;
    return getPerm_tri(ohTriVtx,shTriVtx);
  }
  ///// END Triangles }

  /////// Tetrahedrons ///////// {
  /**
   * tetFace[i] contains the Shards face index for the i'th Omegah face
   */
  const int tetFace[4] = {3,0,1,2};
  /**
   * tetTriVtx.perm[i][j] contains the Shards vertex index for the j'th vertex of the
   * i'th Omegah triangle, permuted to the correct Shards face via tetFace[i], of a tet.
   * For example, given the 1st vertex of the 2nd Omegah face, get the corresponding Shards
   * vertex index:
   * auto shardsVtxIdx = tetTriVtx.perm[tetFace[1]][0];
   */
  const TetTriVtx tetTriVtx;

  TetTriVtx getOmegah2ShardsPerm_tet() {
    const OmegahTetFaceVtx ohFaceVtx;
    const ShardsTetFaceVtx shFaceVtx;
    return getPerm_tet(ohFaceVtx, shFaceVtx, tetFace);
  }
  ///// END Tetrahedrons }

  Omegah2ShardsPerm() :
    triVtx(getOmegah2ShardsPerm_tri()),
    tetTriVtx(getOmegah2ShardsPerm_tet())
  {}
};

/**
 * this is the inverse of Omegah2ShardsPerm
 */
struct Shards2OmegahPerm {
  const std::string name = "Shards-to-Omega_h";

  /////// Triangles /////////
  const TriVtx triVtx;
  TriVtx getShards2OmegahPerm_tri() {
    const ShardsTriVtx shTriVtx;
    const OmegahTriVtx ohTriVtx;
    return getPerm_tri(shTriVtx,ohTriVtx);
  }

  /////// Tetrahedrons /////////
  const int tetFace[4] = {1,2,3,0};
  const TetTriVtx tetTriVtx;

  TetTriVtx getShards2OmegahPerm_tet() {
    const ShardsTetFaceVtx shFaceVtx;
    const OmegahTetFaceVtx ohFaceVtx;
    return getPerm_tet(shFaceVtx, ohFaceVtx, tetFace);
  }
  ///// END Tetrahedrons }

  Shards2OmegahPerm() :
    triVtx(getShards2OmegahPerm_tri()),
    tetTriVtx(getShards2OmegahPerm_tet())
  {}
};

} //end namespace

#endif // ALBANY_OmegahPermutation_HPP
