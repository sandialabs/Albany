#ifndef ALBANY_BLOCK_UTILS_HPP
#define ALBANY_BLOCK_UTILS_HPP

#include <Shards_CellTopology.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_RCPDecl.hpp>

#include <map>

namespace Albany
{

// Block structure is specified by 4 arrays (of same length/nesting structure):
//   names: [U, [pw, h]]
//   fe_types: [P1, [P1, P1]]
//   num_components: [2, [1, 1]]
//   mesh_part: ["volume",["basalside","basalside"]]

enum class FEType {
  P1,
  P0
};

// Whether the block is 'nested' or 'single'.
// E.g., "[[a,b],c]", has 2 blocks:
//  - [a,b]: a blocked block
//  - c: a single block,
enum class BlockType {
  Single,
  Blocked,   
  Unset    // Used as default value, do spot uninited values.
};

struct BlockSpecs {
  FEType                fe_type;
  std::string           mesh_part;
  shards::CellTopology  topo;
  std::string           name;

  int                   num_components = 1;
};

// Store info about the block
//  - global_block_coords contains the indices you need to use
//    to access the block from the overall solution.
//    E.g., if X = [[a,b],c], then block b has coords (0,1),
//    since b = X[0][1].
struct Block {
  // Block (const Teuchos::RCP<BlockSpecs>& bspecs)
  //  : specs(bspecs.getConst())
  //  , type (BlockType::Single)
  // {
  //   // Nothing to do here
  // }
  // Block (const Teuchos::Array<Teuchos::RCP<Block>>& blocks)
  //  : sub_blocks(blocks)
  //  , type (BlockType::Blocked)
  // {
  //   // Nothing to do here
  // }

  // Depending on this->type, only one of the following makes sense
  Teuchos::RCP<BlockSpecs>    specs;
  Teuchos::ArrayRCP<Block>    sub_blocks;
  Teuchos::Array<int>         global_block_coords;

  BlockType type = BlockType::Unset;
};

} // namespace Albany

#endif // ALBANY_BLOCK_UTILS_HPP
