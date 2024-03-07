#include "Albany_ThyraBlockedCrsMatrixFactory.hpp"

#include "Albany_TpetraTypes.hpp"

#include "Albany_Utils.hpp"
#include "Albany_Macros.hpp"

#include "Thyra_DefaultBlockedLinearOp_decl.hpp"
#include "Thyra_DefaultScaledAdjointLinearOp_decl.hpp"
namespace Albany
{

  ThyraBlockedCrsMatrixFactory::
      ThyraBlockedCrsMatrixFactory(const Teuchos::RCP<const Thyra_ProductVectorSpace> domain_vs,
                                   const Teuchos::RCP<const Thyra_ProductVectorSpace> range_vs,
                                   const int /*nonzeros_per_row*/)
      : m_domain_vs(domain_vs), m_range_vs(range_vs), m_filled(false), n_m_blocks(domain_vs->numBlocks())
  {
    block_factories.resize(n_m_blocks);
    for (size_t i_block = 0; i_block < n_m_blocks; ++i_block)
      block_factories[i_block].resize(n_m_blocks);

    for (size_t i_block = 0; i_block < n_m_blocks; ++i_block)
    {
      for (size_t j_block = 0; j_block < n_m_blocks; ++j_block)
      {
        block_factories[i_block][j_block] =
            Teuchos::rcp(new ThyraCrsMatrixFactory(m_domain_vs->getBlock(j_block),
                                                   m_range_vs->getBlock(i_block)));
      }
    }
  }

  ThyraBlockedCrsMatrixFactory::
      ThyraBlockedCrsMatrixFactory(const Teuchos::RCP<const Thyra_ProductVectorSpace> domain_vs,
                                   const Teuchos::RCP<const Thyra_ProductVectorSpace> range_vs,
                                   const Teuchos::RCP<const Thyra_ProductVectorSpace> ov_domain_vs,
                                   const Teuchos::RCP<const Thyra_ProductVectorSpace> ov_range_vs,
                                   const int /*nonzeros_per_row*/)
      : m_domain_vs(domain_vs), m_range_vs(range_vs), m_filled(false), n_m_blocks(domain_vs->numBlocks())
  {
    // block_factories is lower triangular
    block_factories.resize(n_m_blocks);
    for (size_t i_block = 0; i_block < n_m_blocks; ++i_block)
      block_factories[i_block].resize(n_m_blocks);

    for (size_t i_block = 0; i_block < n_m_blocks; ++i_block)
    {
      for (size_t j_block = 0; j_block < n_m_blocks; ++j_block)
      {
        block_factories[i_block][j_block] =
            Teuchos::rcp(new ThyraCrsMatrixFactory(m_domain_vs->getBlock(j_block),
                                                   m_range_vs->getBlock(i_block),
                                                   ov_domain_vs->getBlock(j_block),
                                                   ov_range_vs->getBlock(j_block)));
      }
    }
  }

  void ThyraBlockedCrsMatrixFactory::insertGlobalIndices(const GO row, const Teuchos::ArrayView<const GO> &indices,
                                                         const size_t i_block, const size_t j_block)
  {
    block_factories[i_block][j_block]->insertGlobalIndices(row, indices);
  }

  void ThyraBlockedCrsMatrixFactory::fillComplete()
  {

    for (size_t i_block = 0; i_block < n_m_blocks; ++i_block)
    {
      for (size_t j_block = 0; j_block < n_m_blocks; ++j_block)
      {
        if (!block_factories[i_block][j_block]->is_filled())
          block_factories[i_block][j_block]->fillComplete();
      }
    }

    m_filled = true;
  }

  Teuchos::RCP<Thyra_BlockedLinearOp> ThyraBlockedCrsMatrixFactory::createOp() const
  {
    TEUCHOS_TEST_FOR_EXCEPTION(!m_filled, std::logic_error, "Error! Cannot create a linear operator if the graph is not filled.\n");

    Teuchos::RCP<Thyra_PhysicallyBlockedLinearOp> op = Teuchos::rcp(new Thyra::DefaultBlockedLinearOp<double>());

    op->beginBlockFill(m_range_vs, m_domain_vs);

    for (size_t i_block = 0; i_block < n_m_blocks; ++i_block)
    {
      for (size_t j_block = 0; j_block < n_m_blocks; ++j_block)
      {
        op->setBlock(i_block, j_block, block_factories[i_block][j_block]->createOp());
      }
    }

    op->endBlockFill();

    return op;
  }

} // namespace Albany
