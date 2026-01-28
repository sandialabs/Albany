//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_Utils.hpp"

#include <Teuchos_UnitTestHarness.hpp>
#include <fstream>
#include <sys/stat.h>

namespace Albany
{

TEUCHOS_UNIT_TEST(DebugStream, getDebugFileName)
{
  // Test that getDebugFileName generates correct filenames
  std::string filename1 = getDebugFileName("test", ".txt");
  int rank = getProcRank();
  std::stringstream expected;
  expected << "test_rank" << rank << ".txt";

  TEST_ASSERT(filename1 == expected.str());

  // Test with different suffix
  std::string filename2 = getDebugFileName("debug", ".log");
  std::stringstream expected2;
  expected2 << "debug_rank" << rank << ".log";

  TEST_ASSERT(filename2 == expected2.str());
}

TEUCHOS_UNIT_TEST(DebugStream, getDebugStream_DefaultPrefix)
{
  // Test that getDebugStream creates a file and we can write to it
  auto& stream = getDebugStream();

  // Write test data
  stream << "Test message from rank " << getProcRank() << "\n";
  stream.flush();

  // Check that the file exists
  std::string expectedFilename = getDebugFileName("albany_debug", ".txt");
  struct stat buffer;
  bool fileExists = (stat(expectedFilename.c_str(), &buffer) == 0);

  TEST_ASSERT(fileExists);
  TEST_ASSERT(stream.good());
}

TEUCHOS_UNIT_TEST(DebugStream, getDebugStream_CustomPrefix)
{
  // Test with custom prefix
  auto& stream = getDebugStream("custom_test");

  // Write test data
  stream << "Custom test message\n";
  stream.flush();

  // Check that the file exists
  std::string expectedFilename = getDebugFileName("custom_test", ".txt");
  struct stat buffer;
  bool fileExists = (stat(expectedFilename.c_str(), &buffer) == 0);

  TEST_ASSERT(fileExists);
  TEST_ASSERT(stream.good());
}

TEUCHOS_UNIT_TEST(DebugStream, getDebugStream_MultipleCalls)
{
  // Test that multiple calls to getDebugStream with same prefix return same stream
  auto& stream1 = getDebugStream("singleton_test");
  auto& stream2 = getDebugStream("singleton_test");

  // They should be the same object (same address)
  TEST_ASSERT(&stream1 == &stream2);

  // Write to both (actually same stream)
  stream1 << "Message 1\n";
  stream2 << "Message 2\n";
  stream1.flush();

  TEST_ASSERT(stream1.good());
  TEST_ASSERT(stream2.good());
}

TEUCHOS_UNIT_TEST(DebugStream, getDebugStream_MultiplePrefixes)
{
  // Test that different prefixes create different streams
  auto& stream1 = getDebugStream("prefix1");
  auto& stream2 = getDebugStream("prefix2");

  // They should be different objects
  TEST_ASSERT(&stream1 != &stream2);

  // Write to each
  stream1 << "Stream 1 message\n";
  stream2 << "Stream 2 message\n";
  stream1.flush();
  stream2.flush();

  // Check both files exist
  std::string file1 = getDebugFileName("prefix1", ".txt");
  std::string file2 = getDebugFileName("prefix2", ".txt");

  struct stat buffer;
  bool file1Exists = (stat(file1.c_str(), &buffer) == 0);
  bool file2Exists = (stat(file2.c_str(), &buffer) == 0);

  TEST_ASSERT(file1Exists);
  TEST_ASSERT(file2Exists);
}

} // namespace Albany
