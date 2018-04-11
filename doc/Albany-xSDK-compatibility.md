# xSDK Community Policy Compatibility for Albany

This document summarizes the efforts of current and future xSDK member packages to achieve compatibility with the xSDK community policies. Below only short descriptions of each policy are provided. The full description is available [here](https://docs.google.com/document/d/1DCx2Duijb0COESCuxwEEK1j0BPe2cTIJ-AjtJxt3290/edit#heading=h.2hp5zbf0n3o3)
and should be considered when filling out this form.

Please, provide information on your Compatibility status for each mandatory policy, and if possible also for recommended policies.
If you are not compatible, state what is lacking and what are your plans on how to achieve compliance.
For current xSDK member packages: If you were not compliant at some point, please describe the steps you undertook to fulfill the policy.This information will be helpful for future xSDK member packages.

**Website:**  http://gahansen.github.io/Albany/

### Mandatory Policies

| Policy                 |Support| Notes                   |
|------------------------|-------|-------------------------|
|**M1.** Support xSDK community GNU Autoconf or CMake options. |Partial| Albany uses CMake but does not support the xSDK CMake interface. |
|**M2.** Provide a comprehensive test suite for correctness of installation verification. |Full| Albany has a comprehensive test suite. |
|**M3.** Employ user provided MPI communicator (no MPI_COMM_WORLD). |Full| Albany the application chooses `MPI_COMM_WORLD` at the top level, library code uses a given communicator. |
|**M4.** Give best effort at portability to key architectures (standard Linux distributions, GNU, Clang, vendor compilers, and target machines at ALCF, NERSC, OLCF). |Full| Albany supports a variety of platforms important to DOE and efforts continue to get advanced performance via Kokkos. |
|**M5.** Provide a documented, reliable way to contact the development team. |Full| Users can interact with the development team via Albany's GitHub site. |
|**M6.** Respect system resources and settings made by other previously called packages (e.g. signal handling). |Full| Albany the application chooses its resources and settings, the library code respects them.  |
|**M7.** Come with an open source (BSD style) license. |Full| Albany is BSD licensed. |
|**M8.** Provide a runtime API to return the current version number of the software. |None| Albany library code has no version API|
|**M9.** Use a limited and well-defined symbol, macro, library, and include file name space. |Full| All code is under namespace Albany, or the namespace of a sub-project such as LCM|
|**M10.** Provide an xSDK team accessible repository (not necessarily publicly available). |Full| https://github.com/gahansen/Albany |
|**M11.** Have no hardwired print or IO statements that cannot be turned off. |Partial| Albany has not be carefully reviewed to confirm the absence of such prints |
|**M12.** For external dependencies, allow installing, building, and linking against an outside copy of external software. |Full| Albany uses outside copies of all dependencies.  |
|**M13.** Install headers and libraries under \<prefix\>/include and \<prefix\>/lib. |Full|  |
|**M14.** Be buildable using 64 bit pointers. 32 bit is optional. |Full|  |
|**M15.** All xSDK compatibility changes should be sustainable. |Full|  |
|**M16.** The package must support production-quality installation compatible with the xSDK install tool and xSDK metapackage. |None| Albany is not yet part of Spack. |

### Recommended Policies

| Policy                 |Support| Notes                   |
|------------------------|-------|-------------------------|
|**R1.** Have a public repository. |Full| https://github.com/gahansen/Albany  |
|**R2.** Possible to run test suite under valgrind in order to test for memory corruption issues. |None|  |
|**R3.** Adopt and document consistent system for error conditions/exceptions. |Partial| Exceptions and asserts are mixed. |
|**R4.** Free all system resources acquired as soon as they are no longer needed. |Full|  Manual Valgrind examination supports this. |
|**R5.** Provide a mechanism to export ordered list of library dependencies. |Partial| Albany's CMake needs inspection to confirm this. |
