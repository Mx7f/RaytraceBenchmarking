# Raytrace Benchmark

This program profiles the performance of OptiX Prime and the Embree C++ Stream API in as favorable situation as possible for absolute ray rate (no shading, very little output bandwidth).

This has been developed and tested solely on Windows.

Prerequisites:
CUDA 7.5 (for OptiX)
Visual Studio 2013 (to work with CUDA 7.5)
OptiX 4.1.0 (https://developer.nvidia.com/designworks/optix/download)

Embree and TBB are contained in this distribution and so do not need to be downloaded separately.

To use the program:

RaytraceBenchmark.exe cube.off simple.rff

will load cube.off as a mesh and trace the rays given in simple.rff, using both OptiX Prime and Embree, and report timings (it will run each 10 times and report an average.)

RFF is a simple text format for storing rays consisting a header line, a line with the number of rays, then that many subsequent lines specifying the rays as 6 floats (3 for xyz origin, 3 for xyz direction)

Neither the OFF nor the RFF loaders are robust. The OFF loader can only handle triangle meshes, and neither loader can handle comments or malformed input in any way.

Changelog:

7/20/2017:
- Added code for debugging by outputting a coverage mask (of the OptiX solver) to a PNG (using the public domain library stb_image_write.h); to enable change OUTPUT_COVERAGE_IMAGES to 1; you can change the number of rays per pixel averaged for the coverage mask and the image width using RAYS_PER_PIXEL and IMAGE_WIDTH; if ray counts do not evenly divide the rays into RAYS_PER_PIXEL*IMAGE_WIDTH large segments, the final excess will be cut off.


TODOs:
Improved Documentation