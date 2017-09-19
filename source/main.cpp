// Embree Packet Size; 8 gives best speedup on my machine for perfectly coherent rays
#define PACKET_SIZE 8

// Beta; TODO: pass in from command line
#define OUTPUT_COVERAGE_IMAGES 0

#if OUTPUT_COVERAGE_IMAGES
# define RAYS_PER_PIXEL 4
# define IMAGE_WIDTH 2

# define STB_IMAGE_WRITE_IMPLEMENTATION
# include "stb_image_write.h"
#endif

#define VERBOSE_OUTPUT 0


#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <assert.h> 
#include <sstream>
#include <string>
#include <chrono>

// OptiX setup
#include <cuda_runtime.h>
#include <optix_prime/optix_primepp.h>
#include <optixu/optixu_math_namespace.h>

#include <xmmintrin.h>
#include <pmmintrin.h>

#define TBB_IMPLEMENT_CPP0X 0
#define __TBB_NO_IMPLICIT_LINKAGE 1
#define __TBBMALLOC_NO_IMPLICIT_LINKAGE 1
#include <tbb/tbb.h>


#if OUTPUT_COVERAGE_IMAGES
int extractBit(const std::vector<unsigned char>& bitVector, int index) {
    int byteIndex = index / 8;
    int bitIndex = index % 8;
    return ((bitVector[byteIndex] >> (bitIndex)) & 1);
}
#endif

// Embree setup
#include "embree/rtcore.h"
#include "embree/rtcore_ray.h"
// Set up the linker on Windows
#ifdef _MSC_VER
#   pragma comment(lib, "embree.lib")
#   pragma comment(lib, "tbb.lib")

class SimpleTimer {
public:
    void init() {
        // get ticks per second
        QueryPerformanceFrequency(&frequency);

        // start timer
        QueryPerformanceCounter(&lastTick);
    }
    // Time since last tick in ms
    double tick() {
        LARGE_INTEGER currentTick;
        QueryPerformanceCounter(&currentTick);

        // compute and print the elapsed time in millisec
        double elapsedTime = (currentTick.QuadPart - lastTick.QuadPart) * 1000.0 / frequency.QuadPart;
        lastTick = currentTick;
        return elapsedTime;
    }
protected:
    LARGE_INTEGER frequency;
    LARGE_INTEGER lastTick;
};
#else
class SimpleTimer {
public:
    void init() {}
    // Time since last tick in ms
    double tick() { return nanf(""); }
};
#endif

template <typename T> T min(T a, T b) { return a < b ? a : b; }
template <typename T> T max(T a, T b) { return a > b ? a : b; }

struct vec3 {
    float x, y, z;
};

struct SimpleRay {
    vec3 o;
    vec3 d;
};

struct SimpleMesh {
    std::vector<vec3> vertices;
    std::vector<int> indices;
};

static void loadOFF(std::string filename, SimpleMesh& mesh) {
	if (filename.back() == 'b') {
		// binary
		FILE* file = fopen(filename.c_str(), "rb");
		char header[4] = {};
		fread(header, sizeof(header), 1, file);

		int vertexCount = 0;
		int triCount = 0;
		int edgeCount = 0;
		fread(&vertexCount, sizeof(vertexCount), 1, file);
		fread(&triCount, sizeof(triCount), 1, file);
		fread(&edgeCount, sizeof(edgeCount), 1, file);

		mesh.vertices.resize(vertexCount);
		fread(mesh.vertices.data(), sizeof(vec3) * vertexCount, 1, file);

		// note - assuming 3 vertices per polygon (triangles)
		mesh.indices.resize(triCount * 3);
		fread(mesh.indices.data(), sizeof(int) * 3 * triCount, 1, file);

		fclose(file);
	}
	else {
		std::ifstream inFile(filename.c_str());
		std::string line;
		std::getline(inFile, line);
		// TODO: check that its OFF
		std::getline(inFile, line);
		std::istringstream in(line);
		int numVertices, numTris;
		in >> numVertices >> numTris;

		mesh.vertices.resize(numVertices);
		mesh.indices.resize(numTris * 3);

		for (int i = 0; i < numVertices; ++i) {
			std::string line;
			std::getline(inFile, line);
			std::istringstream in(line);
			float x, y, z;
			in >> x >> y >> z;
			mesh.vertices[i] = { x, y, z };
		}

		for (int i = 0; i < numTris; ++i) {
			std::string line;
			std::getline(inFile, line);
			std::istringstream in(line);

			int dummy, i0, i1, i2;
			in >> dummy >> i0 >> i1 >> i2;
			assert(dummy == 3);
			int t = i * 3;
			mesh.indices[t + 0] = i0;
			mesh.indices[t + 1] = i1;
			mesh.indices[t + 2] = i2;
		}
	}
}

static void loadRFF(std::string filename, std::vector<SimpleRay>& rays) {
	if (filename.back() == 'b') {
		// binary
		FILE* file = fopen(filename.c_str(), "rb");
		char header[4] = {};
		fread(header, sizeof(header), 1, file);

		int rayCount = 0;
		fread(&rayCount, sizeof(rayCount), 1, file);

		rays.resize(rayCount);
		fread(rays.data(), sizeof(SimpleRay) * rayCount, 1, file);

		fclose(file);
	}
	else {
		std::ifstream inFile(filename.c_str());
		std::string line;
		std::getline(inFile, line);
		// TODO: check that its RFF
		std::getline(inFile, line);
		std::istringstream in(line);
		int numRays;
		in >> numRays;

		rays.resize(numRays);

		for (int i = 0; i < numRays; ++i) {
			std::string line;
			std::getline(inFile, line);
			std::istringstream in(line);
			float ox, oy, oz, dx, dy, dz;
			in >> ox >> oy >> oz >> dx >> dy >> dz;
			rays[i].o = { ox, oy, oz };
			rays[i].d = { dx, dy, dz };
		}
	}
}

void traceOptiX(const SimpleMesh& mesh, const std::vector<SimpleRay>& rays, const int timing_iterations) {
    // Type configuration
    RTPcontexttype contextType = RTP_CONTEXT_TYPE_CUDA;
    RTPbuffertype bufferType = RTP_BUFFER_TYPE_CUDA_LINEAR;
    RTPbufferformat rayFormat = RTP_BUFFER_FORMAT_RAY_ORIGIN_DIRECTION;
    RTPbufferformat hitFormat = RTP_BUFFER_FORMAT_HIT_BITMASK;
    RTPbufferformat vertFormat = RTP_BUFFER_FORMAT_VERTEX_FLOAT3;
    RTPbufferformat triBufferFormat = RTP_BUFFER_FORMAT_INDICES_INT3;
    RTPquerytype queryType = RTP_QUERY_TYPE_CLOSEST;


    // Copy to GPU
    SimpleRay* d_rays;
    cudaMalloc(&d_rays, sizeof(SimpleRay) * rays.size());
    cudaMemcpy(d_rays, rays.data(), sizeof(SimpleRay)*rays.size(), cudaMemcpyHostToDevice);
    int* d_tris;
    cudaMalloc(&d_tris, sizeof(int) * mesh.indices.size());
    cudaMemcpy(d_tris, mesh.indices.data(), sizeof(int) * mesh.indices.size(), cudaMemcpyHostToDevice);
    float3* d_verts;
    cudaMalloc(&d_verts, sizeof(vec3) * mesh.vertices.size());
    cudaMemcpy(d_verts, mesh.vertices.data(), sizeof(vec3) * mesh.vertices.size(), cudaMemcpyHostToDevice);
    void* d_hits;
    cudaMalloc(&d_hits, 8 * ((rays.size() / 64) + 1)); // have some buffer space
    cudaMemset(d_hits, 255, 1);


    auto contex = optix::prime::Context::create(contextType);

    // Setup Acceleration Structure
    auto model = contex->createModel();
    auto indexBuffer = contex->createBufferDesc(triBufferFormat, bufferType, d_tris);
    indexBuffer->setRange(0, mesh.indices.size() / 3);
    auto vertexBuffer = contex->createBufferDesc(vertFormat, bufferType, d_verts);
    vertexBuffer->setRange(0, mesh.vertices.size());
    model->setTriangles(indexBuffer, vertexBuffer);
    model->update(0);
    model->finish();

    // Setup Ray Query
    auto rayBuffer = contex->createBufferDesc(rayFormat, bufferType, d_rays);
    auto hitBuffer = contex->createBufferDesc(hitFormat, bufferType, d_hits);
    rayBuffer->setRange(0, rays.size());
    hitBuffer->setRange(0, rays.size());

    SimpleTimer timer;
    timer.init();
    double timeSum = 0.0;
	double timeMin = DBL_MAX;
    for (int i = 0; i < timing_iterations; ++i) {
        auto query = model->createQuery(queryType);
        query->setRays(rayBuffer);
        query->setHits(hitBuffer);

        cudaDeviceSynchronize();
        timer.tick();
        query->execute(0);
        query->finish();
        cudaDeviceSynchronize();
        double timeInMS = timer.tick();

        timeSum += timeInMS;
		timeMin = min(timeMin, timeInMS);
#if VERBOSE_OUTPUT
        std::cout << "OptiX Prime Iteration " << i << std::endl;
        std::cout << timeInMS << " ms" << std::endl;
        std::cout << (rays.size()*1000.0)/timeInMS << " rays/s" << std::endl;
#endif
    }

    double aveTimeInMS = timeSum / timing_iterations;
    std::cout << "------ OptiX Prime API ------- " << std::endl;
    std::cout << "avg: " << aveTimeInMS << "ms" << std::endl;
	std::cout << "min: " << timeMin << "ms" << std::endl;
	std::cout << "avg mrays/s: " << (rays.size() / 1000000.0) / (aveTimeInMS / 1000.0) << std::endl;
	std::cout << "max mrays/s: " << (rays.size() / 1000000.0) / (timeMin / 1000.0) << std::endl;

#if OUTPUT_COVERAGE_IMAGES
    // Chop off any excess (there should be no excess...)
    int pixelCount = int(rays.size() / RAYS_PER_PIXEL);
    int imageHeight = pixelCount / IMAGE_WIDTH;
    std::vector<unsigned char> coverageMask;
    coverageMask.resize(pixelCount);
    std::vector<unsigned char> h_hits;
    h_hits.resize(rays.size() / 8);
    cudaMemcpy(h_hits.data(), d_hits, rays.size() / 8, cudaMemcpyDeviceToHost);
    for (int i = 0; i < pixelCount; ++i) {
        int coveredCount = 0;
        for (int j = 0; j < RAYS_PER_PIXEL; ++j) {
            coveredCount += extractBit(h_hits, i*RAYS_PER_PIXEL + j);
        }
        coverageMask[i] = (unsigned char)((float(coveredCount) / float(RAYS_PER_PIXEL))*255.0f);
    }
    std::cout << "Outputing OptiX Coverage Mask" << std::endl;
    stbi_write_png("optixCoverageMask.png", IMAGE_WIDTH, imageHeight, 1, coverageMask.data(), IMAGE_WIDTH);
#endif


    cudaFree(d_rays);
    cudaFree(d_tris);
    cudaFree(d_verts);
    cudaFree(d_hits);
}

void traceEmbree(const SimpleMesh& mesh, const std::vector<SimpleRay>& rays, const int timing_iterations) {
    size_t triCount = mesh.indices.size() / 3;
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    auto device = rtcNewDevice(nullptr);
	auto scene = rtcDeviceNewScene(device, RTC_SCENE_COHERENT | RTC_SCENE_STATIC | RTC_SCENE_HIGH_QUALITY, RTCAlgorithmFlags(RTC_INTERSECT_STREAM | RTC_INTERSECT_COHERENT));
    auto geomID = rtcNewTriangleMesh(scene, RTC_GEOMETRY_STATIC, triCount, mesh.vertices.size(), 1);

    float4* vtxPtr = (float4*)rtcMapBuffer(scene, geomID, RTC_VERTEX_BUFFER); {
        for (int i = 0; i < mesh.vertices.size(); ++i) {
            vtxPtr[i].x = mesh.vertices[i].x;
            vtxPtr[i].y = mesh.vertices[i].y;
            vtxPtr[i].z = mesh.vertices[i].z;
            vtxPtr[i].w = 1.0f;
        }
    } rtcUnmapBuffer(scene, geomID, RTC_VERTEX_BUFFER);

    int* idxPtr = (int*)rtcMapBuffer(scene, geomID, RTC_INDEX_BUFFER); {
        memcpy(idxPtr, mesh.indices.data(), mesh.indices.size()*sizeof(int));
    } rtcUnmapBuffer(scene, geomID, RTC_INDEX_BUFFER);

    rtcCommit(scene);

    std::vector<RTCRayNt<PACKET_SIZE>> rtcRays;

    const RTCError error = rtcDeviceGetError(device);
    if (error != RTC_NO_ERROR) {
        fprintf(stderr, "Embree error\n");
    }

    SimpleTimer timer;
    timer.init();
    double timeSum = 0.0;
	double timeMin = DBL_MAX;
    for (int i = 0; i < timing_iterations; ++i) {
        RTCIntersectContext context;
        context.flags = RTCIntersectFlags(RTC_INTERSECT_COHERENT);
        // Setup rays
        rtcRays.resize((rays.size() + PACKET_SIZE-1) / PACKET_SIZE);
        for (int r = 0; r < rays.size(); ++r) {
            int p = r / PACKET_SIZE; // Packet Index
            int i = r % PACKET_SIZE; // Index in Packet
            rtcRays[p].orgx[i] = rays[r].o.x;
            rtcRays[p].orgy[i] = rays[r].o.y;
            rtcRays[p].orgz[i] = rays[r].o.z;
            rtcRays[p].dirx[i] = rays[r].d.x;
            rtcRays[p].diry[i] = rays[r].d.y;
            rtcRays[p].dirz[i] = rays[r].d.z;
            rtcRays[p].tnear[i] = 0.0f;
            rtcRays[p].tfar[i] = std::numeric_limits<float>::infinity();
            rtcRays[p].instID[i] = RTC_INVALID_GEOMETRY_ID;
            rtcRays[p].geomID[i] = RTC_INVALID_GEOMETRY_ID;
            rtcRays[p].primID[i] = RTC_INVALID_GEOMETRY_ID;
            rtcRays[p].mask[i] = 0xFFFFFFFF;
            rtcRays[p].time[i] = 0.0f;
        }

        timer.tick();
        // Using raw pointers instead of C++ arrays gave no performance increase for the code below
        static const size_t grainSize = 64;
        tbb::parallel_for(tbb::blocked_range<size_t>(0, rtcRays.size(), grainSize), [&](const tbb::blocked_range<size_t>& r) {
            const size_t start = r.begin();
            const size_t end = r.end();

            static const size_t BLOCK_SIZE = 64;
            for (size_t r = start; r < end; r += BLOCK_SIZE) {

                const size_t numPackets = std::min(BLOCK_SIZE, end - r);
                rtcIntersectNM(scene, &context, (RTCRayN*)&rtcRays[r], PACKET_SIZE, numPackets, sizeof(RTCRayNt<PACKET_SIZE>));
            } // for 
        }); // parallel for
        double timeInMS = timer.tick();

        timeSum += timeInMS;
		timeMin = min(timeMin, timeInMS);
#if VERBOSE_OUTPUT
        std::cout << "Embree Stream API Iteration " << i << std::endl;
        std::cout << timeInMS << " ms" << std::endl;
        std::cout << (rays.size()*1000.0) / timeInMS << " rays/s" << std::endl;
#endif
    }
    double aveTimeInMS = timeSum / timing_iterations;
    std::cout << "------ Embree Stream API ------- " << std::endl;
	std::cout << "avg: " << aveTimeInMS << "ms" << std::endl;
	std::cout << "min: " << timeMin << "ms" << std::endl;
	std::cout << "avg mrays/s: " << (rays.size() / 1000000.0) / (aveTimeInMS / 1000.0) << std::endl;
	std::cout << "max mrays/s: " << (rays.size() / 1000000.0) / (timeMin / 1000.0) << std::endl;

    //cleanup
    rtcDeleteGeometry(scene, geomID);
    rtcDeleteScene(scene);
    rtcDeleteDevice(device);
}

int main(int argc, char *argv[]) {
	assert(argc == 3);
	int timingIterations = 100;

	SimpleTimer timer;
	timer.init();

	// import the scene and rays
	SimpleMesh mesh;
	std::vector<SimpleRay> rays;
	timer.tick();
	std::cout << "loading scene: " << argv[1] << std::endl;
    loadOFF(argv[1], mesh);
	std::cout << "loading rays: " << argv[2] << std::endl;
    loadRFF(argv[2], rays);
	double importTimeS = timer.tick() / 1000.0;
	std::cout << "load time: " << importTimeS << std::endl;
	std::cout << std::endl;

	std::cout << "mesh: " << argv[1] << std::endl;
	std::cout << "vertices: " << mesh.vertices.size() << std::endl;
	std::cout << "triangles: " << mesh.indices.size() / 3 << std::endl;
	std::cout << std::endl;

	std::cout << "rays: " << argv[2] << std::endl;
	std::cout << "ray count: " << rays.size() << std::endl;
	std::cout << std::endl;

	std::cout << "iterations: " << timingIterations << std::endl;
	std::cout << std::endl;

    traceOptiX(mesh, rays, timingIterations);
    std::cout << std::endl;
    traceEmbree(mesh, rays, timingIterations);
	std::cout << std::endl;

    return 0;
}
