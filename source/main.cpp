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
    std::ifstream inFile(filename.c_str());
    std::string line;
    std::getline(inFile, line);
    // TODO: check that its OFF
    std::getline(inFile, line);
    std::istringstream in(line);
    int numVertices, numTris;
    in >> numVertices >> numTris;

    mesh.vertices.resize(numVertices);
    mesh.indices.resize(numTris*3);

    for (int i = 0; i < numVertices; ++i) {
        std::string line;
        std::getline(inFile, line);
        std::istringstream in(line);      
        float x, y, z;
        in >> x >> y >> z;
        mesh.vertices[i] = { x,y,z };
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

static void loadRFF(std::string filename, std::vector<SimpleRay>& rays) {
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
        rays[i].o = { ox,oy,oz };
        rays[i].d = { dx,dy,dz };
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
    cudaMalloc(&d_hits, 64 * ((rays.size() / 64) + 1)); // have some buffer space
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
        std::cout << "OptiX Prime Iteration " << i << std::endl;
        std::cout << timeInMS << " ms" << std::endl;
        std::cout << (rays.size()*1000.0)/timeInMS << " rays/s" << std::endl;
    }

    double aveTimeInMS = timeSum / timing_iterations;
    std::cout << "------ OptiX Prime API Average ------- " << std::endl;
    std::cout << aveTimeInMS << " ms" << std::endl;
    std::cout << (rays.size()*1000.0) / aveTimeInMS << " rays/s" << std::endl;

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
    auto scene = rtcDeviceNewScene(device, RTC_SCENE_COHERENT | RTC_SCENE_STATIC, RTCAlgorithmFlags(RTC_INTERSECT_STREAM | RTC_INTERSECT_COHERENT));
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

    std::vector<RTCRay> rtcRays;

    const RTCError error = rtcDeviceGetError(device);
    if (error != RTC_NO_ERROR) {
        fprintf(stderr, "Embree error\n");
    }

    SimpleTimer timer;
    timer.init();
    double timeSum = 0.0;
    for (int i = 0; i < timing_iterations; ++i) {
        RTCIntersectContext context;
        context.flags = RTCIntersectFlags(RTC_INTERSECT_COHERENT);
        // Setup rays
        rtcRays.resize(rays.size());
        for (int r = 0; r < rays.size(); ++r) {
            RTCRay ray;
            ray.org[0] = rays[r].o.x;
            ray.org[1] = rays[r].o.y;
            ray.org[2] = rays[r].o.z;
            ray.dir[0] = rays[r].d.x;
            ray.dir[1] = rays[r].d.y;
            ray.dir[2] = rays[r].d.z;
            ray.tnear = 0.0f;
            ray.tfar = std::numeric_limits<float>::infinity();
            ray.instID = RTC_INVALID_GEOMETRY_ID;
            ray.geomID = RTC_INVALID_GEOMETRY_ID;
            ray.primID = RTC_INVALID_GEOMETRY_ID;
            ray.mask = 0xFFFFFFFF;
            ray.time = 0.0f;
            rtcRays[r] = ray;
        }

        timer.tick();
        // Using raw pointers instead of C++ arrays gave no performance increase for the code below
        static const size_t grainSize = 64;
        tbb::parallel_for(tbb::blocked_range<size_t>(0, rays.size(), grainSize), [&](const tbb::blocked_range<size_t>& r) {
            const size_t start = r.begin();
            const size_t end = r.end();

            static const size_t BLOCK_SIZE = 64;
            for (size_t r = start; r < end; r += BLOCK_SIZE) {

                const size_t numRays = std::min(BLOCK_SIZE, end - r);
                rtcIntersectNM(scene, &context, (RTCRayN*)&rtcRays[r], 1, numRays, sizeof(RTCRay));
            } // for 
        }); // parallel for

        double timeInMS = timer.tick();
        timeSum += timeInMS;
        std::cout << "Embree Stream API Iteration " << i << std::endl;
        std::cout << timeInMS << " ms" << std::endl;
        std::cout << (rays.size()*1000.0) / timeInMS << " rays/s" << std::endl;
    }
    double aveTimeInMS = timeSum / timing_iterations;
    std::cout << "------ Embree Stream API Average ------- " << std::endl;
    std::cout << aveTimeInMS << " ms" << std::endl;
    std::cout << (rays.size()*1000.0) / aveTimeInMS << " rays/s" << std::endl;

    //cleanup
    rtcDeleteGeometry(scene, geomID);
    rtcDeleteScene(scene);
    rtcDeleteDevice(device);

}

int main(int argc, char *argv[]) {
    assert(argc == 3);
    SimpleMesh mesh;
    std::vector<SimpleRay> rays;
    loadOFF(argv[1], mesh);
    loadRFF(argv[2], rays);
    for (int i = 0; i < 10000000; ++i) {
        rays.push_back(rays[0]);
    }


    int timingIterations = 10;
    traceOptiX(mesh, rays, timingIterations);
    traceEmbree(mesh, rays, timingIterations);
    getchar();
    return 0;
}