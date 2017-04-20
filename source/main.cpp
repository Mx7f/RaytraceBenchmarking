#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <assert.h> 
#include <sstream>
#include <string>

#include <cuda_runtime.h>
#include <optix_prime/optix_primepp.h>
#include <optixu/optixu_math_namespace.h>

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

void traceOptiX(const SimpleMesh& mesh, const std::vector<SimpleRay>& rays) {
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

    for (auto r : rays) {
        printf("o:(%f, %f, %f), d:(%f, %f, %f)\n", r.o.x, r.o.y, r.o.z, r.d.x, r.d.y, r.d.z);
    }

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
    auto model = contex->createModel();
    auto indexBuffer = contex->createBufferDesc(triBufferFormat, bufferType, d_tris);
    indexBuffer->setRange(0, mesh.indices.size() / 3);
    auto vertexBuffer = contex->createBufferDesc(vertFormat, bufferType, d_verts);
    vertexBuffer->setRange(0, mesh.vertices.size());

    model->setTriangles(indexBuffer, vertexBuffer);
    model->update(0);
    model->finish();

    auto rayBuffer = contex->createBufferDesc(rayFormat, bufferType, d_rays);
    auto hitBuffer = contex->createBufferDesc(hitFormat, bufferType, d_hits);
    rayBuffer->setRange(0, rays.size());
    hitBuffer->setRange(0, rays.size());

    auto query = model->createQuery(queryType);
    query->setRays(rayBuffer);
    query->setHits(hitBuffer);

    unsigned char result;
    // TIME THIS:
    query->execute(0);
    query->finish();
    cudaMemcpy(&result, d_hits, 1, cudaMemcpyDeviceToHost);
    printf("Result: %d\n", (int)result);
    printf("Error string: %s\n", contex->getLastErrorString().c_str());

}

void traceEmbree(const SimpleMesh& mesh, const std::vector<SimpleRay>& rays) {
    fprintf(stderr, "TODO: implement\n");
}

int main(int argc, char *argv[]) {
    assert(argc == 3);
    SimpleMesh mesh;
    std::vector<SimpleRay> rays;
    loadOFF(argv[1], mesh);
    loadRFF(argv[2], rays);

    traceOptiX(mesh, rays);
    traceEmbree(mesh, rays);

    return 0;
}