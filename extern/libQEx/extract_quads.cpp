// Extract quad mesh from a triangle mesh with per-corner UV using libQEx.
// Input: OBJ with v/vt/f (seamless or quantized UV)
// Output: OBJ with quad faces
//
// Usage: extract_quads input.obj output.obj

#include <qex.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cstring>

struct Vec3 { double x, y, z; };
struct Vec2 { double u, v; };

bool load_obj(const char* path,
              std::vector<Vec3>& verts,
              std::vector<int>& faces,      // 3 ints per face (vertex indices)
              std::vector<Vec2>& uvs,
              std::vector<int>& face_uvs)   // 3 ints per face (UV indices)
{
    std::ifstream fin(path);
    if (!fin) return false;
    std::string line;
    while (std::getline(fin, line)) {
        if (line.size() < 2) continue;
        if (line[0] == 'v' && line[1] == ' ') {
            Vec3 v;
            sscanf(line.c_str() + 2, "%lf %lf %lf", &v.x, &v.y, &v.z);
            verts.push_back(v);
        } else if (line[0] == 'v' && line[1] == 't') {
            Vec2 uv;
            sscanf(line.c_str() + 3, "%lf %lf", &uv.u, &uv.v);
            uvs.push_back(uv);
        } else if (line[0] == 'f' && line[1] == ' ') {
            // Parse f v/vt or f v/vt/vn or f v
            std::istringstream iss(line.substr(2));
            std::string tok;
            std::vector<int> fv, ft;
            while (iss >> tok) {
                int vi = 0, ti = 0;
                if (tok.find('/') != std::string::npos) {
                    sscanf(tok.c_str(), "%d/%d", &vi, &ti);
                } else {
                    vi = atoi(tok.c_str());
                }
                fv.push_back(vi - 1);
                if (ti > 0) ft.push_back(ti - 1);
            }
            if (fv.size() == 3) {
                faces.insert(faces.end(), fv.begin(), fv.end());
                if (ft.size() == 3) face_uvs.insert(face_uvs.end(), ft.begin(), ft.end());
            }
        }
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s input.obj output.obj\n", argv[0]);
        return 1;
    }

    std::vector<Vec3> verts;
    std::vector<int> faces, face_uvs;
    std::vector<Vec2> uvs;

    fprintf(stderr, "Loading %s...\n", argv[1]);
    if (!load_obj(argv[1], verts, faces, uvs, face_uvs)) {
        fprintf(stderr, "Failed to load %s\n", argv[1]);
        return 1;
    }

    int nV = verts.size();
    int nF = faces.size() / 3;
    int nUV = uvs.size();
    fprintf(stderr, "Loaded: %d V, %d F, %d UV\n", nV, nF, nUV);

    if (face_uvs.size() != faces.size()) {
        fprintf(stderr, "Error: need per-corner UV (f v/vt format)\n");
        return 1;
    }

    // Build libQEx input
    qex_TriMesh triMesh;
    triMesh.vertex_count = nV;
    triMesh.tri_count = nF;
    triMesh.vertices = (qex_Point3*)malloc(sizeof(qex_Point3) * nV);
    triMesh.tris = (qex_Tri*)malloc(sizeof(qex_Tri) * nF);
    triMesh.uvTris = (qex_UVTri*)malloc(sizeof(qex_UVTri) * nF);

    for (int i = 0; i < nV; i++) {
        triMesh.vertices[i].x[0] = verts[i].x;
        triMesh.vertices[i].x[1] = verts[i].y;
        triMesh.vertices[i].x[2] = verts[i].z;
    }

    for (int i = 0; i < nF; i++) {
        triMesh.tris[i].indices[0] = faces[i*3+0];
        triMesh.tris[i].indices[1] = faces[i*3+1];
        triMesh.tris[i].indices[2] = faces[i*3+2];

        for (int j = 0; j < 3; j++) {
            int uv_idx = face_uvs[i*3+j];
            triMesh.uvTris[i].uvs[j].x[0] = uvs[uv_idx].u;
            triMesh.uvTris[i].uvs[j].x[1] = uvs[uv_idx].v;
        }
    }

    // Extract quads
    fprintf(stderr, "Extracting quads...\n");
    qex_QuadMesh quadMesh;
    memset(&quadMesh, 0, sizeof(quadMesh));
    qex_extractQuadMesh(&triMesh, NULL, &quadMesh);

    fprintf(stderr, "Result: %d vertices, %d quads\n",
            quadMesh.vertex_count, quadMesh.quad_count);

    // Write output OBJ
    FILE* fout = fopen(argv[2], "w");
    for (unsigned int i = 0; i < quadMesh.vertex_count; i++) {
        fprintf(fout, "v %.8g %.8g %.8g\n",
                quadMesh.vertices[i].x[0],
                quadMesh.vertices[i].x[1],
                quadMesh.vertices[i].x[2]);
    }
    for (unsigned int i = 0; i < quadMesh.quad_count; i++) {
        fprintf(fout, "f %d %d %d %d\n",
                quadMesh.quads[i].indices[0] + 1,
                quadMesh.quads[i].indices[1] + 1,
                quadMesh.quads[i].indices[2] + 1,
                quadMesh.quads[i].indices[3] + 1);
    }
    fclose(fout);
    fprintf(stderr, "Wrote %s\n", argv[2]);

    free(triMesh.vertices);
    free(triMesh.tris);
    free(triMesh.uvTris);
    free(quadMesh.vertices);
    free(quadMesh.quads);

    return 0;
}
