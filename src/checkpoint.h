// Checkpoint system for debugging GPU mesh processing pipelines.
// Saves mesh state + per-element attributes at each step.
// Pure C++ — no CUDA headers.

#ifndef PYRXMESH_CHECKPOINT_H
#define PYRXMESH_CHECKPOINT_H

#include <string>
#include <vector>
#include <map>
#include <cstdio>
#include <filesystem>

struct Checkpoint {
    std::string name;           // e.g. "pass1_iter0_after_split"
    std::string description;    // human-readable description

    // Mesh geometry
    std::vector<double> vertices;   // flat [x0,y0,z0, ...]
    std::vector<int>    faces;      // flat [v0,v1,v2, ...]
    int num_vertices;
    int num_faces;

    // Per-vertex scalar fields (name → flat array)
    std::map<std::string, std::vector<double>> vertex_scalars;

    // Per-face scalar fields
    std::map<std::string, std::vector<double>> face_scalars;

    // Timing
    double elapsed_ms;   // time since pipeline start
};

class CheckpointLog {
    std::vector<Checkpoint> m_checkpoints;
    std::string m_output_dir;
    bool m_enabled;

public:
    CheckpointLog() : m_enabled(false) {}

    void enable(const std::string& output_dir) {
        m_enabled = true;
        m_output_dir = output_dir;
        std::filesystem::create_directories(output_dir);
    }

    bool is_enabled() const { return m_enabled; }

    void add(const Checkpoint& cp) {
        if (!m_enabled) return;
        m_checkpoints.push_back(cp);

        // Also save OBJ for external viewing
        std::string path = m_output_dir + "/" + cp.name + ".obj";
        FILE* f = fopen(path.c_str(), "w");
        if (f) {
            for (int i = 0; i < cp.num_vertices; i++)
                fprintf(f, "v %.8g %.8g %.8g\n",
                        cp.vertices[i*3], cp.vertices[i*3+1], cp.vertices[i*3+2]);
            for (int i = 0; i < cp.num_faces; i++)
                fprintf(f, "f %d %d %d\n",
                        cp.faces[i*3]+1, cp.faces[i*3+1]+1, cp.faces[i*3+2]+1);
            fclose(f);
        }

        // Save per-vertex scalars as separate files
        for (auto& [name, data] : cp.vertex_scalars) {
            std::string spath = m_output_dir + "/" + cp.name + "_v_" + name + ".raw";
            FILE* sf = fopen(spath.c_str(), "wb");
            if (sf) {
                fwrite(data.data(), sizeof(double), data.size(), sf);
                fclose(sf);
            }
        }

        // Save per-face scalars
        for (auto& [name, data] : cp.face_scalars) {
            std::string spath = m_output_dir + "/" + cp.name + "_f_" + name + ".raw";
            FILE* sf = fopen(spath.c_str(), "wb");
            if (sf) {
                fwrite(data.data(), sizeof(double), data.size(), sf);
                fclose(sf);
            }
        }
    }

    const std::vector<Checkpoint>& checkpoints() const { return m_checkpoints; }
    size_t size() const { return m_checkpoints.size(); }
};

#endif // PYRXMESH_CHECKPOINT_H
