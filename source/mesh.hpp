#pragma once

#include <cstdint>
#include <vector>

#include "math.hpp"

namespace mesh {

struct Vertex {
	math::Vec3 position;
	math::Vec3 color;
	math::Vec3 normal;
};

static_assert(sizeof(Vertex) == 9 * sizeof(float), "Vertex must be tightly packed");

// Builds a regular dodecahedron (variant 10) centered at the origin.
// Every pentagonal face is split into three triangles. Vertices are duplicated
// per face so that flat normals are produced, while colors are generated
// procedurally from the vertex position in the local coordinate system.
std::vector<Vertex> createDodecahedron(float radius = 1.0f);

} // namespace mesh
