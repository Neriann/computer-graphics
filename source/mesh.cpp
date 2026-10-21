#include "mesh.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <set>

namespace mesh {

namespace {

math::Vec3 makeColor(const math::Vec3& position, float radius) {
	const float scale = 0.5f / radius;
	return {
		math::clamp(0.5f + position.x * scale, 0.0f, 1.0f),
		math::clamp(0.5f + position.y * scale, 0.0f, 1.0f),
		math::clamp(0.5f + position.z * scale, 0.0f, 1.0f),
	};
}

} // namespace

std::vector<Vertex> createDodecahedron(float radius) {
	const float phi = (1.0f + std::sqrt(5.0f)) * 0.5f;
	const float inv_phi = 1.0f / phi;

	std::vector<math::Vec3> positions;
	positions.reserve(20);

	for (float x : { -1.0f, 1.0f }) {
		for (float y : { -1.0f, 1.0f }) {
			for (float z : { -1.0f, 1.0f }) {
				positions.emplace_back(x, y, z);
			}
		}
	}
	for (float y : { -inv_phi, inv_phi }) {
		for (float z : { -phi, phi }) {
			positions.emplace_back(0.0f, y, z);
		}
	}
	for (float x : { -inv_phi, inv_phi }) {
		for (float y : { -phi, phi }) {
			positions.emplace_back(x, y, 0.0f);
		}
	}
	for (float x : { -phi, phi }) {
		for (float z : { -inv_phi, inv_phi }) {
			positions.emplace_back(x, 0.0f, z);
		}
	}

	float max_length = 0.0f;
	for (const auto& position : positions) {
		max_length = std::max(max_length, math::length(position));
	}
	const float normalization = radius / max_length;
	for (auto& position : positions) {
		position = position * normalization;
	}

	struct Face {
		std::array<int, 5> indices;
		math::Vec3 normal;
	};

	std::vector<Face> faces;
	std::set<std::array<int, 5>> unique_faces;

	constexpr float plane_epsilon = 1e-3f;
	const int vertex_count = int(positions.size());

	for (int i = 0; i < vertex_count; ++i) {
		for (int j = i + 1; j < vertex_count; ++j) {
			for (int k = j + 1; k < vertex_count; ++k) {
				const math::Vec3 a = positions[i];
				const math::Vec3 b = positions[j];
				const math::Vec3 c = positions[k];

				math::Vec3 normal = math::cross(b - a, c - a);
				const float normal_length = math::length(normal);
				if (normal_length < 1e-6f) {
					continue;
				}
				normal = normal * (1.0f / normal_length);

				const float plane_distance = math::dot(normal, a);

				bool all_below_or_on_plane = true;
				bool all_above_or_on_plane = true;
				std::array<int, 5> face_indices{};
				int face_vertex_count = 0;

				for (int v = 0; v < vertex_count; ++v) {
					const float distance = math::dot(normal, positions[v]) - plane_distance;
					if (distance > plane_epsilon) {
						all_below_or_on_plane = false;
					}
					if (distance < -plane_epsilon) {
						all_above_or_on_plane = false;
					}
					if (std::fabs(distance) <= plane_epsilon) {
						if (face_vertex_count < 5) {
							face_indices[face_vertex_count] = v;
						}
						++face_vertex_count;
					}
				}

				if (face_vertex_count != 5) {
					continue;
				}
				if (!all_below_or_on_plane && !all_above_or_on_plane) {
					continue;
				}
				if (!all_below_or_on_plane) {
					normal = normal * -1.0f;
				}

				std::array<int, 5> sorted_indices = face_indices;
				std::sort(sorted_indices.begin(), sorted_indices.end());
				if (!unique_faces.insert(sorted_indices).second) {
					continue;
				}

				math::Vec3 center;
				for (int index : face_indices) {
					center = center + positions[index];
				}
				center = center * (1.0f / 5.0f);

				const math::Vec3 to_first = math::normalize(positions[face_indices[0]] - center);
				const math::Vec3 bitangent = math::normalize(math::cross(normal, to_first));

				std::sort(face_indices.begin(), face_indices.end(),
					[&](int lhs, int rhs) {
						const math::Vec3 left = positions[lhs] - center;
						const math::Vec3 right = positions[rhs] - center;
						const float left_angle = std::atan2(math::dot(left, bitangent),
						                                     math::dot(left, to_first));
						const float right_angle = std::atan2(math::dot(right, bitangent),
						                                      math::dot(right, to_first));
						return left_angle < right_angle;
					});

				faces.push_back({ face_indices, normal });
			}
		}
	}

	std::vector<Vertex> vertices;
	vertices.reserve(faces.size() * 9);

	for (const auto& face : faces) {
		for (int triangle = 1; triangle < 4; ++triangle) {
			const int triangle_indices[] = {
				face.indices[0],
				face.indices[triangle],
				face.indices[triangle + 1],
			};

			for (int index : triangle_indices) {
				const math::Vec3& position = positions[index];
				vertices.push_back({
					.position = position,
					.color = makeColor(position, radius),
					.normal = face.normal,
				});
			}
		}
	}

	return vertices;
}

} // namespace mesh
