#pragma once

#include <cmath>

namespace math {

constexpr float pi = 3.14159265358979323846f;

inline float radians(float degrees) {
	return degrees * (pi / 180.0f);
}

inline float clamp(float value, float min, float max) {
	return value < min ? min : (value > max ? max : value);
}

struct Vec3 {
	float x = 0.0f;
	float y = 0.0f;
	float z = 0.0f;

	Vec3() = default;
	Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
};

inline Vec3 operator+(const Vec3& a, const Vec3& b) {
	return { a.x + b.x, a.y + b.y, a.z + b.z };
}

inline Vec3 operator-(const Vec3& a, const Vec3& b) {
	return { a.x - b.x, a.y - b.y, a.z - b.z };
}

inline Vec3 operator*(const Vec3& v, float scalar) {
	return { v.x * scalar, v.y * scalar, v.z * scalar };
}

inline float dot(const Vec3& a, const Vec3& b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3 cross(const Vec3& a, const Vec3& b) {
	return {
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x,
	};
}

inline float length(const Vec3& v) {
	return std::sqrt(dot(v, v));
}

inline Vec3 normalize(const Vec3& v) {
	const float len = length(v);
	if (len < 1e-8f) {
		return {};
	}
	return v * (1.0f / len);
}

struct Vec4 {
	float x = 0.0f;
	float y = 0.0f;
	float z = 0.0f;
	float w = 0.0f;
};

struct Mat4 {
	alignas(16) float m[16] = {};

	static Mat4 identity() {
		Mat4 result;
		result.m[0] = result.m[5] = result.m[10] = result.m[15] = 1.0f;
		return result;
	}

	float& at(int column, int row) {
		return m[column * 4 + row];
	}

	float at(int column, int row) const {
		return m[column * 4 + row];
	}
};

inline Mat4 operator*(const Mat4& a, const Mat4& b) {
	Mat4 result;
	for (int column = 0; column < 4; ++column) {
		for (int row = 0; row < 4; ++row) {
			float value = 0.0f;
			for (int k = 0; k < 4; ++k) {
				value += a.at(k, row) * b.at(column, k);
			}
			result.at(column, row) = value;
		}
	}
	return result;
}

inline Vec4 operator*(const Mat4& m, const Vec4& v) {
	return {
		m.at(0, 0) * v.x + m.at(1, 0) * v.y + m.at(2, 0) * v.z + m.at(3, 0) * v.w,
		m.at(0, 1) * v.x + m.at(1, 1) * v.y + m.at(2, 1) * v.z + m.at(3, 1) * v.w,
		m.at(0, 2) * v.x + m.at(1, 2) * v.y + m.at(2, 2) * v.z + m.at(3, 2) * v.w,
		m.at(0, 3) * v.x + m.at(1, 3) * v.y + m.at(2, 3) * v.z + m.at(3, 3) * v.w,
	};
}

inline Mat4 translation(const Vec3& position) {
	Mat4 result = Mat4::identity();
	result.m[12] = position.x;
	result.m[13] = position.y;
	result.m[14] = position.z;
	return result;
}

inline Mat4 scaling(const Vec3& scale) {
	Mat4 result = Mat4::identity();
	result.m[0] = scale.x;
	result.m[5] = scale.y;
	result.m[10] = scale.z;
	return result;
}

inline Mat4 rotationX(float angle) {
	const float c = std::cos(angle);
	const float s = std::sin(angle);
	Mat4 result = Mat4::identity();
	result.at(1, 1) = c;
	result.at(1, 2) = s;
	result.at(2, 1) = -s;
	result.at(2, 2) = c;
	return result;
}

inline Mat4 rotationY(float angle) {
	const float c = std::cos(angle);
	const float s = std::sin(angle);
	Mat4 result = Mat4::identity();
	result.at(0, 0) = c;
	result.at(0, 2) = -s;
	result.at(2, 0) = s;
	result.at(2, 2) = c;
	return result;
}

inline Mat4 rotationZ(float angle) {
	const float c = std::cos(angle);
	const float s = std::sin(angle);
	Mat4 result = Mat4::identity();
	result.at(0, 0) = c;
	result.at(0, 1) = s;
	result.at(1, 0) = -s;
	result.at(1, 1) = c;
	return result;
}

inline Mat4 compose(const Vec3& position, const Vec3& rotation_degrees, const Vec3& scale) {
	return translation(position) *
	       rotationZ(radians(rotation_degrees.z)) *
	       rotationY(radians(rotation_degrees.y)) *
	       rotationX(radians(rotation_degrees.x)) *
	       scaling(scale);
}

inline Mat4 lookAt(const Vec3& eye, const Vec3& center, const Vec3& up) {
	const Vec3 forward = normalize(center - eye);
	const Vec3 side = normalize(cross(forward, up));
	const Vec3 actual_up = cross(side, forward);

	Mat4 result = Mat4::identity();
	result.at(0, 0) = side.x;
	result.at(0, 1) = actual_up.x;
	result.at(0, 2) = -forward.x;
	result.at(1, 0) = side.y;
	result.at(1, 1) = actual_up.y;
	result.at(1, 2) = -forward.y;
	result.at(2, 0) = side.z;
	result.at(2, 1) = actual_up.z;
	result.at(2, 2) = -forward.z;
	result.at(3, 0) = -dot(side, eye);
	result.at(3, 1) = -dot(actual_up, eye);
	result.at(3, 2) = dot(forward, eye);
	return result;
}

// Right-handed camera looking down -Z. Produces Vulkan depth range [0, 1].
inline Mat4 perspective(float fov_y_radians, float aspect, float near_plane, float far_plane) {
	const float f = 1.0f / std::tan(fov_y_radians * 0.5f);
	Mat4 result;
	result.at(0, 0) = f / aspect;
	result.at(1, 1) = f;
	result.at(2, 2) = far_plane / (near_plane - far_plane);
	result.at(2, 3) = -1.0f;
	result.at(3, 2) = (far_plane * near_plane) / (near_plane - far_plane);
	return result;
}

// Right-handed orthographic projection. Produces Vulkan depth range [0, 1].
inline Mat4 orthographic(float left, float right, float bottom, float top,
                         float near_plane, float far_plane) {
	Mat4 result = Mat4::identity();
	result.at(0, 0) = 2.0f / (right - left);
	result.at(1, 1) = 2.0f / (top - bottom);
	result.at(2, 2) = 1.0f / (near_plane - far_plane);
	result.at(3, 0) = -(right + left) / (right - left);
	result.at(3, 1) = -(top + bottom) / (top - bottom);
	result.at(3, 2) = near_plane / (near_plane - far_plane);
	return result;
}

} // namespace math
