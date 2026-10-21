#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 inNormal;

layout(set = 0, binding = 0) uniform UniformBufferObject {
	mat4 mvp;
	mat4 model;
	vec4 baseColor;
	vec4 lightDirection;
	vec4 cameraPosition;
} ubo;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragWorldPosition;

void main() {
	gl_Position = ubo.mvp * vec4(inPosition, 1.0);

	vec4 world_position = ubo.model * vec4(inPosition, 1.0);
	fragWorldPosition = world_position.xyz;

	// Inverse-transpose keeps normals correct under non-uniform stretching.
	mat3 normal_matrix = mat3(transpose(inverse(ubo.model)));
	fragNormal = normal_matrix * inNormal;

	// Procedural per-vertex color is multiplied by the color chosen in ImGui.
	fragColor = inColor * ubo.baseColor.rgb;
}
