#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 inNormal;

layout(set = 0, binding = 0) uniform SceneUniform {
	mat4 viewProjection;
	vec4 cameraPosition;
	vec4 ambientColor;
	vec4 directionalDirection;
	vec4 directionalColor;
	vec4 lightCounts;
} scene;

layout(set = 1, binding = 0) uniform ModelUniform {
	mat4 model;
	vec4 albedo;
	vec4 specularColor;
	vec4 material;
} model;

layout(location = 0) out vec3 fragWorldPosition;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragVertexColor;

void main() {
	vec4 world_position = model.model * vec4(inPosition, 1.0);
	fragWorldPosition = world_position.xyz;
	gl_Position = scene.viewProjection * world_position;

	// Inverse-transpose keeps normals correct under non-uniform stretching.
	mat3 normal_matrix = mat3(transpose(inverse(model.model)));
	fragNormal = normal_matrix * inNormal;
	fragVertexColor = inColor;
}
