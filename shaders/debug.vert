#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;

layout(set = 0, binding = 0) uniform SceneUniform {
	mat4 viewProjection;
	vec4 cameraPosition;
	vec4 ambientColor;
	vec4 directionalDirection;
	vec4 directionalColor;
	vec4 lightCounts;
} scene;

layout(location = 0) out vec3 fragColor;

void main() {
	gl_Position = scene.viewProjection * vec4(inPosition, 1.0);
	fragColor = inColor;
}
