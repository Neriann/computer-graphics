#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragWorldPosition;

layout(set = 0, binding = 0) uniform UniformBufferObject {
	mat4 mvp;
	mat4 model;
	vec4 baseColor;
	vec4 lightDirection;
	vec4 cameraPosition;
} ubo;

layout(location = 0) out vec4 outColor;

void main() {
	vec3 normal = normalize(fragNormal);
	vec3 light = normalize(-ubo.lightDirection.xyz);
	vec3 view_direction = normalize(ubo.cameraPosition.xyz - fragWorldPosition);

	float diffuse = max(dot(normal, light), 0.0);
	float back_light = max(dot(normal, -light), 0.0) * 0.12;
	float ambient = 0.32;

	vec3 half_vector = normalize(light + view_direction);
	float specular = pow(max(dot(normal, half_vector), 0.0), 48.0) *
	                 ubo.cameraPosition.w;

	vec3 color = fragColor * (ambient + diffuse * 0.78 + back_light) +
	             vec3(1.0) * specular;

	outColor = vec4(color, 1.0);
}
