#version 450

layout(location = 0) in vec3 fragWorldPosition;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragVertexColor;

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

struct PointLight {
	vec4 position;
	vec4 colorIntensity;
};

struct SpotLight {
	vec4 positionIntensity;
	vec4 directionInnerCos;
	vec4 colorOuterCos;
};

layout(std430, set = 0, binding = 1) readonly buffer PointLightBuffer {
	PointLight lights[];
} pointLights;

layout(std430, set = 0, binding = 2) readonly buffer SpotLightBuffer {
	SpotLight lights[];
} spotLights;

layout(location = 0) out vec4 outColor;

vec3 blinnPhong(vec3 normal, vec3 view_direction, vec3 light_direction,
                vec3 light_color, float intensity, vec3 albedo,
                vec3 specular_color, float shininess, float specular_strength) {
	float n_dot_l = max(dot(normal, light_direction), 0.0);
	if (n_dot_l <= 0.0) {
		return vec3(0.0);
	}

	vec3 half_vector = normalize(light_direction + view_direction);
	float n_dot_h = max(dot(normal, half_vector), 0.0);
	vec3 diffuse = albedo * n_dot_l;
	vec3 specular = specular_color * specular_strength * pow(n_dot_h, shininess);
	return (diffuse + specular) * light_color * intensity;
}

void main() {
	vec3 normal = normalize(fragNormal);
	vec3 view_direction = normalize(scene.cameraPosition.xyz - fragWorldPosition);

	vec3 albedo = model.albedo.rgb;
	if (model.albedo.w > 0.5) {
		albedo *= fragVertexColor;
	}

	const float shininess = max(model.material.x, 1.0);
	const float specular_strength = model.specularColor.w;

	vec3 color = albedo * scene.ambientColor.rgb * scene.ambientColor.w;

	// Directional light.
	vec3 directional_l = normalize(scene.directionalDirection.xyz);
	color += blinnPhong(normal, view_direction, directional_l,
	                    scene.directionalColor.rgb, scene.directionalDirection.w,
	                    albedo, model.specularColor.rgb, shininess,
	                    specular_strength);

	// Point lights, inverse-square attenuation.
	const int point_count = int(scene.lightCounts.x);
	for (int i = 0; i < point_count; ++i) {
		vec3 to_light = pointLights.lights[i].position.xyz - fragWorldPosition;
		float distance_squared = max(dot(to_light, to_light), 0.001);
		vec3 light_direction = to_light * inversesqrt(distance_squared);
		float attenuation = 1.0 / distance_squared;

		color += blinnPhong(normal, view_direction, light_direction,
		                    pointLights.lights[i].colorIntensity.rgb,
		                    pointLights.lights[i].colorIntensity.w,
		                    albedo, model.specularColor.rgb, shininess,
		                    specular_strength) * attenuation;
	}

	// Spot lights, inverse-square attenuation plus soft angular falloff.
	const int spot_count = int(scene.lightCounts.y);
	for (int i = 0; i < spot_count; ++i) {
		vec3 to_light = spotLights.lights[i].positionIntensity.xyz - fragWorldPosition;
		float distance_squared = max(dot(to_light, to_light), 0.001);
		float distance = sqrt(distance_squared);
		vec3 light_direction = to_light / distance;

		vec3 spot_direction = normalize(spotLights.lights[i].directionInnerCos.xyz);
		float cos_angle = dot(-light_direction, spot_direction);
		float inner_cos = spotLights.lights[i].directionInnerCos.w;
		float outer_cos = spotLights.lights[i].colorOuterCos.w;
		float cone = smoothstep(outer_cos, inner_cos, cos_angle);
		float attenuation = cone / distance_squared;

		color += blinnPhong(normal, view_direction, light_direction,
		                    spotLights.lights[i].colorOuterCos.rgb,
		                    spotLights.lights[i].positionIntensity.w,
		                    albedo, model.specularColor.rgb, shininess,
		                    specular_strength) * attenuation;
	}

	outColor = vec4(color, 1.0);
}
