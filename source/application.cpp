#include "application.hpp"

#include "math.hpp"
#include "mesh.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <imgui.h>

namespace application {

namespace {

constexpr int max_objects = 3;
constexpr uint32_t max_point_lights = 32;
constexpr uint32_t max_spot_lights = 16;
constexpr uint32_t max_debug_triangle_vertices = 4096;
constexpr uint32_t max_debug_line_vertices = 8192;

struct SceneUniform {
	alignas(16) math::Mat4 view_projection;
	alignas(16) float camera_position[4];
	alignas(16) float ambient_color[4];
	alignas(16) float directional_direction[4];
	alignas(16) float directional_color[4];
	alignas(16) float light_counts[4];
};

static_assert(sizeof(SceneUniform) == 144, "Scene UBO layout must match std140");

struct ModelUniform {
	alignas(16) math::Mat4 model;
	alignas(16) float albedo[4];
	alignas(16) float specular_color[4];
	alignas(16) float material[4];
};

static_assert(sizeof(ModelUniform) == 112, "Model UBO layout must match std140");

struct PointLightGpu {
	alignas(16) float position[4];
	alignas(16) float color_intensity[4];
};

static_assert(sizeof(PointLightGpu) == 32, "Point light layout must match std430");

struct SpotLightGpu {
	alignas(16) float position_intensity[4];
	alignas(16) float direction_inner_cos[4];
	alignas(16) float color_outer_cos[4];
};

static_assert(sizeof(SpotLightGpu) == 48, "Spot light layout must match std430");

struct DebugVertex {
	math::Vec3 position;
	math::Vec3 color;
};

static_assert(sizeof(DebugVertex) == 6 * sizeof(float),
              "Debug vertex must be tightly packed");

struct ObjectSettings {
	bool visible = true;
	math::Vec3 position{};
	math::Vec3 rotation{};
	math::Vec3 scale{ 1.0f, 1.0f, 1.0f };
	math::Vec3 albedo{ 0.8f, 0.8f, 0.8f };
	math::Vec3 specular_color{ 1.0f, 1.0f, 1.0f };
	float shininess = 64.0f;
	float specular_strength = 0.5f;
	bool use_vertex_color = false;
};

struct AnimationSettings {
	bool playing = true;
	float speed = 0.8f;
	float radius = 1.6f;
	float vertical_amplitude = 0.9f;
	float spin_speed = 55.0f;
	float phase = 0.0f;
	float time = 0.0f;

	math::Vec3 trajectoryOffset() const {
		return {
			radius * std::sin(time),
			vertical_amplitude * std::sin(2.0f * time + phase),
			radius * std::cos(time),
		};
	}

	math::Vec3 spinOffset() const {
		return {
			spin_speed * time * 0.65f,
			spin_speed * time,
			spin_speed * time * 0.25f,
		};
	}
};

struct CameraSettings {
	math::Vec3 position{ 0.0f, 1.5f, 9.5f };
	float yaw = 0.0f;
	float pitch = -0.156f;
	float fov_degrees = 45.0f;
	float near_plane = 0.1f;
	float far_plane = 100.0f;
	bool perspective = true;
	float ortho_scale = 3.6f;
};

struct PointLightSettings {
	bool enabled = true;
	math::Vec3 position{};
	math::Vec3 color{ 1.0f, 1.0f, 1.0f };
	float intensity = 10.0f;
};

struct SpotLightSettings {
	bool enabled = true;
	math::Vec3 position{};
	math::Vec3 direction{ 0.0f, -1.0f, 0.0f };
	math::Vec3 color{ 1.0f, 1.0f, 1.0f };
	float intensity = 20.0f;
	float inner_angle_degrees = 14.0f;
	float outer_angle_degrees = 26.0f;
};

struct LightingSettings {
	math::Vec3 ambient_color{ 1.0f, 1.0f, 1.0f };
	float ambient_intensity = 0.08f;
	math::Vec3 directional_direction{ 0.35f, 0.85f, 0.4f };
	math::Vec3 directional_color{ 1.0f, 0.96f, 0.88f };
	float directional_intensity = 0.9f;
	bool show_light_sources = true;
	std::vector<PointLightSettings> point_lights;
	std::vector<SpotLightSettings> spot_lights;
};

struct GpuObject {
	VkBuffer uniform_buffer = VK_NULL_HANDLE;
	VmaAllocation uniform_allocation = VK_NULL_HANDLE;
	ModelUniform* uniform_mapped = nullptr;
	VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
};

VkShaderModule vertex_shader_module = VK_NULL_HANDLE;
VkShaderModule fragment_shader_module = VK_NULL_HANDLE;
VkShaderModule debug_vertex_shader_module = VK_NULL_HANDLE;
VkShaderModule debug_fragment_shader_module = VK_NULL_HANDLE;

VkDescriptorSetLayout scene_set_layout = VK_NULL_HANDLE;
VkDescriptorSetLayout model_set_layout = VK_NULL_HANDLE;
VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
VkDescriptorSet scene_descriptor_set = VK_NULL_HANDLE;

VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline graphics_pipeline = VK_NULL_HANDLE;
VkPipelineLayout debug_pipeline_layout = VK_NULL_HANDLE;
VkPipeline debug_triangle_pipeline = VK_NULL_HANDLE;
VkPipeline debug_line_pipeline = VK_NULL_HANDLE;

VkBuffer vertex_buffer = VK_NULL_HANDLE;
VmaAllocation vertex_buffer_allocation = VK_NULL_HANDLE;
uint32_t vertex_count = 0;

VkBuffer scene_uniform_buffer = VK_NULL_HANDLE;
VmaAllocation scene_uniform_allocation = VK_NULL_HANDLE;
SceneUniform* scene_uniform_mapped = nullptr;

VkBuffer point_light_buffer = VK_NULL_HANDLE;
VmaAllocation point_light_allocation = VK_NULL_HANDLE;
PointLightGpu* point_light_mapped = nullptr;

VkBuffer spot_light_buffer = VK_NULL_HANDLE;
VmaAllocation spot_light_allocation = VK_NULL_HANDLE;
SpotLightGpu* spot_light_mapped = nullptr;

VkBuffer debug_triangle_buffer = VK_NULL_HANDLE;
VmaAllocation debug_triangle_allocation = VK_NULL_HANDLE;
DebugVertex* debug_triangle_mapped = nullptr;
uint32_t debug_triangle_vertex_count = 0;

VkBuffer debug_line_buffer = VK_NULL_HANDLE;
VmaAllocation debug_line_allocation = VK_NULL_HANDLE;
DebugVertex* debug_line_mapped = nullptr;
uint32_t debug_line_vertex_count = 0;

GpuObject gpu_objects[max_objects];
ObjectSettings objects[max_objects];
AnimationSettings animation;
CameraSettings camera;
LightingSettings lighting;

std::vector<char> readFile(const std::string& path) {
	std::ifstream file(path, std::ios::ate | std::ios::binary);
	if (!file.is_open()) {
		return {};
	}

	const std::streamsize size = file.tellg();
	if (size <= 0) {
		return {};
	}

	std::vector<char> buffer(static_cast<size_t>(size));
	file.seekg(0, std::ios::beg);
	file.read(buffer.data(), size);
	return buffer;
}

VkShaderModule createShaderModule(const char* name) {
	const std::string candidates[] = {
		std::string("shaders/") + name,
		std::string("../shaders/") + name,
		std::string("../../shaders/") + name,
	};

	std::vector<char> code;
	for (const auto& candidate : candidates) {
		code = readFile(candidate);
		if (!code.empty()) {
			break;
		}
	}

	if (code.empty() || code.size() % sizeof(uint32_t) != 0) {
		std::cerr << "Failed to load shader " << name << '\n';
		return VK_NULL_HANDLE;
	}

	const VkShaderModuleCreateInfo create_info = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = code.size(),
		.pCode = reinterpret_cast<const uint32_t*>(code.data()),
	};

	VkShaderModule module = VK_NULL_HANDLE;
	if (vkCreateShaderModule(graphics::internal::context.device, &create_info,
	                         nullptr, &module) != VK_SUCCESS) {
		std::cerr << "Failed to create shader module " << name << '\n';
		return VK_NULL_HANDLE;
	}
	return module;
}

bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                  VkBuffer& buffer, VmaAllocation& allocation, void** mapped) {
	const VkBufferCreateInfo buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	const VmaAllocationCreateInfo allocation_info = {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
		         VMA_ALLOCATION_CREATE_MAPPED_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
		.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
		                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
	};

	VmaAllocationInfo info{};
	if (vmaCreateBuffer(graphics::internal::context.allocator, &buffer_info,
	                    &allocation_info, &buffer, &allocation, &info) != VK_SUCCESS) {
		return false;
	}
	if (mapped != nullptr) {
		*mapped = info.pMappedData;
	}
	return true;
}

math::Vec3 safeDirection(const math::Vec3& direction, const math::Vec3& fallback) {
	if (math::length(direction) < 1e-5f) {
		return fallback;
	}
	return math::normalize(direction);
}

math::Vec3 lightMarkerColor(const math::Vec3& color) {
	return {
		math::clamp(color.x * 1.4f, 0.0f, 1.0f),
		math::clamp(color.y * 1.4f, 0.0f, 1.0f),
		math::clamp(color.z * 1.4f, 0.0f, 1.0f),
	};
}

void appendTriangle(std::vector<DebugVertex>& vertices,
                    const math::Vec3& a, const math::Vec3& b, const math::Vec3& c,
                    const math::Vec3& color) {
	vertices.push_back({ .position = a, .color = color });
	vertices.push_back({ .position = b, .color = color });
	vertices.push_back({ .position = c, .color = color });
}

void appendCube(std::vector<DebugVertex>& vertices, const math::Vec3& center,
                float half_size, const math::Vec3& color) {
	const math::Vec3 corners[8] = {
		center + math::Vec3{ -half_size, -half_size, -half_size },
		center + math::Vec3{  half_size, -half_size, -half_size },
		center + math::Vec3{  half_size,  half_size, -half_size },
		center + math::Vec3{ -half_size,  half_size, -half_size },
		center + math::Vec3{ -half_size, -half_size,  half_size },
		center + math::Vec3{  half_size, -half_size,  half_size },
		center + math::Vec3{  half_size,  half_size,  half_size },
		center + math::Vec3{ -half_size,  half_size,  half_size },
	};

	const int faces[6][4] = {
		{ 0, 3, 2, 1 },
		{ 4, 5, 6, 7 },
		{ 0, 1, 5, 4 },
		{ 3, 7, 6, 2 },
		{ 0, 4, 7, 3 },
		{ 1, 2, 6, 5 },
	};

	for (const auto& face : faces) {
		const math::Vec3& a = corners[face[0]];
		const math::Vec3& b = corners[face[1]];
		const math::Vec3& c = corners[face[2]];
		const math::Vec3& d = corners[face[3]];
		appendTriangle(vertices, a, b, c, color);
		appendTriangle(vertices, a, c, d, color);
	}
}

void appendDebugLine(std::vector<DebugVertex>& vertices,
                     const math::Vec3& a, const math::Vec3& b,
                     const math::Vec3& color) {
	vertices.push_back({ .position = a, .color = color });
	vertices.push_back({ .position = b, .color = color });
}

void appendSpotCone(std::vector<DebugVertex>& vertices,
                    const SpotLightSettings& light, float angle_degrees,
                    const math::Vec3& color, bool draw_spokes) {
	const math::Vec3 direction = safeDirection(light.direction, { 0.0f, -1.0f, 0.0f });
	const math::Vec3 up = std::fabs(direction.y) > 0.99f
		? math::Vec3{ 0.0f, 0.0f, 1.0f }
		: math::Vec3{ 0.0f, 1.0f, 0.0f };
	const math::Vec3 right = math::normalize(math::cross(direction, up));
	const math::Vec3 cone_up = math::normalize(math::cross(right, direction));

	const float angle = math::radians(math::clamp(angle_degrees, 1.0f, 85.0f));
	const float length = 2.4f;
	const float radius = length * std::tan(angle);
	const math::Vec3 center = light.position + direction * length;

	constexpr int segments = 24;
	math::Vec3 previous = center + right * radius;
	for (int i = 1; i <= segments; ++i) {
		const float a = 2.0f * math::pi * float(i) / float(segments);
		const math::Vec3 point = center +
			right * (std::cos(a) * radius) +
			cone_up * (std::sin(a) * radius);
		if (draw_spokes && i % 3 == 0) {
			appendDebugLine(vertices, light.position, point, color);
		}
		appendDebugLine(vertices, previous, point, color);
		previous = point;
	}
}

bool createDescriptorLayoutsAndPool() {
	const VkDescriptorSetLayoutBinding scene_bindings[] = {
		{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			.binding = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
		{
			.binding = 2,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
	};

	const VkDescriptorSetLayoutCreateInfo scene_layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = uint32_t(sizeof(scene_bindings) / sizeof(scene_bindings[0])),
		.pBindings = scene_bindings,
	};

	if (vkCreateDescriptorSetLayout(graphics::internal::context.device, &scene_layout_info,
	                                nullptr, &scene_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create scene descriptor set layout\n";
		return false;
	}

	const VkDescriptorSetLayoutBinding model_binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
	};

	const VkDescriptorSetLayoutCreateInfo model_layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &model_binding,
	};

	if (vkCreateDescriptorSetLayout(graphics::internal::context.device, &model_layout_info,
	                                nullptr, &model_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create model descriptor set layout\n";
		return false;
	}

	const VkDescriptorPoolSize pool_sizes[] = {
		{ .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1 + max_objects },
		{ .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2 },
	};

	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 1 + max_objects,
		.poolSizeCount = uint32_t(sizeof(pool_sizes) / sizeof(pool_sizes[0])),
		.pPoolSizes = pool_sizes,
	};

	if (vkCreateDescriptorPool(graphics::internal::context.device, &pool_info,
	                           nullptr, &descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor pool\n";
		return false;
	}
	return true;
}

bool createBuffers() {
	if (!createBuffer(sizeof(SceneUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
	                  scene_uniform_buffer, scene_uniform_allocation,
	                  reinterpret_cast<void**>(&scene_uniform_mapped))) {
		std::cerr << "Failed to create scene uniform buffer\n";
		return false;
	}

	if (!createBuffer(sizeof(PointLightGpu) * max_point_lights,
	                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
	                  point_light_buffer, point_light_allocation,
	                  reinterpret_cast<void**>(&point_light_mapped))) {
		std::cerr << "Failed to create point light storage buffer\n";
		return false;
	}

	if (!createBuffer(sizeof(SpotLightGpu) * max_spot_lights,
	                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
	                  spot_light_buffer, spot_light_allocation,
	                  reinterpret_cast<void**>(&spot_light_mapped))) {
		std::cerr << "Failed to create spot light storage buffer\n";
		return false;
	}

	for (int i = 0; i < max_objects; ++i) {
		if (!createBuffer(sizeof(ModelUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		                  gpu_objects[i].uniform_buffer,
		                  gpu_objects[i].uniform_allocation,
		                  reinterpret_cast<void**>(&gpu_objects[i].uniform_mapped))) {
			std::cerr << "Failed to create model uniform buffer #" << i << '\n';
			return false;
		}
	}

	if (!createBuffer(sizeof(DebugVertex) * max_debug_triangle_vertices,
	                  VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
	                  debug_triangle_buffer, debug_triangle_allocation,
	                  reinterpret_cast<void**>(&debug_triangle_mapped))) {
		std::cerr << "Failed to create debug triangle buffer\n";
		return false;
	}

	if (!createBuffer(sizeof(DebugVertex) * max_debug_line_vertices,
	                  VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
	                  debug_line_buffer, debug_line_allocation,
	                  reinterpret_cast<void**>(&debug_line_mapped))) {
		std::cerr << "Failed to create debug line buffer\n";
		return false;
	}
	return true;
}

bool allocateDescriptors() {
	const VkDescriptorSetAllocateInfo scene_allocate = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &scene_set_layout,
	};

	if (vkAllocateDescriptorSets(graphics::internal::context.device, &scene_allocate,
	                             &scene_descriptor_set) != VK_SUCCESS) {
		std::cerr << "Failed to allocate scene descriptor set\n";
		return false;
	}

	VkDescriptorSetLayout model_layouts[max_objects] = {
		model_set_layout, model_set_layout, model_set_layout,
	};

	const VkDescriptorSetAllocateInfo model_allocate = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptor_pool,
		.descriptorSetCount = max_objects,
		.pSetLayouts = model_layouts,
	};

	VkDescriptorSet model_sets[max_objects]{};
	if (vkAllocateDescriptorSets(graphics::internal::context.device, &model_allocate,
	                             model_sets) != VK_SUCCESS) {
		std::cerr << "Failed to allocate model descriptor sets\n";
		return false;
	}

	const VkDescriptorBufferInfo scene_buffer_info = {
		.buffer = scene_uniform_buffer,
		.offset = 0,
		.range = sizeof(SceneUniform),
	};

	const VkDescriptorBufferInfo point_buffer_info = {
		.buffer = point_light_buffer,
		.offset = 0,
		.range = VkDeviceSize(sizeof(PointLightGpu) * max_point_lights),
	};

	const VkDescriptorBufferInfo spot_buffer_info = {
		.buffer = spot_light_buffer,
		.offset = 0,
		.range = VkDeviceSize(sizeof(SpotLightGpu) * max_spot_lights),
	};

	const VkWriteDescriptorSet scene_writes[] = {
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = scene_descriptor_set,
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &scene_buffer_info,
		},
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = scene_descriptor_set,
			.dstBinding = 1,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.pBufferInfo = &point_buffer_info,
		},
		{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = scene_descriptor_set,
			.dstBinding = 2,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.pBufferInfo = &spot_buffer_info,
		},
	};

	vkUpdateDescriptorSets(graphics::internal::context.device,
	                       uint32_t(sizeof(scene_writes) / sizeof(scene_writes[0])),
	                       scene_writes, 0, nullptr);

	for (int i = 0; i < max_objects; ++i) {
		gpu_objects[i].descriptor_set = model_sets[i];

		const VkDescriptorBufferInfo model_buffer_info = {
			.buffer = gpu_objects[i].uniform_buffer,
			.offset = 0,
			.range = sizeof(ModelUniform),
		};

		const VkWriteDescriptorSet write = {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = gpu_objects[i].descriptor_set,
			.dstBinding = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &model_buffer_info,
		};

		vkUpdateDescriptorSets(graphics::internal::context.device, 1, &write, 0, nullptr);
	}
	return true;
}

bool createVertexBuffer() {
	const std::vector<mesh::Vertex> vertices = mesh::createDodecahedron(1.0f);
	if (vertices.empty()) {
		std::cerr << "Failed to generate dodecahedron geometry\n";
		return false;
	}

	void* mapped = nullptr;
	const VkDeviceSize size = VkDeviceSize(vertices.size()) * sizeof(mesh::Vertex);
	if (!createBuffer(size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
	                  vertex_buffer, vertex_buffer_allocation, &mapped)) {
		std::cerr << "Failed to create vertex buffer\n";
		return false;
	}

	std::memcpy(mapped, vertices.data(), size_t(size));
	vertex_count = uint32_t(vertices.size());
	return true;
}

bool createMainPipeline() {
	vertex_shader_module = createShaderModule("shape.vert.spv");
	fragment_shader_module = createShaderModule("shape.frag.spv");
	if (vertex_shader_module == VK_NULL_HANDLE || fragment_shader_module == VK_NULL_HANDLE) {
		return false;
	}

	const VkPipelineShaderStageCreateInfo shader_stages[] = {
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertex_shader_module,
			.pName = "main",
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragment_shader_module,
			.pName = "main",
		},
	};

	const VkVertexInputBindingDescription vertex_binding = {
		.binding = 0,
		.stride = sizeof(mesh::Vertex),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};

	const VkVertexInputAttributeDescription vertex_attributes[] = {
		{
			.location = 0,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(mesh::Vertex, position),
		},
		{
			.location = 1,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(mesh::Vertex, color),
		},
		{
			.location = 2,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(mesh::Vertex, normal),
		},
	};

	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &vertex_binding,
		.vertexAttributeDescriptionCount =
			uint32_t(sizeof(vertex_attributes) / sizeof(vertex_attributes[0])),
		.pVertexAttributeDescriptions = vertex_attributes,
	};

	const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
	};

	const VkViewport viewport = { 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f };
	const VkRect2D scissor = { { 0, 0 }, { 1, 1 } };

	const VkPipelineViewportStateCreateInfo viewport_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.pViewports = &viewport,
		.scissorCount = 1,
		.pScissors = &scissor,
	};

	const VkPipelineRasterizationStateCreateInfo rasterization = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_NONE,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.lineWidth = 1.0f,
	};

	const VkPipelineMultisampleStateCreateInfo multisampling = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};

	const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_LESS,
	};

	const VkPipelineColorBlendAttachmentState color_attachment = {
		.blendEnable = VK_FALSE,
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
		                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
	};

	const VkPipelineColorBlendStateCreateInfo color_blending = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &color_attachment,
	};

	const VkDynamicState dynamic_states[] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
	};

	const VkPipelineDynamicStateCreateInfo dynamic_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = uint32_t(sizeof(dynamic_states) / sizeof(dynamic_states[0])),
		.pDynamicStates = dynamic_states,
	};

	const VkDescriptorSetLayout set_layouts[] = { scene_set_layout, model_set_layout };

	const VkPipelineLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 2,
		.pSetLayouts = set_layouts,
	};

	if (vkCreatePipelineLayout(graphics::internal::context.device, &layout_info,
	                           nullptr, &pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create main pipeline layout\n";
		return false;
	}

	const VkGraphicsPipelineCreateInfo pipeline_info = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = uint32_t(sizeof(shader_stages) / sizeof(shader_stages[0])),
		.pStages = shader_stages,
		.pVertexInputState = &vertex_input,
		.pInputAssemblyState = &input_assembly,
		.pViewportState = &viewport_state,
		.pRasterizationState = &rasterization,
		.pMultisampleState = &multisampling,
		.pDepthStencilState = &depth_stencil,
		.pColorBlendState = &color_blending,
		.pDynamicState = &dynamic_state,
		.layout = pipeline_layout,
		.renderPass = graphics::internal::context.render_pass,
		.subpass = 0,
	};

	if (vkCreateGraphicsPipelines(graphics::internal::context.device, VK_NULL_HANDLE,
	                              1, &pipeline_info, nullptr,
	                              &graphics_pipeline) != VK_SUCCESS) {
		std::cerr << "Failed to create main graphics pipeline\n";
		return false;
	}
	return true;
}

VkPipeline createDebugPipeline(VkPrimitiveTopology topology, bool depth_write) {
	const VkPipelineShaderStageCreateInfo shader_stages[] = {
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = debug_vertex_shader_module,
			.pName = "main",
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = debug_fragment_shader_module,
			.pName = "main",
		},
	};

	const VkVertexInputBindingDescription vertex_binding = {
		.binding = 0,
		.stride = sizeof(DebugVertex),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};

	const VkVertexInputAttributeDescription vertex_attributes[] = {
		{
			.location = 0,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(DebugVertex, position),
		},
		{
			.location = 1,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(DebugVertex, color),
		},
	};

	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &vertex_binding,
		.vertexAttributeDescriptionCount =
			uint32_t(sizeof(vertex_attributes) / sizeof(vertex_attributes[0])),
		.pVertexAttributeDescriptions = vertex_attributes,
	};

	const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = topology,
	};

	const VkViewport viewport = { 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f };
	const VkRect2D scissor = { { 0, 0 }, { 1, 1 } };

	const VkPipelineViewportStateCreateInfo viewport_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.pViewports = &viewport,
		.scissorCount = 1,
		.pScissors = &scissor,
	};

	const VkPipelineRasterizationStateCreateInfo rasterization = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_NONE,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.lineWidth = 1.0f,
	};

	const VkPipelineMultisampleStateCreateInfo multisampling = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};

	const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = depth_write ? VK_TRUE : VK_FALSE,
		.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
	};

	const VkPipelineColorBlendAttachmentState color_attachment = {
		.blendEnable = VK_FALSE,
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
		                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
	};

	const VkPipelineColorBlendStateCreateInfo color_blending = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &color_attachment,
	};

	const VkDynamicState dynamic_states[] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
	};

	const VkPipelineDynamicStateCreateInfo dynamic_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = uint32_t(sizeof(dynamic_states) / sizeof(dynamic_states[0])),
		.pDynamicStates = dynamic_states,
	};

	const VkGraphicsPipelineCreateInfo pipeline_info = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = uint32_t(sizeof(shader_stages) / sizeof(shader_stages[0])),
		.pStages = shader_stages,
		.pVertexInputState = &vertex_input,
		.pInputAssemblyState = &input_assembly,
		.pViewportState = &viewport_state,
		.pRasterizationState = &rasterization,
		.pMultisampleState = &multisampling,
		.pDepthStencilState = &depth_stencil,
		.pColorBlendState = &color_blending,
		.pDynamicState = &dynamic_state,
		.layout = debug_pipeline_layout,
		.renderPass = graphics::internal::context.render_pass,
		.subpass = 0,
	};

	VkPipeline pipeline = VK_NULL_HANDLE;
	if (vkCreateGraphicsPipelines(graphics::internal::context.device, VK_NULL_HANDLE,
	                              1, &pipeline_info, nullptr, &pipeline) != VK_SUCCESS) {
		std::cerr << "Failed to create debug pipeline\n";
		return VK_NULL_HANDLE;
	}
	return pipeline;
}

bool createDebugResources() {
	debug_vertex_shader_module = createShaderModule("debug.vert.spv");
	debug_fragment_shader_module = createShaderModule("debug.frag.spv");
	if (debug_vertex_shader_module == VK_NULL_HANDLE ||
	    debug_fragment_shader_module == VK_NULL_HANDLE) {
		return false;
	}

	const VkPipelineLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &scene_set_layout,
	};

	if (vkCreatePipelineLayout(graphics::internal::context.device, &layout_info,
	                           nullptr, &debug_pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create debug pipeline layout\n";
		return false;
	}

	debug_triangle_pipeline = createDebugPipeline(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, true);
	debug_line_pipeline = createDebugPipeline(VK_PRIMITIVE_TOPOLOGY_LINE_LIST, false);

	return debug_triangle_pipeline != VK_NULL_HANDLE &&
	       debug_line_pipeline != VK_NULL_HANDLE;
}

void resetScene() {
	objects[0] = {
		.visible = true,
		.position = { 0.0f, 0.0f, 0.0f },
		.rotation = { 0.0f, 0.0f, 0.0f },
		.scale = { 1.0f, 1.0f, 1.0f },
		.albedo = { 0.72f, 0.70f, 0.66f },
		.specular_color = { 1.0f, 1.0f, 1.0f },
		.shininess = 64.0f,
		.specular_strength = 0.45f,
		.use_vertex_color = false,
	};
	objects[1] = {
		.visible = true,
		.position = { -2.7f, 1.2f, -0.4f },
		.rotation = { 25.0f, 40.0f, 0.0f },
		.scale = { 0.45f, 0.45f, 0.45f },
		.albedo = { 0.95f, 0.45f, 0.18f },
		.specular_color = { 1.0f, 0.9f, 0.7f },
		.shininess = 48.0f,
		.specular_strength = 0.5f,
		.use_vertex_color = false,
	};
	objects[2] = {
		.visible = true,
		.position = { 2.7f, -1.0f, 0.7f },
		.rotation = { -20.0f, -35.0f, 15.0f },
		.scale = { 0.35f, 0.35f, 0.35f },
		.albedo = { 0.18f, 0.45f, 1.0f },
		.specular_color = { 0.8f, 0.9f, 1.0f },
		.shininess = 48.0f,
		.specular_strength = 0.5f,
		.use_vertex_color = false,
	};

	animation = {};
	camera = {};

	lighting = {};
	lighting.ambient_intensity = 0.06f;
	lighting.directional_direction = { 0.35f, 0.75f, 0.55f };
	lighting.directional_intensity = 0.45f;
	lighting.point_lights = {
		{
			.enabled = true,
			.position = { 2.2f, 1.6f, 2.2f },
			.color = { 1.0f, 0.32f, 0.16f },
			.intensity = 2.2f,
		},
		{
			.enabled = true,
			.position = { -2.2f, 0.5f, 2.0f },
			.color = { 0.16f, 0.45f, 1.0f },
			.intensity = 1.6f,
		},
	};
	lighting.spot_lights = {
		{
			.enabled = true,
			.position = { 0.5f, 2.8f, 2.4f },
			.direction = { 0.0f, -1.0f, -0.45f },
			.color = { 0.55f, 0.8f, 1.0f },
			.intensity = 3.0f,
			.inner_angle_degrees = 14.0f,
			.outer_angle_degrees = 26.0f,
		},
	};
}

void updateAnimation() {
	if (animation.playing) {
		animation.time += ImGui::GetIO().DeltaTime * animation.speed;
	}
}

void aimCameraAtOrigin() {
	const math::Vec3 direction =
		safeDirection(math::Vec3{ 0.0f, 0.0f, 0.0f } - camera.position,
		              { 0.0f, 0.0f, -1.0f });
	camera.pitch = std::asin(math::clamp(direction.y, -1.0f, 1.0f));
	camera.yaw = std::atan2(direction.x, -direction.z);
}

math::Vec3 cameraForward() {
	const float cos_pitch = std::cos(camera.pitch);
	return {
		cos_pitch * std::sin(camera.yaw),
		std::sin(camera.pitch),
		-cos_pitch * std::cos(camera.yaw),
	};
}

void addPointLight() {
	if (lighting.point_lights.size() >= max_point_lights) {
		return;
	}
	const float angle = float(lighting.point_lights.size()) * 1.4f;
	lighting.point_lights.push_back({
		.enabled = true,
		.position = { 2.4f * std::cos(angle), 1.8f, 2.4f * std::sin(angle) },
		.color = { 1.0f, 0.65f, 0.25f },
		.intensity = 2.2f,
	});
}

void addSpotLight() {
	if (lighting.spot_lights.size() >= max_spot_lights) {
		return;
	}
	lighting.spot_lights.push_back({
		.enabled = true,
		.position = { 0.5f, 2.8f, 2.4f },
		.direction = { 0.0f, -1.0f, -0.45f },
		.color = { 0.6f, 0.85f, 1.0f },
		.intensity = 3.0f,
		.inner_angle_degrees = 14.0f,
		.outer_angle_degrees = 26.0f,
	});
}

void drawSceneTab() {
	ImGui::SeparatorText("Projection");
	int projection_mode = camera.perspective ? 0 : 1;
	const char* projection_names[] = { "Perspective", "Orthographic" };
	if (ImGui::Combo("Mode", &projection_mode, projection_names, 2)) {
		camera.perspective = projection_mode == 0;
	}
	if (camera.perspective) {
		ImGui::SliderFloat("Field of view", &camera.fov_degrees, 20.0f, 100.0f, "%.1f deg");
	} else {
		ImGui::SliderFloat("Ortho half-height", &camera.ortho_scale, 1.0f, 12.0f, "%.2f");
	}
	ImGui::SliderFloat("Near plane", &camera.near_plane, 0.01f, 5.0f, "%.2f");
	ImGui::SliderFloat("Far plane", &camera.far_plane, 10.0f, 200.0f, "%.1f");

	ImGui::SeparatorText("Main object transform");
	ImGui::DragFloat3("Position", &objects[0].position.x, 0.02f, -10.0f, 10.0f, "%.2f");
	ImGui::DragFloat3("Rotation", &objects[0].rotation.x, 0.5f, -720.0f, 720.0f, "%.1f deg");
	ImGui::DragFloat3("Stretch / scale", &objects[0].scale.x, 0.01f, 0.05f, 5.0f, "%.2f");

	ImGui::SeparatorText("Trajectory animation");
	ImGui::Checkbox("Play", &animation.playing);
	ImGui::SameLine();
	if (ImGui::Button("Reset trajectory")) {
		animation.time = 0.0f;
	}
	ImGui::SliderFloat("Speed", &animation.speed, 0.0f, 3.0f, "%.2f");
	ImGui::SliderFloat("Trajectory radius", &animation.radius, 0.0f, 6.0f, "%.2f");
	ImGui::SliderFloat("Vertical amplitude", &animation.vertical_amplitude, 0.0f, 4.0f, "%.2f");
	ImGui::SliderFloat("Spin speed", &animation.spin_speed, 0.0f, 360.0f, "%.1f deg/s");
	ImGui::SliderFloat("Trajectory phase", &animation.phase, 0.0f, 2.0f * math::pi, "%.2f");
	ImGui::Text("Animation time: %.2f s", animation.time);

	ImGui::SeparatorText("Additional objects");
	for (int i = 1; i < max_objects; ++i) {
		ImGui::PushID(i);
		ImGui::Text("Object #%d", i + 1);
		ImGui::Checkbox("Visible", &objects[i].visible);
		ImGui::SameLine();
		ImGui::ColorEdit3("Albedo", &objects[i].albedo.x, ImGuiColorEditFlags_NoInputs);
		ImGui::DragFloat3("Position", &objects[i].position.x, 0.02f, -10.0f, 10.0f, "%.2f");
		ImGui::DragFloat3("Scale", &objects[i].scale.x, 0.01f, 0.05f, 5.0f, "%.2f");
		ImGui::PopID();
	}
}

void drawCameraTab() {
	ImGui::SeparatorText("Free-fly camera");
	ImGui::DragFloat3("Position", &camera.position.x, 0.05f, -30.0f, 30.0f, "%.2f");
	ImGui::SliderFloat("Yaw", &camera.yaw, -math::pi, math::pi, "%.2f rad");
	ImGui::SliderFloat("Pitch", &camera.pitch, -1.5f, 1.5f, "%.2f rad");
	ImGui::SliderFloat("Field of view", &camera.fov_degrees, 20.0f, 100.0f, "%.1f deg");
	ImGui::SliderFloat("Near plane", &camera.near_plane, 0.01f, 5.0f, "%.2f");
	ImGui::SliderFloat("Far plane", &camera.far_plane, 10.0f, 200.0f, "%.1f");

	if (ImGui::Button("Look at origin")) {
		aimCameraAtOrigin();
	}
	ImGui::SameLine();
	if (ImGui::Button("Reset camera")) {
		camera = {};
	}

	const math::Vec3 forward = cameraForward();
	ImGui::Text("Forward: %.2f, %.2f, %.2f", forward.x, forward.y, forward.z);
}

void drawMaterialTab() {
	ImGui::SeparatorText("Blinn-Phong material");
	ImGui::ColorEdit3("Albedo", &objects[0].albedo.x);
	ImGui::ColorEdit3("Specular color", &objects[0].specular_color.x);
	ImGui::SliderFloat("Shininess", &objects[0].shininess, 1.0f, 256.0f, "%.1f");
	ImGui::SliderFloat("Specular strength", &objects[0].specular_strength, 0.0f, 2.0f, "%.2f");
	ImGui::Checkbox("Multiply by procedural vertex color", &objects[0].use_vertex_color);

	ImGui::SeparatorText("Global ambient light");
	ImGui::ColorEdit3("Ambient color", &lighting.ambient_color.x);
	ImGui::SliderFloat("Ambient intensity", &lighting.ambient_intensity, 0.0f, 1.0f, "%.2f");
}

void drawLightsTab() {
	ImGui::Checkbox("Show light source helpers", &lighting.show_light_sources);

	ImGui::SeparatorText("Directional light");
	ImGui::DragFloat3("Direction to light", &lighting.directional_direction.x,
	                  0.01f, -5.0f, 5.0f, "%.2f");
	ImGui::ColorEdit3("Directional color", &lighting.directional_color.x);
	ImGui::SliderFloat("Directional intensity", &lighting.directional_intensity,
	                   0.0f, 3.0f, "%.2f");

	ImGui::SeparatorText("Point lights (storage buffer)");
	ImGui::Text("%zu / %u", lighting.point_lights.size(), max_point_lights);
	ImGui::SameLine();
	if (lighting.point_lights.size() < max_point_lights) {
		if (ImGui::Button("Add point light")) {
			addPointLight();
		}
	} else {
		ImGui::TextDisabled("Limit reached");
	}

	ImGui::BeginChild("point_lights", ImVec2(0.0f, 210.0f), true);
	int point_to_remove = -1;
	for (int i = 0; i < int(lighting.point_lights.size()); ++i) {
		auto& light = lighting.point_lights[size_t(i)];
		ImGui::PushID(i);

		const bool open = ImGui::TreeNodeEx("##point_light",
			ImGuiTreeNodeFlags_DefaultOpen, "Point light %d", i + 1);
		ImGui::SameLine();
		if (ImGui::SmallButton("Remove")) {
			point_to_remove = i;
		}

		if (open) {
			ImGui::Checkbox("Enabled", &light.enabled);
			ImGui::DragFloat3("Position", &light.position.x, 0.02f, -30.0f, 30.0f, "%.2f");
			ImGui::ColorEdit3("Color", &light.color.x);
			ImGui::SliderFloat("Intensity", &light.intensity, 0.0f, 60.0f, "%.1f");
			ImGui::TreePop();
		}
		ImGui::PopID();
	}
	ImGui::EndChild();

	if (point_to_remove >= 0) {
		lighting.point_lights.erase(lighting.point_lights.begin() + point_to_remove);
	}

	ImGui::SeparatorText("Spot lights (storage buffer)");
	ImGui::Text("%zu / %u", lighting.spot_lights.size(), max_spot_lights);
	ImGui::SameLine();
	if (lighting.spot_lights.size() < max_spot_lights) {
		if (ImGui::Button("Add spot light")) {
			addSpotLight();
		}
	} else {
		ImGui::TextDisabled("Limit reached");
	}

	ImGui::BeginChild("spot_lights", ImVec2(0.0f, 280.0f), true);
	int spot_to_remove = -1;
	for (int i = 0; i < int(lighting.spot_lights.size()); ++i) {
		auto& light = lighting.spot_lights[size_t(i)];
		ImGui::PushID(i);

		const bool open = ImGui::TreeNodeEx("##spot_light",
			ImGuiTreeNodeFlags_DefaultOpen, "Spot light %d", i + 1);
		ImGui::SameLine();
		if (ImGui::SmallButton("Remove")) {
			spot_to_remove = i;
		}

		if (open) {
			ImGui::Checkbox("Enabled", &light.enabled);
			ImGui::DragFloat3("Position", &light.position.x, 0.02f, -30.0f, 30.0f, "%.2f");
			ImGui::DragFloat3("Direction", &light.direction.x, 0.01f, -5.0f, 5.0f, "%.2f");
			ImGui::ColorEdit3("Color", &light.color.x);
			ImGui::SliderFloat("Intensity", &light.intensity, 0.0f, 80.0f, "%.1f");
			ImGui::SliderFloat("Inner angle", &light.inner_angle_degrees, 1.0f, 80.0f, "%.1f deg");
			ImGui::SliderFloat("Outer angle", &light.outer_angle_degrees, 1.0f, 85.0f, "%.1f deg");

			if (light.outer_angle_degrees <= light.inner_angle_degrees) {
				light.outer_angle_degrees = light.inner_angle_degrees + 0.5f;
			}
			light.outer_angle_degrees = math::clamp(light.outer_angle_degrees, 1.0f, 85.0f);
			ImGui::TreePop();
		}
		ImGui::PopID();
	}
	ImGui::EndChild();

	if (spot_to_remove >= 0) {
		lighting.spot_lights.erase(lighting.spot_lights.begin() + spot_to_remove);
	}
}

void drawInterface() {
	ImGui::SetNextWindowSize(ImVec2(480.0f, 720.0f), ImGuiCond_FirstUseEver);
	ImGui::Begin("Laboratory 2 - Camera & Blinn-Phong");

	ImGui::TextUnformatted("Variant 10: regular dodecahedron");
	ImGui::Text("FPS: %.1f | Point lights: %zu | Spot lights: %zu",
	            ImGui::GetIO().Framerate,
	            lighting.point_lights.size(),
	            lighting.spot_lights.size());

	if (ImGui::BeginTabBar("lab2_tabs")) {
		if (ImGui::BeginTabItem("Scene")) {
			drawSceneTab();
			ImGui::EndTabItem();
		}
		if (ImGui::BeginTabItem("Camera")) {
			drawCameraTab();
			ImGui::EndTabItem();
		}
		if (ImGui::BeginTabItem("Lights")) {
			drawLightsTab();
			ImGui::EndTabItem();
		}
		if (ImGui::BeginTabItem("Material")) {
			drawMaterialTab();
			ImGui::EndTabItem();
		}
		ImGui::EndTabBar();
	}

	ImGui::Separator();
	if (ImGui::Button("Reset scene")) {
		resetScene();
	}
	ImGui::End();
}

math::Mat4 buildProjection(float aspect) {
	if (camera.perspective) {
		return math::perspective(math::radians(camera.fov_degrees), aspect,
		                         camera.near_plane, camera.far_plane);
	}
	const float half_height = camera.ortho_scale;
	const float half_width = half_height * aspect;
	return math::orthographic(-half_width, half_width,
	                          -half_height, half_height,
	                          camera.near_plane, camera.far_plane);
}

void updateSceneUniform(const math::Mat4& view_projection, const math::Vec3& eye) {
	SceneUniform uniform{};
	uniform.view_projection = view_projection;
	uniform.camera_position[0] = eye.x;
	uniform.camera_position[1] = eye.y;
	uniform.camera_position[2] = eye.z;
	uniform.camera_position[3] = 1.0f;

	uniform.ambient_color[0] = lighting.ambient_color.x;
	uniform.ambient_color[1] = lighting.ambient_color.y;
	uniform.ambient_color[2] = lighting.ambient_color.z;
	uniform.ambient_color[3] = lighting.ambient_intensity;

	const math::Vec3 direction =
		safeDirection(lighting.directional_direction, { 0.0f, 1.0f, 0.0f });
	uniform.directional_direction[0] = direction.x;
	uniform.directional_direction[1] = direction.y;
	uniform.directional_direction[2] = direction.z;
	uniform.directional_direction[3] = lighting.directional_intensity;

	uniform.directional_color[0] = lighting.directional_color.x;
	uniform.directional_color[1] = lighting.directional_color.y;
	uniform.directional_color[2] = lighting.directional_color.z;
	uniform.directional_color[3] = 1.0f;

	uint32_t point_count = 0;
	for (const auto& light : lighting.point_lights) {
		if (!light.enabled || point_count >= max_point_lights) {
			continue;
		}
		PointLightGpu& gpu = point_light_mapped[point_count++];
		gpu.position[0] = light.position.x;
		gpu.position[1] = light.position.y;
		gpu.position[2] = light.position.z;
		gpu.position[3] = 1.0f;
		gpu.color_intensity[0] = light.color.x;
		gpu.color_intensity[1] = light.color.y;
		gpu.color_intensity[2] = light.color.z;
		gpu.color_intensity[3] = light.intensity;
	}

	uint32_t spot_count = 0;
	for (const auto& light : lighting.spot_lights) {
		if (!light.enabled || spot_count >= max_spot_lights) {
			continue;
		}
		SpotLightGpu& gpu = spot_light_mapped[spot_count++];
		const math::Vec3 spot_direction =
			safeDirection(light.direction, { 0.0f, -1.0f, 0.0f });
		const float inner = math::radians(math::clamp(light.inner_angle_degrees, 1.0f, 80.0f));
		const float outer = math::radians(math::clamp(light.outer_angle_degrees, 1.0f, 85.0f));

		gpu.position_intensity[0] = light.position.x;
		gpu.position_intensity[1] = light.position.y;
		gpu.position_intensity[2] = light.position.z;
		gpu.position_intensity[3] = light.intensity;
		gpu.direction_inner_cos[0] = spot_direction.x;
		gpu.direction_inner_cos[1] = spot_direction.y;
		gpu.direction_inner_cos[2] = spot_direction.z;
		gpu.direction_inner_cos[3] = std::cos(inner);
		gpu.color_outer_cos[0] = light.color.x;
		gpu.color_outer_cos[1] = light.color.y;
		gpu.color_outer_cos[2] = light.color.z;
		gpu.color_outer_cos[3] = std::cos(outer);
	}

	uniform.light_counts[0] = float(point_count);
	uniform.light_counts[1] = float(spot_count);
	uniform.light_counts[2] = float(max_point_lights);
	uniform.light_counts[3] = float(max_spot_lights);

	std::memcpy(scene_uniform_mapped, &uniform, sizeof(uniform));
}

void updateModelUniforms() {
	for (int i = 0; i < max_objects; ++i) {
		math::Vec3 position = objects[i].position;
		math::Vec3 rotation = objects[i].rotation;

		if (i == 0) {
			position = position + animation.trajectoryOffset();
			rotation = rotation + animation.spinOffset();
		}

		const math::Mat4 model = math::compose(position, rotation, objects[i].scale);

		ModelUniform uniform{};
		uniform.model = model;
		uniform.albedo[0] = objects[i].albedo.x;
		uniform.albedo[1] = objects[i].albedo.y;
		uniform.albedo[2] = objects[i].albedo.z;
		uniform.albedo[3] = objects[i].use_vertex_color ? 1.0f : 0.0f;
		uniform.specular_color[0] = objects[i].specular_color.x;
		uniform.specular_color[1] = objects[i].specular_color.y;
		uniform.specular_color[2] = objects[i].specular_color.z;
		uniform.specular_color[3] = objects[i].specular_strength;
		uniform.material[0] = objects[i].shininess;
		uniform.material[1] = 0.0f;
		uniform.material[2] = 0.0f;
		uniform.material[3] = 1.0f;

		std::memcpy(gpu_objects[i].uniform_mapped, &uniform, sizeof(uniform));
	}
}

void updateDebugGeometry() {
	std::vector<DebugVertex> triangle_vertices;
	std::vector<DebugVertex> line_vertices;

	if (lighting.show_light_sources) {
		for (const auto& light : lighting.point_lights) {
			if (light.enabled) {
				appendCube(triangle_vertices, light.position, 0.07f,
				           lightMarkerColor(light.color));
			}
		}

		for (const auto& light : lighting.spot_lights) {
			if (!light.enabled) {
				continue;
			}
			appendCube(triangle_vertices, light.position, 0.09f,
			           lightMarkerColor(light.color));
			appendSpotCone(line_vertices, light, light.outer_angle_degrees,
			               { 1.0f, 1.0f, 1.0f }, true);
			appendSpotCone(line_vertices, light, light.inner_angle_degrees,
			               { 0.45f, 0.45f, 0.45f }, false);
		}
	}

	triangle_vertices.resize(std::min<size_t>(triangle_vertices.size(),
	                                          max_debug_triangle_vertices));
	line_vertices.resize(std::min<size_t>(line_vertices.size(),
	                                      max_debug_line_vertices));

	debug_triangle_vertex_count = uint32_t(triangle_vertices.size());
	debug_line_vertex_count = uint32_t(line_vertices.size());

	if (!triangle_vertices.empty()) {
		std::memcpy(debug_triangle_mapped, triangle_vertices.data(),
		            triangle_vertices.size() * sizeof(DebugVertex));
	}
	if (!line_vertices.empty()) {
		std::memcpy(debug_line_mapped, line_vertices.data(),
		            line_vertices.size() * sizeof(DebugVertex));
	}
}

} // namespace

bool initialize() {
	resetScene();

	if (!createDescriptorLayoutsAndPool()) {
		return false;
	}
	if (!createBuffers()) {
		return false;
	}
	if (!allocateDescriptors()) {
		return false;
	}
	if (!createVertexBuffer()) {
		return false;
	}
	if (!createMainPipeline()) {
		return false;
	}
	if (!createDebugResources()) {
		return false;
	}
	return true;
}

void shutdown() {
	auto& context = graphics::internal::context;

	vkQueueWaitIdle(context.graphics_queue);

	if (graphics_pipeline != VK_NULL_HANDLE) {
		vkDestroyPipeline(context.device, graphics_pipeline, nullptr);
	}
	if (debug_triangle_pipeline != VK_NULL_HANDLE) {
		vkDestroyPipeline(context.device, debug_triangle_pipeline, nullptr);
	}
	if (debug_line_pipeline != VK_NULL_HANDLE) {
		vkDestroyPipeline(context.device, debug_line_pipeline, nullptr);
	}
	if (pipeline_layout != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
	}
	if (debug_pipeline_layout != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(context.device, debug_pipeline_layout, nullptr);
	}
	if (vertex_shader_module != VK_NULL_HANDLE) {
		vkDestroyShaderModule(context.device, vertex_shader_module, nullptr);
	}
	if (fragment_shader_module != VK_NULL_HANDLE) {
		vkDestroyShaderModule(context.device, fragment_shader_module, nullptr);
	}
	if (debug_vertex_shader_module != VK_NULL_HANDLE) {
		vkDestroyShaderModule(context.device, debug_vertex_shader_module, nullptr);
	}
	if (debug_fragment_shader_module != VK_NULL_HANDLE) {
		vkDestroyShaderModule(context.device, debug_fragment_shader_module, nullptr);
	}

	if (vertex_buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(context.allocator, vertex_buffer, vertex_buffer_allocation);
	}
	if (scene_uniform_buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(context.allocator, scene_uniform_buffer,
		                 scene_uniform_allocation);
	}
	if (point_light_buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(context.allocator, point_light_buffer,
		                 point_light_allocation);
	}
	if (spot_light_buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(context.allocator, spot_light_buffer,
		                 spot_light_allocation);
	}
	if (debug_triangle_buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(context.allocator, debug_triangle_buffer,
		                 debug_triangle_allocation);
	}
	if (debug_line_buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(context.allocator, debug_line_buffer,
		                 debug_line_allocation);
	}
	for (auto& object : gpu_objects) {
		if (object.uniform_buffer != VK_NULL_HANDLE) {
			vmaDestroyBuffer(context.allocator, object.uniform_buffer,
			                 object.uniform_allocation);
		}
	}

	if (descriptor_pool != VK_NULL_HANDLE) {
		vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
	}
	if (model_set_layout != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(context.device, model_set_layout, nullptr);
	}
	if (scene_set_layout != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(context.device, scene_set_layout, nullptr);
	}
}

void update(double time) {
	(void)time;
	updateAnimation();
	drawInterface();
}

void render(const graphics::internal::FrameData& fd) {
	auto& context = graphics::internal::context;

	const float aspect = context.swapchain_extent.height == 0
		? 1.0f
		: float(context.swapchain_extent.width) / float(context.swapchain_extent.height);

	const math::Vec3 eye = camera.position;
	const math::Vec3 forward = cameraForward();
	const math::Mat4 view = math::lookAt(eye, eye + forward, { 0.0f, 1.0f, 0.0f });
	const math::Mat4 projection_matrix = buildProjection(aspect);
	const math::Mat4 view_projection = projection_matrix * view;

	updateSceneUniform(view_projection, eye);
	updateModelUniforms();
	updateDebugGeometry();

	const VkCommandBufferBeginInfo begin = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	vkResetCommandBuffer(fd.command_buffer, 0);
	vkBeginCommandBuffer(fd.command_buffer, &begin);

	const VkClearValue clear_values[] = {
		{ .color = {{ 0.035f, 0.035f, 0.055f, 1.0f }} },
		{ .depthStencil = { 1.0f, 0 } },
	};

	const VkRenderPassBeginInfo render_pass_begin = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = context.render_pass,
		.framebuffer = fd.framebuffer,
		.renderArea = { .extent = context.swapchain_extent },
		.clearValueCount = uint32_t(sizeof(clear_values) / sizeof(clear_values[0])),
		.pClearValues = clear_values,
	};

	vkCmdBeginRenderPass(fd.command_buffer, &render_pass_begin, VK_SUBPASS_CONTENTS_INLINE);

	const VkViewport viewport = {
		.x = 0.0f,
		.y = float(context.swapchain_extent.height),
		.width = float(context.swapchain_extent.width),
		.height = -float(context.swapchain_extent.height),
		.minDepth = 0.0f,
		.maxDepth = 1.0f,
	};
	const VkRect2D scissor = {
		.offset = { 0, 0 },
		.extent = context.swapchain_extent,
	};
	vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
	vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

	vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline);
	vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
	                        pipeline_layout, 0, 1, &scene_descriptor_set, 0, nullptr);

	const VkDeviceSize vertex_buffer_offset = 0;
	vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vertex_buffer, &vertex_buffer_offset);

	for (int i = 0; i < max_objects; ++i) {
		if (!objects[i].visible) {
			continue;
		}
		vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
		                        pipeline_layout, 1, 1, &gpu_objects[i].descriptor_set,
		                        0, nullptr);
		vkCmdDraw(fd.command_buffer, vertex_count, 1, 0, 0);
	}

	if (debug_triangle_vertex_count > 0) {
		vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
		                  debug_triangle_pipeline);
		vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
		                        debug_pipeline_layout, 0, 1, &scene_descriptor_set,
		                        0, nullptr);
		vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &debug_triangle_buffer,
		                       &vertex_buffer_offset);
		vkCmdDraw(fd.command_buffer, debug_triangle_vertex_count, 1, 0, 0);
	}

	if (debug_line_vertex_count > 0) {
		vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
		                  debug_line_pipeline);
		vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
		                        debug_pipeline_layout, 0, 1, &scene_descriptor_set,
		                        0, nullptr);
		vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &debug_line_buffer,
		                       &vertex_buffer_offset);
		vkCmdDraw(fd.command_buffer, debug_line_vertex_count, 1, 0, 0);
	}

	vkCmdEndRenderPass(fd.command_buffer);
	vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application
