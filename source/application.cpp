#include "application.hpp"

#include "math.hpp"
#include "mesh.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <imgui.h>

namespace application {

namespace {

constexpr int max_objects = 3;

struct UniformBufferObject {
	alignas(16) math::Mat4 mvp;
	alignas(16) math::Mat4 model;
	alignas(16) float base_color[4];
	alignas(16) float light_direction[4];
	alignas(16) float camera_position[4];
};

static_assert(sizeof(UniformBufferObject) == 176,
              "Uniform buffer layout must match the GLSL std140 declaration");

struct GpuObject {
	VkBuffer uniform_buffer = VK_NULL_HANDLE;
	VmaAllocation uniform_allocation = VK_NULL_HANDLE;
	UniformBufferObject* uniform_mapped = nullptr;
	VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
};

struct ObjectSettings {
	bool visible = true;
	math::Vec3 position{};
	math::Vec3 rotation{};
	math::Vec3 scale{ 1.0f, 1.0f, 1.0f };
	math::Vec3 color{ 1.0f, 1.0f, 1.0f };
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
	float yaw = 0.65f;
	float pitch = 0.32f;
	float distance = 8.0f;
};

struct ProjectionSettings {
	bool perspective = true;
	float fov_degrees = 45.0f;
	float ortho_scale = 3.6f;
	float near_plane = 0.1f;
	float far_plane = 100.0f;
};

VkShaderModule vertex_shader_module = VK_NULL_HANDLE;
VkShaderModule fragment_shader_module = VK_NULL_HANDLE;
VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline graphics_pipeline = VK_NULL_HANDLE;

VkBuffer vertex_buffer = VK_NULL_HANDLE;
VmaAllocation vertex_buffer_allocation = VK_NULL_HANDLE;
uint32_t vertex_count = 0;

GpuObject gpu_objects[max_objects];
ObjectSettings objects[max_objects];

AnimationSettings animation;
CameraSettings camera;
ProjectionSettings projection;

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

bool createDescriptorResources() {
	const VkDescriptorSetLayoutBinding binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
	};

	const VkDescriptorSetLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &binding,
	};

	if (vkCreateDescriptorSetLayout(graphics::internal::context.device, &layout_info,
	                                nullptr, &descriptor_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor set layout\n";
		return false;
	}

	const VkDescriptorPoolSize pool_size = {
		.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = max_objects,
	};

	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = max_objects,
		.poolSizeCount = 1,
		.pPoolSizes = &pool_size,
	};

	if (vkCreateDescriptorPool(graphics::internal::context.device, &pool_info,
	                           nullptr, &descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor pool\n";
		return false;
	}

	VkDescriptorSetLayout layouts[max_objects] = {
		descriptor_set_layout,
		descriptor_set_layout,
		descriptor_set_layout,
	};

	const VkDescriptorSetAllocateInfo allocate_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptor_pool,
		.descriptorSetCount = max_objects,
		.pSetLayouts = layouts,
	};

	VkDescriptorSet descriptor_sets[max_objects]{};
	if (vkAllocateDescriptorSets(graphics::internal::context.device, &allocate_info,
	                             descriptor_sets) != VK_SUCCESS) {
		std::cerr << "Failed to allocate descriptor sets\n";
		return false;
	}

	for (int i = 0; i < max_objects; ++i) {
		if (!createBuffer(sizeof(UniformBufferObject), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		                  gpu_objects[i].uniform_buffer,
		                  gpu_objects[i].uniform_allocation,
		                  reinterpret_cast<void**>(&gpu_objects[i].uniform_mapped))) {
			std::cerr << "Failed to create uniform buffer #" << i << '\n';
			return false;
		}
		gpu_objects[i].descriptor_set = descriptor_sets[i];

		const VkDescriptorBufferInfo buffer_info = {
			.buffer = gpu_objects[i].uniform_buffer,
			.offset = 0,
			.range = sizeof(UniformBufferObject),
		};

		const VkWriteDescriptorSet write = {
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = gpu_objects[i].descriptor_set,
			.dstBinding = 0,
			.dstArrayElement = 0,
			.descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &buffer_info,
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

bool createPipeline() {
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
		.primitiveRestartEnable = VK_FALSE,
	};

	const VkViewport viewport = {
		.x = 0.0f,
		.y = 0.0f,
		.width = 1.0f,
		.height = 1.0f,
		.minDepth = 0.0f,
		.maxDepth = 1.0f,
	};

	const VkRect2D scissor = {
		.offset = { 0, 0 },
		.extent = { 1, 1 },
	};

	const VkPipelineViewportStateCreateInfo viewport_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.pViewports = &viewport,
		.scissorCount = 1,
		.pScissors = &scissor,
	};

	const VkPipelineRasterizationStateCreateInfo rasterization = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.depthClampEnable = VK_FALSE,
		.rasterizerDiscardEnable = VK_FALSE,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_BACK_BIT,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.depthBiasEnable = VK_FALSE,
		.lineWidth = 1.0f,
	};

	const VkPipelineMultisampleStateCreateInfo multisampling = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
		.sampleShadingEnable = VK_FALSE,
	};

	const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_LESS,
		.depthBoundsTestEnable = VK_FALSE,
		.stencilTestEnable = VK_FALSE,
	};

	const VkPipelineColorBlendAttachmentState color_attachment = {
		.blendEnable = VK_FALSE,
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
		                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
	};

	const VkPipelineColorBlendStateCreateInfo color_blending = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.logicOpEnable = VK_FALSE,
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

	const VkPipelineLayoutCreateInfo pipeline_layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &descriptor_set_layout,
	};

	if (vkCreatePipelineLayout(graphics::internal::context.device, &pipeline_layout_info,
	                           nullptr, &pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create pipeline layout\n";
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
		std::cerr << "Failed to create graphics pipeline\n";
		return false;
	}

	return true;
}

void resetScene() {
	objects[0] = {
		.visible = true,
		.position = { 0.0f, 0.0f, 0.0f },
		.rotation = { 0.0f, 0.0f, 0.0f },
		.scale = { 1.0f, 1.0f, 1.0f },
		.color = { 1.0f, 1.0f, 1.0f },
	};
	objects[1] = {
		.visible = true,
		.position = { -2.7f, 1.2f, -0.4f },
		.rotation = { 25.0f, 40.0f, 0.0f },
		.scale = { 0.45f, 0.45f, 0.45f },
		.color = { 1.0f, 0.55f, 0.25f },
	};
	objects[2] = {
		.visible = true,
		.position = { 2.7f, -1.0f, 0.7f },
		.rotation = { -20.0f, -35.0f, 15.0f },
		.scale = { 0.35f, 0.35f, 0.35f },
		.color = { 0.25f, 0.65f, 1.0f },
	};

	animation = {};
	camera = {};
	projection = {};
}

void updateAnimation() {
	if (animation.playing) {
		animation.time += ImGui::GetIO().DeltaTime * animation.speed;
	}
}

void drawInterface() {
	ImGui::Begin("Laboratory 1 - Dodecahedron", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

	ImGui::TextUnformatted("Variant 10: regular dodecahedron");
	ImGui::Text("Triangles: %u", vertex_count / 3);
	ImGui::Text("VkDescriptorSets used: %d", max_objects);

	ImGui::SeparatorText("Projection");
	int projection_mode = projection.perspective ? 0 : 1;
	const char* projection_names[] = { "Perspective", "Orthographic" };
	if (ImGui::Combo("Mode", &projection_mode, projection_names, 2)) {
		projection.perspective = projection_mode == 0;
	}
	if (projection.perspective) {
		ImGui::SliderFloat("Field of view", &projection.fov_degrees, 20.0f, 100.0f, "%.1f deg");
	} else {
		ImGui::SliderFloat("Ortho half-height", &projection.ortho_scale, 1.0f, 10.0f, "%.2f");
	}
	ImGui::SliderFloat("Near plane", &projection.near_plane, 0.01f, 5.0f, "%.2f");
	ImGui::SliderFloat("Far plane", &projection.far_plane, 10.0f, 200.0f, "%.1f");

	ImGui::SeparatorText("Camera");
	ImGui::SliderFloat("Yaw", &camera.yaw, -3.14f, 3.14f, "%.2f rad");
	ImGui::SliderFloat("Pitch", &camera.pitch, -1.5f, 1.5f, "%.2f rad");
	ImGui::SliderFloat("Distance", &camera.distance, 2.0f, 25.0f, "%.2f");

	ImGui::SeparatorText("Main object transform");
	ImGui::DragFloat3("Position", &objects[0].position.x, 0.02f, -10.0f, 10.0f, "%.2f");
	ImGui::DragFloat3("Rotation", &objects[0].rotation.x, 0.5f, -720.0f, 720.0f, "%.1f deg");
	ImGui::DragFloat3("Stretch / scale", &objects[0].scale.x, 0.01f, 0.05f, 5.0f, "%.2f");

	ImGui::SeparatorText("Color");
	ImGui::ColorEdit3("Base color (multiplies vertex colors)", &objects[0].color.x);

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

	ImGui::SeparatorText("Additional objects (separate VkDescriptorSet)");
	for (int i = 1; i < max_objects; ++i) {
		ImGui::PushID(i);
		ImGui::Text("Object #%d", i + 1);
		ImGui::Checkbox("Visible", &objects[i].visible);
		ImGui::SameLine();
		ImGui::ColorEdit3("Color", &objects[i].color.x, ImGuiColorEditFlags_NoInputs);
		ImGui::DragFloat3("Position", &objects[i].position.x, 0.02f, -10.0f, 10.0f, "%.2f");
		ImGui::DragFloat3("Scale", &objects[i].scale.x, 0.01f, 0.05f, 5.0f, "%.2f");
		ImGui::PopID();
	}

	ImGui::Separator();
	if (ImGui::Button("Reset scene")) {
		resetScene();
	}

	ImGui::End();
}

math::Mat4 buildProjection(float aspect) {
	if (projection.perspective) {
		return math::perspective(math::radians(projection.fov_degrees), aspect,
		                         projection.near_plane, projection.far_plane);
	}

	const float half_height = projection.ortho_scale;
	const float half_width = half_height * aspect;
	return math::orthographic(-half_width, half_width,
	                          -half_height, half_height,
	                          projection.near_plane, projection.far_plane);
}

math::Vec3 cameraPosition() {
	const float cos_pitch = std::cos(camera.pitch);
	return {
		camera.distance * cos_pitch * std::sin(camera.yaw),
		camera.distance * std::sin(camera.pitch),
		camera.distance * cos_pitch * std::cos(camera.yaw),
	};
}

void updateUniformBuffers(const math::Mat4& view, const math::Mat4& projection_matrix,
                          const math::Vec3& eye) {
	for (int i = 0; i < max_objects; ++i) {
		math::Vec3 position = objects[i].position;
		math::Vec3 rotation = objects[i].rotation;

		if (i == 0) {
			position = position + animation.trajectoryOffset();
			rotation = rotation + animation.spinOffset();
		}

		const math::Mat4 model = math::compose(position, rotation, objects[i].scale);

		UniformBufferObject uniform{};
		uniform.mvp = projection_matrix * view * model;
		uniform.model = model;
		uniform.base_color[0] = objects[i].color.x;
		uniform.base_color[1] = objects[i].color.y;
		uniform.base_color[2] = objects[i].color.z;
		uniform.base_color[3] = 1.0f;
		uniform.light_direction[0] = -0.35f;
		uniform.light_direction[1] = -0.80f;
		uniform.light_direction[2] = -0.45f;
		uniform.light_direction[3] = 0.0f;
		uniform.camera_position[0] = eye.x;
		uniform.camera_position[1] = eye.y;
		uniform.camera_position[2] = eye.z;
		uniform.camera_position[3] = 0.35f;

		std::memcpy(gpu_objects[i].uniform_mapped, &uniform, sizeof(uniform));
	}
}

} // namespace

bool initialize() {
	resetScene();

	if (!createDescriptorResources()) {
		return false;
	}
	if (!createVertexBuffer()) {
		return false;
	}
	if (!createPipeline()) {
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
	if (pipeline_layout != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
	}
	if (vertex_shader_module != VK_NULL_HANDLE) {
		vkDestroyShaderModule(context.device, vertex_shader_module, nullptr);
	}
	if (fragment_shader_module != VK_NULL_HANDLE) {
		vkDestroyShaderModule(context.device, fragment_shader_module, nullptr);
	}
	if (vertex_buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(context.allocator, vertex_buffer, vertex_buffer_allocation);
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
	if (descriptor_set_layout != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(context.device, descriptor_set_layout, nullptr);
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

	const math::Vec3 eye = cameraPosition();
	const math::Mat4 view = math::lookAt(eye, { 0.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f });
	const math::Mat4 projection_matrix = buildProjection(aspect);

	updateUniformBuffers(view, projection_matrix, eye);

	const VkCommandBufferBeginInfo begin = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	vkResetCommandBuffer(fd.command_buffer, 0);
	vkBeginCommandBuffer(fd.command_buffer, &begin);

	const VkClearValue clear_values[] = {
		{ .color = {{ 0.05f, 0.05f, 0.08f, 1.0f }} },
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

	vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline);

	const VkViewport viewport = {
		.x = 0.0f,
		.y = 0.0f,
		.width = float(context.swapchain_extent.width),
		.height = float(context.swapchain_extent.height),
		.minDepth = 0.0f,
		.maxDepth = 1.0f,
	};

	const VkRect2D scissor = {
		.offset = { 0, 0 },
		.extent = context.swapchain_extent,
	};

	vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
	vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);

	const VkDeviceSize vertex_buffer_offset = 0;
	vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vertex_buffer, &vertex_buffer_offset);

	for (int i = 0; i < max_objects; ++i) {
		if (!objects[i].visible) {
			continue;
		}

		vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
		                        pipeline_layout, 0, 1,
		                        &gpu_objects[i].descriptor_set, 0, nullptr);
		vkCmdDraw(fd.command_buffer, vertex_count, 1, 0, 0);
	}

	vkCmdEndRenderPass(fd.command_buffer);
	vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application
