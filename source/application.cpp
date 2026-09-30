#include "application.hpp"

#include <imgui.h>

namespace application {

bool initialize() {
	return true;
}

void shutdown() {
	auto& context = graphics::internal::context;
	vkQueueWaitIdle(context.graphics_queue);
}

void update([[maybe_unused]] double time) {
	ImGui::ShowDemoWindow();
}

void render(const graphics::internal::FrameData& fd) {
	auto& context = graphics::internal::context;

	const VkCommandBufferBeginInfo begin = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	vkResetCommandBuffer(fd.command_buffer, 0);
	vkBeginCommandBuffer(fd.command_buffer, &begin);

	// Основной render pass очищает и цвет, и глубину (loadOp = CLEAR), значит нужны 2 clear value
	const VkClearValue clear_values[] = {
		{ .color = {{ 0.05f, 0.05f, 0.08f, 1.0f }} },
		{ .depthStencil = { 1.0f, 0 } },
	};

	const VkRenderPassBeginInfo rp = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = context.render_pass,
		.framebuffer = fd.framebuffer,
		.renderArea = { .extent = context.swapchain_extent },
		.clearValueCount = 2,
		.pClearValues = clear_values,
	};

	vkCmdBeginRenderPass(fd.command_buffer, &rp, VK_SUBPASS_CONTENTS_INLINE);
	// сюда позже добавите свои vkCmdBindPipeline / vkCmdDraw
	vkCmdEndRenderPass(fd.command_buffer);

	vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application