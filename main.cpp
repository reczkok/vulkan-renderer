#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs           = std::filesystem;
using QueueFamilyIndex = uint32_t;

//---------------------------------------------------------------------
// Structures for queue families and swap chain support
//---------------------------------------------------------------------
struct QueueFamilyIndices
{
	std::optional<QueueFamilyIndex> graphicsFamily;
	std::optional<QueueFamilyIndex> presentFamily;

	[[nodiscard]] bool isComplete() const
	{
		return graphicsFamily.has_value() && presentFamily.has_value();
	}
};

struct SwapChainSupportDetails
{
	VkSurfaceCapabilitiesKHR        capabilities;
	std::vector<VkSurfaceFormatKHR> formats;
	std::vector<VkPresentModeKHR>   presentModes;
};

// Constants
constexpr uint32_t WIDTH  = 1920;
constexpr uint32_t HEIGHT = 1080;

#ifdef NDEBUG
static inline constexpr bool enableValidationLayers = false;
#else
static inline constexpr bool enableValidationLayers = true;
#endif

static inline const std::vector<const char *> validationLayers = {
    "VK_LAYER_KHRONOS_validation",
};

#ifdef __APPLE__
static inline const std::vector<const char *> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    "VK_KHR_portability_subset",
};
#else
static inline const std::vector<const char *> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
};
#endif

class VulkanInstance
{
  public:
	VulkanInstance(const VkInstanceCreateInfo &createInfo)
	{
		if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS)
			throw std::runtime_error("Failed to create Vulkan instance.");
	}
	~VulkanInstance()
	{
		if (instance != VK_NULL_HANDLE)
			vkDestroyInstance(instance, nullptr);
	}
	VulkanInstance(const VulkanInstance &)            = delete;
	VulkanInstance &operator=(const VulkanInstance &) = delete;
	VulkanInstance(VulkanInstance &&other) noexcept :
	    instance(other.instance)
	{
		other.instance = VK_NULL_HANDLE;
	}
	VulkanInstance &operator=(VulkanInstance &&other) noexcept
	{
		if (this != &other)
		{
			if (instance != VK_NULL_HANDLE)
				vkDestroyInstance(instance, nullptr);
			instance       = other.instance;
			other.instance = VK_NULL_HANDLE;
		}
		return *this;
	}
	// Allow implicit conversion to VkInstance
	operator VkInstance() const
	{
		return instance;
	}

  private:
	VkInstance instance = VK_NULL_HANDLE;
};

VkResult CreateDebugUtilsMessengerEXT(VkInstance                                instance,
                                      const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
                                      const VkAllocationCallbacks              *pAllocator,
                                      VkDebugUtilsMessengerEXT                 *pDebugMessenger)
{
	auto func = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
	    vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
	if (func != nullptr)
	{
		return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
	}
	return VK_ERROR_EXTENSION_NOT_PRESENT;
}

void DestroyDebugUtilsMessengerEXT(VkInstance                   instance,
                                   VkDebugUtilsMessengerEXT     debugMessenger,
                                   const VkAllocationCallbacks *pAllocator)
{
	auto func = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
	    vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
	if (func != nullptr)
	{
		func(instance, debugMessenger, pAllocator);
	}
}

class DebugMessenger
{
  public:
	DebugMessenger(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT &createInfo) :
	    instance(instance)
	{
		if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &messenger) != VK_SUCCESS)
			throw std::runtime_error("Failed to set up debug messenger!");
	}
	~DebugMessenger()
	{
		if (messenger != VK_NULL_HANDLE)
			DestroyDebugUtilsMessengerEXT(instance, messenger, nullptr);
	}
	DebugMessenger(const DebugMessenger &)            = delete;
	DebugMessenger &operator=(const DebugMessenger &) = delete;
	DebugMessenger(DebugMessenger &&other) noexcept :
	    instance(other.instance), messenger(other.messenger)
	{
		other.messenger = VK_NULL_HANDLE;
	}
	DebugMessenger &operator=(DebugMessenger &&other) noexcept
	{
		if (this != &other)
		{
			if (messenger != VK_NULL_HANDLE)
				DestroyDebugUtilsMessengerEXT(instance, messenger, nullptr);
			instance        = other.instance;
			messenger       = other.messenger;
			other.messenger = VK_NULL_HANDLE;
		}
		return *this;
	}

  private:
	VkInstance               instance;
	VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
};

class VulkanDevice
{
  public:
	VulkanDevice(VkPhysicalDevice                            physicalDevice,
	             const std::vector<const char *>            &requiredExtensions,
	             const std::vector<VkDeviceQueueCreateInfo> &queueCreateInfos,
	             const VkPhysicalDeviceFeatures             &deviceFeatures) :
	    physicalDevice(physicalDevice)
	{
		VkDeviceCreateInfo createInfo{};
		createInfo.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		createInfo.queueCreateInfoCount    = static_cast<uint32_t>(queueCreateInfos.size());
		createInfo.pQueueCreateInfos       = queueCreateInfos.data();
		createInfo.pEnabledFeatures        = &deviceFeatures;
		createInfo.enabledExtensionCount   = static_cast<uint32_t>(requiredExtensions.size());
		createInfo.ppEnabledExtensionNames = requiredExtensions.data();
		if (enableValidationLayers)
		{
			createInfo.enabledLayerCount   = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();
		}
		else
		{
			createInfo.enabledLayerCount = 0;
		}

		if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS)
			throw std::runtime_error("Failed to create logical device!");
	}
	~VulkanDevice()
	{
		if (device != VK_NULL_HANDLE)
			vkDestroyDevice(device, nullptr);
	}
	VulkanDevice(const VulkanDevice &)            = delete;
	VulkanDevice &operator=(const VulkanDevice &) = delete;
	VulkanDevice(VulkanDevice &&other) noexcept :
	    device(other.device)
	{
		other.device = VK_NULL_HANDLE;
	}
	VulkanDevice &operator=(VulkanDevice &&other) noexcept
	{
		if (this != &other)
		{
			if (device != VK_NULL_HANDLE)
				vkDestroyDevice(device, nullptr);
			device       = other.device;
			other.device = VK_NULL_HANDLE;
		}
		return *this;
	}
	// Implicit conversion to VkDevice
	operator VkDevice() const
	{
		return device;
	}

  private:
	VkPhysicalDevice physicalDevice;
	VkDevice         device = VK_NULL_HANDLE;
};

class HelloTriangleApplication
{
  public:
	void run()
	{
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}

  private:
	GLFWwindow *window = nullptr;

	std::unique_ptr<VulkanInstance> vkInstance;
	std::optional<DebugMessenger>   debugMessenger;
	std::unique_ptr<VulkanDevice>   device;

	VkPhysicalDevice           physicalDevice = VK_NULL_HANDLE;
	VkQueue                    graphicsQueue;
	VkQueue                    presentQueue;
	VkSurfaceKHR               surface   = VK_NULL_HANDLE;
	VkSwapchainKHR             swapChain = VK_NULL_HANDLE;
	std::vector<VkImage>       swapChainImages;
	std::vector<VkImageView>   swapChainImageViews;
	VkFormat                   swapChainImageFormat;
	VkExtent2D                 swapChainExtent;
	VkPipelineLayout           pipelineLayout   = VK_NULL_HANDLE;
	VkRenderPass               renderPass       = VK_NULL_HANDLE;
	VkPipeline                 graphicsPipeline = VK_NULL_HANDLE;
	std::vector<VkFramebuffer> swapChainFramebuffers;
	VkCommandPool              commandPool             = VK_NULL_HANDLE;
	VkCommandBuffer            commandBuffer           = VK_NULL_HANDLE;
	VkSemaphore                imageAvailableSemaphore = VK_NULL_HANDLE;
	VkSemaphore                renderFinishedSemaphore = VK_NULL_HANDLE;
	VkFence                    inFlightFence           = VK_NULL_HANDLE;

	static std::vector<char> readFile(const std::string &filename)
	{
		std::cout << "Current working directory: " << fs::current_path() << "\n";
		fs::path              filePath{filename};
		std::vector<fs::path> searchPaths{
		    filePath,                                                                 // As provided
		    fs::current_path() / filePath,                                            // Current directory
		    fs::current_path().parent_path() / filePath,                              // Parent directory
		    fs::current_path().parent_path().parent_path() / filePath,                // Grandparent directory
		    fs::current_path().parent_path() / "shaders" / filePath.filename()        // Common shader location
		};
		std::ifstream file;
		std::string   attemptedPaths;
		for (const auto &path : searchPaths)
		{
			attemptedPaths += path.string() + "\n";
			file.open(path, std::ios::ate | std::ios::binary);
			if (file.is_open())
			{
				std::cout << "Successfully opened shader file: " << path << "\n";
				break;
			}
		}
		if (!file.is_open())
			throw std::runtime_error("Failed to open file: " + filename + "\nAttempted paths:\n" + attemptedPaths);

		size_t            fileSize = static_cast<size_t>(file.tellg());
		std::vector<char> buffer(fileSize);
		file.seekg(0);
		file.read(buffer.data(), fileSize);
		return buffer;
	}

	void initWindow()
	{
		if (!glfwInit())
			throw std::runtime_error("Failed to initialize GLFW.");
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
		if (!window)
			throw std::runtime_error("Failed to create GLFW window.");
	}

	void initVulkan()
	{
		createInstance();
		setupDebugMessenger();
		createSurface();
		pickPhysicalDevice();
		createLogicalDevice();
		createSwapChain();
		createImageViews();
		createRenderPass();
		createGraphicsPipeline();
		createFramebuffers();
		createCommandPool();
		createCommandBuffer();
		createSyncObjects();
	}

	void createInstance()
	{
		if (enableValidationLayers && !checkValidationLayerSupport())
			throw std::runtime_error("Validation layers requested, but not available!");

		VkApplicationInfo appInfo{};
		appInfo.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName   = "Hello Triangle";
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.pEngineName        = "No Engine";
		appInfo.engineVersion      = VK_MAKE_VERSION(1, 0, 0);
		appInfo.apiVersion         = VK_API_VERSION_1_0;

		auto extensions = getRequiredExtensions();

		VkInstanceCreateInfo createInfo{};
		createInfo.sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		createInfo.pApplicationInfo = &appInfo;
#ifdef __APPLE__
		createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif
		createInfo.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
		createInfo.ppEnabledExtensionNames = extensions.data();

		VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
		if (enableValidationLayers)
		{
			createInfo.enabledLayerCount   = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();
			populateDebugMessengerCreateInfo(debugCreateInfo);
			createInfo.pNext = &debugCreateInfo;
		}
		else
		{
			createInfo.enabledLayerCount = 0;
			createInfo.pNext             = nullptr;
		}

		vkInstance = std::make_unique<VulkanInstance>(createInfo);
	}

	// Get required extensions from GLFW plus debug extension if needed.
	std::vector<const char *> getRequiredExtensions()
	{
		uint32_t                  glfwExtensionCount = 0;
		const char              **glfwExtensions     = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
		std::vector<const char *> extensions(glfwExtensions,
		                                     glfwExtensions + glfwExtensionCount);
#ifdef __APPLE__
		extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
		extensions.push_back("VK_KHR_get_physical_device_properties2");
#endif
		if (enableValidationLayers)
			extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		return extensions;
	}

	// Populate debug messenger create info.
	void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT &createInfo)
	{
		createInfo                 = {};
		createInfo.sType           = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
		                             VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
		                             VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
		                         VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
		                         VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		createInfo.pfnUserCallback = debugCallback;
	}

	// Debug callback (prints messages to std::cerr).
	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
	    VkDebugUtilsMessageSeverityFlagBitsEXT      messageSeverity,
	    VkDebugUtilsMessageTypeFlagsEXT             messageType,
	    const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
	    void                                       *pUserData)
	{
		std::cerr << "Validation layer: " << pCallbackData->pMessage << "\n";
		return VK_FALSE;
	}

	// Setup debug messenger using RAII.
	void setupDebugMessenger()
	{
		if (!enableValidationLayers)
			return;
		VkDebugUtilsMessengerCreateInfoEXT createInfo;
		populateDebugMessengerCreateInfo(createInfo);
		debugMessenger.emplace(static_cast<VkInstance>(*vkInstance), createInfo);
	}

	// Create window surface.
	void createSurface()
	{
		if (glfwCreateWindowSurface(static_cast<VkInstance>(*vkInstance), window, nullptr, &surface) != VK_SUCCESS)
			throw std::runtime_error("Failed to create window surface!");
	}

	// Check device extension support.
	bool checkDeviceExtensionSupport(VkPhysicalDevice device) const
	{
		uint32_t extensionCount;
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
		std::vector<VkExtensionProperties> availableExtensions(extensionCount);
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());
		std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());
		for (const auto &extension : availableExtensions)
			requiredExtensions.erase(extension.extensionName);
		return requiredExtensions.empty();
	}

	// Query swap chain support for a device.
	SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device)
	{
		SwapChainSupportDetails details;
		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);
		uint32_t formatCount;
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
		if (formatCount != 0)
		{
			details.formats.resize(formatCount);
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
		}
		uint32_t presentModeCount;
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
		if (presentModeCount != 0)
		{
			details.presentModes.resize(presentModeCount);
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
		}
		return details;
	}

	// Find required queue families.
	QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) const
	{
		QueueFamilyIndices indices;
		uint32_t           queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

		for (QueueFamilyIndex i = 0; i < static_cast<QueueFamilyIndex>(queueFamilies.size()); ++i)
		{
			VkBool32 presentSupport = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
			if (presentSupport)
				indices.presentFamily = i;
			if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
				indices.graphicsFamily = i;
			if (indices.isComplete())
				break;
		}
		return indices;
	}

	// Check if a physical device is suitable.
	bool isDeviceSuitable(VkPhysicalDevice device)
	{
		bool extensionsSupported = checkDeviceExtensionSupport(device);
		bool swapChainAdequate   = false;
		if (extensionsSupported)
		{
			SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
			swapChainAdequate                        = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
		}
		return findQueueFamilies(device).isComplete() && extensionsSupported && swapChainAdequate;
	}

	// Pick a physical device.
	void pickPhysicalDevice()
	{
		uint32_t deviceCount = 0;
		vkEnumeratePhysicalDevices(static_cast<VkInstance>(*vkInstance), &deviceCount, nullptr);
		if (deviceCount == 0)
			throw std::runtime_error("Failed to find GPUs with Vulkan support!");

		std::vector<VkPhysicalDevice> devices(deviceCount);
		vkEnumeratePhysicalDevices(static_cast<VkInstance>(*vkInstance), &deviceCount, devices.data());

		std::cout << "Available Vulkan devices:\n";
		std::map<int, std::pair<std::string, VkPhysicalDevice>> deviceMap;
		for (int i = 0; i < static_cast<int>(devices.size()); i++)
		{
			VkPhysicalDeviceProperties properties;
			vkGetPhysicalDeviceProperties(devices[i], &properties);
			deviceMap[i] = {properties.deviceName, devices[i]};
			std::cout << "  [" << i << "] " << properties.deviceName << "\n";
		}
		std::cout << "Select device by entering its number: ";
		int selectedId;
		if (!(std::cin >> selectedId) || selectedId < 0 || selectedId >= static_cast<int>(devices.size()))
			throw std::runtime_error("Invalid device selection!");
		physicalDevice = deviceMap[selectedId].second;
		std::cout << "Selected device: " << deviceMap[selectedId].first << "\n";
		if (!isDeviceSuitable(physicalDevice))
			throw std::runtime_error("Selected device is not suitable!");
	}

	// Create logical device.
	void createLogicalDevice()
	{
		QueueFamilyIndices                   indices             = findQueueFamilies(physicalDevice);
		std::set<uint32_t>                   uniqueQueueFamilies = {indices.graphicsFamily.value(), indices.presentFamily.value()};
		std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
		float                                queuePriority = 1.0f;
		for (uint32_t queueFamily : uniqueQueueFamilies)
		{
			VkDeviceQueueCreateInfo queueCreateInfo{};
			queueCreateInfo.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queueCreateInfo.queueFamilyIndex = queueFamily;
			queueCreateInfo.queueCount       = 1;
			queueCreateInfo.pQueuePriorities = &queuePriority;
			queueCreateInfos.push_back(queueCreateInfo);
		}

		VkPhysicalDeviceFeatures deviceFeatures{};
		device = std::make_unique<VulkanDevice>(physicalDevice, deviceExtensions, queueCreateInfos, deviceFeatures);

		vkGetDeviceQueue(static_cast<VkDevice>(*device), indices.graphicsFamily.value(), 0, &graphicsQueue);
		vkGetDeviceQueue(static_cast<VkDevice>(*device), indices.presentFamily.value(), 0, &presentQueue);
	}

	// Choose the best surface format.
	VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &availableFormats)
	{
		for (const auto &availableFormat : availableFormats)
		{
			if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB &&
			    availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
				return availableFormat;
		}
		return availableFormats[0];
	}

	// Choose the best present mode.
	VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR> &availablePresentModes)
	{
		for (const auto &availablePresentMode : availablePresentModes)
		{
			if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
				return availablePresentMode;
		}
		return VK_PRESENT_MODE_FIFO_KHR;
	}

	// Choose the swap extent.
	VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities)
	{
		if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
			return capabilities.currentExtent;
		else
		{
			int width, height;
			glfwGetFramebufferSize(window, &width, &height);
			VkExtent2D actualExtent = {
			    static_cast<uint32_t>(width),
			    static_cast<uint32_t>(height)};
			actualExtent.width  = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
			actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
			return actualExtent;
		}
	}

	// Create swap chain.
	void createSwapChain()
	{
		SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);
		VkSurfaceFormatKHR      surfaceFormat    = chooseSwapSurfaceFormat(swapChainSupport.formats);
		VkPresentModeKHR        presentMode      = chooseSwapPresentMode(swapChainSupport.presentModes);
		VkExtent2D              extent           = chooseSwapExtent(swapChainSupport.capabilities);
		swapChainImageFormat                     = surfaceFormat.format;
		swapChainExtent                          = extent;
		uint32_t imageCount                      = swapChainSupport.capabilities.minImageCount + 1;
		if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
			imageCount = swapChainSupport.capabilities.maxImageCount;

		VkSwapchainCreateInfoKHR createInfo{};
		createInfo.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		createInfo.surface          = surface;
		createInfo.minImageCount    = imageCount;
		createInfo.imageFormat      = surfaceFormat.format;
		createInfo.imageColorSpace  = surfaceFormat.colorSpace;
		createInfo.imageExtent      = extent;
		createInfo.imageArrayLayers = 1;
		createInfo.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

		QueueFamilyIndices indices              = findQueueFamilies(physicalDevice);
		uint32_t           queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};
		if (indices.graphicsFamily != indices.presentFamily)
		{
			createInfo.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
			createInfo.queueFamilyIndexCount = 2;
			createInfo.pQueueFamilyIndices   = queueFamilyIndices;
		}
		else
		{
			createInfo.imageSharingMode      = VK_SHARING_MODE_EXCLUSIVE;
			createInfo.queueFamilyIndexCount = 0;
			createInfo.pQueueFamilyIndices   = nullptr;
		}
		createInfo.preTransform   = swapChainSupport.capabilities.currentTransform;
		createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		createInfo.presentMode    = presentMode;
		createInfo.clipped        = VK_TRUE;
		createInfo.oldSwapchain   = VK_NULL_HANDLE;
		if (vkCreateSwapchainKHR(static_cast<VkDevice>(*device), &createInfo, nullptr, &swapChain) != VK_SUCCESS)
			throw std::runtime_error("Failed to create swap chain!");

		vkGetSwapchainImagesKHR(static_cast<VkDevice>(*device), swapChain, &imageCount, nullptr);
		swapChainImages.resize(imageCount);
		vkGetSwapchainImagesKHR(static_cast<VkDevice>(*device), swapChain, &imageCount, swapChainImages.data());
	}

	// Create image views for each swap chain image.
	void createImageViews()
	{
		swapChainImageViews.resize(swapChainImages.size());
		for (size_t i = 0; i < swapChainImages.size(); i++)
		{
			VkImageViewCreateInfo createInfo{};
			createInfo.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			createInfo.image                           = swapChainImages[i];
			createInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
			createInfo.format                          = swapChainImageFormat;
			createInfo.components.r                    = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.g                    = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.b                    = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.a                    = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
			createInfo.subresourceRange.baseMipLevel   = 0;
			createInfo.subresourceRange.levelCount     = 1;
			createInfo.subresourceRange.baseArrayLayer = 0;
			createInfo.subresourceRange.layerCount     = 1;
			if (vkCreateImageView(static_cast<VkDevice>(*device), &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS)
				throw std::runtime_error("Failed to create image views!");
		}
	}

	// Create render pass.
	void createRenderPass()
	{
		VkAttachmentDescription colorAttachment{};
		colorAttachment.format         = swapChainImageFormat;
		colorAttachment.samples        = VK_SAMPLE_COUNT_1_BIT;
		colorAttachment.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAttachment.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
		colorAttachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		colorAttachment.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
		colorAttachment.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentReference colorAttachmentRef{};
		colorAttachmentRef.attachment = 0;
		colorAttachmentRef.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpass{};
		subpass.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments    = &colorAttachmentRef;

		VkSubpassDependency dependency{};
		dependency.srcSubpass    = VK_SUBPASS_EXTERNAL;
		dependency.dstSubpass    = 0;
		dependency.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.srcAccessMask = 0;
		dependency.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

		VkRenderPassCreateInfo renderPassInfo{};
		renderPassInfo.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = 1;
		renderPassInfo.pAttachments    = &colorAttachment;
		renderPassInfo.subpassCount    = 1;
		renderPassInfo.pSubpasses      = &subpass;
		renderPassInfo.dependencyCount = 1;
		renderPassInfo.pDependencies   = &dependency;

		if (vkCreateRenderPass(static_cast<VkDevice>(*device), &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS)
			throw std::runtime_error("Failed to create render pass!");
	}

	// Create graphics pipeline.
	void createGraphicsPipeline()
	{
		auto vertShaderCode = readFile("shaders/vert.spv");
		auto fragShaderCode = readFile("shaders/frag.spv");

		VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
		VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

		VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
		vertShaderStageInfo.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage  = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderStageInfo.module = vertShaderModule;
		vertShaderStageInfo.pName  = "main";

		VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
		fragShaderStageInfo.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderStageInfo.stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageInfo.module = fragShaderModule;
		fragShaderStageInfo.pName  = "main";

		VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

		VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
		vertexInputInfo.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertexInputInfo.vertexBindingDescriptionCount   = 0;
		vertexInputInfo.pVertexBindingDescriptions      = nullptr;
		vertexInputInfo.vertexAttributeDescriptionCount = 0;
		vertexInputInfo.pVertexAttributeDescriptions    = nullptr;

		VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
		inputAssembly.sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology               = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		inputAssembly.primitiveRestartEnable = VK_FALSE;

		VkViewport viewport{};
		viewport.x        = 0.0f;
		viewport.y        = 0.0f;
		viewport.width    = static_cast<float>(swapChainExtent.width);
		viewport.height   = static_cast<float>(swapChainExtent.height);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		VkRect2D scissor{};
		scissor.offset = {0, 0};
		scissor.extent = swapChainExtent;

		VkPipelineViewportStateCreateInfo viewportState{};
		viewportState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.pViewports    = &viewport;
		viewportState.scissorCount  = 1;
		viewportState.pScissors     = &scissor;

		VkPipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable        = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.polygonMode             = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth               = 1.0f;
		rasterizer.cullMode                = VK_CULL_MODE_BACK_BIT;
		rasterizer.frontFace               = VK_FRONT_FACE_CLOCKWISE;
		rasterizer.depthBiasEnable         = VK_FALSE;

		VkPipelineMultisampleStateCreateInfo multisampling{};
		multisampling.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable  = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		VkPipelineColorBlendAttachmentState colorBlendAttachment{};
		colorBlendAttachment.colorWriteMask =
		    VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
		    VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_FALSE;

		VkPipelineColorBlendStateCreateInfo colorBlending{};
		colorBlending.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlending.logicOpEnable   = VK_FALSE;
		colorBlending.attachmentCount = 1;
		colorBlending.pAttachments    = &colorBlendAttachment;

		VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount         = 0;
		pipelineLayoutInfo.pushConstantRangeCount = 0;

		if (vkCreatePipelineLayout(static_cast<VkDevice>(*device), &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
			throw std::runtime_error("Failed to create pipeline layout!");

		VkGraphicsPipelineCreateInfo pipelineInfo{};
		pipelineInfo.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.stageCount          = 2;
		pipelineInfo.pStages             = shaderStages;
		pipelineInfo.pVertexInputState   = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssembly;
		pipelineInfo.pViewportState      = &viewportState;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState   = &multisampling;
		pipelineInfo.pDepthStencilState  = nullptr;
		pipelineInfo.pColorBlendState    = &colorBlending;
		pipelineInfo.layout              = pipelineLayout;
		pipelineInfo.renderPass          = renderPass;
		pipelineInfo.subpass             = 0;
		pipelineInfo.basePipelineHandle  = VK_NULL_HANDLE;

		if (vkCreateGraphicsPipelines(static_cast<VkDevice>(*device), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS)
			throw std::runtime_error("Failed to create graphics pipeline!");

		vkDestroyShaderModule(static_cast<VkDevice>(*device), fragShaderModule, nullptr);
		vkDestroyShaderModule(static_cast<VkDevice>(*device), vertShaderModule, nullptr);
	}

	// Create a shader module from code.
	VkShaderModule createShaderModule(const std::vector<char> &code)
	{
		VkShaderModuleCreateInfo createInfo{};
		createInfo.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		createInfo.codeSize = code.size();
		createInfo.pCode    = reinterpret_cast<const uint32_t *>(code.data());
		VkShaderModule shaderModule;
		if (vkCreateShaderModule(static_cast<VkDevice>(*device), &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
			throw std::runtime_error("Failed to create shader module!");
		return shaderModule;
	}

	// Create framebuffers for each swap chain image view.
	void createFramebuffers()
	{
		swapChainFramebuffers.resize(swapChainImageViews.size());
		for (size_t i = 0; i < swapChainImageViews.size(); i++)
		{
			VkImageView             attachments[] = {swapChainImageViews[i]};
			VkFramebufferCreateInfo framebufferInfo{};
			framebufferInfo.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			framebufferInfo.renderPass      = renderPass;
			framebufferInfo.attachmentCount = 1;
			framebufferInfo.pAttachments    = attachments;
			framebufferInfo.width           = swapChainExtent.width;
			framebufferInfo.height          = swapChainExtent.height;
			framebufferInfo.layers          = 1;
			if (vkCreateFramebuffer(static_cast<VkDevice>(*device), &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS)
				throw std::runtime_error("Failed to create framebuffer!");
		}
	}

	// Create command pool.
	void createCommandPool()
	{
		QueueFamilyIndices      queueFamilyIndices = findQueueFamilies(physicalDevice);
		VkCommandPoolCreateInfo poolInfo{};
		poolInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInfo.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
		if (vkCreateCommandPool(static_cast<VkDevice>(*device), &poolInfo, nullptr, &commandPool) != VK_SUCCESS)
			throw std::runtime_error("Failed to create command pool!");
	}

	// Allocate command buffer.
	void createCommandBuffer()
	{
		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool        = commandPool;
		allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandBufferCount = 1;
		if (vkAllocateCommandBuffers(static_cast<VkDevice>(*device), &allocInfo, &commandBuffer) != VK_SUCCESS)
			throw std::runtime_error("Failed to allocate command buffer!");
	}

	// Record the command buffer for a given swap chain image index.
	void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex)
	{
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
			throw std::runtime_error("Failed to begin recording command buffer!");

		VkClearValue          clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
		VkRenderPassBeginInfo renderPassInfo{};
		renderPassInfo.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		renderPassInfo.renderPass        = renderPass;
		renderPassInfo.framebuffer       = swapChainFramebuffers[imageIndex];
		renderPassInfo.renderArea.offset = {0, 0};
		renderPassInfo.renderArea.extent = swapChainExtent;
		renderPassInfo.clearValueCount   = 1;
		renderPassInfo.pClearValues      = &clearColor;

		vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
		vkCmdDraw(commandBuffer, 3, 1, 0, 0);
		vkCmdEndRenderPass(commandBuffer);
		if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
			throw std::runtime_error("Failed to record command buffer!");
	}

	// Create semaphores and fence for synchronization.
	void createSyncObjects()
	{
		VkSemaphoreCreateInfo semaphoreInfo{};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
		VkFenceCreateInfo fenceInfo{};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
		if (vkCreateSemaphore(static_cast<VkDevice>(*device), &semaphoreInfo, nullptr, &imageAvailableSemaphore) != VK_SUCCESS ||
		    vkCreateSemaphore(static_cast<VkDevice>(*device), &semaphoreInfo, nullptr, &renderFinishedSemaphore) != VK_SUCCESS ||
		    vkCreateFence(static_cast<VkDevice>(*device), &fenceInfo, nullptr, &inFlightFence) != VK_SUCCESS)
			throw std::runtime_error("Failed to create semaphores!");
	}

	void mainLoop()
	{
		while (!glfwWindowShouldClose(window))
		{
			glfwPollEvents();
			drawFrame();
		}
		vkDeviceWaitIdle(static_cast<VkDevice>(*device));
	}

	void drawFrame()
	{
		vkWaitForFences(static_cast<VkDevice>(*device), 1, &inFlightFence, VK_TRUE, UINT64_MAX);
		vkResetFences(static_cast<VkDevice>(*device), 1, &inFlightFence);

		uint32_t imageIndex;
		vkAcquireNextImageKHR(static_cast<VkDevice>(*device), swapChain, UINT64_MAX, imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex);
		vkResetCommandBuffer(commandBuffer, 0);
		recordCommandBuffer(commandBuffer, imageIndex);

		VkSemaphore          waitSemaphores[] = {imageAvailableSemaphore};
		VkPipelineStageFlags waitStages[]     = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
		VkSubmitInfo         submitInfo{};
		submitInfo.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.waitSemaphoreCount   = 1;
		submitInfo.pWaitSemaphores      = waitSemaphores;
		submitInfo.pWaitDstStageMask    = waitStages;
		submitInfo.commandBufferCount   = 1;
		submitInfo.pCommandBuffers      = &commandBuffer;
		VkSemaphore signalSemaphores[]  = {renderFinishedSemaphore};
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores    = signalSemaphores;

		if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFence) != VK_SUCCESS)
			throw std::runtime_error("Failed to submit draw command buffer!");

		VkPresentInfoKHR presentInfo{};
		presentInfo.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores    = signalSemaphores;
		VkSwapchainKHR swapChains[]    = {swapChain};
		presentInfo.swapchainCount     = 1;
		presentInfo.pSwapchains        = swapChains;
		presentInfo.pImageIndices      = &imageIndex;
		vkQueuePresentKHR(presentQueue, &presentInfo);
	}

	void cleanup()
	{
		vkDestroySemaphore(static_cast<VkDevice>(*device), imageAvailableSemaphore, nullptr);
		vkDestroySemaphore(static_cast<VkDevice>(*device), renderFinishedSemaphore, nullptr);
		vkDestroyFence(static_cast<VkDevice>(*device), inFlightFence, nullptr);
		vkDestroyCommandPool(static_cast<VkDevice>(*device), commandPool, nullptr);
		for (auto framebuffer : swapChainFramebuffers)
			vkDestroyFramebuffer(static_cast<VkDevice>(*device), framebuffer, nullptr);
		vkDestroyPipeline(static_cast<VkDevice>(*device), graphicsPipeline, nullptr);
		vkDestroyRenderPass(static_cast<VkDevice>(*device), renderPass, nullptr);
		vkDestroyPipelineLayout(static_cast<VkDevice>(*device), pipelineLayout, nullptr);
		for (auto imageView : swapChainImageViews)
			vkDestroyImageView(static_cast<VkDevice>(*device), imageView, nullptr);
		vkDestroySwapchainKHR(static_cast<VkDevice>(*device), swapChain, nullptr);
		vkDestroySurfaceKHR(static_cast<VkInstance>(*vkInstance), surface, nullptr);

		glfwDestroyWindow(window);
		glfwTerminate();
	}

	bool checkValidationLayerSupport()
	{
		uint32_t layerCount;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
		std::vector<VkLayerProperties> availableLayers(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());
		for (const char *layerName : validationLayers)
		{
			bool layerFound = false;
			for (const auto &layerProperties : availableLayers)
				if (std::string(layerName) == layerProperties.layerName)
				{
					layerFound = true;
					break;
				}
			if (!layerFound)
				return false;
		}
		return true;
	}
};

int main()
{
	try
	{
		HelloTriangleApplication app;
		app.run();
	}
	catch (const std::exception &e)
	{
		std::cerr << std::format("Error: {}\n", e.what());
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
