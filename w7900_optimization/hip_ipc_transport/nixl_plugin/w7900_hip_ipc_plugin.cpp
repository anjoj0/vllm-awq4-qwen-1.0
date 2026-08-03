#include "backend/backend_plugin.h"
#include "w7900_hip_ipc_backend.h"

using Plugin = nixlBackendPluginCreator<W7900HipIpcBackend>;

extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin* nixl_plugin_init() {
    return Plugin::create(NIXL_PLUGIN_API_VERSION, "W7900_HIP_IPC", "0.1.0", {},
                          {VRAM_SEG});
}

extern "C" NIXL_PLUGIN_EXPORT void nixl_plugin_fini() {}
