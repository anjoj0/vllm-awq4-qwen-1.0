#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace {

thread_local std::string last_error;

int fail(hipError_t status, const char* operation) {
    last_error = std::string(operation) + ": " + hipGetErrorString(status);
    return -static_cast<int>(status == hipSuccess ? hipErrorUnknown : status);
}

struct Context {
    int device = 0;
    hipStream_t stream = nullptr;
};

struct Transfer {
    hipEvent_t start = nullptr;
    hipEvent_t end = nullptr;
};

struct CopyDesc {
    uint64_t destination;
    uint64_t source;
    uint64_t bytes;
};

}  // namespace

extern "C" {

size_t w7900_hip_ipc_handle_size() {
    return sizeof(hipIpcMemHandle_t);
}

const char* w7900_hip_ipc_last_error() {
    return last_error.c_str();
}

int w7900_hip_ipc_export(uint64_t pointer, void* handle_out,
                         uint64_t* allocation_base_out,
                         uint64_t* allocation_size_out, int* device_out) {
    void* allocation_base = nullptr;
    size_t allocation_size = 0;
    int device = 0;
    hipError_t status = hipGetDevice(&device);
    if (status != hipSuccess) {
        return fail(status, "hipGetDevice");
    }
    status = hipMemGetAddressRange(&allocation_base, &allocation_size,
                                   reinterpret_cast<void*>(pointer));
    if (status != hipSuccess) {
        return fail(status, "hipMemGetAddressRange");
    }

    hipIpcMemHandle_t handle{};
    status = hipIpcGetMemHandle(&handle, allocation_base);
    if (status != hipSuccess) {
        return fail(status, "hipIpcGetMemHandle");
    }
    std::memcpy(handle_out, &handle, sizeof(handle));
    *allocation_base_out = reinterpret_cast<uint64_t>(allocation_base);
    *allocation_size_out = static_cast<uint64_t>(allocation_size);
    *device_out = device;
    return 0;
}

int w7900_hip_ipc_context_create(int device, void** context_out) {
    auto* context = new Context();
    context->device = device;
    hipError_t status = hipSetDevice(device);
    if (status != hipSuccess) {
        delete context;
        return fail(status, "hipSetDevice context create");
    }
    status = hipStreamCreateWithFlags(&context->stream, hipStreamNonBlocking);
    if (status != hipSuccess) {
        delete context;
        return fail(status, "hipStreamCreateWithFlags");
    }
    *context_out = context;
    return 0;
}

int w7900_hip_ipc_context_destroy(void* opaque_context) {
    auto* context = static_cast<Context*>(opaque_context);
    if (context == nullptr) {
        return 0;
    }
    hipError_t status = hipSetDevice(context->device);
    if (status == hipSuccess) {
        status = hipStreamSynchronize(context->stream);
    }
    if (status == hipSuccess) {
        status = hipStreamDestroy(context->stream);
    }
    delete context;
    return status == hipSuccess ? 0 : fail(status, "destroy HIP IPC context");
}

int w7900_hip_ipc_open(void* opaque_context, const void* handle_bytes,
                       uint64_t* mapped_base_out) {
    auto* context = static_cast<Context*>(opaque_context);
    hipError_t status = hipSetDevice(context->device);
    if (status != hipSuccess) {
        return fail(status, "hipSetDevice IPC open");
    }
    hipIpcMemHandle_t handle{};
    std::memcpy(&handle, handle_bytes, sizeof(handle));
    void* mapped_base = nullptr;
    status = hipIpcOpenMemHandle(&mapped_base, handle,
                                 hipIpcMemLazyEnablePeerAccess);
    if (status != hipSuccess) {
        return fail(status, "hipIpcOpenMemHandle");
    }
    *mapped_base_out = reinterpret_cast<uint64_t>(mapped_base);
    return 0;
}

int w7900_hip_ipc_close(void* opaque_context, uint64_t mapped_base) {
    auto* context = static_cast<Context*>(opaque_context);
    hipError_t status = hipSetDevice(context->device);
    if (status != hipSuccess) {
        return fail(status, "hipSetDevice IPC close");
    }
    status = hipIpcCloseMemHandle(reinterpret_cast<void*>(mapped_base));
    return status == hipSuccess ? 0 : fail(status, "hipIpcCloseMemHandle");
}

int w7900_hip_ipc_submit(void* opaque_context, const CopyDesc* copies,
                         size_t count, void** transfer_out) {
    auto* context = static_cast<Context*>(opaque_context);
    hipError_t status = hipSetDevice(context->device);
    if (status != hipSuccess) {
        return fail(status, "hipSetDevice submit");
    }
    auto* transfer = new Transfer();
    status = hipEventCreate(&transfer->start);
    if (status != hipSuccess) {
        delete transfer;
        return fail(status, "hipEventCreate start");
    }
    status = hipEventCreate(&transfer->end);
    if (status != hipSuccess) {
        (void)hipEventDestroy(transfer->start);
        delete transfer;
        return fail(status, "hipEventCreate end");
    }
    status = hipEventRecord(transfer->start, context->stream);
    if (status != hipSuccess) {
        (void)hipEventDestroy(transfer->end);
        (void)hipEventDestroy(transfer->start);
        delete transfer;
        return fail(status, "hipEventRecord start");
    }

    for (size_t i = 0; i < count; ++i) {
        status = hipMemcpyAsync(reinterpret_cast<void*>(copies[i].destination),
                                reinterpret_cast<const void*>(copies[i].source),
                                static_cast<size_t>(copies[i].bytes),
                                hipMemcpyDeviceToDevice, context->stream);
        if (status != hipSuccess) {
            (void)hipEventDestroy(transfer->end);
            (void)hipEventDestroy(transfer->start);
            delete transfer;
            return fail(status, "hipMemcpyAsync device-to-device");
        }
    }
    status = hipEventRecord(transfer->end, context->stream);
    if (status != hipSuccess) {
        (void)hipEventDestroy(transfer->end);
        (void)hipEventDestroy(transfer->start);
        delete transfer;
        return fail(status, "hipEventRecord end");
    }
    *transfer_out = transfer;
    return 0;
}

// 0: complete, 1: in progress, negative: error.
int w7900_hip_ipc_query(void* opaque_transfer) {
    auto* transfer = static_cast<Transfer*>(opaque_transfer);
    const hipError_t status = hipEventQuery(transfer->end);
    if (status == hipSuccess) {
        return 0;
    }
    if (status == hipErrorNotReady) {
        return 1;
    }
    return fail(status, "hipEventQuery");
}

int w7900_hip_ipc_wait(void* opaque_transfer) {
    auto* transfer = static_cast<Transfer*>(opaque_transfer);
    const hipError_t status = hipEventSynchronize(transfer->end);
    return status == hipSuccess ? 0 : fail(status, "hipEventSynchronize");
}

int w7900_hip_ipc_elapsed_us(void* opaque_transfer, float* elapsed_us_out) {
    auto* transfer = static_cast<Transfer*>(opaque_transfer);
    float elapsed_ms = 0.0f;
    const hipError_t status =
        hipEventElapsedTime(&elapsed_ms, transfer->start, transfer->end);
    if (status != hipSuccess) {
        return fail(status, "hipEventElapsedTime");
    }
    *elapsed_us_out = elapsed_ms * 1000.0f;
    return 0;
}

int w7900_hip_ipc_release(void* opaque_transfer) {
    auto* transfer = static_cast<Transfer*>(opaque_transfer);
    if (transfer == nullptr) {
        return 0;
    }
    hipError_t status = hipEventSynchronize(transfer->end);
    if (status == hipSuccess) {
        status = hipEventDestroy(transfer->end);
    }
    if (status == hipSuccess) {
        status = hipEventDestroy(transfer->start);
    }
    delete transfer;
    return status == hipSuccess ? 0 : fail(status, "release HIP IPC transfer");
}

}  // extern "C"
