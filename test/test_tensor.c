#include <stdio.h>
#include <stdlib.h>
#include <onnxruntime_c_api.h>

// Load the ORT api
const OrtApi* g_ort = NULL;

void check_status(OrtStatus* status) {
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        printf("❌ ONNX Runtime Error: %s\n", msg);
        g_ort->ReleaseStatus(status);
        exit(1);
    }
}

void check_node_dim0(OrtSession* session, OrtAllocator* allocator, size_t index, int is_input) {
    char* node_name = NULL;
    OrtTypeInfo* type_info = NULL;
    const OrtTensorTypeAndShapeInfo* tensor_info = NULL;

    if (is_input) {
        check_status(g_ort->SessionGetInputName(session, index, allocator, &node_name));
        check_status(g_ort->SessionGetInputTypeInfo(session, index, &type_info));
    } else {
        check_status(g_ort->SessionGetOutputName(session, index, allocator, &node_name));
        check_status(g_ort->SessionGetOutputTypeInfo(session, index, &type_info));
    }

    check_status(g_ort->CastTypeInfoToTensorInfo(type_info, &tensor_info));

    // Get the number of the dim
    size_t num_dims = 0;
    check_status(g_ort->GetDimensionsCount(tensor_info, &num_dims));

    if (num_dims == 0) {
        printf("[%s %zu] Name: %s -> dimension: 0D Scalar \n", 
               is_input ? "Input " : "Output", index, node_name);
    } else {
        int64_t* dims = (int64_t*)malloc(num_dims * sizeof(int64_t));
        check_status(g_ort->GetDimensions(tensor_info, dims, num_dims));

        int64_t dim_0 = dims[0];

        // Check dim[0] dynamic
        if (dim_0 == -1) {
            printf("[%s %zu] Name: %s -> dimension[0]: -1 (Dynamic Batch Size)\n", 
                   is_input ? "Input " : "Output", index, node_name);
        } else {
            printf("[%s %zu] Name: %s -> dimension[0]: %lld (Fixed Batch Size)\n", 
                   is_input ? "Input " : "Output", index, node_name, (long long)dim_0);
        }

        free(dims);
    }

    // Release 
    check_status(g_ort->AllocatorFree(allocator, node_name));
    g_ort->ReleaseTypeInfo(type_info);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Enter path: %s <path_to_onnx_model>\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];

    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort) {
        printf("❌ Can't access ONNX Runtime API\n");
        return 1;
    }

    OrtEnv* env = NULL;
    OrtSessionOptions* session_options = NULL;
    OrtSession* session = NULL;
    OrtAllocator* allocator = NULL;

    check_status(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "onnx_dim0_checker", &env));
    check_status(g_ort->CreateSessionOptions(&session_options));
    
    printf("=== Model Path: %s ===\n\n", model_path);
    check_status(g_ort->CreateSession(env, model_path, session_options, &session));
    check_status(g_ort->GetAllocatorWithDefaultOptions(&allocator));

    // Check the input dim[0]
    size_t num_inputs = 0;
    check_status(g_ort->SessionGetInputCount(session, &num_inputs));
    for (size_t i = 0; i < num_inputs; i++) {
        check_node_dim0(session, allocator, i, 1);
    }

    // Check the output dim[0]
    size_t num_outputs = 0;
    check_status(g_ort->SessionGetOutputCount(session, &num_outputs));
    for (size_t i = 0; i < num_outputs; i++) {
        check_node_dim0(session, allocator, i, 0);
    }

    // Release
    g_ort->ReleaseAllocator(allocator);
    g_ort->ReleaseSession(session);
    g_ort->ReleaseSessionOptions(session_options);
    g_ort->ReleaseEnv(env);

    printf("\n✅ Check Completed!\n");
    return 0;
}