#include <stdio.h>
#include <stdlib.h>
#include <onnxruntime_c_api.h>

const OrtApi* g_ort = NULL;

#define CHECK_STATUS(status_expr) \
    do { \
        OrtStatus* _status = (status_expr); \
        if (_status != NULL) { \
            printf("❌ ONNX Error at line %d: %s\n", __LINE__, g_ort->GetErrorMessage(_status)); \
            g_ort->ReleaseStatus(_status); \
            exit(1); \
        } \
    } while(0)

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Enter the model path: %s <path_to_integrated_onnx_model>\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];

    // 1. 初始化 OrtApi
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort) {
        printf("❌ Can't access ONNX Runtime API\n");
        return 1;
    }

    OrtEnv* env = NULL;
    OrtSessionOptions* session_options = NULL;
    OrtSession* session = NULL;
    OrtMemoryInfo* memory_info = NULL;
    OrtAllocator* allocator = NULL;

    CHECK_STATUS(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "onnx_test", &env));
    CHECK_STATUS(g_ort->CreateSessionOptions(&session_options));
    CHECK_STATUS(g_ort->CreateSession(env, model_path, session_options, &session));
    CHECK_STATUS(g_ort->GetAllocatorWithDefaultOptions(&allocator));

    // ==========================================
    // ⭐️ Dynamic input
    // ==========================================
    char* input_name = NULL;
    OrtTypeInfo* type_info = NULL;
    const OrtTensorTypeAndShapeInfo* tensor_info = NULL;

    CHECK_STATUS(g_ort->SessionGetInputName(session, 0, allocator, &input_name));
    CHECK_STATUS(g_ort->SessionGetInputTypeInfo(session, 0, &type_info));
    CHECK_STATUS(g_ort->CastTypeInfoToTensorInfo(type_info, &tensor_info));

    size_t num_dims = 0;
    CHECK_STATUS(g_ort->GetDimensionsCount(tensor_info, &num_dims));

    if (num_dims == 0) num_dims = 1;
    
    int64_t* input_shape = (int64_t*)calloc(num_dims, sizeof(int64_t));
    CHECK_STATUS(g_ort->GetDimensions(tensor_info, input_shape, num_dims));

    printf("🔍 Check input: '%s', original dimension: [", input_name);
    for (size_t i = 0; i < num_dims; ++i) {
        printf("%lld%s", (long long)input_shape[i], i == num_dims - 1 ? "" : ", ");
    }
    printf("]\n");

    size_t total_elements = 1;
    for (size_t i = 0; i < num_dims; ++i) {
        if (input_shape[i] <= 0) {
            if (i == 0) {
                printf("   ⚠️ Dynamic Batch batch size (Dimension[0]: %lld), modify to 1\n", (long long)input_shape[i]);
                input_shape[i] = 1;
            } else {
                printf("❌ Error, unsupported dynamic batch size\n");
                return 1;
            }
        }
        total_elements *= (size_t)input_shape[i];
    }
    g_ort->ReleaseTypeInfo(type_info); 

    // ==========================================
    // ⭐️ Dynamic output
    // ==========================================
    char* output_name = NULL;
    CHECK_STATUS(g_ort->SessionGetOutputName(session, 0, allocator, &output_name));

    // Reference: [train_val_process.ipynb](https://github.com/Ryan-z-Feng-ccsf/chemical-regressor/blob/main/nn_regressor/train_val_process.ipynb) H+ = 1e-7 mol/L (pH=7), Zn = 1e-4 mol/L
    double input_data[2] = {1.0e-7, 1.0e-4};
    if (total_elements != 2) {
        printf("❌ Error! \n");
        return 1;
    }

    CHECK_STATUS(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
    OrtValue* input_tensor = NULL;
    CHECK_STATUS(g_ort->CreateTensorWithDataAsOrtValue(
        memory_info,
        input_data,
        total_elements * sizeof(double),
        input_shape,
        num_dims,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
        &input_tensor
    ));

    const char* input_names[] = {input_name};
    const char* output_names[] = {output_name};
    OrtValue* output_tensor = NULL;

    printf("🚀 Running Inference...\n");
    CHECK_STATUS(g_ort->Run(
        session,
        NULL,
        input_names, (const OrtValue* const*)&input_tensor, 1,
        output_names, 1,
        &output_tensor
    ));

    // Print
    double* output_data = NULL;
    CHECK_STATUS(g_ort->GetTensorMutableData(output_tensor, (void**)&output_data));

    printf("\n✅ Inference Success!\n");
    printf("  └─ SURF-H+  (H_sorbed) : %.9e mol/L\n", output_data[0]);
    printf("  └─ SURF-Zn++ (Zn_sorbed): %.9e mol/L\n", output_data[1]);

    free(input_shape);
    g_ort->AllocatorFree(allocator, input_name);
    g_ort->AllocatorFree(allocator, output_name);
    g_ort->ReleaseValue(input_tensor);
    g_ort->ReleaseValue(output_tensor);
    g_ort->ReleaseMemoryInfo(memory_info);
    g_ort->ReleaseSession(session);
    g_ort->ReleaseSessionOptions(session_options);
    g_ort->ReleaseEnv(env);

    return 0;
}