/**
 * @file inference_mlp.cc
 * @brief int8 MLP inference via TFLite Micro (Change E).
 *
 * Compiled as C++ because TFLite Micro headers require C++.
 * Guarded by MODEL_USE_MLP — provides compile-time stubs when 0.
 */

#include "inference_mlp.h"
#include "model_weights.h"

#if MODEL_USE_MLP

#include "emg_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

static uint8_t tensor_arena[48 * 1024];  /* 48 KB — tune down if memory is tight */
static tflite::MicroInterpreter *s_interpreter = nullptr;
static TfLiteTensor *s_input  = nullptr;
static TfLiteTensor *s_output = nullptr;

void inference_mlp_init(void) {
    const tflite::Model *model = tflite::GetModel(g_model);
    static tflite::MicroMutableOpResolver<4> resolver;
    resolver.AddFullyConnected();
    resolver.AddRelu();
    resolver.AddSoftmax();
    resolver.AddDequantize();
    static tflite::MicroInterpreter interp(model, resolver,
                                            tensor_arena, sizeof(tensor_arena));
    s_interpreter = &interp;
    s_interpreter->AllocateTensors();
    s_input  = s_interpreter->input(0);
    s_output = s_interpreter->output(0);
}

int inference_mlp_predict(const float *features, int n_feat, float *conf_out) {
    /* Quantise input: int8 = round(float / scale) + zero_point */
    const float iscale = s_input->params.scale;
    const int   izp    = s_input->params.zero_point;
    for (int i = 0; i < n_feat; i++) {
        int q = static_cast<int>(roundf(features[i] / iscale)) + izp;
        if (q < -128) q = -128;
        if (q >  127) q =  127;
        s_input->data.int8[i] = static_cast<int8_t>(q);
    }

    s_interpreter->Invoke();

    /* Dequantise output and find winning class */
    const float oscale = s_output->params.scale;
    const int   ozp    = s_output->params.zero_point;
    float max_p = -1e9f;
    int   max_c = 0;
    const int n_cls = s_output->dims->data[1];
    for (int c = 0; c < n_cls; c++) {
        float p = (s_output->data.int8[c] - ozp) * oscale;
        if (p > max_p) { max_p = p; max_c = c; }
    }
    *conf_out = max_p;
    return max_c;
}

#else  /* MODEL_USE_MLP == 0 — compile-time stubs */

void inference_mlp_init(void) {}

int inference_mlp_predict(const float * /*features*/, int /*n_feat*/, float *conf_out) {
    if (conf_out) *conf_out = 0.0f;
    return 0;
}

#endif  /* MODEL_USE_MLP */
