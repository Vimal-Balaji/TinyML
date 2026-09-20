/*
  TinyStudentDSCNN inference on ESP8266
  --------------------------------------
  - All conv/fc weights are stored as int8_t (quantized) to save flash/RAM.
  - Each layer is dequantized on-the-fly during the MAC:  real_w = int8_w * scale
    (zero_point is 0 for every layer in this checkpoint, so it's omitted from
    the dequant formula; if you requantize with a nonzero zero_point, subtract
    it from the int8 value before multiplying by scale.)
  - Biases are kept as float32 (they are tiny in count, so no benefit to
    quantizing them, and doing so would only hurt accuracy).
  - Architecture (mirrors manual_cpy.py's manual_forward exactly):
        input (13 x 101 MFCC)
          -> stem:      Conv1d(13->22, k=5, s=2, p=2) + ReLU        -> (22 x 51)
          -> block1.dw: Conv1d(22->22, k=3, s=1, p=1, groups=22) + ReLU -> (22 x 51)
          -> block1.pw: Conv1d(22->22, k=1)                          + ReLU -> (22 x 51)
          -> block2.dw: Conv1d(22->22, k=3, s=2, p=1, groups=22) + ReLU -> (22 x 26)
          -> block2.pw: Conv1d(22->44, k=1)                          + ReLU -> (44 x 26)
          -> global average pool                                    -> (44)
          -> fc: Linear(44->12)                                      -> (12) logits
  - Prints the logits vector and the argmax class to the Serial Monitor.
    Values should match the numpy reference in manual_cpy.py bit-for-bit
    within float rounding error.

  Flash usage for weights (int8): 1430 + 66 + 484 + 66 + 968 + 528 = 3542 bytes
  Bias usage (float32):           (22+22+22+44+12)*4              =  488 bytes
*/

#include "model_weights.h"
#include "sample_input.h"

// ---------------------------------------------------------------------
// Scratch buffers sized for the largest intermediate tensor at each stage
// ---------------------------------------------------------------------
static float stem_out[STEM_COUT][51];      // after stem:      22 x 51
static float b1dw_out[B1DW_C][51];         // after block1.dw: 22 x 51
static float b1pw_out[B1PW_COUT][51];      // after block1.pw: 22 x 51
static float b2dw_out[B2DW_C][26];         // after block2.dw: 22 x 26
static float b2pw_out[B2PW_COUT][26];      // after block2.pw: 44 x 26
static float pooled[B2PW_COUT];            // global avg pool: 44
static float logits[FC_COUT];              // final logits:    12

// Input MFCC copied out of PROGMEM into RAM for easy indexing
static float input_mfcc[MFCC_CHANNELS][MFCC_TIME];

// ---------------------------------------------------------------------
// Generic 1D convolution with on-the-fly int8 dequantization.
//   x        : input tensor, shape (C_in, L), flattened row-major
//   x_cin,x_l: dims of x
//   w_q      : int8 weight tensor, shape (C_out, C_in/groups, K), flattened
//   w_scale  : single per-tensor scale (zero_point assumed 0)
//   bias     : float32 bias, shape (C_out)
//   out      : output buffer, shape (C_out, L_out), flattened row-major
//   c_out,k,stride,padding,groups : conv hyperparameters
//   apply_relu : whether to clamp negatives to 0 after adding bias
// ---------------------------------------------------------------------
void conv1d_int8(const float *x, int x_cin, int x_l,
                  const int8_t *w_q, float w_scale,
                  const float *bias,
                  float *out, int c_out, int k, int stride, int padding, int groups,
                  bool apply_relu) {
  int in_per_group  = x_cin / groups;
  int out_per_group = c_out / groups;
  int l_pad = x_l + 2 * padding;
  int l_out = (l_pad - k) / stride + 1;

  for (int g = 0; g < groups; g++) {
    for (int oc_local = 0; oc_local < out_per_group; oc_local++) {
      int oc = g * out_per_group + oc_local;
      for (int t = 0; t < l_out; t++) {
        float acc = 0.0f;
        int start = t * stride - padding;  // position in the (unpadded) input
        for (int ic_local = 0; ic_local < in_per_group; ic_local++) {
          int ic = g * in_per_group + ic_local;
          for (int kk = 0; kk < k; kk++) {
            int in_pos = start + kk;
            if (in_pos < 0 || in_pos >= x_l) continue;  // implicit zero padding
            float xv = x[ic * x_l + in_pos];
            // --- dequantize weight on the fly: real_w = int8_w * scale ---
            int8_t wq = w_q[(oc * in_per_group + ic_local) * k + kk];
            float wv = (float)wq * w_scale;
            acc += xv * wv;
          }
        }
        acc += bias[oc];
        if (apply_relu && acc < 0.0f) acc = 0.0f;
        out[oc * l_out + t] = acc;
      }
    }
  }
}

void global_avg_pool(const float *x, int c, int l, float *out) {
  for (int ch = 0; ch < c; ch++) {
    float sum = 0.0f;
    for (int t = 0; t < l; t++) sum += x[ch * l + t];
    out[ch] = sum / (float)l;
  }
}

void linear_int8(const float *x, int c_in,
                  const int8_t *w_q, float w_scale,
                  const float *bias,
                  float *out, int c_out) {
  for (int oc = 0; oc < c_out; oc++) {
    float acc = 0.0f;
    for (int ic = 0; ic < c_in; ic++) {
      int8_t wq = w_q[oc * c_in + ic];
      float wv = (float)wq * w_scale;
      acc += x[ic] * wv;
    }
    out[oc] = acc + bias[oc];
  }
}

void run_inference() {
  unsigned long t0 = micros();

  // --- stem: Conv1d(13->22, k=5, s=2, p=2) + ReLU ---
  conv1d_int8(&input_mfcc[0][0], STEM_CIN, MFCC_TIME,
              stem_w_q, stem_w_scale, stem_bias,
              &stem_out[0][0], STEM_COUT, STEM_K, /*stride*/2, /*pad*/2, /*groups*/1,
              /*relu*/true);

  // --- block1.depthwise: Conv1d(22->22, k=3, s=1, p=1, groups=22) + ReLU ---
  conv1d_int8(&stem_out[0][0], STEM_COUT, 51,
              b1dw_w_q, b1dw_w_scale, b1dw_bias,
              &b1dw_out[0][0], B1DW_C, B1DW_K, /*stride*/1, /*pad*/1, /*groups*/22,
              /*relu*/true);

  // --- block1.pointwise: Conv1d(22->22, k=1) + ReLU ---
  conv1d_int8(&b1dw_out[0][0], B1DW_C, 51,
              b1pw_w_q, b1pw_w_scale, b1pw_bias,
              &b1pw_out[0][0], B1PW_COUT, /*k*/1, /*stride*/1, /*pad*/0, /*groups*/1,
              /*relu*/true);

  // --- block2.depthwise: Conv1d(22->22, k=3, s=2, p=1, groups=22) + ReLU ---
  conv1d_int8(&b1pw_out[0][0], B1PW_COUT, 51,
              b2dw_w_q, b2dw_w_scale, b2dw_bias,
              &b2dw_out[0][0], B2DW_C, B2DW_K, /*stride*/2, /*pad*/1, /*groups*/22,
              /*relu*/true);

  // --- block2.pointwise: Conv1d(22->44, k=1) + ReLU ---
  conv1d_int8(&b2dw_out[0][0], B2DW_C, 26,
              b2pw_w_q, b2pw_w_scale, b2pw_bias,
              &b2pw_out[0][0], B2PW_COUT, /*k*/1, /*stride*/1, /*pad*/0, /*groups*/1,
              /*relu*/true);

  // --- global average pool: (44 x 26) -> (44) ---
  global_avg_pool(&b2pw_out[0][0], B2PW_COUT, 26, pooled);

  // --- fc: Linear(44->12) ---
  linear_int8(pooled, FC_CIN, fc_w_q, fc_w_scale, fc_bias, logits, FC_COUT);

  unsigned long dt = micros() - t0;

  // --- find argmax ---
  int best = 0;
  for (int i = 1; i < FC_COUT; i++) if (logits[i] > logits[best]) best = i;

  Serial.println();
  Serial.println(F("===== ESP8266 DS-CNN Inference ====="));
  Serial.print(F("Inference time (us): "));
  Serial.println(dt);
  Serial.println(F("Logits:"));
  for (int i = 0; i < FC_COUT; i++) {
    Serial.print(F("  logit["));
    Serial.print(i);
    Serial.print(F("] = "));
    Serial.println(logits[i], 6);
  }
  Serial.print(F("Predicted class index: "));
  Serial.println(best);
  Serial.print(F("Predicted class name : "));
  Serial.println(CLASS_NAMES[best]);
  Serial.println(F("====================================="));
}

void setup() {
  Serial.begin(115200);
  delay(2000);
   Serial.println("\n[1] DEVICE INFORMATION");

    Serial.printf("CPU Frequency     : %u MHz\n",
                  ESP.getCpuFreqMHz());

    Serial.printf("Chip ID           : %08X\n",
                  ESP.getChipId());

    Serial.printf("Flash Size        : %u bytes\n",
                  ESP.getFlashChipSize());

    Serial.printf("Flash Speed       : %u Hz\n",
                  ESP.getFlashChipSpeed());
  Serial.println(F("Loading sample MFCC input from PROGMEM..."));

  // Copy the PROGMEM sample MFCC into RAM
  for (int c = 0; c < MFCC_CHANNELS; c++) {
    for (int t = 0; t < MFCC_TIME; t++) {
      input_mfcc[c][t] = pgm_read_float(&sample_mfcc[c * MFCC_TIME + t]);
    }
  }

  Serial.println(F("Running inference..."));
  run_inference();
}

void loop() {
  // Nothing to do; inference runs once in setup().
  // To run continuously (e.g. with live MFCC features from a microphone),
  // move the body of setup()'s "load input + run_inference()" logic here
  // and replace input_mfcc[][] with freshly computed MFCC frames.
  delay(5000);
}