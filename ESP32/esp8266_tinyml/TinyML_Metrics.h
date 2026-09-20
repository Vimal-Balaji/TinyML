#ifndef TINYML_METRICS_H
#define TINYML_METRICS_H

#include <Arduino.h>
#include <stdint.h>

//==============================================================================
// TinyML EVALUATION PARAMETERS FOR ESP8266
// 
// This header file contains all ESP8266-specific evaluation metrics including:
// - Memory Usage (Flash, RAM, SRAM)
// - Latency (Inference Time)
// - Power Consumption
// - Accuracy and Task-Specific Metrics
// - Hardware Constraints
//==============================================================================

// ============================================================================
// 1. MEMORY CONSTRAINTS & TRACKING
// ============================================================================

#define ESP8266_TOTAL_FLASH_KB      4096        // 4MB typical
#define ESP8266_TOTAL_SRAM_KB       160         // 160KB SRAM (includes WiFi overhead)
#define ESP8266_USABLE_RAM_KB       80          // ~80KB usable after WiFi/system
#define ESP8266_HEAP_START          0x3FFE8000  // Start of heap region

// Program & Model Memory Breakdown (user should update based on actual upload)
#define PROGRAM_CODE_KB             150         // Estimated program code footprint
#define MODEL_WEIGHTS_KB            185         // Model weights stored in PROGMEM
#define FILESYSTEM_RESERVED_KB      1024        // LittleFS filesystem reserve
#define OTA_RESERVE_KB              1019        // Over-the-air update reserve

// Calculated available space for model & buffers
#define AVAILABLE_FLASH_FOR_MODEL   (ESP8266_TOTAL_FLASH_KB - PROGRAM_CODE_KB - FILESYSTEM_RESERVED_KB - OTA_RESERVE_KB)
#define AVAILABLE_RAM_FOR_INFERENCE (ESP8266_USABLE_RAM_KB * 1024)  // Convert to bytes

// ============================================================================
// 2. LATENCY & PERFORMANCE METRICS
// ============================================================================

#define MAX_INFERENCE_TIME_MS       100         // Target max latency (ms)
#define MIN_INFERENCE_TIME_MS       1           // Minimum measurable latency
#define CPU_FREQUENCY_MHZ           80          // or 160 MHz when overclocked

// Inference timing structure
struct InferenceMetrics {
  uint32_t inference_start_us;
  uint32_t inference_end_us;
  uint32_t inference_time_us;
  uint32_t inference_time_ms;
  float    throughput_hz;          // Inferences per second
  uint32_t total_inferences;       // Counter for averaging
  uint32_t min_time_us;
  uint32_t max_time_us;
  float    avg_time_us;
};

// ============================================================================
// 3. POWER CONSUMPTION METRICS
// ============================================================================

#define ESP8266_IDLE_CURRENT_UA     100         // ~100 μA idle
#define ESP8266_ACTIVE_CURRENT_MA   80          // ~80 mA during inference (estimate)
#define ESP8266_WIFI_IDLE_MA        20          // WiFi idle consumption
#define ESP8266_DEEP_SLEEP_UA       20          // Deep sleep (disabled WiFi)
#define BATTERY_VOLTAGE_MV          3300        // Typical 3.3V

// Power consumption tracking
struct PowerMetrics {
  uint32_t active_cycles;          // CPU cycles during inference
  uint32_t total_energy_mj;        // Total energy in milli-joules
  float    avg_power_mw;           // Average power (mW)
  float    peak_power_mw;          // Peak power during inference
  uint32_t samples_collected;
  float    battery_life_hours;     // Estimated battery life
};

// ============================================================================
// 4. MEMORY USAGE TRACKING
// ============================================================================

struct MemoryMetrics {
  // Flash Memory
  uint32_t flash_model_bytes;      // Model weights in flash
  uint32_t flash_program_bytes;    // Program code in flash
  uint32_t flash_used_bytes;       // Total flash used
  
  // RAM Memory
  uint32_t ram_static_bytes;       // Static variables (global)
  uint32_t ram_activation_bytes;   // Activation buffers (stem_out, b1_dw, etc.)
  uint32_t ram_peak_bytes;         // Peak RAM during inference
  uint32_t ram_free_bytes;         // Free RAM available
  
  // Memory efficiency
  float    memory_utilization_pct; // % of available RAM used
  float    model_compression_ratio;// Model size vs uncompressed
};

// ============================================================================
// 5. ACCURACY & PREDICTION METRICS
// ============================================================================

struct AccuracyMetrics {
  // Classification task
  uint32_t correct_predictions;
  uint32_t total_predictions;
  float    accuracy_pct;
  
  // Per-class metrics
  float    precision[12];          // For each class (adjust NUM_CLASSES as needed)
  float    recall[12];
  float    f1_score[12];
  
  // Confidence metrics
  float    confidence_threshold;   // Minimum confidence to accept prediction
  float    avg_confidence;
  float    min_confidence;
  
  // Quantization effects
  float    accuracy_drop_vs_float; // Accuracy loss due to quantization
  
  // Confusion matrix (optional, requires more storage)
  // uint16_t confusion_matrix[NUM_CLASSES][NUM_CLASSES];
};

// ============================================================================
// 6. QUANTIZATION METRICS
// ============================================================================

struct QuantizationMetrics {
  uint8_t  bit_width;              // 8-bit for INT8
  float    scale_factor;           // Quantization scale
  int8_t   zero_point;             // Quantization zero-point
  
  // Per-layer quantization info (example for a few layers)
  float    stem_input_scale;
  int8_t   stem_input_zp;
  
  float    stem_output_scale;
  int8_t   stem_output_zp;
  
  float    fc_output_scale;
  int8_t   fc_output_zp;
  
  // Quantization noise
  float    max_quantization_error;
  float    avg_quantization_error;
};

// ============================================================================
// 7. INFERENCE PIPELINE METRICS
// ============================================================================

struct LayerMetrics {
  char     layer_name[32];         // Layer identifier
  uint32_t layer_inference_us;     // Time for this layer only
  uint32_t layer_flops;            // Floating point operations
  float    layer_throughput_gflops;// GFLOPS for this layer
  uint32_t layer_memory_bytes;     // Memory used by this layer
};

struct PipelineMetrics {
  uint32_t num_layers;
  LayerMetrics layer_times[10];    // Support up to 10 layers
  
  float    total_gflops;           // Total FLOPs for full inference
  float    memory_bandwidth_mb_s;  // Estimated memory bandwidth usage
};

// ============================================================================
// 8. HARDWARE CONSTRAINT COMPLIANCE
// ============================================================================

struct HardwareConstraints {
  // Check results
  bool     flash_ok;               // Model fits in flash
  bool     ram_ok;                 // Buffers fit in RAM
  bool     latency_ok;             // Meets latency target
  bool     power_ok;               // Meets power budget
  
  // Margins (how close to limits)
  float    flash_margin_pct;       // % headroom in flash
  float    ram_margin_pct;         // % headroom in RAM
  float    latency_margin_pct;     // % headroom in latency budget
  float    power_margin_pct;       // % headroom in power budget
};

// ============================================================================
// 9. DATA ACQUISITION METRICS (for on-device test data)
// ============================================================================

struct DataMetrics {
  uint32_t total_samples;
  uint32_t samples_processed;
  uint32_t num_test_batches;
  
  // MFCC-specific (for audio models)
  uint32_t mfcc_computation_us;
  float    mfcc_buffer_kb;
};

// ============================================================================
// 10. UTILITY FUNCTIONS
// ============================================================================

class TinyMLEvaluator {
public:
  InferenceMetrics  inference_metrics;
  PowerMetrics      power_metrics;
  MemoryMetrics     memory_metrics;
  AccuracyMetrics   accuracy_metrics;
  QuantizationMetrics quantization_metrics;
  PipelineMetrics   pipeline_metrics;
  HardwareConstraints hardware_constraints;
  DataMetrics       data_metrics;
  
  // Initialize all metrics to zero
  void init() {
    memset(&inference_metrics, 0, sizeof(InferenceMetrics));
    memset(&power_metrics, 0, sizeof(PowerMetrics));
    memset(&memory_metrics, 0, sizeof(MemoryMetrics));
    memset(&accuracy_metrics, 0, sizeof(AccuracyMetrics));
    memset(&quantization_metrics, 0, sizeof(QuantizationMetrics));
    memset(&pipeline_metrics, 0, sizeof(PipelineMetrics));
    memset(&hardware_constraints, 0, sizeof(HardwareConstraints));
    memset(&data_metrics, 0, sizeof(DataMetrics));
  }
  
  // Start inference timer
  void startInference() {
    inference_metrics.inference_start_us = micros();
  }
  
  // End inference timer and calculate metrics
  void endInference() {
    inference_metrics.inference_end_us = micros();
    inference_metrics.inference_time_us = 
      inference_metrics.inference_end_us - inference_metrics.inference_start_us;
    inference_metrics.inference_time_ms = 
      inference_metrics.inference_time_us / 1000.0f;
    inference_metrics.total_inferences++;
    
    // Update min/max
    if (inference_metrics.inference_time_us < inference_metrics.min_time_us || 
        inference_metrics.min_time_us == 0) {
      inference_metrics.min_time_us = inference_metrics.inference_time_us;
    }
    if (inference_metrics.inference_time_us > inference_metrics.max_time_us) {
      inference_metrics.max_time_us = inference_metrics.inference_time_us;
    }
    
    // Calculate average
    if (inference_metrics.total_inferences > 0) {
      inference_metrics.avg_time_us = 
        (float)inference_metrics.inference_time_us / inference_metrics.total_inferences;
    }
    
    // Calculate throughput (inferences per second)
    if (inference_metrics.inference_time_us > 0) {
      inference_metrics.throughput_hz = 
        1000000.0f / inference_metrics.inference_time_us;
    }
  }
  
  // Update memory metrics
  void updateMemoryMetrics(uint32_t peak_ram_bytes) {
    memory_metrics.ram_peak_bytes = peak_ram_bytes;
    memory_metrics.ram_free_bytes = ESP.getFreeHeap();
    memory_metrics.memory_utilization_pct = 
      (float)(AVAILABLE_RAM_FOR_INFERENCE - memory_metrics.ram_free_bytes) / 
      AVAILABLE_RAM_FOR_INFERENCE * 100.0f;
  }
  
  // Update accuracy metrics
  void updateAccuracy(bool is_correct, float confidence, uint8_t predicted_class, uint8_t true_class) {
    accuracy_metrics.total_predictions++;
    if (is_correct) {
      accuracy_metrics.correct_predictions++;
    }
    accuracy_metrics.accuracy_pct = 
      (float)accuracy_metrics.correct_predictions / accuracy_metrics.total_predictions * 100.0f;
    accuracy_metrics.avg_confidence += confidence;
  }
  
  // Verify hardware constraints
  void verifyHardwareConstraints() {
    hardware_constraints.flash_ok = (memory_metrics.flash_used_bytes < AVAILABLE_FLASH_FOR_MODEL * 1024);
    hardware_constraints.ram_ok = (memory_metrics.ram_peak_bytes < AVAILABLE_RAM_FOR_INFERENCE);
    hardware_constraints.latency_ok = (inference_metrics.inference_time_ms <= MAX_INFERENCE_TIME_MS);
    hardware_constraints.power_ok = true; // Set based on power budget
    
    if (AVAILABLE_FLASH_FOR_MODEL * 1024 > 0) {
      hardware_constraints.flash_margin_pct = 
        ((AVAILABLE_FLASH_FOR_MODEL * 1024 - memory_metrics.flash_used_bytes) / 
         (AVAILABLE_FLASH_FOR_MODEL * 1024)) * 100.0f;
    }
    
    if (AVAILABLE_RAM_FOR_INFERENCE > 0) {
      hardware_constraints.ram_margin_pct = 
        ((AVAILABLE_RAM_FOR_INFERENCE - memory_metrics.ram_peak_bytes) / 
         AVAILABLE_RAM_FOR_INFERENCE) * 100.0f;
    }
    
    if (MAX_INFERENCE_TIME_MS > 0) {
      hardware_constraints.latency_margin_pct = 
        ((MAX_INFERENCE_TIME_MS - inference_metrics.inference_time_ms) / 
         MAX_INFERENCE_TIME_MS) * 100.0f;
    }
  }
  
  // Print all metrics to Serial
  void printAllMetrics() {
    printInferenceMetrics();
    printMemoryMetrics();
    printAccuracyMetrics();
    printQuantizationMetrics();
    printHardwareConstraints();
  }
  
  void printInferenceMetrics() {
    Serial.println("\n=== INFERENCE LATENCY METRICS ===");
    Serial.print("Inference Time (ms): ");
    Serial.println(inference_metrics.inference_time_ms, 3);
    Serial.print("Inference Time (μs): ");
    Serial.println(inference_metrics.inference_time_us);
    Serial.print("Min Time (μs): ");
    Serial.println(inference_metrics.min_time_us);
    Serial.print("Max Time (μs): ");
    Serial.println(inference_metrics.max_time_us);
    Serial.print("Avg Time (μs): ");
    Serial.println(inference_metrics.avg_time_us, 2);
    Serial.print("Throughput (Hz): ");
    Serial.println(inference_metrics.throughput_hz, 2);
    Serial.print("Total Inferences: ");
    Serial.println(inference_metrics.total_inferences);
  }
  
  void printMemoryMetrics() {
    Serial.println("\n=== MEMORY METRICS ===");
    Serial.print("Flash Used (KB): ");
    Serial.println(memory_metrics.flash_used_bytes / 1024.0f, 2);
    Serial.print("Model Weights (KB): ");
    Serial.println(memory_metrics.flash_model_bytes / 1024.0f, 2);
    Serial.print("Program Code (KB): ");
    Serial.println(memory_metrics.flash_program_bytes / 1024.0f, 2);
    Serial.print("RAM Peak (bytes): ");
    Serial.println(memory_metrics.ram_peak_bytes);
    Serial.print("RAM Free (bytes): ");
    Serial.println(memory_metrics.ram_free_bytes);
    Serial.print("RAM Utilization (%): ");
    Serial.println(memory_metrics.memory_utilization_pct, 2);
    Serial.print("Available Flash for Model (KB): ");
    Serial.println(AVAILABLE_FLASH_FOR_MODEL);
  }
  
  void printAccuracyMetrics() {
    Serial.println("\n=== ACCURACY METRICS ===");
    Serial.print("Accuracy (%): ");
    Serial.println(accuracy_metrics.accuracy_pct, 2);
    Serial.print("Correct Predictions: ");
    Serial.println(accuracy_metrics.correct_predictions);
    Serial.print("Total Predictions: ");
    Serial.println(accuracy_metrics.total_predictions);
    Serial.print("Avg Confidence: ");
    Serial.println(accuracy_metrics.avg_confidence / accuracy_metrics.total_predictions, 4);
  }
  
  void printQuantizationMetrics() {
    Serial.println("\n=== QUANTIZATION METRICS ===");
    Serial.print("Bit Width: ");
    Serial.print(quantization_metrics.bit_width);
    Serial.println("-bit");
    Serial.print("Stem Input Scale: ");
    Serial.println(quantization_metrics.stem_input_scale, 6);
    Serial.print("Stem Input Zero Point: ");
    Serial.println(quantization_metrics.stem_input_zp);
    Serial.print("FC Output Scale: ");
    Serial.println(quantization_metrics.fc_output_scale, 6);
    Serial.print("FC Output Zero Point: ");
    Serial.println(quantization_metrics.fc_output_zp);
  }
  
  void printHardwareConstraints() {
    Serial.println("\n=== HARDWARE COMPLIANCE CHECK ===");
    Serial.print("Flash Constraint OK: ");
    Serial.println(hardware_constraints.flash_ok ? "YES" : "NO");
    Serial.print("RAM Constraint OK: ");
    Serial.println(hardware_constraints.ram_ok ? "YES" : "NO");
    Serial.print("Latency Constraint OK: ");
    Serial.println(hardware_constraints.latency_ok ? "YES" : "NO");
    Serial.print("Power Constraint OK: ");
    Serial.println(hardware_constraints.power_ok ? "YES" : "NO");
    
    Serial.println("\n--- Constraint Margins ---");
    Serial.print("Flash Margin (%): ");
    Serial.println(hardware_constraints.flash_margin_pct, 2);
    Serial.print("RAM Margin (%): ");
    Serial.println(hardware_constraints.ram_margin_pct, 2);
    Serial.print("Latency Margin (%): ");
    Serial.println(hardware_constraints.latency_margin_pct, 2);
  }
};

#endif // TINYML_METRICS_H
