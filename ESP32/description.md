# ESP8266 TinyML CNN Inference Implementation

The `esp8266_tinyml.ino` file contains a highly optimized, standalone C++ implementation of a Convolutional Neural Network (CNN) specifically tailored for microcontrollers like the ESP8266. It implements quantized neural network operations directly from scratch without relying on large machine learning frameworks like TensorFlow Lite for Microcontrollers (TFLite Micro). This significantly reduces the memory and storage footprint.

## Core Characteristics
- **Bare-Metal Inference**: No external ML libraries used. The mathematical operations are hand-written for simplicity and minimal overhead.
- **Quantization**: Most of the operations are quantized (int8/uint8) to speed up inference and save RAM. Real-valued operations are only performed where absolutely necessary (like reading the raw float MFCC input).
- **Flash Memory Usage**: The network weights and biases are stored in the flash memory (`PROGMEM`) to save precious SRAM. They are accessed using `pgm_read_byte` and `pgm_read_dword`.

## Operations

The model architecture expects 13-channel MFCC features as input and outputs logits for 12 classes. The network operations consist of customized 1D convolutions, average pooling, and a fully connected layer.

### 1. `conv1d_float_in_q_out` (Stem Convolution)
This operation handles the very first layer (stem layer) of the network. It takes the real-valued (floating-point) MFCC inputs and computes the convolutions using quantized weights. 
- **Input**: `float` array of MFCC features.
- **Output**: Quantized `uint8_t` activations.
- **Purpose**: Bridges the gap between the unquantized input and the rest of the quantized network, performing a standard 1D convolution directly on the float buffer.

### 2. `conv1d_q` (General Quantized Conv1D)
This is the workhorse of the model, used for all subsequent convolutional layers. It performs standard and depthwise 1D convolutions entirely using quantized integer arithmetic. 
- **Input**: Quantized `uint8_t` activations.
- **Output**: Quantized `uint8_t` activations.
- **Usage in Model**:
  - **Depthwise Convolution**: Filters each input channel separately (using groups).
  - **Pointwise Convolution**: Combines the output of depthwise layers using 1x1 convolutions (kernel size = 1).

### 3. `avgpool_q` (Quantized Average Pooling)
Reduces the temporal/spatial dimensions of the feature maps by calculating the average over the length dimension.
- **Operation**: Sums up the quantized values for each channel over time and divides by the length to get the average.

### 4. `linear_q` (Fully Connected / Dense Layer)
The final classification layer of the network.
- **Operation**: Performs a quantized matrix multiplication between the pooled feature vectors and the final layer's weights.
- **Output**: Generates quantized logits for the 12 possible classes.

### 5. `predict` 
The main orchestrator function that defines the model architecture. It sequentially chains the operations together in the RAM buffers:
1. **Stem Conv**: 5-kernel, stride 2, padding 2.
2. **Depthwise Conv 1**: 3-kernel, stride 1, padding 1.
3. **Pointwise Conv 1**: 1-kernel, stride 1.
4. **Depthwise Conv 2**: 3-kernel, stride 2, padding 1.
5. **Pointwise Conv 2**: 1-kernel, stride 1.
6. **Average Pooling**.
7. **Linear Classifier**.

It loops through the final output `logits_q` and returns the index of the class with the highest probability.

