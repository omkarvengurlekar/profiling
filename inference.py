import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import numpy as np
import pandas as pd
import argparse
import sys

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class TensorRTInfer:
    def __init__(self, model, engine_path, batch_size, precision):
        # Initialize logger
        self.logger = trt.Logger(trt.Logger.VERBOSE)

        # Load the TRT engine
        with open(engine_path, "rb") as f:
            self.engine_data = f.read()
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(self.engine_data)

        # Get input and output bindings
        self.inputs, self.outputs, self.bindings, self.stream = [], [], [], cuda.Stream()
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append(HostDeviceMem(host_mem, device_mem)) 
            else:
                self.outputs.append(HostDeviceMem(host_mem, device_mem))

                # Create execution context
        self.context = self.engine.create_execution_context()

        self.profiler = MyProfiler(model, batch_size, precision)
        self.context.profiler = self.profiler

    def infer(self, input_data):
        # Set input data
        np.copyto(self.inputs[0].host, input_data.ravel())

        [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]

        # Run inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]

        self.stream.synchronize()

        self.profiler.get_data()

        return [out.host for out in self.outputs]
    
class MyProfiler(trt.IProfiler):
    def __init__(self, model, batch_size, precision):
        trt.IProfiler.__init__(self)
        self.profile_data = {}
        self.model, self.batch_size, self.precision = model, batch_size, precision

    def report_layer_time(self, layer_name, ms):
        self.profile_data[layer_name] = ms

    def get_data(self):
        total_time = 0
        for layer, ms in self.profile_data.items():
            print(f"Layer {layer} and time: {ms}")
            total_time+=ms
        
        metadata = {
        'Model Name': self.model,
        'Batch Size': self.batch_size,
        'Precision': self.precision,
        'Total Time': total_time
        }
        combined_data = {**metadata, **self.profile_data}

        df = pd.DataFrame(list(combined_data.items()), columns=['Layer Name', 'Execution Time (ms)'])
        df.to_csv(f"profiler_model_{self.model}_b{self.batch_size}_{self.precision}.csv", index=False)


def main(model, engine_path, batch_size, precision):

    model_input_map = {'yolov8':(3,640,640),
                       'x3d_m':(3,16,256,256),
                       'VSR':(10,3,144,180),
                       'scrfd-2.5g':(3,640,640),
                       'liteHRNet30':(3,256,192)}
    
    if model!="all": model_input_map = {model:model_input_map[model]}
    
    for model_, shape in model_input_map.items():
        trt_infer = TensorRTInfer(model_, engine_path, batch_size, precision)
        input_data = np.random.randn(batch_size, *shape).astype(np.float32)

        output = trt_infer.infer(input_data)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="TensorRT Inference with Batch Size")
    parser.add_argument("model_name", choices=["yolov8", "x3d_m", "VSR", "scrfd-2.5g", "liteHRNet30", "all"], help="Name of the model")
    parser.add_argument("--engine", type=str, default="", help="Path to the TRT engine file")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--precision", type=str, default="int8")
    args = parser.parse_args()
    main(args.model_name, args.engine, args.batch, args.precision)
