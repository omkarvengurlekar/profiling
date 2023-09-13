import subprocess
import argparse
from jtop import jtop
import multiprocessing
import time
import pandas as pd

parser = argparse.ArgumentParser(description="Model Options")

# parser.add_argument("model_name", choices=["yolov8", "x3d_m", "VSR", "scrfd-2.5g", "liteHRNet30", "all"], help="Name of the model")
parser.add_argument("--precision", choices=["int8", "fp16", "fp32"], help="model precision")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size")

args = parser.parse_args()

model_dict = {
    "yolov8": {
        "onnx_file": "/home/jetson/profiling/onnx_files/0_dynamic/yolo8n-seg-dynamic.onnx",
        "opt_shapes": f"images:{args.batch_size}x3x640x640",
        "precision":args.precision,
        "save_engine": f"./yolo8n_b{args.batch_size}_{args.precision}.engine"
    },
    # "x3d_m": {
    #     "onnx_file": "/home/jetson/profiling/onnx_files/0_dynamic/x3d_m-dynamic.onnx",
    #     "opt_shapes": f"input.1:{args.batch_size}x3x16x256x256",
    #     "precision":args.precision,
    #     "save_engine": f"./x3d_m_b{args.batch_size}_{args.precision}.engine"
    # },
    # "VSR": {
    #     "onnx_file": "/home/jetson/profiling/onnx_files/0_dynamic/RealBasicVSR-dynamic.onnx",
    #     "opt_shapes": f"input:{args.batch_size}x10x3x144x180",
    #     "precision":args.precision,
    #     "save_engine": f"./RealBasicVSR_b{args.batch_size}_{args.precision}.engine"
    # },
    # "scrfd-2.5g": {
    #     "onnx_file": "/home/jetson/profiling/onnx_files/0_dynamic/scrfd-2.5g-dynamic.onnx",
    #     "opt_shapes": f"input.1:{args.batch_size}x3x640x640",
    #     "precision":args.precision,
    #     "save_engine": f"./scrfd_b{args.batch_size}_{args.precision}.engine"
    # },
    # "liteHRNet30": {
    #     "onnx_file": "/home/jetson/profiling/onnx_files/0_dynamic/liteHRNet30-dynamic.onnx",
    #     "opt_shapes": f"input:{args.batch_size}x3x256x192",
    #     "precision":args.precision,
    #     "save_engine": f"./liteHRNet_b{args.batch_size}_{args.precision}.engine"
    # }
}

for k,v in model_dict.items():
    jetson = jtop()
    jetson.start()
    if args.precision=='fp32':
            subprocess.run(["trtexec", f"--onnx={v['onnx_file']}", f"--optShapes={v['opt_shapes']}", 
                    f"--saveEngine={v['save_engine']}"])
    else:
        subprocess.run(["trtexec", f"--onnx={v['onnx_file']}", f"--optShapes={v['opt_shapes']}", 
                        f"--{v['precision']}", f"--saveEngine={v['save_engine']}"])
    print(f"<---------------------------------- PROCESS FOR {k} IS DONE!! ---------------------------------->")

# def run_model(model_name, model_entry):
#     try:
#         if args.precision == 'fp32':
#             subprocess.run(["trtexec", f"--onnx={model_entry['onnx_file']}", f"--optShapes={model_entry['opt_shapes']}", 
#                             f"--saveEngine={model_entry['save_engine']}"])
#         else:
#             subprocess.run(["trtexec", f"--onnx={model_entry['onnx_file']}", f"--optShapes={model_entry['opt_shapes']}", 
#                             f"--{model_entry['precision']}", f"--saveEngine={model_entry['save_engine']}"])
#     except Exception as e:
#         print(f"Error running {model_name}: {str(e)}")

# def monitor_jetson_stats(model_name, model_entry):
#     jetson = jtop()
#     jetson.start()

#     stats_df = pd.DataFrame(columns=["Timestamp", "Model", "Stat Name", "Stat Value"])

#     try:
#         while True:
#             # Get and print jtop statistics
#             stats = jetson.stats()
#             print("Power Statistics:")
#             for stat_name, stat_value in stats.items():
#                 print(stat_name, stat_value)
            
#             # time.sleep(180)  # Adjust the sleep interval as needed

#     except KeyboardInterrupt:
#         pass
#     finally:
#         # Stop jtop when monitoring is done
#         jetson.stop()


# if __name__ == "__main__":
#     # Create a process for running the Jetson stats monitor
#     model_name, model_entry = model_dict.popitem()
#     jetson_stats_process = multiprocessing.Process(target=monitor_jetson_stats)
#     jetson_stats_process.start()

#     model_process = multiprocessing.Process(target=run_model, args=(model_name, model_entry))
#     model_process.start()

#     model_process.join()

#     jetson_stats_process.terminate()


