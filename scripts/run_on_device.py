import subprocess
import os
import re
import sys
import time

model_directory = sys.argv[1]
all_output = open("output.txt","w")

for filename in os.listdir(model_directory):
    model = os.path.join(model_directory,filename)
    try:
        output = subprocess.check_output(["edgetpu_compiler",model])
        onchip = "0.0M"
        offchip = "0.0M"
        output_file = ""
        cpu_runtime = "0"
        tpu_runtime = "0"
        
        m = re.match(".*used for caching model parameters: (\d*\.\d*\w)iB.*",str(output))
        if m:
            onchip = m.group(1)
        m = re.match(".*used for streaming uncached model parameters: (\d*\.\d*\w)iB.*",str(output))
        if m:
            offchip = m.group(1)
        m = re.match(".*Output\s+model\:\s+(.+\.tflite).*",str(output))
        if m:
            output_file = m.group(1)


        os.system("rm -f runtime.txt")
        # Run on CPU
        os.system("mdt push " + model)
        os.system("timeout 10m mdt exec python3 run_model.py --model=" + filename + " --device=CPU --runtype=Time")
        os.system("mdt exec rm -f " + filename)
        os.system("mdt pull runtime.txt .")
        
        if os.path.exists("runtime.txt"):
            with open('runtime.txt') as f:
                cpu_runtime = next(f).strip()
        else:
            cpu_runtime = "Timeout"

        os.system("rm -f runtime.txt")
            
        os.system("mdt push " + output_file)
        os.system("timeout 10m mdt exec python3 run_model.py --model=" +output_file + " --device=TPU --runtype=Time")
        os.system("mdt exec rm -f " + output_file)
        os.system("mdt pull runtime.txt .")

        if os.path.exists("runtime.txt"):
            with open('runtime.txt') as f:
                tpu_runtime = next(f).strip()
        else:
            tpu_runtime = "Timeout"
        
        print(filename + "," + cpu_runtime + "," + tpu_runtime + "," + onchip + "," + offchip)
        all_output.write(filename + "," + cpu_runtime + "," + tpu_runtime + "," + onchip + "," + offchip + "\n")
    except subprocess.CalledProcessError as e:
        print("Error!")
        print(e.output)

all_output.close()
