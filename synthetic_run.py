### Import Packages

import sys
import json
import os
import datetime
import time
import logging
import subprocess
import requests
import uuid
import re

from tqdm import tqdm
from slugify import slugify
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

### Configure Request Adapter

retry_strategy = Retry(total=0)
adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=1, pool_maxsize=1)
http = requests.Session()
http.mount("http://", adapter)

### Generate Benchmark Run UID

uid = datetime.datetime.now().strftime("%m-%d-%H:%M:%S")

### Declare Constants

NUM_REQUESTS = int(sys.argv[1])
REQUEST_DELAY = int(sys.argv[2]) # in ms
test = sys.argv[4]
filename = "data/"
if sys.argv[3] == "pypy": 
  filename += "python-" + test + ".csv"
else:
   filename += "java-" + test + ".csv"
BENCHMARKS = sys.argv[5:]
# STRATEGIES = [
#     "cold",
#     "fixed&request_to_checkpoint=1",
#     "request_centric&max_capacity=12"
# ]
STRATEGIES = [
    "dynamic_system",
    "request_centric&max_capacity=12"
]
# RATES = [20, 4, 1]
RATES = [20]

### Configure Logging Handlers

log_directory = "logs"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

data_directory = "data"
if not os.path.exists(data_directory):
    os.makedirs(data_directory)

logger = logging.getLogger()
if sys.argv[3] == "pypy": 
  logging.basicConfig(filename="logs/python-" + test + ".log", format='%(asctime)s %(filename)s: %(message)s', filemode='a+')
else:
   logging.basicConfig(filename="logs/java-" + test + ".log", format='%(asctime)s %(filename)s: %(message)s', filemode='a+')
logger.setLevel(logging.DEBUG)

def check_namespace_pods():
    namespace = "openfaas-fn"
    cmd = f"kubectl get pods -n {namespace} --no-headers | wc -l"
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return int(result.stdout.strip())

# user="pronghornae"
user="potatocabage"

def get_pool_sizes_from_logs(benchmark, strategy, rate, total_requests):
    """Parse container logs to extract pool sizes at each request number."""
    cmd = f"kubectl logs -n openfaas-fn -l faas_function={benchmark} --tail=-1"
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Save raw logs for later analysis
    log_path = os.path.join(log_directory, f"{benchmark}_{strategy}_{rate}_container.log")
    with open(log_path, "w") as f:
        f.write(result.stdout)

    # Parse pool sizes from log lines like "Pool (req=5, size=3):" or "Current Pool (req=5, size=3):"
    pool_pattern = re.compile(r"(?:Current )?Pool \(req=(\d+), size=(\d+)\):")
    entries = []
    for line in result.stdout.splitlines():
        match = pool_pattern.search(line)
        if match:
            req_num = int(match.group(1))
            pool_size = int(match.group(2))
            entries.append((req_num, pool_size))

    if not entries:
        logger.warning(f"No pool size entries found in logs for {benchmark}")
        return []

    # Deduplicate: keep the last entry for each request number (post-checkpoint state)
    by_req = {}
    for req_num, pool_size in entries:
        by_req[req_num] = pool_size

    sorted_reqs = sorted(by_req.keys())

    # Build pool_sizes_list weighted by request duration
    pool_sizes_list = []
    for i, req in enumerate(sorted_reqs):
        if i + 1 < len(sorted_reqs):
            duration = sorted_reqs[i + 1] - req
        else:
            duration = max(1, total_requests - req)
        pool_sizes_list.extend([by_req[req]] * duration)

    return pool_sizes_list

# Load existing pool sizes or create a new dict
pool_sizes_file = "data/pool_sizes.json"
if os.path.exists(pool_sizes_file):
    try:
        with open(pool_sizes_file, "r") as f:
            all_pool_sizes = json.load(f)
    except:
        all_pool_sizes = {}
else:
    all_pool_sizes = {}

with open(filename, "a") as output_file:
   for benchmark in BENCHMARKS:
      if benchmark not in all_pool_sizes:
          all_pool_sizes[benchmark] = {}
      for strategy in STRATEGIES:
          if strategy not in all_pool_sizes[benchmark]:
              all_pool_sizes[benchmark][strategy] = {}
          for rate in RATES:
  
                logger.info("Deploying %s function", benchmark)
                deploy_cmd = f"faas-cli deploy --image={user}/{benchmark} --name={benchmark} --env=ENV={strategy},true,{rate}"
                deploy_proc = subprocess.run(deploy_cmd.split(" "), capture_output=True)
                logger.debug("Deploy command stdout: %s", deploy_proc.stdout.decode("UTF-8"))
                logger.debug("Deploy command stderr: %s", deploy_proc.stderr.decode("UTF-8"))
                
                time.sleep(5)

                logger.info("Executing strategy: %s for benchmark %s with rate %s and mutability %s", strategy, benchmark, rate, "1")
                
                nums = re.compile(r"\d+ ms")
                url = f"http://127.0.0.1:8080/function/{benchmark}?mutability=1"
                for index, request in tqdm(enumerate(range(NUM_REQUESTS))):
                  for retry in range(3):
                    retries = 0
                    try:
                      start_time = datetime.datetime.now()
                      response = http.get(url)
                      end_time = datetime.datetime.now()
                      search = nums.search(response.text)
                      if search is None: # PyPy benchmark
                        body = json.loads(response.text)
                        server_side = body.get('server_time')
                        overhead = body.get('client_overhead', 0)
                      else: # Java benchmark
                        server_side = int(search.group(0).split(" ")[0])
                        overhead = 0
                      client_side = (end_time - start_time) / datetime.timedelta(microseconds=1)
                      logger.debug("%s %s %s", server_side, overhead, client_side)
                      
                      output_file.write(f"{index+ 1},{benchmark},1,{strategy},{rate},{client_side},{server_side},{overhead}\n")
                      time.sleep(REQUEST_DELAY/1000)
                    except:
                      retries += 1
                      time.sleep(min(retries ** 2, 10))
                    else:
                      break
                output_file.flush()

                # Get pool sizes from container logs
                pool_sizes_list = get_pool_sizes_from_logs(benchmark, strategy, rate, NUM_REQUESTS)

                # Calculate and record pool size stats
                if len(pool_sizes_list) > 0:
                    avg_size = sum(pool_sizes_list) / len(pool_sizes_list)
                    max_size = max(pool_sizes_list)
                else:
                    avg_size = 0.0
                    max_size = 0

                all_pool_sizes[benchmark][strategy][str(rate)] = {
                    "average": avg_size,
                    "max": max_size
                }

                # Save dynamically to JSON so it is preserved even if the script crashes
                with open(pool_sizes_file, "w") as f:
                    json.dump(all_pool_sizes, f, indent=4)

                logger.info(f"Pool sizes for {benchmark} {strategy} {rate}: avg={avg_size:.2f}, max={max_size}")
                logger.info("Completed strategy: %s for benchmark %s with mutability %s", strategy, benchmark, "1")
                clean_cmd = f"faas-cli remove {benchmark}"
                clean_proc = subprocess.run(clean_cmd.split(" "), capture_output=True)
                logger.debug("Clean command stdout: %s", clean_proc.stdout.decode("UTF-8"))
                logger.debug("Clean command stderr: %s", clean_proc.stderr.decode("UTF-8"))

                # Update the delete and redeploy commands
                delete_cmd = f"kubectl delete -f {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database/pod.yaml')}"
                delete_proc = subprocess.run(delete_cmd.split(" "), capture_output=True)
                logger.debug("Delete command stdout: %s", delete_proc.stdout.decode("UTF-8"))
                logger.debug("Delete command stderr: %s", delete_proc.stderr.decode("UTF-8"))

                redeploy_cmd = f"kubectl apply -f {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database/pod.yaml')}"
                redeploy_proc = subprocess.run(redeploy_cmd.split(" "), capture_output=True)
                logger.debug("Redeploy command stdout: %s", redeploy_proc.stdout.decode("UTF-8"))
                logger.debug("Redeploy command stderr: %s", redeploy_proc.stderr.decode("UTF-8"))

                minio_cleanup_cmd = f"mc rb myminio/checkpoints --force"
                minio_cleanup_proc = subprocess.run(minio_cleanup_cmd.split(" "), capture_output=True)
                logger.debug("MinIO cleanup command stdout: %s", minio_cleanup_proc.stdout.decode("UTF-8"))
                logger.debug("MinIO cleanup command stderr: %s", minio_cleanup_proc.stderr.decode("UTF-8"))

                # Check if there are pods in the openfaas-fn namespace
                while check_namespace_pods() > 0:
                    print("Waiting for pods to terminate...")
                    time.sleep(10)