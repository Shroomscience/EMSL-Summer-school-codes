# AI Summer School 2025

# Getting Started on Tahoma

- About Tahoma: https://www.emsl.pnnl.gov/MSC/UserGuide/tahoma/tahoma_overview.html
    - 184 Intel Cascade Lake nodes
    - nearly 100TB memory
    - 10 PB of global storage in a BeeGFS file system, and an aggregate of 536TB of local disk. 
    - Peak performance is 1015 Teraflops.
    - 24 ML/AI nodes each with 2 NVIDIA Tesla V100 32GB GPUs (48 GPUs total). The GPUs are attached via a PCI-express bus (PCIe x16) to the Xeon© cores and their 1.5TB of memory.
    - We will use the ML/AI nodes for some of our instruction sessions starting day 1 afternoon

- Logging in using ssh: https://www.emsl.pnnl.gov/MSC/UserGuide/tahoma/getting_started.html

    - ssh –X your_userid@tahoma.emsl.pnnl.gov

- For credentials follow the instructions in the packages
    - You'll need the token generated number
    - PIN 
    - Password


- RSA token
    - Please do not lose or mix tokens. Tokens are tied to your accounts
    - Please return your tokens Thursday

# EMSL Tahoma OnDemand
EMSL Ondemand provides a web portal to the system where a user can manage files, jobs, run GUI applications, and connect to an interactive shell.

Link to Tahoma OnDemand: https://tahoma-ondemand.emsl.pnnl.gov/

Details of Tahoma OnDemand can be found here: https://www.emsl.pnnl.gov/MSC/UserGuide/ood/overview.html

# Day 1 ML session

1. Launch Jupyterlab on Tahoma OOD
2. Select 'Run on AI nodes', enter 2 under 'Number of hours', enter 1 under 'Number of Nodes', and click 'Launch'
3. Then you should see Sessions screen. Here's where your job is in the queue.
4. Click 'Connect to Jupyter'. This open JupyterLab.
5. Open Terminal inside JupyterLab and enter the following

```
eval "$(/tahoma/aisummerschool/miniconda3/bin/conda shell.bash hook)"
conda activate aisummerschool_day1
python -m ipykernel install --user --name aisummerschool_day1 --display-name aisummerschool_day1
```

6. Reload the webpage
7. Load the kernel `aisummerschool_day1`
8. The kernel should have the packages needed for running today's notebooks.


Open terminal inside jupyterlab

	cp -r /tahoma/aisummerschool/day1 .