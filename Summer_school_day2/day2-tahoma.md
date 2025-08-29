# Day 2 LLM session

### Starting Tahoma OnDemand

1. Log into Tahoma OnDemand: https://tahoma-ondemand.emsl.pnnl.gov/ and Launch 'Jupyter Lab Workshop' on Tahoma OnDemand.
2. Select 'Run on AI nodes', enter 3 under 'Number of hours' and click 'Launch'
3. Then you should see Sessions screen. Here's where your job is in the queue.
4. Click 'Connect to Jupyter' (you may have to wait for a minute for this to show up). This will open JupyterLab.
5. Open Terminal inside JupyterLab and enter the following

```
eval "$(/tahoma/aisummerschool/miniconda3/bin/conda shell.bash hook)"
conda activate aisummerschool_day2
python -m ipykernel install --user --name aisummerschool_day2 --display-name aisummerschool_day2
```

6. Now, reload the webpage on your browser
7. You should see `aisummerschool_day2` as one of the 'Notebook' options when you press the blue '+' button in JupyterLab. This will be the kernel needed run today's notebooks.

### Getting notebooks and data for day 2 LLM session
Open terminal inside JupyterLab

```
cp -r /tahoma/aisummerschool/day2 $HOME/.
```
This will copy the Jupyter notebooks for day 2 LLM session to your home under a directory named 'day2'. Make sure when you are running today's notebooks the kernel loaded on the top right is `aisummerschool_day2`


### Starting an Ollama server for running LLMs
Open another terminal inside JupyterLab to run a Ollama server

```
apptainer run --nv --writable-tmpfs --env OLLAMA_HOST=127.0.0.1:11434 --env OLLAMA_MODELS=/cluster/ollama_models --env OLLAMA_KEEP_ALIVE=-1 library://ondemand/ollama:development serve

```

Now, you have the LLM server and the notebooks/data needed for today's session.