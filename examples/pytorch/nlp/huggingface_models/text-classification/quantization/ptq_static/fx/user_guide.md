# User Guide for Launching Distributed Tuning on Kubernetes (K8s)

In this user guide, we will walk you through the steps required to set up and launch a Distributed Tuning task on a GCP Kubernetes cluster. This guide assumes that you have access to a Kubernetes cluster, and we recommend using Google Cloud Platform (GCP) for setting up your cluster.

## Table of Contents

- Setting Up a Kubernetes Cluster on GCP
- The Main Difference
- Preparing Task Files
- Deploying and Monitoring the Task

### Setting Up a Kubernetes Cluster on GCP

Before you can launch a distributed tuning task, you need a Kubernetes cluster. Follow these steps to set up a Kubernetes cluster on Google Cloud Platform:

1. Access the GCP console.
2. Navigate to Kubernetes Engine and create a new cluster.
3. There are two cluster types to choose from: [Autopilot mode](https://cloud.google.com/kubernetes-engine/docs/concepts/choose-cluster-mode#why-autopilot) and [Standard Mode](https://cloud.google.com/kubernetes-engine/docs/concepts/choose-cluster-mode#why-standard). We recommend using `Autopilot` mode to simplify cluster management and focus on the task at hand.
4. (Optional for `Standard` mode) Configure the cluster according to your requirements, including the number of nodes and machine types.
5. Once the cluster is created, configure `kubectl` to interact with your cluster by running `kubectl cluster-info`.



### The Main Difference

When running distributed tasks on a Kubernetes (K8s) cluster, explicit hostname specification is not required. Instead, we can utilize the [Kubeflow Training Operator](https://github.com/kubeflow/training-operator) to manage the resources. We can allocate resources for tasks by specifying the **number of replicas** and **resource limitations (the number of CPUs and memory size)**. The Kubeflow Training Operator will allocate resources (Pods) for new tasks and recycle the Pods of completed tasks.



### Preparing Task Files

Before you can deploy your distributed tuning task, you need to prepare the necessary files and configurations. Follow these steps:

- Ensure that your task is ready and tested on your local development environment.

    ```shell
    # test it on local machine
    mpirun -np 3 python -u ./run_glue.py --model_name_or_path distilbert_mrpc --task_name mrpc --do_eval  --max_seq_length 128 --per_device_eval_batch_size 16 --no_cuda --output_dir ./int8_model_dir --tune --overwrite_output_dir
    ```



- Create a Dockerfile that describes how to build a Docker image for your task. Ensure it includes all dependencies and code required to run your tuning job. Please refer to the `./Dockerfile` for details.

- Prepare the Kubernetes YAML files needed for deploying your task on the cluster. This includes defining custom resources for the training operators, specifying resource requirements, and configuring any necessary environment variables.

  ```yaml
  apiVersion: kubeflow.org/v1
  kind: MPIJob
  metadata:
    name: inc-distributed-tuning-v41-1
  spec:
    slotsPerWorker: 1
    runPolicy:
      cleanPodPolicy: Running
    mpiReplicaSpecs:
      Launcher:
        replicas: 1
        template:
          spec:
            containers:
            - image: register_name/image_name:tag
              name: mpi-launcher
              command:
              - mpirun
              args:
              ... # the command line to run task
      Worker:
        replicas: 3
        template:
          spec:
            containers:
            - image: register_name/image_name:tag
              name: mpi-worker
  ```

  

### Deploying and Monitoring the Task

Now that you have your Kubernetes cluster set up and your task files prepared, it's time to deploy and monitor your Distributed Tuning task. Follow these steps:

- Apply the Kubernetes custom resources (training operators) to your cluster.

  ```shell
  # deploy the operator
  kubectl apply -k "github.com/kubeflow/training-operator/manifests/overlays/standalone?ref=v1.5.0"
  # check the custom resource is installed or not
  kubectl get crd
  ```

- Build your task's Docker image using the Dockerfile you created and push it to an image registry accessible from your Kubernetes cluster.

  ```shell
  # build docker 
  bash build_docker.sh
  # push docker image 
  docker push registry_name/image_name:tag
  ```

- Create a Kubernetes Job or Deployment that uses the Docker image you built to run your tuning task.

  ```shell
  kubectl create -f ./inc-dist-v2.yaml
  ```

- Query the job's status

  ```shell
  user@cloudshell:~/.../ptq_static/fx$ kubectl  get pods
  NAME                                    READY   STATUS    RESTARTS   AGE
  inc-distributed-tuning-v41-1-launcher   1/1     Running   0          22m
  inc-distributed-tuning-v41-1-worker-0   1/1     Running   0          22m
  inc-distributed-tuning-v41-1-worker-1   1/1     Running   0          22m
  inc-distributed-tuning-v41-1-worker-2   1/1     Running   0          22m
  ```

- Once the job is completed, query the finished job logs using `kubectl logs <pod-name>` to access any logs or output generated during the tuning process.

Congratulations! You've successfully launched Distributed Tuning on your Kubernetes cluster. You can now analyze the results and make any necessary adjustments to your tuning task.



EOD