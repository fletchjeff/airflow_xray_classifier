from airflow import DAG
from datetime import datetime, timedelta
from airflow.contrib.operators.kubernetes_pod_operator import KubernetesPodOperator
from airflow import configuration as conf
from airflow.models import base
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python import task
from airflow.operators.python_operator import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount
#from kubernetes.client import models as k8s
from airflow.providers.ssh.operators.ssh import SSHOperator
from airflow.providers.ssh.hooks.ssh import SSHHook
from airflow.providers.amazon.aws.operators.ec2_start_instance import EC2StartInstanceOperator
from airflow.providers.amazon.aws.operators.ec2_stop_instance import EC2StopInstanceOperator
from airflow.contrib.operators.sftp_operator import SFTPOperator
import boto3
#from docker.types import Mount
import sys
import os
#from airflow.providers.sftp.operators.sftp import SFTPOperator
#from airflow.providers.ssh.operators.ssh import SSHOperator

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2021, 12, 29),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
    'catchup_by_default':False
}

namespace = conf.get('kubernetes', 'NAMESPACE')

if namespace =='default':
    config_file = '/usr/local/airflow/include/config/kubectl/config.local'#/home/jeff/airflow/include/.kube/config'
    in_cluster=False
else:
    in_cluster=True
    config_file=None

compute_resources = {
    'request_cpu': '6000m',
    'request_memory': '32Gi',
    'limit_cpu': '6000m',
    'limit_memory': '32Gi',
    'limit_gpu': '1'
  }

# volume_mount = k8s.V1VolumeMount(
#     name='task-pv-storage', mount_path='/tf', sub_path=None, read_only=False
# )

# volume = k8s.V1Volume(
#     name='task-pv-storage',
#     persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(claim_name='task-pv-claim'),
# )

def _build_bentoml_image(run_path,**context):
    sys.path.insert(-1, "/usr/local/airflow/include/code")
    import tensorflow as tf
    from tensorflow_xray import TensorflowXray
    model = tf.keras.models.load_model('/usr/local/airflow/xray/models/{}/xray_classifier_model.h5'.format(run_path))
    bento_svc = TensorflowXray()
    bento_svc.pack("model", model)
    saved_path = bento_svc.save()
    return saved_path

def _get_public_ip(**context):
    client = boto3.client('ec2')
    public_ip = client.describe_instances(Filters=[{
        'Name':'instance-type', 
        'Values': ['g4dn.2xlarge']}])['Reservations'][0]['Instances'][0]['PublicDnsName']
    public_ip = "192.168.1.100"
    context["task_instance"].xcom_push(key="public_ip", value=public_ip)
    print(public_ip)

with DAG('xray_classifier_dag',
          schedule_interval=None,
          default_args=default_args) as dag:



    fetch_data = BashOperator(
        task_id = 'fetch_data',
        bash_command='curl -o /usr/local/airflow/xray/xray_data.tgz https://jfletcher-datasets.s3.eu-central-1.amazonaws.com/xray_data.tgz'
        # ssh_hook = SSHHook(remote_host = "192.168.1.100",username='jeff', key_file='/usr/local/airflow/include/config/ssh/jf-key-astro-us-east-1.pem'),
        #remote_host="{{task_instance.xcom_pull(task_ids='get_public_ip', key='public_ip')}}",
    )

    unpack_data = BashOperator(
        task_id = 'unpack_data',
        bash_command='tar -xzf /usr/local/airflow/xray/xray_data.tgz -C /usr/local/airflow/xray/',
        # ssh_hook = SSHHook(remote_host = "192.168.1.100",username='jeff', key_file='/usr/local/airflow/include/config/ssh/jf-key-astro-us-east-1.pem'),
        #ssh_hook = SSHHook(remote_host = "0.0.0.0",username='ubuntu', key_file='/usr/local/airflow/include/config/ssh/jf-key-astro-us-east-1.pem'),
        #remote_host="{{task_instance.xcom_pull(task_ids='get_public_ip', key='public_ip')}}"
    )

    # push_latest_code = SFTPOperator(
    #     task_id="push_latest_code",
    #     #ssh_conn_id="ssh_default",
    #     ssh_hook = SSHHook(remote_host = "192.168.1.100",username='jeff', key_file='/usr/local/airflow/include/config/ssh/jf-key-astro-us-east-1.pem'),
    #     local_filepath="/usr/local/airflow/include/code/xray_classifier_train_model.py",
    #     remote_filepath="/data/xray/xray_classifier_train_model.py",
    #     operation="put",
    #     create_intermediate_dirs=True
    # )

    # xray_model_train = KubernetesPodOperator(
    #     namespace=namespace,
    #     image="tensorflow/tensorflow:latest-gpu",
    #     name="airflow-tf-test-pod",
    #     cmds=["python","/tf/xray_classifier_train_model.py","/tf",'{{dag_run.logical_date.strftime("%Y%m%d-%H%M%S")}}'],
    #     task_id="xray_model_train",
    #     in_cluster=in_cluster,
    #     cluster_context='kubernetes-admin@kubernetes',
    #     config_file=config_file,
    #     is_delete_operator_pod=True,
    #     get_logs=True,
    #     resources=compute_resources,
    #     volume_mounts=[volume_mount],
    #     volumes = [volume]
    # )

    xray_model_train = DockerOperator(
        task_id='xray_model_train',  
        image='tensorflow/tensorflow:latest-gpu',
        api_version='auto',
        auto_remove=True,
        command=[
            'python', 
            '/xray/include/code/xray_classifier_train_model.py',
            '/xray',
            '{{dag_run.logical_date.strftime("%Y%m%d-%H%M%S")}}'
            ],
        docker_url="tcp://172.17.0.1:2375",#unix://var/run/docker.sock",
        network_mode="bridge",
        mounts=[
            Mount(source="{}".format(os.environ["ASTRO_HOST_DIR"]), target="/xray", type="bind")
            ]
        )

    # get_latest_model = SFTPOperator(
    #     task_id="get_latest_model",
    #     #ssh_conn_id="ssh_default",
    #     ssh_hook = SSHHook(remote_host = "192.168.1.100",username='jeff', key_file='/usr/local/airflow/include/config/ssh/jf-key-astro-us-east-1.pem'),
    #     local_filepath="/usr/local/airflow/training_runs/models/{{dag_run.logical_date.strftime('%Y%m%d-%H%M%S')}}/xray_classifier_model.h5",
    #     remote_filepath="/data/xray/models/{{dag_run.logical_date.strftime('%Y%m%d-%H%M%S')}}/xray_classifier_model.h5",
    #     operation="get",
    #     create_intermediate_dirs=True
    # )

    # fetch_training_run = BashOperator(
    #     task_id = 'fetch_training_run',
    #     bash_command="echo 'sftp something'"
    # )

    build_bentoml_image = PythonOperator(
        task_id = 'build_bentoml_image',
        python_callable= _build_bentoml_image,
        op_kwargs={'run_path': '{{dag_run.logical_date.strftime("%Y%m%d-%H%M%S")}}'},
    )

    create_bentoml_container = BashOperator(
        task_id = 'create_bentoml_container',
        bash_command='export LATEST_RUN=`ls -Art /home/astro/bentoml/repository/TensorflowXray/ | tail -n 1` && docker build -t tensorflow-xray:$LATEST_RUN /home/astro/bentoml/repository/TensorflowXray/$LATEST_RUN/ -f /home/astro/bentoml/repository/TensorflowXray/$LATEST_RUN/Dockerfile'
    )

    restart_bentoml_server = BashOperator(
        task_id='restart_bentoml_server',
        bash_command="export LATEST_RUN=`ls -Art /home/astro/bentoml/repository/TensorflowXray/ | tail -n 1` &&docker kill `(docker ps | awk ' /tensorflow-xray/ { print $1 }')` && docker run -d -p 5000:5000 tensorflow-xray:$LATEST_RUN"
    )

    fetch_data >> unpack_data >> xray_model_train >> build_bentoml_image >> create_bentoml_container >> restart_bentoml_server
