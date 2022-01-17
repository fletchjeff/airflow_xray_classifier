from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python import task
from airflow.operators.python_operator import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount
import sys, os

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

def _build_bentoml_image(run_path,**context):
    sys.path.insert(-1, "/usr/local/airflow/include/code")
    import tensorflow as tf
    from tensorflow_xray import TensorflowXray
    model = tf.keras.models.load_model('/usr/local/airflow/xray/models/{}/xray_classifier_model.h5'.format(run_path))
    bento_svc = TensorflowXray()
    bento_svc.pack("model", model)
    saved_path = bento_svc.save()
    return saved_path


with DAG('xray_classifier_dag',
          schedule_interval=None,
          default_args=default_args) as dag:

    fetch_data = BashOperator(
        task_id = 'fetch_data',
        bash_command='curl -o /usr/local/airflow/xray/xray_data.tgz https://jfletcher-datasets.s3.eu-central-1.amazonaws.com/xray_data.tgz'
    )

    unpack_data = BashOperator(
        task_id = 'unpack_data',
        bash_command='tar --skip-old-files -xzf /usr/local/airflow/xray/xray_data.tgz -C /usr/local/airflow/xray/'
    )

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
        network_mode="bridge",
        mounts=[
            Mount(source="{}".format(os.environ["ASTRO_HOST_DIR"]), target="/xray", type="bind")
            ]
        )

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
        bash_command="scripts/restart_bentoml_service.sh"
    )

    fetch_data >> unpack_data >> xray_model_train >> build_bentoml_image >> create_bentoml_container >> restart_bentoml_server