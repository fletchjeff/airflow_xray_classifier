import pendulum
from airflow.decorators import dag, task
from astronomer.providers.amazon.aws.sensors.s3 import S3KeySensorAsync
from airflow.operators.email import EmailOperator
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from kubernetes.client import models as k8s
import os

STORAGE_PATH = os.environ["STORAGE_PATH"]
PVC_NAME = os.environ["PVC_NAME"]
CLUSTER_CONTEXT = os.environ["CLUSTER_CONTEXT"]
RAY_SERVER = os.environ["RAY_SERVER"]
MLFLOW_SERVER = os.environ["MLFLOW_SERVER"]

volume_mount = k8s.V1VolumeMount(
    name='storage',
    mount_path=STORAGE_PATH,
    sub_path=None,
    read_only=False
)

volume = k8s.V1Volume(
    name='storage',
    persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(
        claim_name=PVC_NAME),
)

container_resources = k8s.V1ResourceRequirements(
    limits={
        'nvidia.com/gpu': '1'
    }
)

kpo_defaults = {
    "cluster_context": CLUSTER_CONTEXT,
    "namespace": "default",
    "labels": {"airflow_kpo_in_cluster": "False"},
    "get_logs": True,
    "is_delete_operator_pod": True,
    "in_cluster": False,
    "config_file": "/home/astro/config",
    "volume_mounts": [volume_mount],
    "volumes": [volume]
}


@dag(
    schedule_interval=None,
    start_date=pendulum.datetime(2022, 7, 20, tz="UTC"),
    catchup=False,
    tags=['xray_classifier'],
)
def xray_classifier_dag():
    """
    This DAG will will check if new xray image files exist, and then trigger the Xray Classifier model training process.
    """

    check_for_new_s3_data = S3KeySensorAsync(
        bucket_key="s3://jfletcher-datasets/xray_data_2_class.tgz",
        task_id="check_for_new_s3_data",
        aws_conn_id="my_aws_conn"
    )

    fetch_data = KubernetesPodOperator(
        task_id=f"fetch_data",
        name="fetch_data_pod",
        image="fletchjeffastro/xray_services:0.0.3",
        cmds=["/bin/bash", "-c", "--",
              f"cd {STORAGE_PATH} && curl -C - -O 'https://jfletcher-datasets.s3.eu-central-1.amazonaws.com/xray_data_2_class.tgz' && tar -xzf xray_data_2_class.tgz --skip-old-files"],
        **kpo_defaults
    )

    fetch_code = KubernetesPodOperator(
        task_id=f"fetch_code",
        name="fetch_code_pod",
        image="fletchjeffastro/xray_services:0.0.3",
        cmds=["/bin/bash", "-c", "--",
              f"cd {STORAGE_PATH} && curl -O https://raw.githubusercontent.com/fletchjeff/airflow_xray_classifier/main/include/code/model_training.py"],
        **kpo_defaults
    )

    train_xray_model_on_gpu = KubernetesPodOperator(
        image="fletchjeffastro/tfmlflow:0.0.4",
        name="train_xray_model_on_gpu_pod",
        cmds=["/bin/bash", "-c", "--",
              "python {}/model_training.py {} {{{{dag_run.logical_date.strftime('%Y%m%d-%H%M%S')}}}} {}".format(STORAGE_PATH, STORAGE_PATH, MLFLOW_SERVER)],
        task_id="train_xray_model_on_gpu",
        startup_timeout_seconds=600,
        container_resources=container_resources,
        **kpo_defaults
    )

    @task()
    def update_ray(current_run):
        import ray
        from ray import serve

        ray.init(f"ray://{RAY_SERVER}:10001")
        serve.start(detached=True, http_options={"host": "0.0.0.0"})

        @serve.deployment
        async def predictor(request):
            from PIL import Image
            import tensorflow as tf
            from io import BytesIO
            import numpy as np
            import base64
            image_data = await request.json()
            model = tf.keras.models.load_model(
                "{}/models/{}/xray_classifier_model.h5".format(STORAGE_PATH, current_run))
            im = Image.open(
                BytesIO(base64.b64decode(image_data['image'][22:])))
            im = im.resize((224, 224), Image.ANTIALIAS)
            im = im.convert("RGB")
            im_array = tf.keras.preprocessing.image.img_to_array(im)
            im_array = np.expand_dims(im_array, axis=0)
            outputs = model.predict(im_array)
            class_names = ['normal', 'pneumonia']
            return {'result': class_names[tf.where(tf.nn.sigmoid(outputs) < 0.5, 0, 1).numpy()[0][0]]}

        if 'predictor' in serve.list_deployments().keys():
            predictor.delete()

        predictor.deploy()
        return "Done"

    ray_updated = update_ray(
        "{{dag_run.logical_date.strftime('%Y%m%d-%H%M%S')}}")

    fetch_updated_streamlit_code = KubernetesPodOperator(
        task_id=f"fetch_updated_streamlit_code",
        name="fetch_updated_streamlit_code_pod",
        image="fletchjeffastro/xray_services:0.0.3",
        cmds=["/bin/bash", "-c", "--", "cd {} && curl -O https://raw.githubusercontent.com/fletchjeff/airflow_xray_classifier/main/include/code/streamlit_app.py && sed -i \"s/RAY_SERVER=''/RAY_SERVER='{}'/\" streamlit_app.py && sed -i \"s#STORAGE_PATH=''#STORAGE_PATH='{}'#\" streamlit_app.py && sed -i \"s/CURRENT_RUN=''/CURRENT_RUN='{{{{dag_run.logical_date.strftime('%Y%m%d-%H%M%S')}}}}'\"/ streamlit_app.py".format(
            STORAGE_PATH, RAY_SERVER, STORAGE_PATH)],
        **kpo_defaults
    )

    email_on_completion = EmailOperator(
        task_id="email_on_completion",
        to='fletch.jeff@gmail.com',
        subject='Model Training Complete',
        html_content="Xray Classifier Model training completed for model run {{dag_run.logical_date.strftime('%Y%m%d-%H%M%S')}}",
    )

    check_for_new_s3_data >> [fetch_data, fetch_code] >> train_xray_model_on_gpu >> [
        ray_updated, fetch_updated_streamlit_code] >> email_on_completion


xray_classifier = xray_classifier_dag()
