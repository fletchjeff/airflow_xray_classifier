import pendulum
from airflow.decorators import dag, task
from airflow.operators.dummy_operator import DummyOperator
from astronomer.providers.amazon.aws.sensors.s3 import S3KeySensorAsync
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from kubernetes.client import models as k8s

volume_mount = k8s.V1VolumeMount(
    name='data-storage', 
    mount_path='/efs', 
    sub_path=None, 
    read_only=False
)

volume = k8s.V1Volume(
    name='data-storage',
    persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(claim_name='efs-claim'),
)

container_resources = k8s.V1ResourceRequirements(
    limits={
        'nvidia.com/gpu': '1'
    }
)

kpo_defaults = {
        "cluster_context":'arn:aws:eks:eu-central-1:559345414282:cluster/jf-eks', 
        "namespace":"default",
        "labels":{"cluster": "jf-eks"},
        "get_logs":True,
        "is_delete_operator_pod":True,
        "in_cluster":False,
        "config_file":'/usr/local/airflow/include/kube_config',
        "volume_mounts":[volume_mount],
        "volumes":[volume]
}

@dag(
    schedule_interval=None,
    start_date=pendulum.datetime(2022, 7, 20, tz="UTC"),
    catchup=False,
    tags=['xray_classifier'],
)
def xray_classifier_dag():
    """

    """

    check_for_new_s3_data = S3KeySensorAsync(
        bucket_key="s3://jfletcher-datasets/xray_data_2_class.tgz",task_id="check_for_new_s3_data",aws_conn_id="my_aws_conn"
    )
    
    fetch_data_and_code = KubernetesPodOperator(
        task_id=f"fetch_data_and_code",
        name="fetch_data_and_code_pod",
        image="fletchjeffastro/xray_services:0.0.3",
        cmds=[ "/bin/bash", "-c", "--" , "cd /efs && curl -C - -O 'https://jfletcher-datasets.s3.eu-central-1.amazonaws.com/xray_data_2_class.tgz' && tar -xzf xray_data_2_class.tgz --skip-old-files && curl -O https://raw.githubusercontent.com/fletchjeff/airflow_xray_classifier/main/include/code/model_training.py"],
        **kpo_defaults
    )

    train_xray_model_on_gpu = KubernetesPodOperator(
        image="tensorflow/tensorflow:latest-gpu",
        name="airflow-xray-pod",
        cmds=["/bin/bash", "-c", "--", 
            "pip install mlflow && python /efs/model_training.py /efs/data {{dag_run.logical_date.strftime('%Y%m%d-%H%M%S')}}"],
        task_id="train_xray_model_on_gpu",
        startup_timeout_seconds=600,
        container_resources = container_resources,
        **kpo_defaults
    )

    @task()
    def update_ray():
        import ray
        from ray import serve

        ray.init("ray://a48892de9f3b44901a5fbf0d47d2f24a-100577308.eu-central-1.elb.amazonaws.com:10001")
        serve.start(detached=True,http_options={"host":"0.0.0.0"})

        @serve.deployment
        async def predictor(request):
            from PIL import Image
            import tensorflow as tf
            from io import BytesIO
            import numpy as np
            import base64
            image_data = await request.json()
            model = tf.keras.models.load_model('/data/models/20220719-125500/xray_classifier_model.h5')
            class_names = ['bacteria','normal','virus']
            im = Image.open(BytesIO(base64.b64decode(image_data['image'][22:])))
            im = im.resize((224,224),Image.ANTIALIAS)
            im = im.convert("RGB")
            im_array = tf.keras.preprocessing.image.img_to_array(im)
            im_array = np.expand_dims(im_array, axis=0)
            outputs = model.predict(im_array)
            output_classes = tf.math.argmax(outputs, axis=1)
            return {'result' : [class_names[c] for c in output_classes]}    


        if 'predictor' in serve.list_deployments().keys():
            predictor.delete()

        predictor.deploy()
        return "Done"
 
    ray_updated = update_ray()

    fetch_streamlit_code = KubernetesPodOperator(
        task_id=f"fetch_streamlit_code",
        cluster_context='arn:aws:eks:eu-central-1:559345414282:cluster/jf-eks', 
        namespace="default",
        name="fetch_data_and_code_pod",
        image="fletchjeffastro/xray_services:0.0.2",
        cmds=[ "/bin/bash", "-c", "--" , "cd /data && curl -C - -O 'https://jfletcher-datasets.s3.eu-central-1.amazonaws.com/xray_data_2_class.tgz' && tar -xzf xray_data_2_class.tgz --skip-old-files && curl -O https://raw.githubusercontent.com/fletchjeff/airflow_xray_classifier/main/include/code/xray_classifier_train_model.py"],
        labels={"example": "example", "airflow_kpo_in_cluster": "False"},
        get_logs=True,
        is_delete_operator_pod=True,
        in_cluster=False,
        config_file='/usr/local/airflow/include/kube_config',
        startup_timeout_seconds=240,
        volume_mounts=[volume_mount],
        volumes = [volume]
    )    

    my_dummy_task = DummyOperator(task_id="thankless_task")

    check_for_new_s3_data >> fetch_data_and_code >> train_xray_model_on_gpu >> ray_updated >> fetch_streamlit_code >> my_dummy_task

    # order_data = extract()
    # order_summary = transform(order_data)
    # load(order_summary["total_order_value"])
    
xray_classifier = xray_classifier_dag()