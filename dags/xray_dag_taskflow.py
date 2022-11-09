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
def xray_classifier_dag_taskflow():
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

    @task.kubernetes(
        image="fletchjeffastro/tfmlflow:0.0.4",
        name="train_xray_model_on_gpu_pod_tf",
        task_id="train_xray_model_on_gpu_tf",
        startup_timeout_seconds=600,
        container_resources=container_resources,
        env_vars={
            "STORAGE_PATH": f"{STORAGE_PATH}",
            "RUN_DATE":"{{dag_run.logical_date.strftime('%Y%m%d-%H%M%S')}}",
            "MLFLOW_SERVER":f"{MLFLOW_SERVER}"
            },
        **kpo_defaults
    )
    def train_xray_model_on_gpu_tf():
        import sys
        import pathlib
        import tensorflow as tf
        from tensorflow.keras.preprocessing import image_dataset_from_directory
        import os
        import mlflow

        # Show GPU
        print(tf.test.gpu_device_name())

        # Read and create datasets
        data_path_args = os.environ["STORAGE_PATH"]
        run_date = os.environ["RUN_DATE"]
        mlflow_server = os.environ["MLFLOW_SERVER"]
        data_dir = pathlib.Path("{}/data/train/".format(data_path_args))
        batch_size = 32
        img_height = 224
        img_width = 224
        img_size = (224, 224)

        train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(img_height, img_width)
        )

        train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size
        )

        validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size
        )
        #class_names = train_dataset.class_names
        val_batches = tf.data.experimental.cardinality(validation_dataset)
        test_dataset = validation_dataset.take(val_batches // 5)
        validation_dataset = validation_dataset.skip(val_batches // 5)

        # Configure the dataset for performance
        AUTOTUNE = tf.data.AUTOTUNE
        train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
        validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
        test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

        # Data augmentation
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_height, img_width,3)),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
            tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
        ])

        # Rescale pixel values
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        img_shape = img_size + (3,)

        # Load base model
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=img_shape,
            include_top=False,
            weights='imagenet'
        )

        image_batch, label_batch = next(iter(train_dataset))
        feature_batch = base_model(image_batch)
        print(feature_batch.shape)

        ## Feature extraction
        # Freeze the convolutional base
        base_model.trainable = False

        # Add a classification head
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        feature_batch_average = global_average_layer(feature_batch)
        prediction_layer = tf.keras.layers.Dense(1)
        prediction_batch = prediction_layer(feature_batch_average)

        # Build model
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = data_augmentation(inputs)
        x = preprocess_input(x)
        x = base_model(x, training=False)
        x = global_average_layer(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = prediction_layer(x)
        model = tf.keras.Model(inputs, outputs)

        # Compile the model
        base_learning_rate = 0.0001
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        # Train the model
        initial_epochs = 10
        loss0, accuracy0 = model.evaluate(validation_dataset)
        print("initial loss: {:.2f}".format(loss0))
        print("initial accuracy: {:.2f}".format(accuracy0))

        # Start mlflow
        mlflow.set_tracking_uri(f"http://{mlflow_server}:5000")
        mlflow.keras.autolog(registered_model_name=f"xray_model_train_{run_date}")
        if mlflow.get_experiment_by_name(f"run_{run_date}") == None:
            mlflow.create_experiment(f"run_{run_date}")
        mlflow.set_experiment(f"run_{run_date}")
        with mlflow.start_run() as run:

            history = model.fit(
                train_dataset,
                epochs=initial_epochs,
                validation_data=validation_dataset
            )

            # Record metrics
            acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']
            loss = history.history['loss']
            val_loss = history.history['val_loss']

        ## Fine tuning
        # Un-freeze the top layers of the model
        base_model.trainable = True
        fine_tune_at = 100
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable =  False

        # Re-compile the model
        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
            metrics=['accuracy']
        )

        # Continue training the model
        fine_tune_epochs = 15
        total_epochs =  initial_epochs + fine_tune_epochs

        with mlflow.start_run() as run:
            history_fine = model.fit(
                train_dataset,
                epochs=total_epochs,
                initial_epoch=history.epoch[-1],
                validation_data=validation_dataset
            )

            acc += history_fine.history['accuracy']
            val_acc += history_fine.history['val_accuracy']
            loss += history_fine.history['loss']
            val_loss += history_fine.history['val_loss']

        loss, accuracy = model.evaluate(test_dataset)
        print('Test accuracy :', accuracy)

        # Save the model
        os.makedirs('{}/models/{}/'.format(data_path_args,run_date))
        model.save("{}/models/{}/xray_classifier_model.h5".format(data_path_args,run_date))

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

    @task.kubernetes(
        task_id="fetch_updated_streamlit_code",
        name="fetch_updated_streamlit_code_pod",
        image="fletchjeffastro/xray_services:0.0.5",
        env_vars={
            "STORAGE_PATH": f"{STORAGE_PATH}",
            "CURRENT_RUN":"{{dag_run.logical_date.strftime('%Y%m%d-%H%M%S')}}",
            "RAY_SERVER":f"{RAY_SERVER}"
            },
        **kpo_defaults
    )
    def fetch_updated_streamlit_code():
        import os
        from smart_open import open
        storage_path = os.environ["STORAGE_PATH"]
        current_run = os.environ["CURRENT_RUN"]
        ray_server = os.environ["RAY_SERVER"]
        with open('https://raw.githubusercontent.com/fletchjeff/airflow_xray_classifier/main/include/code/streamlit_app.py') as streamlit_app_in:
            with open(f'{storage_path}/streamlit_app.py', 'w') as streamlit_app_out:
                for line in streamlit_app_in:
                    if line == "RAY_SERVER=''\n":
                        streamlit_app_out.write(f"RAY_SERVER='{ray_server}'\n")  
                    elif line == "STORAGE_PATH=''\n":
                        streamlit_app_out.write(f"STORAGE_PATH='{storage_path}'\n")
                    elif line == "CURRENT_RUN=''\n":
                        streamlit_app_out.write(f"CURRENT_RUN='{current_run}'\n")   
                    else:                            
                        streamlit_app_out.write(line)        

    email_on_completion = EmailOperator(
        task_id="email_on_completion",
        to='fletch.jeff@gmail.com',
        subject='Model Training Complete',
        html_content="Xray Classifier Model training completed for model run {{dag_run.logical_date.strftime('%Y%m%d-%H%M%S')}}",
    )

    check_for_new_s3_data >> fetch_data >> train_xray_model_on_gpu_tf() >> [
        update_ray("{{dag_run.logical_date.strftime('%Y%m%d-%H%M%S')}}"), 
        fetch_updated_streamlit_code()
    ] >> email_on_completion
       
xray_classifier_dag_taskflow()
