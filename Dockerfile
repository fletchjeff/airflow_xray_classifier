FROM quay.io/astronomer/astro-runtime:8.5.0

COPY --chown=astro:astro include/.aws /home/astro/.aws
COPY --chown=astro:astro include/config /home/astro/config

ENV CLUSTER_CONTEXT=arn:aws:eks:us-east-1:285860431378:cluster/ce-ml \
    STORAGE_PATH=/efs \
    RAY_SERVER=a2b5f8ececf5d43a8aa5317f9044ef61-1667943142.us-east-1.elb.amazonaws.com \
    MLFLOW_SERVER=add2f2378d8604988b8bd8fdcb5afa93-609300651.us-east-1.elb.amazonaws.com \
    PVC_NAME=efs-claim \
    AIRFLOW__EMAIL__EMAIL_BACKEND=airflow.providers.sendgrid.utils.emailer.send_email \
    AIRFLOW__EMAIL__EMAIL_CONN_ID=sendgrid_default \
    SENDGRID_MAIL_FROM=jeff.fletcher@astronomer.io