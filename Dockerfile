FROM quay.io/astronomer/astro-runtime:5.0.6

COPY --chown=astro:astro include/.aws /home/astro/.aws
COPY --chown=astro:astro include/config /home/astro/config

ENV CLUSTER_CONTEXT=arn:aws:eks:us-east-2:016012822754:cluster/cosmicenergy-ml-demo \
    STORAGE_PATH=/efs \
    RAY_SERVER=a3ef591d9cc7146c3a01606677c4a758-966325034.us-east-2.elb.amazonaws.com \
    MLFLOW_SERVER=a480f098df66f4b859ee4acefc299eaa-1252271903.us-east-2.elb.amazonaws.com \
    PVC_NAME=efs-claim \
    AIRFLOW__EMAIL__EMAIL_BACKEND=airflow.providers.sendgrid.utils.emailer.send_email \
    AIRFLOW__EMAIL__EMAIL_CONN_ID=sendgrid_default \
    SENDGRID_MAIL_FROM=jeff.fletcher@astronomer.io