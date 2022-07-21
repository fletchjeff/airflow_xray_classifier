FROM quay.io/astronomer/astro-runtime:5.0.5

COPY --chown=astro:astro include/.aws /home/astro/.aws

ENV CLUSTER_CONTEXT=arn:aws:eks:eu-central-1:559345414282:cluster/jf-eks
ENV STORAGE_PATH=/efs
ENV RAY_SERVER=aec1a277e48474b80ad9a713faf74991-411668911.eu-central-1.elb.amazonaws.com
ENV MLFLOW_SERVER=a98c118fb63bc44ee92b85511d2bcfb2-2060505287.eu-central-1.elb.amazonaws.com
ENV PVC_CLAIM=efs-claim