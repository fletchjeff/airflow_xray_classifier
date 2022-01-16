export LATEST_RUN=`ls -Art /home/astro/bentoml/repository/TensorflowXray/ | tail -n 1`

if [[ $(docker ps | awk ' /tensorflow-xray/ { print $1 }') ]]; then
    docker kill `(docker ps | awk ' /tensorflow-xray/ { print $1 }')` && docker run -d -p 5000:5000 tensorflow-xray:$LATEST_RUN
else
    docker run -d -p 5000:5000 tensorflow-xray:$LATEST_RUN
fi