JAEGER_CONTAINER_NAME="jaeger"

if [ $(docker inspect --format '{{json .State.Running}}' ${JAEGER_CONTAINER_NAME}) ]; then
  echo "Jaeger is already running"
else
  echo "Run Jaeger"
  docker run \
    --name $JAEGER_CONTAINER_NAME \
    --rm \
    -d \
    -e COLLECTOR_ZIPKIN_HOST_PORT=:9411 \
    -p 16686:16686 \
    -p 4317:4317 \
    -p 4318:4318 \
    -p 9411:9411 \
    jaegertracing/all-in-one:latest
fi
