docker run -d \
  --name ollaya \
  -p 11435:11435 \
  -e OLLAYA_DEVICE=cpu \
  --memory=4g \
  -v ollaya:/home/ollaya/.ollaya \
  ghcr.io/ollaya-dev/ollaya:latest
