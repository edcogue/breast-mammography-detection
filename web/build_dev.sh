docker rmi $(docker images | grep dev_tfm)
docker build . -t dev_tfm