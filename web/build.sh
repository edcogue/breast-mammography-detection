docker rmi $(docker images | grep mammographies_web)
docker build . -t mammographies_web