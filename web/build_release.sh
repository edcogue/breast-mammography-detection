docker rmi $(docker images | grep mammographies_web)
docker build -f Dockerfile.release -t mammographies_web .