docker run -p 8080:8080 mammographies_web

docker rm $(docker stop $(docker ps -a -q --filter ancestor=mammographies_web --format="{{.ID}}"))