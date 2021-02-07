# Docker


- sudo
- docker images
- docker container ls
- docker ps -a

---

- docker run --name [container-name] --shm-size=4g --network localhost -v $PWD:/master -it [image:tag]/[image-id] /bin/bash

---
- docker start [container-name]
- docker restart [container-name]
- docker attach [container-name]
- docker exec -it [container-name] /bin/bash

- exit / ctrl+D


---

- docker commit -a "user-name" -m "commit" [container-id] [image:tag]
