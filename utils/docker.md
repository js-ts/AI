# Docker


- sudo
- docker images ls
- docker container ls
- docker ps

---

- docker run --name [container-name] --shm-size=4g --network localhost -v $PWD:/master -it [image:tag] /bin/bash

- docker attach [container-id]
- docker exec [container-id] -it /bin/bash

---

- docker commit -a "user-name" -m "commit" [container-id] [image:tag]
