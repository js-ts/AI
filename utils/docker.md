# Docker


- sudo
- docker images ls
- docker container ls
- docker ps

---

- docker run --name [container-name] --shm-size=4g --network localhost -v $PWD:/master -it [image:tag] /bin/bash

---
- docker start [container-name]
- docker attach [container-name]
- docker exec -it [container-name] /bin/bash
- exit / ctrl+D


---

- docker commit -a "user-name" -m "commit" [container-id] [image:tag]
