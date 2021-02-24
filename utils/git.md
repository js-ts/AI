
## ssh
- ls ~/.ssh/
- ssh-keygen -t rsa -b 2048 -C “xxx@gmail.com”
- eval "$(ssh-agent -s)"
- ssh-add ~/.ssh/id_rsa
---
- ssh-keygen -t rsa -C "user.email"
- cat ~/.ssh/id_rsa.pub
- ssh -T git@github.com


## config 
- --global
- git config user.name "xxx"
- git config user.email "xxx"
- git config --global core.editor "vim"

## folk
- sync with orginal repo
- git folk 
- git clone xxx
- git remote add upstream git@github.com:PaddlePaddle/Paddle.git
- git remote -v
---
- git fetch upstream
- git pull upstream develop
---
- git branch -r
- git checkout -b newbranch upstream/branchname
- git checkout -b newbranch
---
- git checkout originlocalbranch
- git diff newbranch
- git merge newbranch
- git push 

---
- git push origin newbranch:newbranch
- git push origin :newbranch

## branch
- git checkout xxx
- git branch -d xxx

## tag
- git tag xx
- git tag -a xx -m 'xx'
- git push origin xx
- git show xx
---
- git tag -d xx
- git push origin :xx
---
- git checkout -b [new-brach-name] [tag-name]

## commit
- git commit -m ""
- git commit --amend


## pull
- Pull is not possible because you have unmerged files.
- git reset --hard FETCH_HEAD
- git pull


## detach

- git branch -b tmp
- git checkout dev
- git merge tmp
- git push /dev




