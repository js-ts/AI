
## ssh
- ls ~/.ssh/
- ssh-keygen -t rsa -b 2048 -C “xxx@gmail.com”
- eval "$(ssh-agent -s)"
- ssh-add ~/.ssh/id_rsa

## config 
- --global
- git config user.name ""
- git config user.email ""
- git config --global core.editor "vim"

## folk
- sync with orginal repo
- git folk 
- git clone xxx
- git remote add upstream git@github.com:PaddlePaddle/Paddle.git
- git remote -v
- git fetch upstream
- git branch -r
- git checkout -b newbranch upstream/branchname
- git checkout originlocalbranch
- git diff newbranch
- git merge newbranch
- git push 

## branch
- git checkout xxx
- git branch -d xxx

## commit
- git commit -m ""
- git commit --amend
