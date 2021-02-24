# Linux commands

- find ./ -name "*.py" -or -name "*.cpp" | xargs grep "xxx" | wc 

---
- nohup command &
- ps aux 

---
- command &
- jobs -l
- fg %num
- kill %num

---

- scp -r /path/to/file user@ip:/home/path/to/file
- scp -r user@ip:/home/path/to/file /path/to/file


---

 df -lh /


---

## tmux
- tmux new -s L
- tmux detach
- tmux attach -t L
