docker run -p 8888:8888 -v `pwd`:/home/jovyan/mnt pnp jupyter notebook --ip=0.0.0.0 --NotebookApp.token=\"\" --NotebookApp.disable_check_xsrf=True 
