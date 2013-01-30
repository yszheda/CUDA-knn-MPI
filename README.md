# k-nearest neighbor on multiple GPUs using CUDA and MPI
Default setting is for:
*	2 nodes
*	2 GPU cards per node
You may need to alter the code if you want to run the program on a different environment.

## Compile the program
```shell
	$ make
```
## Execute the program
Put the hostnames in a file and then set EXEC_PATH and HOST_FILE in Makefile.
```shell
	$ make run IN="<inputfile>" OUT="<outputfile>"
```

Shuai YUAN <yszheda@gmail.com>
